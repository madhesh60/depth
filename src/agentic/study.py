"""
study.py — the timed user study that turns "minutes saved" from an assumption into a measurement
(review X-2 / C: "measure the minutes for real").

Protocol (built into the app's **Study** mode, ~5–10 minutes per person):

* frames: the labelled sample frames (``webui/samples`` ships 8; set ``$DEPTH_STUDY_DIR`` to a folder
  of labelled frames for a bigger study);
* **counterbalanced**: the frames are split into two halves; each participant reviews one half
  **manually** (the frame only — click every pot) and the other half as **DEPTH cards** (the agent's
  REVIEW/CONFIRMED cards — real pot / not a pot), so nobody sees a frame twice. Which half goes to
  which arm, and which arm comes first, rotate with the session number (a 2×2 Latin square);
* the browser times every frame and every card (ms, from display to answer);
* scoring is server-side against ground truth: manual clicks within a labelled box (+8 px) find that
  pot (one click per pot); a card is real if it overlaps a labelled pot at IoU ≥ 0.3.

Per session → ``runs/study/<session>.json``; :func:`summary` pools sessions → median s/frame and
s/card, manual recall, card accuracy, DEPTH-arm recall (bounded by what the detector proposes) and
minutes per survey-hour both ways. ``src/agentic/effort.py`` uses these instead of its assumptions.
"""
from __future__ import annotations

import json
import os
import statistics
import time
import uuid
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
CLICK_TOL = 8            # px a click may fall outside a labelled box and still count
CARD_IOU = 0.3


def study_dir() -> Path:
    return Path(os.environ.get("DEPTH_STUDY_RESULTS", REPO / "runs" / "study"))


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _gt_from(img: Path, labels: Path) -> Optional[list]:
    """Class-0 (pot) boxes in pixels from a YOLO label file next to the image; None if unlabelled."""
    lp = labels / f"{img.stem}.txt"
    if not lp.exists():
        return None
    import cv2
    im = cv2.imread(str(img))
    if im is None:
        return None
    H, W = im.shape[:2]
    out = []
    for line in lp.read_text().splitlines():
        q = line.split()
        if len(q) >= 5 and int(float(q[0])) == 0:
            cx, cy, bw, bh = map(float, q[1:5])
            out.append((int((cx - bw / 2) * W), int((cy - bh / 2) * H), int((cx + bw / 2) * W), int((cy + bh / 2) * H)))
    return out


def study_frames() -> list[dict]:
    """Labelled frames for the study: [{frame_id, sample_id, path, gt}] — ``$DEPTH_STUDY_DIR``
    (``images/`` + ``labels/``) for a bigger study, else the shipped labelled samples."""
    sd = os.environ.get("DEPTH_STUDY_DIR")
    out = []
    if sd:
        root = Path(sd)
        for ip in sorted((root / "images").glob("*.jpg")):
            gt = _gt_from(ip, root / "labels")
            if gt is not None:
                out.append({"sample_id": None, "frame_id": ip.stem, "path": str(ip), "gt": gt})
        return out
    from src.dashboard import samples as samples_mod
    for smp in samples_mod.list_samples():
        gt = samples_mod.sample_gt(smp["id"])
        if gt is not None:
            out.append({"sample_id": smp["id"], "frame_id": samples_mod.frame_id(smp["id"]),
                        "path": str(samples_mod.sample_path(smp["id"])), "gt": gt})
    return out


def frame_path(frame_id: str) -> Optional[Path]:
    for f in study_frames():
        if f["frame_id"] == frame_id:
            return Path(f["path"])
    return None


def sessions() -> list[dict]:
    d = study_dir()
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def plan(participant: str) -> dict:
    """Assign halves + order for the next session (2×2 Latin square over session index)."""
    frames = study_frames()
    if len(frames) < 2:
        raise ValueError("the study needs at least 2 labelled frames")
    n = len(sessions())
    half_a, half_b = frames[0::2], frames[1::2]
    manual, cards = (half_a, half_b) if n % 2 == 0 else (half_b, half_a)
    order = ["manual", "cards"] if (n // 2) % 2 == 0 else ["cards", "manual"]
    return {"session_id": f"s{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}",
            "participant": (participant or "anon")[:40], "index": n, "order": order,
            "manual_frames": [{"frame_id": f["frame_id"]} for f in manual],
            "card_frames": [{"frame_id": f["frame_id"]} for f in cards]}


def score(p: dict, result: dict, cards_by_id: dict[str, dict]) -> dict:
    """Score one session against ground truth. ``cards_by_id`` = the cards the server showed."""
    gt = {f["frame_id"]: f["gt"] for f in study_frames()}
    # ---- manual arm ------------------------------------------------------------------------
    m_found = m_gt = m_fp = 0
    m_ms = []
    for fr in result.get("manual", []):
        g = gt.get(fr.get("frame_id"), [])
        m_gt += len(g)
        used = set()
        for x, y in fr.get("clicks", []):
            hit = next((j for j, b in enumerate(g) if j not in used and
                        b[0] - CLICK_TOL <= x <= b[2] + CLICK_TOL and b[1] - CLICK_TOL <= y <= b[3] + CLICK_TOL), None)
            if hit is None:
                m_fp += 1
            else:
                used.add(hit)
        m_found += len(used)
        if fr.get("ms"):
            m_ms.append(float(fr["ms"]))
    # ---- DEPTH cards arm --------------------------------------------------------------------
    c_ms, conf_real, conf_fake, dism_real, dism_fake = [], 0, 0, 0, 0
    # every pot in the card-arm frames counts — also those in frames where the detector produced no
    # card at all (otherwise DEPTH's recall would silently exclude its misses)
    card_frames = {f["frame_id"] for f in p.get("card_frames", [])} or {c["frame_id"] for c in cards_by_id.values()}
    c_gt = sum(len(gt.get(f, [])) for f in card_frames)
    matched: dict[str, set] = {}
    for ans in result.get("cards", []):
        c = cards_by_id.get(ans.get("card_id"))
        if not c:
            continue
        g = gt.get(c["frame_id"], [])
        used = matched.setdefault(c["frame_id"], set())
        j = max(range(len(g)), key=lambda k: _iou(c["bbox"], g[k]), default=None)
        real = j is not None and j not in used and _iou(c["bbox"], g[j]) >= CARD_IOU
        if real:
            used.add(j)
        yes = ans.get("decision") == "real"
        conf_real += yes and real; conf_fake += yes and not real
        dism_real += (not yes) and real; dism_fake += (not yes) and not real
        if ans.get("ms"):
            c_ms.append(float(ans["ms"]))
    real_cards = conf_real + dism_real
    return {
        "session_id": p["session_id"], "participant": p["participant"], "index": p["index"], "order": p["order"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manual": {"frames": len(result.get("manual", [])), "pots": m_gt, "found": m_found, "false_clicks": m_fp,
                   "recall": round(m_found / m_gt, 3) if m_gt else None,
                   "sec_per_frame": round(statistics.median(m_ms) / 1000, 2) if m_ms else None,
                   "total_s": round(sum(m_ms) / 1000, 1)},
        "cards": {"frames": len(card_frames), "cards": len(result.get("cards", [])), "pots": c_gt,
                  "confirmed_real": conf_real, "confirmed_fake": conf_fake,
                  "dismissed_real": dism_real, "dismissed_fake": dism_fake,
                  "accuracy_on_real": round(conf_real / real_cards, 3) if real_cards else None,
                  "recall": round(conf_real / c_gt, 3) if c_gt else None,
                  "sec_per_card": round(statistics.median(c_ms) / 1000, 2) if c_ms else None,
                  "total_s": round(sum(c_ms) / 1000, 1)},
        "raw": result,
    }


def save(rec: dict) -> Path:
    d = study_dir(); d.mkdir(parents=True, exist_ok=True)
    p = d / f"{rec['session_id']}.json"
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return p


def summary(frames_per_survey_hour: Optional[float] = None, cards_per_frame: Optional[float] = None) -> dict:
    """Pool every session: medians of the per-session timings, pooled recalls, per-survey-hour minutes."""
    ss = sessions()
    if not ss:
        return {"sessions": 0}

    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.median(xs), 2) if xs else None

    m_found = sum(s["manual"]["found"] for s in ss); m_gt = sum(s["manual"]["pots"] for s in ss)
    c_real = sum(s["cards"]["confirmed_real"] for s in ss)
    c_realcards = sum(s["cards"]["confirmed_real"] + s["cards"]["dismissed_real"] for s in ss)
    c_gt = sum(s["cards"]["pots"] for s in ss)
    spf, spc = med([s["manual"]["sec_per_frame"] for s in ss]), med([s["cards"]["sec_per_card"] for s in ss])
    out = {"sessions": len(ss), "participants": len({s["participant"] for s in ss}),
           "manual": {"sec_per_frame": spf, "recall": round(m_found / m_gt, 3) if m_gt else None,
                      "false_clicks_per_frame": round(sum(s["manual"]["false_clicks"] for s in ss)
                                                      / max(1, sum(s["manual"]["frames"] for s in ss)), 2)},
           "cards": {"sec_per_card": spc, "accuracy_on_real": round(c_real / c_realcards, 3) if c_realcards else None,
                     "recall": round(c_real / c_gt, 3) if c_gt else None,
                     "cards_per_frame": round(sum(s["cards"]["cards"] for s in ss)
                                              / max(1, sum(s["cards"]["frames"] for s in ss)), 2)}}
    fph = frames_per_survey_hour or 15.0 * 3600 / 640 * 2
    cpf = cards_per_frame or out["cards"]["cards_per_frame"]
    if spf and spc:
        out["per_survey_hour_min"] = {"manual": round(fph * spf / 60, 1), "depth": round(fph * cpf * spc / 60, 1)}
        out["speedup"] = round(spf / (cpf * spc), 2) if cpf and spc else None
    out["caveat"] = ("small sample" if len(ss) < 3 else "pooled") + \
        "; frames are the shipped labelled samples unless $DEPTH_STUDY_DIR is set"
    return out
