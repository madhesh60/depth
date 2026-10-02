"""
study_scale_tta.py — STUDY-13: does EXP-001 find more pots with a larger input or a flipped second
view? (no retraining; informs EXP-002's input size)

EXP-001 (YOLO11s, trained at 640) on its unseen crab-pot sonograms, unique frames, through the deploy
path (``YoloOnnxDetector`` / ``cv2.dnn``), detector floor 0.05, match IoU ≥ 0.3 (as calibration):

* **input size** 640 (deployed) · 800 · 960 · 1024 — the same weights exported at each static size;
* **flip TTA** — a second pass on the frame mirrored along-track (pings reversed: physically valid for a
  side-scan sonogram, unlike a vertical flip, which would swap near and far range), mapped back and
  fused two ways: ``union`` (class NMS, max confidence) and ``mean`` (a box and its flipped twin
  averaged; a box the other view does not confirm keeps half its confidence — a full-frame re-look).

Protocol (fixed before looking): every variant is scored on the **calibration split** (v1 val, Rec19)
against the deployed 640/single baseline with a paired frame-bootstrap 95% CI. A variant is adopted
only if its AP@0.3 gain has CI lower bound > 0 AND its recall ceiling is not lower. Only then is the
**verification split** (v1 test) run — once, for the winner and the baseline. Latency is measured on
the same run (forward passes only, this machine).

    python -m src.detection.study_scale_tta [--sizes 640,800,960,1024]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from src.agentic.calibrate import FLOOR, MATCH_IOU, V1, _gt, list_frames
from src.detection.evaluate import voc_ap
from src.detection.infer import DEFAULT_ONNX, Detection, YoloOnnxDetector, nms, _iou

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "study13"
PATTERN = "crabpot_*wcp_ss_*.jpg"
CLS = "fishing_gear"


def onnx_for(size: int) -> Path:
    if size == 640:
        return DEFAULT_ONNX
    p = OUT / f"exp001_{size}.onnx"
    if not p.exists():                               # export a COPY of the weights at this static size
        from ultralytics import YOLO
        OUT.mkdir(parents=True, exist_ok=True)
        pt = OUT / "exp001_copy.pt"
        shutil.copy(DEFAULT_ONNX.parent / "best.pt", pt)
        made = Path(YOLO(str(pt)).export(format="onnx", imgsz=size, opset=12, dynamic=False, simplify=False))
        made.replace(p)
    return p


def run_split(split: str, size: int) -> list[dict]:
    cache = OUT / f"{split}_{size}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    det = YoloOnnxDetector(onnx_for(size), imgsz=size, conf_thres={CLS: FLOOR})
    det.warmup()
    root = V1 / split
    out = []
    for p in list_frames(root, PATTERN):
        img = cv2.imread(str(p))
        if img is None:
            continue
        H, W = img.shape[:2]
        t0 = time.perf_counter()
        a = [d for d in det.detect(img) if d.cls_name == CLS]
        t1 = time.perf_counter()
        b = [d for d in det.detect(cv2.flip(img, 1)) if d.cls_name == CLS]
        t2 = time.perf_counter()
        fb = [[W - d.bbox[2], d.bbox[1], W - d.bbox[0], d.bbox[3], d.conf] for d in b]    # back to the frame
        out.append({"name": p.name, "gt": _gt(root / "labels" / f"{p.stem}.txt", W, H, 0),
                    "orig": [[*d.bbox, d.conf] for d in a], "flip": fb,
                    "ms": [round((t1 - t0) * 1000, 1), round((t2 - t1) * 1000, 1)]})
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def fuse(fr: dict, mode: str) -> list[list[float]]:
    a, b = fr["orig"], fr["flip"]
    if mode == "single":
        return a
    if mode == "union":
        allx = a + b
        if not allx:
            return []
        boxes = np.array([[x[0], x[1], x[2] - x[0], x[3] - x[1]] for x in allx], np.float32)
        confs = np.array([x[4] for x in allx], np.float32)
        keep = nms(boxes, confs, np.zeros(len(allx), np.int32), 0.0, 0.45, True)
        return [allx[i] for i in keep]
    # mean: pair each box with its flipped twin (IoU ≥ 0.5, greedy by confidence)
    used, out = set(), []
    for x in sorted(a, key=lambda x: -x[4]):
        j = max(((i, _iou(x[:4], y[:4])) for i, y in enumerate(b) if i not in used), key=lambda t: t[1], default=(None, 0))
        if j[0] is not None and j[1] >= 0.5:
            used.add(j[0])
            y = b[j[0]]
            out.append([(x[k] + y[k]) / 2 for k in range(4)] + [(x[4] + y[4]) / 2])
        else:
            out.append(x[:4] + [x[4] / 2])
    out += [y[:4] + [y[4] / 2] for i, y in enumerate(b) if i not in used]
    return out


def score(frames: list[dict], mode: str) -> dict:
    pts, n, found, cands = [], 0, 0, 0
    per = []
    for fr in frames:
        dets = fuse(fr, mode)
        gts = fr["gt"]
        n += len(gts)
        cands += len(dets)
        used, fpts = set(), []
        for d in sorted(dets, key=lambda d: -d[4]):
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if j not in used and _iou(d[:4], g) > best:
                    best, bj = _iou(d[:4], g), j
            if bj >= 0 and best >= MATCH_IOU:
                used.add(bj); fpts.append((d[4], 1))
            else:
                fpts.append((d[4], 0))
        reach = sum(1 for g in gts if any(_iou(d[:4], g) >= MATCH_IOU for d in dets))
        found += reach
        pts += fpts
        per.append({"pts": fpts, "n": len(gts), "reach": reach})
    ms = [sum(fr["ms"]) if mode != "single" else fr["ms"][0] for fr in frames]
    return {"pots": n, "ceiling": found / max(1, n), "ap30": voc_ap(pts, n), "cands_per_frame": cands / max(1, len(frames)),
            "ms_p50": float(np.median(ms)) if ms else None, "_per": per}


def paired_ci(base: dict, var: dict, reps: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    B, V = base["_per"], var["_per"]
    d_ap, d_ceil = [], []
    for _ in range(reps):
        idx = rng.integers(0, len(B), len(B))
        agg = lambda P: (voc_ap([p for i in idx for p in P[i]["pts"]], sum(P[i]["n"] for i in idx)),
                         sum(P[i]["reach"] for i in idx) / max(1, sum(P[i]["n"] for i in idx)))
        (ab, cb), (av, cv) = agg(B), agg(V)
        d_ap.append(av - ab); d_ceil.append(cv - cb)
    q = lambda v: [round(float(np.mean(v)), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"d_ap30": q(d_ap), "d_ceiling": q(d_ceil)}


def clean(r: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("_")}


def report(out: dict) -> str:
    fmt = lambda ci: f"{ci[0]:+.3f} ({ci[1]:+.3f} .. {ci[2]:+.3f})"
    L = ["# STUDY-13 — Larger input and a flipped second view for EXP-001 (no retraining)", "",
         "_`python -m src.detection.study_scale_tta` · EXP-001 (trained at 640) exported at each size, run through "
         "`cv2.dnn` · floor 0.05 · match IoU ≥ 0.3 · unique frames · paired frame-bootstrap 95% CI · "
         "regenerated by the script_", "",
         "**Protocol (fixed before looking):** every variant is scored on the **calibration split** (v1 val, "
         "Rec19) against the deployed 640 / single pass; a variant qualifies only if its AP@0.3 gain has a CI "
         "lower bound > 0 and its recall ceiling does not drop. Only the best qualifier is then run **once** on "
         "the **verification split** (v1 test).", "",
         "Flip = the frame mirrored along-track (pings reversed; physically valid for a side-scan sonogram). "
         "`union` = both views, class NMS; `mean` = a box and its flipped twin averaged, an unconfirmed box keeps "
         "half its confidence (a full-frame re-look).", "",
         "## Calibration split (v1 val, Rec19)", "",
         "| variant | recall ceiling | AP@0.3 | candidates / frame | Δ AP@0.3 (95% CI) | Δ ceiling (95% CI) |",
         "|---|--:|--:|--:|:--:|:--:|"]
    for k, v in out["val"].items():
        vb = v.get("vs_baseline")
        L.append(f"| {k}{' (deployed)' if k == '640/single' else ''}{' **← qualifier**' if k == out['winner'] else ''} | "
                 f"{v['ceiling']:.3f} | {v['ap30']:.3f} | {v['cands_per_frame']:.2f} | "
                 f"{fmt(vb['d_ap30']) if vb else '—'} | {fmt(vb['d_ceiling']) if vb else '—'} |")
    L.append("")
    if out.get("winner"):
        t, w = out["test"], out["winner"]
        b, x, d = t["640/single"], t[w], t["vs_baseline"]
        held = d["d_ap30"][1] > 0 and x["ceiling"] >= b["ceiling"]
        L += [f"## Verification split (v1 test) — run once: {w} vs the deployed 640 / single", "",
              "| | recall ceiling | AP@0.3 | candidates / frame |", "|---|--:|--:|--:|",
              f"| 640 / single (deployed) | {b['ceiling']:.3f} | {b['ap30']:.3f} | {b['cands_per_frame']:.2f} |",
              f"| {w} | {x['ceiling']:.3f} | {x['ap30']:.3f} | {x['cands_per_frame']:.2f} |",
              f"| Δ (95% CI) | {fmt(d['d_ceiling'])} | {fmt(d['d_ap30'])} | |", "",
              "## Decision", "",
              ("**Adopted.**" if held else
               f"**Not adopted — the gain did not hold on unseen data.** On the calibration recording (Rec19, the "
               f"hardest: ceiling 0.72) the larger input and the flipped view reach more pots (+0.08 ceiling, +0.10 AP). "
               f"On the verification recordings the ceiling barely moves ({fmt(d['d_ceiling'])}) and AP falls "
               f"({fmt(d['d_ap30'])}): the extra candidates are mostly false alarms. It would also cost ~3× the "
               f"compute (two passes at 1.56× the pixels). Same lesson as STUDY-07 — a second look that helps one "
               f"recording is not a product change until it holds on another."), "",
              "**For EXP-002:** test-time upscaling helps only the hard recording, so a 1024-px model is not a safe bet "
              "by default. Train both 640 and 1024 on v2b and let the held-out recordings (Rec10/12/16) and the "
              "latency gate decide (`docs/exp003_kaggle.md`).", "",
              "_Outcome (2026-09-30): both sizes were trained (EXP-002 / EXP-002s). Both were underfit and tied "
              "on validation (ghost AP 0.25 vs 0.25; `docs/exp002_diagnosis.md`), so EXP-003 trains at 640 px and "
              "the resolution question stays open._", ""]
    L += ["---", "_Latency is not reported: this run shared the laptop with other work (640-px forwards ranged "
          "270–1650 ms), so only the relative cost (passes × pixels) is used above._", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--sizes", default="640,800,960,1024")
    ap.add_argument("--report-only", action="store_true", help="rewrite docs/scale_tta.md from runs/study13/result.json")
    a = ap.parse_args()
    if a.report_only:
        (REPO / "docs" / "scale_tta.md").write_text(report(json.loads((OUT / "result.json").read_text())), encoding="utf-8")
        print("wrote docs/scale_tta.md")
        return
    sizes = [int(s) for s in a.sizes.split(",")]
    res = {"protocol": "select on v1 val (Rec19) with paired bootstrap; verify once on v1 test", "val": {}, "test": {}}
    base_val = None
    for s in sizes:
        print(f"val @ {s}px ...", flush=True)
        fr = run_split("val", s)
        for mode in ("single", "union", "mean"):
            r = score(fr, mode)
            if s == 640 and mode == "single":
                base_val = r
            res["val"][f"{s}/{mode}"] = r
    for k, r in res["val"].items():
        r["vs_baseline"] = paired_ci(base_val, r) if k != "640/single" else None
    ok = {k: r for k, r in res["val"].items() if k != "640/single" and r["vs_baseline"]["d_ap30"][1] > 0
          and r["ceiling"] >= base_val["ceiling"]}
    winner = max(ok, key=lambda k: ok[k]["ap30"]) if ok else None
    res["winner"] = winner
    print("winner on the calibration split:", winner, flush=True)
    if winner:
        s_w, m_w = winner.split("/")
        base_t = score(run_split("test", 640), "single")
        win_t = score(run_split("test", int(s_w)), m_w)
        res["test"] = {"640/single": base_t, winner: win_t, "vs_baseline": paired_ci(base_t, win_t)}
    out = {"protocol": res["protocol"], "winner": winner,
           "val": {k: {**clean(r), "vs_baseline": r.get("vs_baseline")} for k, r in res["val"].items()},
           "test": {k: (clean(r) if isinstance(r, dict) and "_per" in r else r) for k, r in res["test"].items()}}
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    (REPO / "docs" / "scale_tta.md").write_text(report(out), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
