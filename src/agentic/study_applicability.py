"""
study_applicability.py — STUDY-18: does the agent know when its recall promise does not apply?

The promise (``calibration.json``) is fit on three crab-pot recordings. On a sonar the detector has
never seen it can fail silently: on ``test_xsonar`` the validation threshold keeps 11% of the pots. The
agent therefore checks each survey against the calibration recordings before it keeps the promise
(``applicability.py``: OpenCV image statistics → conformal p-value per frame → withdraw the promise
when the 95% lower bound of the atypical share exceeds a limit).

**Protocol (fixed in this file and committed before any verification data was read)**

1. **Fit** on the calibration split only: v2b ``val`` ``*wcp_ss_*`` (163 unique frames, Rec10/12/16) —
   the frames the promise was fit on. Frame rule: atypical if the conformal p ≤ 0.05.
2. **Survey rule** from design data only (v2b ``train``; never test): ``share_limit`` = the largest
   atypical share over the in-distribution training recordings (``Rec*`` grayscale crab-pot recordings
   with ≥ 10 unique frames), rounded **up** to the next 0.05. A survey's promise is withdrawn when the
   95% Clopper-Pearson lower bound of its atypical share exceeds ``share_limit``.
3. **Verify once**: (a) each v2b ``test`` recording (Rec8, Rec9, Rec14) as one survey, and the 8 demo
   frames as one survey; (b) ``test_xsonar`` (555 frames of another sonar) as one survey, plus its
   25-frame chunks (flag rate); (c) the raw recordings ``Test-Small-DS`` (Humminbird, Colorado River) and
   ``Test-Large-DS`` (Solix, Pearl River) as rendered by ``humminbird.frames_and_track`` — reported
   with no gate (no labels: whether the promise holds there is unknown).
4. **Gates** — G1: no test recording and not the demo frames is flagged. G2: ``test_xsonar`` is
   flagged. Only if both pass is ``models/<MODEL>/applicability.json`` written; the agent then checks
   every survey by default (``core.py``).
5. Context, not a gate: recall at the standard threshold on ``test_xsonar`` from the cached, once-run
   predictions — what the promise would have claimed there.

Disclosed before the run: the eight features and their scale floors were chosen on design data
(calibration + training splits). Colour saturation was added after the first design look, when the
grey-only features left colour-palette crab-pot surveys (baycove, BC_POST) inside the in-distribution
range (atypical share 0.12–0.18 vs ≤ 0.14); with it they reached 1.00.

Usage:
    python -m src.agentic.study_applicability              # fit + design (calibration + train only)
    python -m src.agentic.study_applicability --verify     # + verification, once (+ raw recordings)
    python -m src.agentic.study_applicability --verify --write   # + applicability.json if G1 and G2 pass
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.detection.calibration import DEFAULT_MODEL, load_calibration
from src.detection.frames import recording_of, unique_frames
from .applicability import Reference, frame_stats, reference_path
from .calibrate import MATCH_IOU
from src.detection.infer import _iou

REPO = Path(__file__).resolve().parents[2]
V2B = REPO / "DATASET" / "03_yolo_ready_dataset_v2b"
SAMPLES = REPO / "webui" / "samples"
REPORT = REPO / "docs" / "reports" / "applicability.md"
ALPHA_FRAME, DELTA, MIN_ID_FRAMES, CHUNK = 0.05, 0.05, 10, 25
OTHER_SOURCES = ("seabed", "TI", "baycove", "shipwreck", "BB_", "BC_POST")


def stats_of(paths) -> list[dict]:
    out = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is not None:
            out.append(frame_stats(im))
    return out


def train_groups() -> dict[str, list[Path]]:
    """Design data: training frames grouped by recording (grayscale crab-pot ``Rec*``) or source."""
    g = defaultdict(list)
    for p in unique_frames(sorted((V2B / "train" / "images").glob("*.jpg"))):
        r = recording_of(p.name)
        if r and "wcp_ss" in p.name:
            g[r].append(p)
        else:
            src = next((s for s in OTHER_SOURCES if s in p.name), "other")
            g["source:" + src.strip("_")].append(p)
    return dict(g)


def raw_frames(dat: Path) -> list:
    from src.cv_pipeline.humminbird import frames_and_track, read_recording
    frames, _, _ = frames_and_track(read_recording(dat), crosscheck=False)
    return [im for _, im in frames]


def xsonar_recall_at_tau(model: str, tau: float) -> dict | None:
    """Recall at the standard threshold on test_xsonar, from the cached once-run predictions."""
    p = REPO / "runs" / "calib" / f"{model}_test_xsonar.json"
    if not p.exists():
        return None
    frames = json.loads(p.read_text())["frames"]
    pots = []
    for fr in frames:
        for g in fr["gt"]:
            ov = [c["conf"] for c in fr["cands"] if _iou(tuple(c["bbox"]), tuple(g)) >= MATCH_IOU]
            pots.append(max(ov) if ov else 0.0)
    pots = np.asarray(pots)
    return {"pots": int(len(pots)), "frames": len(frames), "recall_at_tau": round(float(np.mean(pots >= tau)), 4)}


def report(res: dict) -> str:
    L = ["# STUDY-18 — does the promise apply here? (applicability check)\n",
         "_Generated by `python -m src.agentic.study_applicability --verify` · protocol fixed in the module docstring "
         "and committed before any verification data was read · per-frame conformal p-value vs the calibration "
         f"recordings, atypical if p ≤ {ALPHA_FRAME} · survey rule: withdraw the promise when the 95% lower bound of "
         "the atypical share exceeds the limit_\n",
         "Features (OpenCV 5, frame resized to 320²): brightness, contrast, speckle texture (`cv2.Laplacian`), "
         "along- vs across-track gradient (`cv2.Sobel`), dark and saturated pixel shares, colour saturation "
         "(`cv2.cvtColor` → HSV), grey-histogram distance to the calibration recordings (`cv2.calcHist`, "
         "`cv2.compareHist` Bhattacharyya).\n",
         "## 1. Fit and survey rule (calibration + training splits only)\n",
         f"Reference: {res['fit']['split']}. Share limit = the largest atypical share over the in-distribution "
         f"training recordings, rounded up to 0.05: **{res['fit']['share_limit']:.2f}**.\n",
         "| training group | frames | atypical share | 95% LCB | promise |", "|---|--:|--:|--:|:--:|"]
    for name, r in sorted(res["design"].items(), key=lambda kv: (kv[0].startswith("source:"), kv[0])):
        L.append(f"| {name} | {r['frames']} | {r['share']:.3f} | {r['share_lcb']:.3f} | "
                 f"{'applies' if r['applies'] else '**withdrawn**'} |")
    if res.get("verify"):
        v = res["verify"]
        L += ["", "## 2. Verification (read once)\n",
              "| survey | frames | atypical share | 95% LCB | promise | expected |", "|---|--:|--:|--:|:--:|:--:|"]
        for name, r, exp in v["rows"]:
            L.append(f"| {name} | {r['frames']} | {r['share']:.3f} | {r['share_lcb']:.3f} | "
                     f"{'applies' if r['applies'] else '**withdrawn**'} | {exp} |")
        if v.get("xsonar_chunks"):
            c = v["xsonar_chunks"]
            L.append(f"\n`test_xsonar` in {CHUNK}-frame surveys: the promise is withdrawn in **{c['flagged']} of "
                     f"{c['n']}** chunks.")
        if v.get("xsonar_recall"):
            x = v["xsonar_recall"]
            L.append(f"\nWhat the check prevents: on `test_xsonar` the standard threshold (τ = {res['tau']:.3f}) keeps "
                     f"**{x['recall_at_tau']:.1%}** of {x['pots']} pots while the tiers would print "
                     f"\"≥ {res['promise']:.0%}\" (cached once-run predictions, {x['frames']} frames).")
        L += ["", f"**G1** — no test recording and not the demo frames flagged: **{'PASS' if res['gates']['G1'] else 'FAIL'}**",
              f"**G2** — `test_xsonar` flagged: **{'PASS' if res['gates']['G2'] else 'FAIL'}**\n"]
    else:
        L += ["", "## 2. Verification\n", "_Not read yet (`--verify`)._\n"]
    L += ["## 3. Decision\n", res["decision"], "",
          "Limits: a coarse guard against a different sonar or rendering. It does not detect a subtle site or gain "
          "change inside the same sonar family, and a recording that passes is not proof that the promise holds "
          "there — only that the frames look like those it was fit on. Frames of one recording are correlated "
          "(the Clopper-Pearson bound assumes independence).", "", "---",
          "_Regenerated by the script; do not edit by hand._"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--verify", action="store_true", help="read the verification data (once)")
    ap.add_argument("--write", action="store_true", help="write applicability.json if G1 and G2 pass")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cal = load_calibration(a.model)
    tau = float(cal.tiers["tau_review"])
    promise = float(cal.tiers["guarantees"]["recall_promise"])

    cal_paths = unique_frames(sorted((V2B / "val" / "images").glob("*wcp_ss_*.jpg")))
    recs = sorted({recording_of(p.name) for p in cal_paths if recording_of(p.name)})
    split = f"`DATASET/03_yolo_ready_dataset_v2b/val` *wcp_ss_* — {len(cal_paths)} unique frames, recordings {recs}"
    print(f"fit: {len(cal_paths)} calibration frames")
    ref = Reference.fit(stats_of(cal_paths), ALPHA_FRAME, 1.0, DELTA, split=split, model=a.model)

    print("design: training groups")
    groups = {k: stats_of(v) for k, v in train_groups().items()}
    shares = {k: ref.check(s) for k, s in groups.items() if s}
    id_shares = [r["share"] for k, r in shares.items() if not k.startswith("source:") and r["frames"] >= MIN_ID_FRAMES]
    limit = math.ceil(max(id_shares) / 0.05 - 1e-9) * 0.05
    ref = Reference({**ref.data, "share_limit": round(limit, 2)})
    design = {k: {kk: vv for kk, vv in ref.check(groups[k]).items() if kk != "p_values"} for k in shares}
    res = {"fit": {"split": split, "share_limit": ref.share_limit}, "design": design, "tau": tau, "promise": promise,
           "gates": {"G1": None, "G2": None}}

    if a.verify:
        print("verification (read once)")
        rows = []
        test = unique_frames(sorted((V2B / "test" / "images").glob("*wcp_ss_*.jpg")))
        by = defaultdict(list)
        for p in test:
            by[recording_of(p.name)].append(p)
        g1 = True
        for r in sorted(by):
            c = ref.check(stats_of(by[r]))
            rows.append((f"test {r}", c, "applies"))
            g1 &= c["applies"]
        c = ref.check(stats_of(sorted(SAMPLES.glob("*.jpg"))))
        rows.append(("the 8 demo frames (test Rec9)", c, "applies"))
        g1 &= c["applies"]
        xs = stats_of(unique_frames(sorted((V2B / "test_xsonar" / "images").glob("*.jpg"))))
        cx = ref.check(xs)
        rows.append(("test_xsonar (another sonar)", cx, "withdrawn"))
        chunks = [ref.check(xs[i:i + CHUNK]) for i in range(0, len(xs) - CHUNK + 1, CHUNK)]
        for name, dat in (("Test-Small-DS (Humminbird, Colorado River)", REPO / "DATASET" / "external" / "pingmapper_sample" / "Test-Small-DS.DAT"),
                          ("Test-Large-DS (Solix, Pearl River)", REPO / "DATASET" / "external" / "pingmapper_large" / "Test-Large-DS.DAT")):
            if dat.exists():
                print(f"  raw recording {dat.name}")
                rows.append((name, ref.check([frame_stats(im) for im in raw_frames(dat)]), "reported"))
        res["verify"] = {"rows": [(n, {k: v for k, v in c.items() if k != "p_values"}, e) for n, c, e in rows],
                         "xsonar_chunks": {"n": len(chunks), "flagged": sum(not c["applies"] for c in chunks)},
                         "xsonar_recall": xsonar_recall_at_tau(a.model, tau)}
        res["gates"] = {"G1": bool(g1), "G2": not cx["applies"]}
    ok = bool(res["gates"]["G1"] and res["gates"]["G2"])
    res["decision"] = ("**Ship**: the agent checks every survey and withdraws the recall promise where the frames do "
                       "not look like the calibration recordings, then asks a person to approve a full review."
                       if ok else ("**Not shipped**: a gate failed." if a.verify else "_Pending verification._"))
    REPORT.write_text(report(res), encoding="utf-8")
    out = REPO / "runs" / "applicability" / f"{a.model}_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"share_limit": ref.share_limit, "gates": res["gates"], "decision": res["decision"]}, indent=1))
    if a.write and ok:
        ref.data["verified"] = {n: {"applies": c["applies"], "share": c["share"], "share_lcb": c["share_lcb"]}
                                for n, c, _ in res["verify"]["rows"]}
        ref.data["report"] = REPORT.relative_to(REPO).as_posix()
        print(f"wrote {ref.save(reference_path(a.model))}")


if __name__ == "__main__":
    main()
