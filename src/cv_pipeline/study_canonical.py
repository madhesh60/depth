"""
study_canonical.py — STUDY-08: does Stage-1 canonicalisation measure real sonar geometry, and does
it help the deployed detector?

Part A — bottom tracking, no labels needed, on **un-augmented originals only**: the raw Hugging Face
archive (all splits) holds 1–12 Roboflow copies per frame, and crop/zoom copies change the pixel
scale (an altitude in px is only meaningful on the original geometry). A frame whose key has exactly
ONE copy in the archive is an original; the port/starboard check uses pairs where both are originals.
  * coverage: share of frames where a water-column → seabed step is found;
  * **port/starboard agreement**: both channels are recorded on the same pings, so a frame pair with
    the same recording + chunk index must show the same altitude (an independent physical check);
  * along-track continuity: consecutive chunks of one channel should change altitude smoothly;
  * per-op latency.
Part B — with EXP-001 on its unseen crab-pot sonograms (v1 val Rec19 + v1 test, unique frames):
  * water column: where do labelled pots and detector candidates (TP/FP) sit relative to the seabed?
  * detector input ``raw`` vs ``gray`` vs ``gain`` (range-gain normalised): recall ceiling at the
    detector floor (conf 0.05) and AP@0.3/0.5 for crab pots, with a paired frame-bootstrap CI.

Writes ``docs/stage1_canonical.md`` + ``runs/stage1/study.json``.

    python -m src.cv_pipeline.study_canonical [--skip-detector]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.detection.frames import unique_frames, frame_key
from .canonical import Canonicaliser

REPO = Path(__file__).resolve().parents[2]
V1 = REPO / "DATASET" / "03_yolo_ready_dataset_v1"
RAW = REPO / "DATASET" / "01_raw_archives" / "crab-pot-dataset"
PAT = "*wcp_ss_*.jpg"
# series = everything before "_wcp" (keeps e.g. "Rec09_Sensor_Depth" apart from "Rec9"), side, chunk
_KEY = re.compile(r"^(.*?)_wcp_ss_(port|star)\w*?_0*(\d+)", re.I)


def _parse(name: str):
    m = _KEY.search(frame_key(name))
    return (m.group(1), m.group(2).lower(), int(m.group(3))) if m else None


# ============================================================================ Part A
def part_a(C: Canonicaliser) -> dict:
    copies: dict[tuple, list[Path]] = defaultdict(list)
    for split in ("train", "valid", "test"):
        for p in sorted((RAW / split).rglob(PAT)):
            k = _parse(p.name)
            if k:
                copies[k].append(p)
    originals = {k: ps[0] for k, ps in copies.items() if len(ps) == 1}
    alt: dict[tuple, float] = {}
    n_meas, times = 0, defaultdict(list)
    for k, p in sorted(originals.items()):
        im = cv2.imread(str(p))
        if im is None:
            continue
        cf = C.process(im, p.stem)
        for op, ms in cf.timings_ms.items():
            times[op].append(ms)
        if cf.measured:
            n_meas += 1
            alt[k] = cf.altitude_px
    keys = set(originals)

    # port / starboard agreement at the same (recording, chunk)
    pairs = [(alt[(r, "port", c)], alt[(r, "star", c)]) for (r, s, c) in alt
             if s == "port" and (r, "star", c) in alt]
    d_pair = np.array([abs(a - b) for a, b in pairs])
    # null: port vs starboard of a DIFFERENT recording (what agreement looks like by chance)
    rng = np.random.default_rng(0)
    ports = [(r, v) for (r, s, c), v in alt.items() if s == "port"]
    stars = [(r, v) for (r, s, c), v in alt.items() if s == "star"]
    null = []
    for _ in range(2000):
        (ra, va), (rb, vb) = ports[rng.integers(len(ports))], stars[rng.integers(len(stars))]
        if ra != rb:
            null.append(abs(va - vb))
    null = np.array(null)
    # continuity: consecutive chunks of one channel
    d_next = np.array([abs(alt[(r, s, c)] - alt[(r, s, c + 1)]) for (r, s, c) in alt if (r, s, c + 1) in alt])
    alts = np.array(list(alt.values()))
    corr = float(np.corrcoef(np.array(pairs).T)[0, 1]) if len(pairs) > 2 else None
    return {
        "frames": len(keys), "keys_total": len(copies), "measured": n_meas, "coverage": round(n_meas / max(1, len(keys)), 3),
        "altitude_px": {"median": round(float(np.median(alts)), 1), "p05": round(float(np.percentile(alts, 5)), 1),
                        "p95": round(float(np.percentile(alts, 95)), 1)},
        "pairs": len(pairs), "pair_abs_diff_median": round(float(np.median(d_pair)), 2) if len(d_pair) else None,
        "pair_within_2px": round(float((d_pair <= 2).mean()), 3) if len(d_pair) else None,
        "pair_corr": round(corr, 3) if corr is not None else None,
        "null_abs_diff_median": round(float(np.median(null)), 2) if len(null) else None,
        "null_within_2px": round(float((null <= 2).mean()), 3) if len(null) else None,
        "consecutive": len(d_next), "consecutive_abs_diff_median": round(float(np.median(d_next)), 2) if len(d_next) else None,
        "ms_median": {op: round(float(np.median(v)), 2) for op, v in times.items()},
    }


# ============================================================================ Part B
def _gt(label: Path, w: int, h: int, cls_id: int = 0):
    if not label.exists():
        return []
    out = []
    for line in label.read_text().splitlines():
        p = line.split()
        if len(p) >= 5 and int(float(p[0])) == cls_id:
            cx, cy, bw, bh = map(float, p[1:5])
            out.append((int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h)))
    return out


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _label(dets, gts, thr):
    """Greedy one-to-one matching → [(conf, tp)] and the GT indices that were matched."""
    used, out = set(), []
    for conf, box in sorted(dets, key=lambda d: -d[0]):
        best, bj = 0.0, -1
        for j, g in enumerate(gts):
            if j not in used:
                i = _iou(box, g)
                if i > best:
                    best, bj = i, j
        if bj >= 0 and best >= thr:
            used.add(bj); out.append((conf, 1))
        else:
            out.append((conf, 0))
    return out


def _ap(pts, n_gt):
    from src.detection.evaluate import voc_ap
    return voc_ap(pts, n_gt)


def part_b(C: Canonicaliser, modes=("raw", "gray", "gain"), reps: int = 1000) -> dict:
    from src.detection.infer import YoloOnnxDetector, DEFAULT_ONNX
    det = YoloOnnxDetector(DEFAULT_ONNX, conf_thres={"fishing_gear": 0.05})
    det.warmup()
    frames = []
    for split in ("val", "test"):
        frames += unique_frames(sorted((V1 / split / "images").glob(PAT)))
    per_frame = []                       # [{mode: {"pts@.3", "pts@.5", "n_gt"}}]
    wc = {"gt_total": 0, "gt_in_wc": 0, "tp_in_wc": 0, "fp_in_wc": 0, "tp": 0, "fp": 0, "frames_measured": 0}
    t0 = time.time()
    for i, p in enumerate(frames):
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        gts = _gt(V1 / p.parent.parent.name / "labels" / f"{p.stem}.txt", w, h)
        cf = C.process(im, p.stem)
        rec = {}
        for m in modes:
            x = C.detector_input(im, cf, m)
            dets = [(d.conf, d.bbox) for d in det.detect(x) if d.cls_name == "fishing_gear"]
            rec[m] = {"p3": _label(dets, gts, 0.3), "p5": _label(dets, gts, 0.5), "n_gt": len(gts)}
            if m == "raw" and cf.measured:
                lab = dict(zip([b for _, b in sorted(dets, key=lambda d: -d[0])], [t for _, t in rec[m]["p3"]]))
                for conf, box in dets:
                    inwc = cf.in_water_column(box)
                    tp = lab.get(box, 0)
                    wc["tp" if tp else "fp"] += 1
                    if inwc:
                        wc["tp_in_wc" if tp else "fp_in_wc"] += 1
        if cf.measured:
            wc["frames_measured"] += 1
            wc["gt_total"] += len(gts)
            wc["gt_in_wc"] += sum(bool(cf.in_water_column(g)) for g in gts)
        per_frame.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  part B: {i + 1}/{len(frames)} frames ({time.time() - t0:.0f}s)")

    def summ(idx, m):
        n = sum(per_frame[j][m]["n_gt"] for j in idx)
        p3 = [x for j in idx for x in per_frame[j][m]["p3"]]
        p5 = [x for j in idx for x in per_frame[j][m]["p5"]]
        ceiling = sum(t for _, t in p3) / max(1, n)
        return {"ap30": _ap(p3, n), "ap50": _ap(p5, n), "ceiling": ceiling, "cands_per_frame": len(p3) / max(1, len(idx))}

    allidx = list(range(len(per_frame)))
    res = {"frames": len(per_frame), "pots": sum(r["raw"]["n_gt"] for r in per_frame), "modes": {}}
    for m in modes:
        res["modes"][m] = {k: round(v, 4) for k, v in summ(allidx, m).items()}
    rng = np.random.default_rng(0)
    for m in modes:
        if m == "raw":
            continue
        gains = {"ap30": [], "ceiling": []}
        for _ in range(reps):
            idx = list(rng.integers(0, len(per_frame), len(per_frame)))
            a, b = summ(idx, m), summ(idx, "raw")
            gains["ap30"].append(a["ap30"] - b["ap30"]); gains["ceiling"].append(a["ceiling"] - b["ceiling"])
        res["modes"][m]["gain_vs_raw"] = {k: [round(float(np.mean(v)), 4), round(float(np.percentile(v, 2.5)), 4),
                                              round(float(np.percentile(v, 97.5)), 4)] for k, v in gains.items()}
    res["water_column"] = wc
    return res


# ============================================================================ report
def report(a: dict, b: dict | None) -> str:
    L = ["# STUDY-08 — Stage-1 sonar canonicalisation: measured geometry + detector input",
         "",
         "_Generated by `python -m src.cv_pipeline.study_canonical` · EXP-001 via `cv2.dnn` · unique frames only_",
         "",
         "## A. Bottom tracking (sonar altitude in pixels) — no labels needed",
         "",
         f"Frames: **{a['frames']}** un-augmented original `*wcp_ss*` sonograms (single-copy keys of "
         f"{a['keys_total']} in the raw archive, all splits). "
         f"Bottom tracked on **{a['measured']} ({a['coverage']:.0%})**; the rest show no water-column → seabed "
         "step and are reported as *not measured* (never guessed).",
         "",
         f"Altitude: median **{a['altitude_px']['median']} px** (5–95%: {a['altitude_px']['p05']}–{a['altitude_px']['p95']} px) "
         "of a 640-px slant-range axis — shallow bays, as expected for these surveys.",
         "",
         "| physical check | pairs | median abs difference | within 2 px |",
         "|---|--:|--:|--:|",
         f"| **port vs starboard, same recording + chunk** (same pings) | {a['pairs']} | **{a['pair_abs_diff_median']} px** | **{a['pair_within_2px']:.0%}** |",
         f"| null: port vs starboard of *different* recordings | 2000 draws | {a['null_abs_diff_median']} px | {a['null_within_2px']:.0%} |",
         f"| consecutive chunks of one channel (along-track) | {a['consecutive']} | {a['consecutive_abs_diff_median']} px | — |",
         "",
         "Median latency per op (this laptop, 1 frame = 640×640): " +
         " · ".join(f"`{k}` {v} ms" for k, v in a["ms_median"].items()) + ".",
         ""]
    if b:
        L += ["## B. With the deployed detector (EXP-001) on its unseen sonograms",
              "",
              f"{b['frames']} unique frames (v1 val Rec19 + v1 test), {b['pots']} labelled pots · detector floor conf 0.05 · "
              "paired frame-bootstrap 95% CI of the difference vs `raw`.",
              "",
              "| detector input | recall ceiling (IoU 0.3) | AP@0.3 | AP@0.5 | candidates/frame | Δ AP@0.3 vs raw (95% CI) | Δ ceiling vs raw (95% CI) |",
              "|---|--:|--:|--:|--:|:--:|:--:|"]
        for m, r in b["modes"].items():
            g = r.get("gain_vs_raw")
            gs = (lambda k: f"{g[k][0]:+.3f} ({g[k][1]:+.3f} .. {g[k][2]:+.3f})") if g else (lambda k: "—")
            L.append(f"| `{m}` | {r['ceiling']:.3f} | {r['ap30']:.3f} | {r['ap50']:.3f} | {r['cands_per_frame']:.2f} | {gs('ap30')} | {gs('ceiling')} |")
        wc = b["water_column"]
        L += ["",
              f"**Water column** ({wc['frames_measured']} frames with a tracked bottom): "
              f"{wc['gt_in_wc']} of {wc['gt_total']} labelled pots lie in the water column; "
              f"detector candidates in the water column: {wc['tp_in_wc']} TP / {wc['fp_in_wc']} FP "
              f"(of {wc['tp']} TP / {wc['fp']} FP overall).",
              ""]
    L += ["---", "_Numbers in this file are regenerated by the script; do not edit by hand._", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--skip-detector", action="store_true")
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()
    C = Canonicaliser()
    print("Part A: bottom tracking …")
    a = part_a(C)
    print(json.dumps(a, indent=1))
    b = None
    if not args.skip_detector:
        print("Part B: detector input …")
        b = part_b(C, reps=args.reps)
        print(json.dumps({k: v for k, v in b.items()}, indent=1))
    out = REPO / "runs" / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "study.json").write_text(json.dumps({"part_a": a, "part_b": b}, indent=1))
    (REPO / "docs" / "stage1_canonical.md").write_text(report(a, b), encoding="utf-8")
    print("wrote docs/stage1_canonical.md")


if __name__ == "__main__":
    main()
