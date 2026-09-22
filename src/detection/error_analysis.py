"""
error_analysis.py — deep error analysis of the Stage-2 detector on the test split.

Answers the questions the aggregate mAP hides:
  1. Per-class P / R / F1 at the operating point (conf 0.25).
  2. **Domain split** — sonar vs optical recall per class (the product is side-scan SONAR;
     optical frames were added only for volume and may be dragging the aggregate down).
  3. **Per-source** recall (which of the 7 datasets the detector actually handles).
  4. **Confidence sweep** — recall at conf 0.05 / 0.10 / 0.25, overall and per domain
     (quantifies the free recall lever of lowering the threshold for a safety-critical class).
  5. Missed-box size distribution per class (confirms the small-object hypothesis).

Runs one detector pass per image at conf 0.05, then evaluates each threshold by filtering,
so the whole report is a single sweep over the test set.

Usage:
    python -m src.detection.error_analysis
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))

from src.detection.infer import YoloOnnxDetector, DEFAULT_ONNX          # noqa: E402
from src.detection.ablation_fp import load_gt, match, Tally            # noqa: E402

DEFAULT_DATA = REPO / "DATASET" / "03_yolo_ready_dataset_v1" / "data.yaml"

# source -> sensor modality (mirrors DATASET/scripts/build_dataset_v1.py)
PREFIXES = ["crabpot", "uatd", "icra", "mpulse", "seabed", "shipwreck", "vid"]
SENSOR = {"crabpot": "sonar", "uatd": "sonar", "mpulse": "sonar", "seabed": "sonar",
          "shipwreck": "sonar", "icra": "optical", "vid": "optical"}


def source_of(stem: str) -> str:
    low = stem.lower()
    for p in PREFIXES:
        if low.startswith(p):
            return p
    return "other"


def main():
    ap = argparse.ArgumentParser(description="Deep error analysis of the Stage-2 detector.")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--split", default="test")
    ap.add_argument("--base-conf", type=float, default=0.05, help="single detector pass at this conf")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = yaml.safe_load(Path(args.data).read_text())
    names = data["names"]; nc = len(names)
    root = Path(args.data).parent
    img_dir = root / data[args.split]
    lbl_dir = Path(str(img_dir).replace("images", "labels"))
    imgs = sorted(q for q in img_dir.rglob("*") if q.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        imgs = imgs[: args.limit]

    det = YoloOnnxDetector(args.onnx, names=names, conf_thres=args.base_conf)
    THRS = [0.05, 0.10, 0.25]

    # tallies[thr][domain][class]  and per-source recall at 0.25
    tallies = {t: {d: [Tally() for _ in range(nc)] for d in ("sonar", "optical", "all")} for t in THRS}
    src_tally = defaultdict(lambda: [Tally() for _ in range(nc)])      # source -> per-class @0.25
    miss_sizes = [[] for _ in range(nc)]                                # per-class missed long-side frac

    for k, ip in enumerate(imgs):
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        dom = SENSOR.get(source_of(ip.stem), "optical")
        src = source_of(ip.stem)
        gt = load_gt(lbl_dir / (ip.stem + ".txt"), w, h)
        dets_all = det.detect(img)                                      # conf >= base_conf

        for thr in THRS:
            dets = [d for d in dets_all if d.conf >= thr]
            for c in range(nc):
                d_c = [d for d in dets if d.cls_id == c]
                g_c = [g for g in gt if g[0] == c]
                tp, fp = match(d_c, g_c, args.match_iou)
                for scope in (dom, "all"):
                    tallies[thr][scope][c].add(tp, fp, len(g_c))
                if thr == 0.25:
                    src_tally[src][c].add(tp, fp, len(g_c))
                    if len(g_c) > tp:            # some GT of this class missed on this frame
                        # record sizes of unmatched GT boxes (approx: smallest-first heuristic)
                        matched_ct = tp
                        gbx = sorted(((gb[2]-gb[0])*(gb[3]-gb[1]), gb) for _, gb in g_c)
                        for area, gb in gbx[: len(g_c) - matched_ct]:
                            miss_sizes[c].append(max(gb[2]-gb[0], gb[3]-gb[1]) / max(w, h))
        if (k + 1) % 300 == 0:
            print(f"  {k+1}/{len(imgs)}...")

    def line(label, T: Tally):
        return (f"{label:<22} P {T.precision:5.3f}  R {T.recall:5.3f}  "
                f"F1 {2*T.precision*T.recall/(T.precision+T.recall) if (T.precision+T.recall) else 0:5.3f}"
                f"  (TP {T.tp}  FP {T.fp}  FN {T.n_gt - T.tp})")

    print("\n" + "=" * 70)
    print(f"DEEP ERROR ANALYSIS — {args.split} split, {len(imgs)} images")
    print("=" * 70)

    print("\n[1] Per-class @ conf 0.25 (operating point)")
    for c in range(nc):
        print("  " + line(names[c], tallies[0.25]["all"][c]))
    overall = Tally()
    for c in range(nc):
        T = tallies[0.25]["all"][c]; overall.tp += T.tp; overall.fp += T.fp; overall.n_gt += T.n_gt
    print("  " + line("ALL", overall))

    print("\n[2] Domain split @ conf 0.25 — recall per class (product = SONAR)")
    print(f"  {'class':<22}{'sonar R':>10}{'optical R':>12}   (sonar TP/GT | optical TP/GT)")
    for c in range(nc):
        s = tallies[0.25]["sonar"][c]; o = tallies[0.25]["optical"][c]
        print(f"  {names[c]:<22}{s.recall:>10.3f}{o.recall:>12.3f}   "
              f"({s.tp}/{s.n_gt} | {o.tp}/{o.n_gt})")

    print("\n[3] Per-source recall @ conf 0.25")
    print(f"  {'source':<12}{'domain':<9}" + "".join(f"{n[:8]:>10}" for n in names))
    for src in sorted(src_tally):
        row = "".join(f"{src_tally[src][c].recall:>10.3f}" for c in range(nc))
        print(f"  {src:<12}{SENSOR.get(src,'optical'):<9}{row}")

    print("\n[4] Confidence sweep — recall (overall / sonar-only)")
    print(f"  {'class':<22}" + "".join(f"{'@'+str(t):>9}" for t in THRS)
          + "   |  sonar-only " + " ".join(f"@{t}" for t in THRS))
    for c in range(nc):
        allr = "".join(f"{tallies[t]['all'][c].recall:>9.3f}" for t in THRS)
        sonr = " ".join(f"{tallies[t]['sonar'][c].recall:.3f}" for t in THRS)
        print(f"  {names[c]:<22}{allr}   |   {sonr}")

    print("\n[5] Missed-box long-side (frac of frame) @ conf 0.25")
    for c in range(nc):
        s = np.array(miss_sizes[c]) if miss_sizes[c] else np.array([np.nan])
        if np.isnan(s).all():
            print(f"  {names[c]:<22} (no misses recorded)")
        else:
            print(f"  {names[c]:<22} n={len(s):<5} median {np.median(s):.3f}  "
                  f"p75 {np.percentile(s,75):.3f}  <0.10: {100*np.mean(s<0.10):.0f}%")


if __name__ == "__main__":
    main()
