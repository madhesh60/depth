"""
train.py — Stage 2 detector training (YOLOv8 / YOLO11) for marine-debris detection.

Trains on the cleaned, leakage-free v1 dataset with augmentation tuned for side-scan
sonar imagery, then validates and reports per-class metrics.

Design notes
------------
* Sonar-aware augmentation: sonar frames are effectively grayscale and have a fixed
  range/along-track geometry, so colour and rotation augmentation are disabled or minimised:
    - hsv_h / hsv_s = 0        (no hue/saturation — images carry no real colour)
    - hsv_v kept small          (mild brightness jitter ~ gain variation)
    - degrees = 0, flipud = 0   (rotation / vertical flip break sonar geometry)
    - fliplr = 0.5              (along-track flip is plausible)
    - mosaic / scale / translate kept (help small-object robustness)
* Class imbalance (fishing_gear ~11x pipe_cylinder): reported per class every epoch.
  Ultralytics has no per-class loss-weight arg; the levers are augmentation (set here) and,
  if pipe/rope recall stays low, oversampling minority-class images (see --oversample).
  Recommended inverse-frequency weights (for reference / a custom sampler) are printed at
  startup and stored in the dataset manifest.
* Every run is logged under runs/<name>/; record the result in experiments.md (EXP-NNN).

Usage
-----
    pip install -r requirements.txt          # needs ultralytics
    python src/detection/train.py --model yolo11s.pt --epochs 100 --batch 16 --name EXP-001
    python src/detection/train.py --seg --model yolo11s-seg.pt --name EXP-001-seg
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO / "DATASET" / "03_yolo_ready_dataset_v1" / "data.yaml"

# Inverse-frequency class weights (normalised to mean 1.0), for reference / custom samplers.
# Regenerate with: python DATASET/scripts/analyze_quality.py
CLASS_WEIGHTS = {
    "fishing_gear": 0.22,
    "pipe_cylinder": 2.52,
    "structural_fragment": 0.42,
    "natural_formation": 0.84,
}


def parse_args():
    p = argparse.ArgumentParser(description="Train the Stage 2 marine-debris detector.")
    p.add_argument("--model", default="yolo11s.pt",
                   help="base weights (e.g. yolo11s.pt, yolov8s.pt, or *-seg.pt with --seg)")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="path to data.yaml")
    p.add_argument("--seg", action="store_true", help="instance-segmentation task")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="e.g. 0, 0,1, or cpu (auto if omitted)")
    p.add_argument("--name", default="EXP-001", help="run name under runs/")
    p.add_argument("--patience", type=int, default=20, help="early-stopping patience")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"data.yaml not found: {data}\n"
                         f"Build the dataset first: python DATASET/scripts/build_dataset_v1.py")

    print("Recommended inverse-frequency class weights (imbalance reference):")
    for k, v in CLASS_WEIGHTS.items():
        print(f"  {k:22} {v}")

    from ultralytics import YOLO  # imported here so --help works without the dep installed

    model = YOLO(args.model)
    model.train(
        data=str(data),
        task="segment" if args.seg else "detect",
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        seed=args.seed,
        project=str(REPO / "runs"),
        name=args.name,
        # ---- sonar-aware augmentation ----
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
        degrees=0.0, flipud=0.0, fliplr=0.5,
        translate=0.1, scale=0.5,
        mosaic=1.0, close_mosaic=10,
        plots=True,
    )

    # final validation with per-class metrics
    metrics = model.val(data=str(data), imgsz=args.imgsz, split="test", plots=True)
    print("\nTest metrics:")
    print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"  precision    : {metrics.box.mp:.4f}")
    print(f"  recall       : {metrics.box.mr:.4f}")
    print("\nLog this run in experiments.md (EXP-NNN) with per-class results.")


if __name__ == "__main__":
    main()
