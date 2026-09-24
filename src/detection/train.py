"""
train.py — Stage-2 detector training (YOLO11) with sonar-aware settings, then ONNX export + packaging.

EXP-002 recipe (review I-3 — the recall ceiling is the binding constraint: EXP-001 never proposes
~20-28% of pots even at conf 0.05, and no triage can recover those):

* **data:** dataset v2b (2 classes, sonar only, one copy per Roboflow frame, val = held-out
  recordings Rec10/12/16 → ``best.pt`` is selected on the SAME sonar as test), optionally the tiled
  build (``DATASET/scripts/build_tiles.py``: full frames + 2×2 overlapping tiles; val/test stay full
  frame) and the sonar-aware copy-paste (``--paste`` in build_tiles: pots + their shadow tails pasted
  onto empty seabed at the same range with Poisson blending).
* **imgsz 1024** (small objects), 30 epochs, patience 8 (EXP-001 overfit after ~epoch 16-20),
  cosine LR, mosaic off for the last 5 epochs.
* **sonar-aware augmentation:** no hue/saturation (no real colour), no rotation / vertical flip
  (break range geometry), along-track flip only, mild brightness (gain), scale + translate for size.
  NOTE: ultralytics ``copy_paste`` is a no-op for box-only labels (it returns early without masks),
  so it is not used — the offline, range-preserving copy-paste in ``build_tiles.py`` replaces it.
* **after training:** ``best.pt`` → ONNX (opset 12, static ``imgsz``) → verified through ``cv2.dnn``
  (the deploy path) → ``model_meta.json`` (names, imgsz, data, args, best epoch, val metrics, sha256)
  → ``<name>_complete.zip`` ready to download. Locally, ONE command plugs it in:
  ``python -m src.detection.onboard_model --zip <name>_complete.zip``.

Usage (Kaggle T4 — see docs/exp002_kaggle.md):
    python src/detection/train.py --data /kaggle/working/v2b_tiles/data.yaml --name EXP-002
    python src/detection/train.py --data DATASET/03_yolo_ready_dataset_v2b/data.yaml --fraction 0.05 \
        --epochs 1 --imgsz 320 --name smoke          # a 2-minute smoke test on CPU
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys                                            # noqa: E402 — run as a script on Kaggle
sys.path.insert(0, str(REPO))
V2B =REPO / "DATASET" / "03_yolo_ready_dataset_v2b"
DEFAULT_DATA = (V2B.parent / "03_yolo_ready_dataset_v2b_tiles" / "data.yaml"
                if (V2B.parent / "03_yolo_ready_dataset_v2b_tiles" / "data.yaml").exists() else V2B / "data.yaml")

SONAR_AUG = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, degrees=0.0, flipud=0.0, fliplr=0.5,
                 translate=0.1, scale=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0)


def parse_args():
    p = argparse.ArgumentParser(description="Train the DEPTH detector (sonar-aware) + export/verify ONNX.")
    p.add_argument("--model", default="yolo11s.pt", help="base weights")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="path to data.yaml (made absolute)")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--close-mosaic", type=int, default=5)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--no-cos-lr", action="store_true")
    p.add_argument("--fraction", type=float, default=1.0, help="train on a fraction (smoke tests)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache", default="disk", help="ram | disk | none")
    p.add_argument("--device", default=None, help="0 | cpu | auto")
    p.add_argument("--name", default="EXP-002", help="run name under runs/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--notes", default="", help="free text stored in model_meta.json")
    return p.parse_args()


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _names(data_yaml: Path) -> list[str]:
    names = []
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("- "):
            names.append(line.strip()[2:].strip())
    return names


def _best_epoch(run: Path) -> dict:
    """Best epoch by ultralytics fitness (0.1·mAP50 + 0.9·mAP50-95) from results.csv."""
    csv = run / "results.csv"
    if not csv.exists():
        return {}
    rows = [ln.split(",") for ln in csv.read_text().splitlines()]
    head = [h.strip() for h in rows[0]]
    best, best_f = None, -1.0
    for r in rows[1:]:
        d = dict(zip(head, (x.strip() for x in r)))
        try:
            f = 0.1 * float(d["metrics/mAP50(B)"]) + 0.9 * float(d["metrics/mAP50-95(B)"])
        except (KeyError, ValueError):
            continue
        if f > best_f:
            best_f, best = f, d
    if not best:
        return {}
    return {"epoch": int(float(best["epoch"])), "epochs_run": len(rows) - 1,
            "val_mAP50": round(float(best["metrics/mAP50(B)"]), 4),
            "val_mAP50_95": round(float(best["metrics/mAP50-95(B)"]), 4),
            "val_precision": round(float(best["metrics/precision(B)"]), 4),
            "val_recall": round(float(best["metrics/recall(B)"]), 4)}


def main():
    a = parse_args()
    data = Path(a.data).resolve()                      # absolute: ultralytics resolves relative paths vs CWD
    if not data.exists():
        raise SystemExit(f"data.yaml not found: {data}\n"
                         f"Build it: python DATASET/scripts/build_dataset_v2b.py "
                         f"(+ DATASET/scripts/build_tiles.py for the tiled set)")
    names = _names(data)
    print(f"data {data} | classes {names} | imgsz {a.imgsz} | epochs {a.epochs} | batch {a.batch}")

    from ultralytics import YOLO                       # train-time dependency only
    import ultralytics
    t0 = time.time()
    model = YOLO(a.model)
    model.train(
        data=str(data), task="detect", imgsz=a.imgsz, epochs=a.epochs, batch=a.batch,
        device=a.device, patience=a.patience, seed=a.seed, deterministic=True,
        project=str(REPO / "runs"), name=a.name, exist_ok=True,
        cos_lr=not a.no_cos_lr, lr0=a.lr0, close_mosaic=a.close_mosaic, fraction=a.fraction,
        workers=a.workers, cache=(False if a.cache == "none" else a.cache), plots=True,
        **SONAR_AUG,
    )
    run = REPO / "runs" / a.name
    best = run / "weights" / "best.pt"
    meta = {
        "name": a.name, "names": names, "imgsz": a.imgsz, "data": str(data), "base": a.model,
        "args": vars(a), "augmentation": SONAR_AUG, "best": _best_epoch(run),
        "train_minutes": round((time.time() - t0) / 60, 1),
        "ultralytics": ultralytics.__version__, "python": platform.python_version(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "notes": a.notes,
    }
    print("best epoch:", json.dumps(meta["best"]))

    if not a.no_export and best.exists():
        from src.detection.export_onnx import export, verify
        onnx = export(str(best), imgsz=a.imgsz, opset=a.opset, out=str(run / "weights" / "best.onnx"))
        ok, msg = verify(onnx, a.imgsz)
        print(msg)
        if not ok:
            raise SystemExit("ONNX verification through cv2.dnn FAILED - do not deploy")
        meta.update({"onnx": "weights/best.onnx", "onnx_sha256": _sha256(onnx), "opset": a.opset,
                     "cv2_dnn_verified": msg})
    (run / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # one zip to download: weights (pt + onnx), meta, curves
    pkg = REPO / "runs" / f"{a.name}_complete"
    shutil.make_archive(str(pkg), "zip", root_dir=str(REPO / "runs"), base_dir=a.name)
    print(f"\npackage -> {pkg}.zip")
    print(f"next (locally): python -m src.detection.onboard_model --zip {a.name}_complete.zip")


if __name__ == "__main__":
    main()
