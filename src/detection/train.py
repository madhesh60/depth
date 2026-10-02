"""
train.py — Stage-2 detector training (YOLO11) with sonar-aware settings, then ONNX export + packaging.

EXP-003 recipe — the EXP-002 post-mortem (``docs/exp002_diagnosis.md``) fixed in code:

* **explicit optimizer (SGD, lr0 0.01).** EXP-002 passed ``lr0 0.01`` but left ``optimizer=auto``,
  which in ultralytics 8.4 silently picks **AdamW at lr 0.002·5/(4+nc) = 0.00167** whenever the run
  has ≤ 10 000 optimizer steps (v2b: 2 700) and ignores ``lr0``. The models were badly underfit
  (ghost-gear AP 0.42 on their OWN training frames; train cls loss 1.72 vs EXP-001's 0.51). The
  optimizer actually built is now recorded in ``model_meta.json`` and must match the request.
* **enough steps:** ~9-10 k SGD steps at nominal batch 64, about EXP-001's 16.6 k scale: 150 epochs
  on the tiled set, or 300 on full frames only. Patience is a quarter of the epochs.
* **data:** dataset v2b (2 classes, sonar only, one copy per Roboflow frame, val = held-out
  recordings Rec10/12/16). Training uses either the tiled build (``DATASET/scripts/build_tiles.py``)
  or the full frames only (``--no-tiles``). The tiled build no longer leaves objects that a tile
  cuts unlabelled. Val and test stay full frame.
* **imgsz 640.** At 1024 px EXP-002 matched 640 px (ghost AP 0.25 vs 0.25 on val) and cost about
  2.6× the inference time.
* **sonar-aware augmentation:** no hue/saturation (no real colour), no rotation / vertical flip
  (break range geometry), along-track flip only, mild brightness (gain), scale + translate for size.
  NOTE: ultralytics ``copy_paste`` is a no-op for box-only labels (it returns early without masks),
  so it is not used — the offline, range-preserving copy-paste in ``build_tiles.py`` replaces it.
* **checkpoint selection for the product, not the leaderboard:** ultralytics keeps the epoch with
  the best ``0.9·mAP50-95 + 0.1·mAP50`` over *both* classes — box tightness, dominated by the easy wreck
  class. DEPTH needs *ghost gear found* at a loose match (IoU ≥ 0.3).
  - During training, a callback reads ultralytics' own per-class val AP after every epoch.
  - It keeps the epoch with the best **ghost-gear AP@0.5** as ``best_ghost.pt``, with no per-epoch
    files.
  - After training, the shortlist (``best_ghost.pt``, ``best.pt``, ``last.pt``) is re-validated with
    fixed settings. The winner becomes ``best_depth.pt`` (tie-break: all-class mAP50).
  - The per-epoch table and ultralytics' own pick are both recorded.
  - The val recordings also host the calibration later; the promise is still verified once on test,
    so this selection cannot inflate it.
* **after training:** ``best_depth.pt`` → ONNX (opset 12, static ``imgsz``) → verified through
  ``cv2.dnn`` (the deploy path) → ``model_meta.json`` (names, imgsz, data, args, selection, val
  metrics, sha256) → ``<name>_complete.zip`` (per-epoch checkpoints excluded) ready to download.
  Locally, ONE command plugs it in: ``python -m src.detection.onboard_model --zip <name>_complete.zip``.
* **Kaggle-safe defaults:** no image cache (``--cache none``: a disk cache of ~9 k tiles at 1024 px
  is ~27 GB and would overflow Kaggle's ~20 GB working disk hours into the run).

Usage (Kaggle T4 — see docs/exp003_kaggle.md):
    python src/detection/train.py --data /kaggle/working/v2b_tiles/data.yaml --name EXP-003
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
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=0, help="early-stop patience (0 = a quarter of the epochs)")
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--optimizer", default="SGD", choices=["SGD", "AdamW", "auto"],
                   help="explicit by default: 'auto' silently swaps in AdamW at lr 0.00167 for short runs")
    p.add_argument("--lr0", type=float, default=0.01, help="honoured only with an explicit --optimizer")
    p.add_argument("--no-cos-lr", action="store_true")
    p.add_argument("--fraction", type=float, default=1.0, help="train on a fraction (smoke tests)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache", default="none", help="none | ram | disk (disk/ram at 1024 px can exhaust Kaggle)")
    p.add_argument("--select", default="ghost_ap50", choices=["ghost_ap50", "fitness"],
                   help="checkpoint to export: best ghost-gear AP@0.5 on val (default) or ultralytics' fitness")
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


def ghost_class(names: list[str]) -> int:
    return names.index("ghost_gear") if "ghost_gear" in names else 0


class GhostTracker:
    """ultralytics ``on_model_save`` callback: after each epoch's validation (``last.pt`` has just
    been written), read the per-class AP@0.5 and keep the best ghost-gear epoch as
    ``weights/best_ghost.pt``. Replaces ``save_period=1``: no per-epoch files, so no 4-GB disk use.
    It also records the optimizer ultralytics really built."""

    def __init__(self, cls_id: int):
        self.cls_id, self.best, self.rows, self.optimizer = cls_id, -1.0, [], None

    def on_train_start(self, trainer):
        opt = getattr(trainer, "optimizer", None)
        if opt is not None:
            self.optimizer = {"type": type(opt).__name__,
                              "lr": float(opt.param_groups[0].get("initial_lr", opt.param_groups[0]["lr"])),   # per group; defaults is unused
                              "nominal_batch": int(trainer.args.nbs),
                              "accumulate": int(getattr(trainer, "accumulate", 1))}

    def on_model_save(self, trainer):
        m = getattr(trainer.validator, "metrics", None)
        raw = getattr(m, "ap_class_index", None) if m is not None else None      # a numpy array
        idx = [int(i) for i in raw] if raw is not None else []
        ap50 = list(m.box.ap50) if idx else []
        ghost = float(ap50[idx.index(self.cls_id)]) if self.cls_id in idx else 0.0
        self.rows.append({"epoch": trainer.epoch + 1, "ghost_ap50": round(ghost, 4),
                          "map50": round(float(m.box.map50), 4) if idx else 0.0})
        if ghost > self.best and Path(trainer.last).exists():
            self.best = ghost
            shutil.copy2(trainer.last, Path(trainer.wdir) / "best_ghost.pt")


def select_checkpoint(run: Path, data: Path, names: list[str], imgsz: int, device, batch: int = 16) -> dict:
    """Re-validate the shortlist (``best_ghost.pt`` from training, ultralytics' ``best.pt``,
    ``last.pt``; plus any ``epoch*.pt`` from older runs) on the held-out val split with fixed settings.
    Return the table and the pick: best ghost-gear AP@0.5, tie-break all-class mAP50. Writes
    ``selection.csv`` next to the run."""
    import re as _re
    from ultralytics import YOLO
    wdir = run / "weights"
    g = ghost_class(names)
    cks = sorted(wdir.glob("epoch*.pt"), key=lambda p: int(_re.sub(r"\D", "", p.stem) or 0))
    cks += [p for p in (wdir / "best_ghost.pt", wdir / "best.pt", wdir / "last.pt") if p.exists()]
    rows = []
    for ck in cks:
        m = YOLO(str(ck)).val(data=str(data), imgsz=imgsz, batch=batch, conf=0.001, iou=0.6, split="val",
                              device=device, plots=False, verbose=False, project=str(run / "select"),
                              name=ck.stem, exist_ok=True)
        idx = list(getattr(m.box, "ap_class_index", []))
        ghost = float(m.box.ap50[idx.index(g)]) if g in idx else 0.0
        rows.append({"checkpoint": ck.name, "ghost_ap50": round(ghost, 4), "map50": round(float(m.box.map50), 4),
                     "map50_95": round(float(m.box.map), 4)})
        print(f"  {ck.name:>12}: ghost_gear AP50 {ghost:.4f} | mAP50 {m.box.map50:.4f}")
    shutil.rmtree(run / "select", ignore_errors=True)
    (run / "selection.csv").write_text("checkpoint,ghost_ap50,map50,map50_95\n" + "\n".join(
        f"{r['checkpoint']},{r['ghost_ap50']},{r['map50']},{r['map50_95']}" for r in rows) + "\n")
    pick = max(rows, key=lambda r: (r["ghost_ap50"], r["map50"])) if rows else None
    ub = next((r for r in rows if r["checkpoint"] == "best.pt"), None)
    return {"criterion": "best ghost_gear AP@0.5 on the held-out val recordings (tie-break mAP50)",
            "picked": pick, "ultralytics_best": ub, "table": rows}


def main():
    a = parse_args()
    data = Path(a.data).resolve()                      # absolute: ultralytics resolves relative paths vs CWD
    if not data.exists():
        raise SystemExit(f"data.yaml not found: {data}\n"
                         f"Build it: python DATASET/scripts/build_dataset_v2b.py "
                         f"(+ DATASET/scripts/build_tiles.py for the tiled set)")
    names = _names(data)
    patience = a.patience or max(10, a.epochs // 4)
    print(f"data {data} | classes {names} | imgsz {a.imgsz} | epochs {a.epochs} (patience {patience}) | "
          f"batch {a.batch} | optimizer {a.optimizer} lr0 {a.lr0}")

    from ultralytics import YOLO                       # train-time dependency only
    import ultralytics
    t0 = time.time()
    model = YOLO(a.model)
    tracker = GhostTracker(ghost_class(names))
    if a.select == "ghost_ap50":
        model.add_callback("on_train_start", tracker.on_train_start)
        model.add_callback("on_model_save", tracker.on_model_save)
    model.train(
        data=str(data), task="detect", imgsz=a.imgsz, epochs=a.epochs, batch=a.batch,
        device=a.device, patience=patience, seed=a.seed, deterministic=True,
        project=str(REPO / "runs"), name=a.name, exist_ok=True, optimizer=a.optimizer,
        cos_lr=not a.no_cos_lr, lr0=a.lr0, close_mosaic=a.close_mosaic, fraction=a.fraction,
        workers=a.workers, cache=(False if a.cache == "none" else a.cache), plots=True, save_period=-1,
        **SONAR_AUG,
    )
    run = REPO / "runs" / a.name
    best = run / "weights" / "best.pt"
    if tracker.optimizer:
        print("optimizer built by ultralytics:", json.dumps(tracker.optimizer))
        if a.optimizer != "auto" and tracker.optimizer["type"] != a.optimizer:
            print(f"!! WARNING: asked for {a.optimizer}, ultralytics built {tracker.optimizer['type']}")
    meta = {
        "name": a.name, "names": names, "imgsz": a.imgsz, "data": str(data), "base": a.model,
        "args": vars(a), "patience_used": patience, "optimizer_built": tracker.optimizer,
        "ghost_ap50_per_epoch": tracker.rows, "augmentation": SONAR_AUG, "best": _best_epoch(run),
        "train_minutes": round((time.time() - t0) / 60, 1),
        "ultralytics": ultralytics.__version__, "python": platform.python_version(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "notes": a.notes,
    }
    print("ultralytics' best epoch (fitness):", json.dumps(meta["best"]))
    if a.select == "ghost_ap50" and best.exists():
        print("\nselecting the checkpoint for DEPTH (ghost-gear AP@0.5 on the held-out val recordings) ...")
        sel = select_checkpoint(run, data, names, a.imgsz, a.device)
        meta["selection"] = sel
        if sel["picked"]:
            from ultralytics.utils.torch_utils import strip_optimizer
            src = run / "weights" / sel["picked"]["checkpoint"]
            chosen = run / "weights" / "best_depth.pt"
            shutil.copy2(src, chosen)
            strip_optimizer(str(chosen))
            best = chosen
            print(f"picked {sel['picked']['checkpoint']}: ghost_gear AP50 {sel['picked']['ghost_ap50']} "
                  f"(ultralytics best.pt: {(sel['ultralytics_best'] or {}).get('ghost_ap50')})")
        for ep in [*(run / "weights").glob("epoch*.pt"), run / "weights" / "best_ghost.pt"]:   # not shipped
            ep.unlink(missing_ok=True)

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
