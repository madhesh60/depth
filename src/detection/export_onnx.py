"""
export_onnx.py — export a trained YOLO11 ``.pt`` to ONNX and verify the ``cv2.dnn`` load path.

This is the bridge between training (ultralytics/torch, GPU) and deployment (OpenCV `cv2.dnn`,
torch-free — the path used by ``src/detection/infer.py`` and ``infra/lambda_handler.py``). It
exports the checkpoint, then **verifies** OpenCV can read it and run a forward pass whose output
shape matches the detect head (``(1, 4+nc, N)``), so a broken export is caught here, not in prod.

Usage:
    python src/detection/export_onnx.py runs/EXP-002/weights/best.pt
    python src/detection/export_onnx.py best.pt --imgsz 1024 --opset 12 --out best.onnx
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def export(pt_path: str, imgsz: int = 640, opset: int = 12, out: str | None = None) -> Path:
    """Export ``pt_path`` to ONNX (ultralytics). Returns the ONNX path."""
    from ultralytics import YOLO  # train-time dep; not needed at inference
    model = YOLO(pt_path)
    onnx_path = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=False)
    onnx_path = Path(onnx_path)
    if out:
        dst = Path(out)
        if dst.resolve() != onnx_path.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            onnx_path.replace(dst)
            onnx_path = dst
    return onnx_path


def verify(onnx_path: Path, imgsz: int = 640) -> tuple[bool, str]:
    """Load through ``cv2.dnn`` and run one forward pass — the exact deploy path."""
    import cv2
    import numpy as np
    try:
        net = cv2.dnn.readNetFromONNX(str(onnx_path))
        blob = cv2.dnn.blobFromImage(
            np.zeros((imgsz, imgsz, 3), np.uint8), 1 / 255.0, (imgsz, imgsz), swapRB=True)
        net.setInput(blob)
        out = net.forward()
        shape = tuple(out.shape)
        # detect head: (1, 4+nc, N) — 4 box + nc class scores, N anchors
        ok = len(shape) == 3 and shape[0] == 1 and shape[1] >= 5
        return ok, f"cv2.dnn forward OK, output {shape} (expected (1, 4+nc, N))"
    except Exception as e:  # noqa: BLE001 — report any load/forward failure to the user
        return False, f"cv2.dnn load/forward FAILED: {e}"


def _sha256(p: Path, n: int = 12) -> str:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    return h[:n]


def main():
    ap = argparse.ArgumentParser(description="Export YOLO .pt -> ONNX and verify the cv2.dnn path.")
    ap.add_argument("weights", help="path to best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--out", default=None, help="optional output .onnx path")
    a = ap.parse_args()

    if not Path(a.weights).exists():
        raise SystemExit(f"weights not found: {a.weights}")
    print(f"exporting {a.weights}  (imgsz={a.imgsz}, opset={a.opset}) ...")
    onnx_path = export(a.weights, a.imgsz, a.opset, a.out)
    ok, msg = verify(onnx_path, a.imgsz)
    print(f"  -> {onnx_path}  (sha256 {_sha256(onnx_path)}, {onnx_path.stat().st_size/1e6:.1f} MB)")
    print(f"  -> {msg}")
    if not ok:
        raise SystemExit("verification failed — do NOT deploy this ONNX")
    print("OK: ready for src/detection/infer.py and infra/lambda_handler.py")


if __name__ == "__main__":
    main()
