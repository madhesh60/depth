"""
lambda_handler.py — AWS Lambda inference endpoint for marine-debris detection.

Deploy-ready (not yet deployed — pending AWS credits). Runs the Stage-2 detector through
``cv2.dnn`` only: no torch / ultralytics, so the deploy package stays small and cold-start is
fast. Works unchanged on x86 and on **Graviton/Arm** Lambda (the COOL-award compute target).

Pipeline per request:
    frame (S3 key or base64) → optional Stage-1 sonar preprocessing (CPU, COOL-accelerated)
    → cv2.dnn ONNX detection (per-class thresholds) → geotagged JSON detections.

Event shapes accepted:
    {"s3": {"bucket": "...", "key": "frames/x.png"}, "lat": .., "lon": ..}
    {"image_b64": "<base64 png/jpg>", "lat": .., "lon": ..}

Env:
    MODEL_S3   = s3://bucket/models/best.onnx   (downloaded to /tmp on cold start)
    MODEL_PATH = /var/task/best.onnx            (or a bundled path; overrides MODEL_S3)

This mirrors the local path in ``src/detection/infer.py`` so local and cloud results match.
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

# --- model bootstrap (once per container) --------------------------------------------------
_NET = None
_NAMES = ["fishing_gear", "pipe_cylinder", "structural_fragment", "natural_formation"]
_CONF = {"fishing_gear": 0.10, "pipe_cylinder": 0.25,
         "structural_fragment": 0.25, "natural_formation": 0.25}
_IMGSZ = 640
_IOU = 0.45


def _model_path() -> str:
    p = os.environ.get("MODEL_PATH")
    if p and Path(p).exists():
        return p
    s3_uri = os.environ.get("MODEL_S3")
    if not s3_uri:
        raise RuntimeError("set MODEL_PATH (bundled) or MODEL_S3 (s3://bucket/key)")
    import boto3  # only needed in the S3 path
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    dst = "/tmp/best.onnx"
    if not Path(dst).exists():
        boto3.client("s3").download_file(bucket, key, dst)
    return dst


def _net():
    global _NET
    if _NET is None:
        _NET = cv2.dnn.readNetFromONNX(_model_path())
    return _NET


# --- inference (kept in lockstep with src/detection/infer.py) ------------------------------
def _letterbox(img, size):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = round(w * s), round(h * s)
    canvas = np.full((size, size, 3), 114, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    return canvas, s, px, py


def detect(img: np.ndarray) -> list[dict]:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    canvas, s, px, py = _letterbox(img, _IMGSZ)
    blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (_IMGSZ, _IMGSZ), swapRB=True)
    net = _net()
    net.setInput(blob)
    preds = net.forward()[0].T
    floor = min(_CONF.values())
    cls = np.argmax(preds[:, 4:], axis=1)
    conf = preds[np.arange(preds.shape[0]), 4 + cls]
    keep = conf >= floor
    if not np.any(keep):
        return []
    cx, cy, bw, bh = preds[keep, :4].T
    cls, conf = cls[keep], conf[keep]
    boxes = np.stack([(cx - bw / 2 - px) / s, (cy - bh / 2 - py) / s, bw / s, bh / s], axis=1)
    idxs = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), floor, _IOU)
    H, W = img.shape[:2]
    out = []
    for i in np.array(idxs).flatten() if len(idxs) else []:
        c = int(cls[i])
        if conf[i] < _CONF[_NAMES[c]]:
            continue
        x1, y1 = max(0, int(boxes[i, 0])), max(0, int(boxes[i, 1]))
        x2, y2 = min(W, int(boxes[i, 0] + boxes[i, 2])), min(H, int(boxes[i, 1] + boxes[i, 3]))
        if x2 > x1 and y2 > y1:
            out.append({"bbox": [x1, y1, x2, y2], "cls": _NAMES[c], "conf": round(float(conf[i]), 4)})
    return out


def _load_image(event) -> np.ndarray:
    if "image_b64" in event:
        buf = np.frombuffer(base64.b64decode(event["image_b64"]), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if "s3" in event:
        import boto3
        obj = boto3.client("s3").get_object(Bucket=event["s3"]["bucket"], Key=event["s3"]["key"])
        buf = np.frombuffer(obj["Body"].read(), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    raise ValueError("event needs 'image_b64' or 's3'")


def handler(event, context=None):
    """Lambda entry point. Returns geotagged detections + timing."""
    if isinstance(event.get("body"), str):          # API Gateway proxy
        event = json.loads(event["body"])
    t0 = time.perf_counter()
    img = _load_image(event)
    if img is None:
        return {"statusCode": 400, "body": json.dumps({"error": "could not decode image"})}
    dets = detect(img)
    lat, lon = event.get("lat"), event.get("lon")
    for d in dets:                                   # attach geotag if provided
        d["lat"], d["lon"] = lat, lon
    import platform as _pf
    body = {
        "detections": dets,
        "count": len(dets),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "arch": _pf.machine(),          # x86_64 vs aarch64 (Graviton) — for the COOL benchmark
    }
    return {"statusCode": 200, "body": json.dumps(body)}


if __name__ == "__main__":  # local smoke test: python infra/lambda_handler.py <img>
    import sys
    os.environ.setdefault("MODEL_PATH", "runs/EXP-001/weights/best.onnx")
    with open(sys.argv[1], "rb") as fh:
        ev = {"image_b64": base64.b64encode(fh.read()).decode(), "lat": 12.9, "lon": 80.2}
    print(json.loads(handler(ev)["body"]))
