"""
lambda_handler.py — AWS Lambda inference endpoint for marine-debris detection.

Deploy-ready (not yet deployed — pending AWS credits). Runs the Stage-2 detector through
``cv2.dnn`` only: no torch / ultralytics, so the deploy package stays small and cold-start is
fast. Works unchanged on x86 and on **Graviton/Arm** Lambda (the COOL-award compute target).

Pipeline per request:
    frame (S3 key or base64) → cv2.dnn ONNX detection (per-class thresholds)
    → **honest** geotag (across-track offset from the boat fix; no coordinates when no GPS) → JSON.

Detection-only by design: STUDY-01 retired classical Stage-1 ROI-gating (it has no discriminative
power on this sonar), so this endpoint does **not** run Stage 1. The Stage-1 sonar-preprocessing
pass is a *separate* CPU workload benchmarked for the COOL award
(``infra/benchmark_graviton.sh``) — it is not a detection gate, so wiring it here would only slow
inference without changing results.

Geotagging is honest (mirrors ``src/agentic/geo.py``): each detection sits at an across-track
ground range from the boat track, offset on the port/starboard side — it is **not** the same
lat/lon stamped onto every object. With no ``lat``/``lon`` in the event we return
``gps_available=False`` and null coordinates rather than inventing a position.

Event shapes accepted (``lat``/``lon`` optional — the boat fix for this frame):
    {"s3": {"bucket": "...", "key": "frames/x.png"}, "lat": .., "lon": ..,
     "heading": deg, "side": "port"|"starboard", "nadir": "top"|.., "m_per_px": ..}
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


# --- honest geotag (mirrors src/agentic/geo.py; no fake "one lat/lon on every object") --------
_EARTH_R = 6_371_000.0


def _offset_latlon(lat, lon, dist_m, bearing_deg):
    import math
    b = math.radians(bearing_deg)
    dlat = (dist_m * math.cos(b)) / _EARTH_R
    dlon = (dist_m * math.sin(b)) / (_EARTH_R * math.cos(math.radians(lat)))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


def _calibrate_nadir(gray) -> str:
    """Darkest + flattest edge band = the nadir / water column; range grows away from it."""
    h, w = gray.shape[:2]
    bh, bw = max(3, h // 12), max(3, w // 12)
    bands = {"top": gray[:bh, :], "bottom": gray[-bh:, :], "left": gray[:, :bw], "right": gray[:, -bw:]}
    return min(bands, key=lambda e: float(bands[e].mean()) + float(bands[e].std()))


def _range_px_from_nadir(bbox, nadir, w, h) -> float:
    cx, cy = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
    return {"top": cy, "bottom": h - cy, "left": cx, "right": w - cx}.get(nadir, cy)


def geotag(dets, img, event) -> bool:
    """Fill each detection's lat/lon/geo_error_m from the boat fix. Returns gps_available."""
    lat, lon = event.get("lat"), event.get("lon")
    if lat is None or lon is None:
        for d in dets:
            d["lat"] = d["lon"] = d["geo_error_m"] = None
        return False
    heading = float(event.get("heading", 0.0))
    side = (event.get("side") or "starboard").lower()
    m_per_px = float(event.get("m_per_px", 0.05))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nadir = event.get("nadir") or _calibrate_nadir(gray)
    H, W = img.shape[:2]
    cross = heading + (90.0 if side.startswith("star") else -90.0)
    for d in dets:
        rng = _range_px_from_nadir(d["bbox"], nadir, W, H) * m_per_px
        dlat, dlon = _offset_latlon(lat, lon, rng, cross)
        d["lat"], d["lon"] = round(dlat, 6), round(dlon, 6)
        d["geo_error_m"] = round(max(3.0, 0.25 * rng), 1)   # coarse, honest, grows with range
    return True


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
    gps_available = geotag(dets, img, event)         # honest per-detection geotag (or nulls)
    import platform as _pf
    body = {
        "detections": dets,
        "count": len(dets),
        "gps_available": gps_available,              # False ⇒ coordinates are null, not invented
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
