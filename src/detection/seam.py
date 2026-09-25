"""
seam.py — seam inference across chunk boundaries (STUDY-11b).

Side-scan recordings are cut into consecutive chunks (PINGMapper: ``…_ss_port_00005`` then
``…_00006``), pings running left → right, so chunk k's right edge continues into chunk k+1's left
edge. A pot cut by that boundary is half an object in each frame — the largest single failure mode of
EXP-001 (STUDY-11a: 32% of missed pots touch the frame edge vs 11% of found ones).

:func:`seam_detections` places a full-size detector window **straddling each boundary** — half from
the neighbour, half from this frame — so the cut object is seen whole at the training scale, maps
every detection back into this frame's coordinates, keeps only those that reach into the frame near
the boundary, clips them to the frame and merges them with the frame's own detections (class-aware
NMS). No retraining; +1 inference per boundary (shared by both frames in a survey).
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

import cv2
import numpy as np

from .infer import Detection, YoloOnnxDetector, nms

_KEY = re.compile(r"^(.*?_ss_(?:port|star)\w*?_)(\d+)", re.I)


def chunk_key(frame_id: str) -> Optional[tuple[str, int]]:
    """(series prefix incl. side, chunk index) for PINGMapper-style names; None otherwise."""
    m = _KEY.search(frame_id or "")
    return (m.group(1).lower(), int(m.group(2))) if m else None


def neighbours(frame_ids: Sequence[str]) -> dict[str, tuple[Optional[str], Optional[str]]]:
    """frame_id → (previous chunk id, next chunk id) within the given set."""
    idx = {}
    for f in frame_ids:
        k = chunk_key(f)
        if k:
            idx[k] = f
    out = {}
    for f in frame_ids:
        k = chunk_key(f)
        out[f] = (idx.get((k[0], k[1] - 1)), idx.get((k[0], k[1] + 1))) if k else (None, None)
    return out


def _same_height(a: np.ndarray, h: int) -> np.ndarray:
    return a if a.shape[0] == h else cv2.resize(a, (int(round(a.shape[1] * h / a.shape[0])), h), interpolation=cv2.INTER_AREA)


def seam_detections(det: YoloOnnxDetector, cur: np.ndarray, prev: Optional[np.ndarray] = None,
                    nxt: Optional[np.ndarray] = None, reach: float = 0.5) -> tuple[list[Detection], int]:
    """Detections from windows straddling the left (prev|cur) and right (cur|next) boundaries, in
    ``cur`` pixel coordinates, clipped to ``cur``. Only boxes that reach into the frame from the seam
    (their near edge within ``reach`` × window half-width of the boundary) are kept. Returns
    (detections, inferences spent)."""
    H, W = cur.shape[:2]
    half = W // 2
    out, n = [], 0
    for side, nb in (("left", prev), ("right", nxt)):
        if nb is None:
            continue
        nb = _same_height(nb, H)
        if side == "left":
            win = np.hstack([nb[:, nb.shape[1] - half:], cur[:, :W - half]])
            off = -half                                    # window x → cur x
        else:
            win = np.hstack([cur[:, half:], nb[:, :W - half]])
            off = half
        n += 1
        for d in det.detect(win):
            x1, y1, x2, y2 = d.bbox
            cx1, cx2 = x1 + off, x2 + off
            crosses = cx1 < 0 < cx2 if side == "left" else cx1 < W < cx2        # cut by the boundary
            near = (cx1 < reach * half) if side == "left" else (cx2 > W - reach * half)
            if not (crosses or near) or cx2 <= 0 or cx1 >= W:
                continue
            out.append(Detection(bbox=(max(0, cx1), y1, min(W, cx2), y2), cls_id=d.cls_id,
                                 cls_name=d.cls_name, conf=d.conf))
    return out, n


def merge(frame_dets: list[Detection], extra: list[Detection], iou: float = 0.45) -> list[Detection]:
    """Class-aware NMS over the frame's own detections + the seam detections."""
    allx = list(frame_dets) + list(extra)
    if not allx:
        return []
    boxes = np.array([[d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]] for d in allx], np.float32)
    confs = np.array([d.conf for d in allx], np.float32)
    cls = np.array([d.cls_id for d in allx], np.int32)
    keep = nms(boxes, confs, cls, 0.0, iou, True)
    return [allx[i] for i in sorted(keep, key=lambda i: -confs[i])]
