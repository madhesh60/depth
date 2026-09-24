"""
perception.py — the **See** stage and the agent's re-look tool.

``Perceptor`` wraps the existing ``YoloOnnxDetector`` (``cv2.dnn``, no torch) and adds the two
operations the Prove/Decide stages need:

* :meth:`Perceptor.perceive`    — full-frame detection (per-class thresholds, hot for fishing_gear).
* :meth:`Perceptor.zoom_relook` — crop a padded window around a candidate, upscale it, and re-detect.
  A real object re-fires strongly at higher effective resolution; a speckle false-positive does
  not. This *re-look persistence* lifts precision from 0.60 (raw) to 0.71 at its CONFIRMED tier;
  combined with the high-confidence path the CONFIRMED tier is ~0.74 at ~30% recall (``experiments.md``
  STUDY-03/04) — the parameters below were tuned on the real test split and stay in lockstep with
  ``calibrate.py``.
* :meth:`Perceptor.enhance_contrast` — CLAHE, offered to the agent as a secondary re-look aid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from src.detection.infer import YoloOnnxDetector, Detection, PER_CLASS_CONF, DEFAULT_ONNX
from .types import RelookResult

# Re-look geometry (validated on the crab-pot test split — see STUDY-03).
RELOOK_PAD_FRAC = 1.5        # context pad = this × max(box side)
RELOOK_TARGET_PX = 220       # upscale the crop so its long side ≈ this (object ~detector scale)
RELOOK_MATCH_PX = 12         # min half-window (px) to accept a re-fire as "the same object"


class Perceptor:
    """See-stage sensor + re-look tool. Reuse one instance across frames."""

    def __init__(
        self,
        onnx_path: str | Path = DEFAULT_ONNX,
        conf_thres: "float | dict[str, float]" = PER_CLASS_CONF,
        iou_thres: float = 0.45,
        relook_conf: float = 0.05,
    ):
        # main detector runs at the per-class thresholds (hot for fishing_gear, 0.10)
        self.detector = YoloOnnxDetector(onnx_path, conf_thres=conf_thres, iou_thres=iou_thres)
        # the re-look detector runs even hotter so a faint re-fire is still measurable
        rl = dict(PER_CLASS_CONF)
        rl["fishing_gear"] = min(rl.get("fishing_gear", 0.10), relook_conf)
        self.relook_detector = YoloOnnxDetector(onnx_path, conf_thres=rl, iou_thres=iou_thres)

    # -- See ---------------------------------------------------------------------------------
    def perceive(self, frame: np.ndarray) -> list[Detection]:
        """Full-frame detection (the See stage)."""
        return self.detector.detect(frame)

    # -- re-look tool (used by Prove/Decide) -------------------------------------------------
    def zoom_relook(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        cls_name: str = "fishing_gear",
        enhance: bool = False,
    ) -> RelookResult:
        """Crop a padded window around ``bbox``, upscale it, re-detect, and report the best
        same-class re-fire that lands on the original object centre.

        ``enhance=True`` first runs CLAHE on the crop — the agent's "try harder" pass for an
        otherwise uncertain candidate (local-contrast boost before the higher-resolution look).
        """
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        H, W = frame.shape[:2]
        pad = int(max(bw, bh) * RELOOK_PAD_FRAC) + 10
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return RelookResult(found=False, conf=0.0, gain=0.0, scale=1.0)
        if enhance:
            crop = self.enhance_contrast(crop)       # CLAHE → single-channel; detector re-colours

        long_side = max(1, max(crop.shape[:2]))
        scale = max(1.0, RELOOK_TARGET_PX / long_side)
        big = cv2.resize(
            crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
        # object centre expressed in crop pixels
        ocx, ocy = (x1 + x2) / 2 - cx1, (y1 + y2) / 2 - cy1
        tol_x, tol_y = max(bw, RELOOK_MATCH_PX), max(bh, RELOOK_MATCH_PX)

        best = 0.0
        for d in self.relook_detector.detect(big):
            if d.cls_name != cls_name:
                continue
            dcx = 0.5 * (d.bbox[0] + d.bbox[2]) / scale
            dcy = 0.5 * (d.bbox[1] + d.bbox[3]) / scale
            if abs(dcx - ocx) <= tol_x and abs(dcy - ocy) <= tol_y:
                best = max(best, d.conf)
        return RelookResult(found=best > 0.0, conf=float(best), gain=0.0, scale=float(scale))

    # -- secondary re-look aid ---------------------------------------------------------------
    @staticmethod
    def enhance_contrast(crop: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
        """CLAHE local-contrast boost — offered to the agent as an extra re-look aid."""
        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        return clahe.apply(gray)

    # -- the agent's-eye view (display-only; the images zoom_relook detects on) ---------------
    @staticmethod
    def relook_view(
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        target_px: int = 340,
    ) -> Optional[dict]:
        """Reproduce the exact re-look crop geometry (:meth:`zoom_relook`) as **display images**
        for the UI — no inference, so this never touches the calibrated agent path.

        Returns ``{"zoom", "enhanced", "scale", "obj_box"}`` where ``zoom`` is the plain
        high-quality upscale (LANCZOS4 — what a naive zoom shows) and ``enhanced`` is the agent's
        CLAHE "try-harder" pass, both BGR at ``obj_box`` = the object rectangle in crop pixels.
        This lets the dashboard show *what the agent actually saw* when it zoomed in.
        """
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        H, W = frame.shape[:2]
        pad = int(max(bw, bh) * RELOOK_PAD_FRAC) + 10
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        long_side = max(1, max(crop.shape[:2]))
        scale = max(1.0, target_px / long_side)
        dsize = (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale)))
        zoom = cv2.resize(crop, dsize, interpolation=cv2.INTER_LANCZOS4)
        enh_gray = Perceptor.enhance_contrast(crop)                       # CLAHE on the native crop
        enh = cv2.resize(cv2.cvtColor(enh_gray, cv2.COLOR_GRAY2BGR), dsize,
                         interpolation=cv2.INTER_LANCZOS4)
        obj_box = [int((x1 - cx1) * scale), int((y1 - cy1) * scale),
                   int((x2 - cx1) * scale), int((y2 - cy1) * scale)]
        return {"zoom": zoom, "enhanced": enh, "scale": float(scale), "obj_box": obj_box}
