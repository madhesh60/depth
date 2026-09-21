"""
pipeline.py — Stage 1 classical-CV candidate generation for marine-debris detection.

This is the **COOL core workload**: the classical OpenCV pass that runs CPU-only on AWS
Graviton (via COOL) and produces candidate Regions of Interest (ROIs) for the Stage 2 YOLO
verifier. It ingests a side-scan sonar frame and returns geometry-filtered candidate boxes,
plus a per-op latency breakdown used by the Arm-vs-x86 / COOL-vs-stock benchmark.

Design
------
* Every op is timed individually (``Stage1Result.timings_ms``) so the benchmark can prove
  *where* the compute goes — the ops COOL accelerates (resize, adaptive-gaussian threshold,
  contour detection) should dominate, which is what makes the COOL speedup attributable.
* Behaviour is fully driven by ``Stage1Config`` — no magic numbers here.
* Pure OpenCV + NumPy, CPU-only: no GPU-only ops, so the identical code runs on Graviton.

Pipeline: to-gray → resize → denoise → adaptive threshold → morphology → findContours →
geometry filter → padded candidate ROIs.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import Stage1Config


@dataclass
class Candidate:
    """A single candidate ROI handed to Stage 2."""
    bbox: tuple[int, int, int, int]      # x1, y1, x2, y2 in ORIGINAL image pixels
    area_frac: float                     # contour area / frame area (resized space)
    solidity: float
    extent: float
    aspect: float


@dataclass
class Stage1Result:
    candidates: list[Candidate]
    mask: np.ndarray                     # binary mask after morphology (resized space)
    timings_ms: dict[str, float]         # per-op latency, milliseconds
    scale: float                         # resized->original coordinate scale factor
    frame_shape: tuple[int, int]         # original (h, w)

    @property
    def total_ms(self) -> float:
        return float(sum(self.timings_ms.values()))


class Stage1Pipeline:
    """Stateless classical-CV candidate generator. Reuse one instance across frames."""

    def __init__(self, cfg: Optional[Stage1Config] = None):
        self.cfg = (cfg or Stage1Config()).validate()

    # -- lightweight per-op timer --------------------------------------------------------
    @contextmanager
    def _timed(self, name: str, sink: dict[str, float]):
        t0 = time.perf_counter()
        yield
        sink[name] = (time.perf_counter() - t0) * 1000.0

    # -- main entry ----------------------------------------------------------------------
    def process(self, image: np.ndarray) -> Stage1Result:
        cfg = self.cfg
        t: dict[str, float] = {}
        h0, w0 = image.shape[:2]

        with self._timed("to_gray", t):
            gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # resize (COOL-accelerated). scale maps resized coords back to the original frame.
        with self._timed("resize", t):
            scale = 1.0
            if cfg.resize_long_side and max(h0, w0) != cfg.resize_long_side:
                scale = max(h0, w0) / float(cfg.resize_long_side)
                new_wh = (round(w0 / scale), round(h0 / scale))
                gray = cv2.resize(gray, new_wh, interpolation=cv2.INTER_AREA)

        with self._timed("denoise", t):
            gray = self._denoise(gray)

        with self._timed("threshold", t):
            binary = self._threshold(gray)

        with self._timed("morphology", t):
            binary = self._morphology(binary)

        with self._timed("contours", t):
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        with self._timed("geometry_filter", t):
            candidates = self._filter(contours, gray.shape, scale)

        return Stage1Result(
            candidates=candidates, mask=binary, timings_ms=t,
            scale=scale, frame_shape=(h0, w0),
        )

    # -- ops -----------------------------------------------------------------------------
    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        c = self.cfg
        if c.denoise == "gaussian":
            k = c.gaussian_ksize | 1
            return cv2.GaussianBlur(gray, (k, k), 0)
        if c.denoise == "median":
            return cv2.medianBlur(gray, c.median_ksize | 1)
        if c.denoise == "bilateral":
            return cv2.bilateralFilter(gray, c.bilateral_d, c.bilateral_sigma, c.bilateral_sigma)
        if c.denoise == "nlmeans":
            return cv2.fastNlMeansDenoising(gray, None, c.nlmeans_h, 7, 21)
        return gray  # "none"

    def _threshold(self, gray: np.ndarray) -> np.ndarray:
        c = self.cfg
        if c.thresh_method == "adaptive_gaussian":
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, c.adaptive_block, c.adaptive_C,
            )
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _morphology(self, binary: np.ndarray) -> np.ndarray:
        c = self.cfg
        if c.morph_op == "none":
            return binary
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c.morph_ksize, c.morph_ksize))
        it = c.morph_iterations
        if c.morph_op in ("open", "both"):
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=it)
        if c.morph_op in ("close", "both"):
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=it)
        return binary

    def _filter(self, contours, shape, scale: float) -> list[Candidate]:
        c = self.cfg
        h, w = shape
        frame_area = float(h * w)
        pad = c.roi_pad_frac
        out: list[Candidate] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_frac = area / frame_area
            if area_frac < c.min_area_frac or area_frac > c.max_area_frac:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw == 0 or bh == 0:
                continue
            extent = area / float(bw * bh)
            if extent < c.min_extent:
                continue
            aspect = max(bw, bh) / float(min(bw, bh))
            if aspect > c.max_aspect:
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < c.min_solidity:
                continue
            # pad, map resized->original pixels, clamp
            px, py = int(bw * pad), int(bh * pad)
            x1 = max(0, int((x - px) * scale));       y1 = max(0, int((y - py) * scale))
            x2 = min(int(w * scale), int((x + bw + px) * scale))
            y2 = min(int(h * scale), int((y + bh + py) * scale))
            out.append(Candidate((x1, y1, x2, y2), area_frac, solidity, extent, aspect))
        out.sort(key=lambda cd: cd.area_frac, reverse=True)
        return out[: c.max_candidates]


def draw_candidates(image: np.ndarray, result: Stage1Result) -> np.ndarray:
    """Render candidate ROIs on a copy of the frame (for the transparency/demo view)."""
    vis = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    vis = vis.copy()
    for cd in result.candidates:
        x1, y1, x2, y2 = cd.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"sol{cd.solidity:.2f}", (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(vis, f"{len(result.candidates)} ROIs | {result.total_ms:.1f} ms",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return vis
