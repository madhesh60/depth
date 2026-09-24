"""
canonical.py — **Stage 1: sonar canonicalisation** (classical OpenCV, CPU-only; the COOL workload
that now sits in the product path).

STUDY-01 retired Stage 1 as an ROI gate (no discriminative power). Review I-4 gave it a real job:
turn "a picture from some sonar" into **measured sonar geometry** that the later stages need.
Per frame, in order:

1. **palette → luminance** — colour-palette sonars (e.g. the orange ``Contact_*_sslo`` crops) are
   mapped back to a monotonic intensity (HSV value = max channel); greyscale passes through.
2. **canonical orientation** — the frame is viewed with the nadir (sonar track) at the TOP and far
   range DOWN (``np.rot90`` by the source rule from :mod:`orientation`; never guessed). Unknown
   orientation ⇒ steps 3–5 are skipped and reported as not measured.
3. **bottom tracking** — PINGMapper ``wcp`` sonograms keep the water column: a bright transducer
   ring-down, a dark water-column dip, then a sharp first seabed return. A frame-level profile finds
   the dip + return; each ping (column) is then tracked in a window around it and median-filtered
   along track. Output: the per-ping first-return row (**sonar altitude in pixels**) + a tracking
   confidence. No water column / no clear step ⇒ ``altitude_px = None`` (honest "not measured").
4. **slant → ground range** — ``ground = sqrt(slant² − altitude²)`` per ping, as a ``cv2.remap``
   (display + geotag range) and as :meth:`CanonicalFrame.ground_range_px` for single points.
5. **range-gain normalisation** — side-scan returns fade with range; a per-range-row gain
   (median profile below the seabed, smoothed) equalises it. Offered as a detector-input option;
   whether it helps the *deployed* model is measured, not assumed (``experiments.md`` STUDY-08).
6. **water-column mask** — a box entirely above the tracked seabed is in the water column (fish,
   bubbles, surface noise), not on the seabed.

The detector still sees the frame it was trained on unless ``calibration.json`` selects another
``detector_input``; Stage 1 is about *measurement* (altitude → relative height, ground range →
geotag, water column → evidence), and every op is timed for the Graviton/COOL benchmark.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .orientation import Orientation, resolve_orientation

# rot90 count that brings each nadir edge to the top (np.rot90 is counter-clockwise)
_TO_TOP_K = {"top": 0, "left": 3, "bottom": 2, "right": 1}

BAND_FRAC = 0.35          # the seabed's first return is searched in the top 35% of the range axis
MIN_STEP = 8.0            # grey levels between water-column dip and seabed for a valid track
PING_SMOOTH = 15          # pings (columns) averaged before tracking — speckle suppression
TRACK_WIN = 8             # per-ping search half-window (rows) around the frame-level bottom
RISE_ROWS = 6             # the seabed return must rise ≥ MIN_STEP within this many rows of the dip
ALONG_MEDIAN = 31         # along-track median filter of the bottom line (pings)
PALETTE_SAT = 40.0        # mean HSV saturation above which a frame is treated as colour palette


@dataclass
class CanonicalFrame:
    """Stage-1 output: the measured geometry of one sonar frame (native frame coordinates kept)."""
    orientation: Orientation
    palette: str                                  # "gray" | "color"
    gray: np.ndarray                              # luminance, native orientation (uint8)
    height: int
    width: int
    bottom_line: Optional[np.ndarray] = None      # per-ping first-return row, CANONICAL view (float)
    altitude_px: Optional[float] = None           # median of bottom_line (slant px), None = not measured
    track_conf: float = 0.0                       # share of pings with a clear water→seabed step
    step: float = 0.0                             # frame-level seabed - water-column contrast (grey levels)
    timings_ms: dict[str, float] = field(default_factory=dict)

    # -- geometry helpers ---------------------------------------------------------------------
    @property
    def measured(self) -> bool:
        return self.altitude_px is not None

    def _k(self) -> Optional[int]:
        return _TO_TOP_K.get(self.orientation.nadir or "")

    def to_canonical(self, x: float, y: float) -> Optional[tuple[float, float]]:
        """Native pixel → canonical (ping index, slant-range row). None if orientation unknown."""
        k, H, W = self._k(), self.height, self.width
        if k is None:
            return None
        return {0: (x, y), 1: (y, W - 1 - x), 2: (W - 1 - x, H - 1 - y), 3: (H - 1 - y, x)}[k]

    def altitude_at(self, ping: float) -> Optional[float]:
        if self.bottom_line is None:
            return self.altitude_px
        i = int(np.clip(round(ping), 0, len(self.bottom_line) - 1))
        return float(self.bottom_line[i])

    def ground_range_px(self, x: float, y: float) -> Optional[float]:
        """Across-track GROUND range (px) of a native pixel: sqrt(slant² − altitude²) at its ping.
        Without a measured altitude the slant range is returned (the old behaviour, flagged by
        :attr:`measured`); None if orientation is unknown."""
        c = self.to_canonical(x, y)
        if c is None:
            return None
        ping, slant = c
        alt = self.altitude_at(ping)
        if alt is None:
            return float(slant)
        return float(np.sqrt(max(slant * slant - alt * alt, 0.0)))

    def in_water_column(self, bbox) -> Optional[bool]:
        """True if the whole box lies above the tracked seabed (suspended in the water column).
        None when the bottom was not tracked."""
        if self.bottom_line is None:
            return None
        x1, y1, x2, y2 = bbox
        corners = [self.to_canonical(x, y) for x, y in ((x1, y1), (x2, y2), (x1, y2), (x2, y1))]
        if any(c is None for c in corners):
            return None
        pings = [c[0] for c in corners]
        far = max(c[1] for c in corners)                     # deepest slant row of the box
        lo, hi = int(max(0, min(pings))), int(min(len(self.bottom_line) - 1, max(pings)))
        seabed = float(np.min(self.bottom_line[lo:hi + 1])) if hi >= lo else self.altitude_px
        return bool(far < seabed - 1)

    def to_dict(self) -> dict:
        d = {"orientation": self.orientation.to_dict(), "palette": self.palette,
             "altitude_px": None if self.altitude_px is None else round(self.altitude_px, 1),
             "track_conf": round(self.track_conf, 3), "step": round(self.step, 1),
             "measured": self.measured, "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()}}
        if self.bottom_line is not None:
            d["bottom_line_native"] = self.bottom_polyline(max_points=64)
        return d

    def bottom_polyline(self, max_points: int = 64) -> list[list[int]]:
        """The tracked seabed line in NATIVE pixel coordinates (for the UI overlay)."""
        if self.bottom_line is None:
            return []
        n = len(self.bottom_line)
        idx = np.linspace(0, n - 1, min(n, max_points)).round().astype(int)
        k, H, W = self._k(), self.height, self.width
        pts = []
        for i in idx:
            r = float(self.bottom_line[i])
            x, y = {0: (i, r), 1: (W - 1 - r, i), 2: (W - 1 - i, H - 1 - r), 3: (r, H - 1 - i)}[k]
            pts.append([int(round(x)), int(round(y))])
        return pts


# ================================================================================ ops
def to_luminance(img: np.ndarray) -> tuple[np.ndarray, str]:
    """Palette → monotonic intensity. Greyscale-looking frames use the standard luma conversion;
    colour-palette frames (orange/copper sonar palettes) use HSV value (= max channel), which is
    monotonic in echo strength for those palettes where luma is not."""
    if img.ndim == 2:
        return img, "gray"
    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    sat = float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1].mean())
    if sat > PALETTE_SAT:
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2], "color"
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), "gray"


def canonical_view(gray: np.ndarray, nadir: Optional[str]) -> Optional[np.ndarray]:
    """Rotate so the nadir is the top edge (far range points down). None if orientation unknown."""
    k = _TO_TOP_K.get(nadir or "")
    if k is None:
        return None
    return np.ascontiguousarray(np.rot90(gray, k)) if k else gray


def bottom_track(gc: np.ndarray) -> tuple[Optional[np.ndarray], Optional[float], float, float]:
    """Track the first seabed return in a CANONICAL view (nadir at top).

    Returns ``(bottom_line, altitude_px, track_conf, step)``; ``(None, None, 0, step)`` when the frame
    shows no water-column → seabed step (water column removed, or not a sonogram)."""
    H, W = gc.shape
    band = int(max(12, BAND_FRAC * H))
    sm = cv2.blur(gc, (PING_SMOOTH, 3)).astype(np.float32)

    # frame level: robust row profile. Near the transducer the profile goes ring-down (bright) →
    # water-column dip (dark) → first seabed return (sharp rise). The seabed further out can be
    # darker than the water column, so the dip is the FIRST local minimum that is followed by a
    # rise of ≥ MIN_STEP within RISE_ROWS — not the global minimum.
    prof = np.median(sm[:band], axis=1)
    prof = np.convolve(np.pad(prof, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    d0 = b0 = None
    step = 0.0
    for i in range(2, band - RISE_ROWS - 1):                         # rows 0-1: image edge
        if prof[i] > prof[i - 1] or prof[i] > prof[i + 1]:
            continue                                                # not a local minimum
        rise = float(prof[i + 1:i + 1 + RISE_ROWS].max() - prof[i])
        if rise >= MIN_STEP:
            d0, step = i, rise
            half = prof[i] + 0.5 * rise
            b0 = i + 1 + int(np.nonzero(prof[i + 1:i + 1 + RISE_ROWS] >= half)[0][0])
            break
    if d0 is None:
        return None, None, 0.0, step

    # per ping: strongest dark→bright step in a window around the frame-level bottom
    lo, hi = max(1, d0, b0 - TRACK_WIN), min(band - 1, b0 + TRACK_WIN)   # never above the dip (ring-down)
    grad = sm[lo + 1:hi + 1] - sm[lo - 1:hi - 1]                     # centred vertical difference
    rows = lo + np.argmax(grad, axis=0).astype(np.float32) + 0.5    # centred diff peaks 1 row early
    strength = grad.max(axis=0)
    ok = strength >= 0.25 * step                                     # a real step, not flat speckle
    track_conf = float(ok.mean())
    if track_conf < 0.3:
        return None, None, track_conf, step
    line = rows.copy()
    if not ok.all():                                                 # fill weak pings by interpolation
        idx = np.arange(W)
        line[~ok] = np.interp(idx[~ok], idx[ok], rows[ok])
    k = min(ALONG_MEDIAN, W if W % 2 else W - 1)
    if k >= 5:                                                       # median (spikes) then mean (jitter)
        line = cv2.medianBlur(line.reshape(1, -1).astype(np.float32), 5).ravel()
        line = np.convolve(np.pad(line, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[:W]
    return line, float(np.median(line)), track_conf, step


def ground_range_remap(gc: np.ndarray, bottom_line: np.ndarray) -> np.ndarray:
    """Slant-range → ground-range resampling of a canonical view (``cv2.remap``): output row g of
    ping x samples slant row ``sqrt(g² + alt_x²)``. The water column collapses to the nadir."""
    H, W = gc.shape[:2]
    g = np.arange(H, dtype=np.float32).reshape(-1, 1)
    alt = bottom_line.astype(np.float32).reshape(1, -1)
    map_y = np.sqrt(g * g + alt * alt)
    map_x = np.broadcast_to(np.arange(W, dtype=np.float32).reshape(1, -1), (H, W)).copy()
    return cv2.remap(gc, map_x, map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def range_gain(gc: np.ndarray, altitude_px: Optional[float], target: Optional[float] = None) -> np.ndarray:
    """Equalise brightness along range (a data-driven TVG): each range row is scaled so its median
    matches the frame's seabed median. Rows above the seabed (water column) are left untouched."""
    H = gc.shape[0]
    start = int(altitude_px) + 2 if altitude_px else 0
    prof = np.median(gc, axis=1).astype(np.float32)
    k = max(5, (H // 20) | 1)
    prof = np.convolve(np.pad(prof, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[:H]
    tgt = float(target if target is not None else np.median(prof[start:]))
    gain = np.ones(H, np.float32)
    gain[start:] = np.clip(tgt / np.maximum(prof[start:], 1.0), 0.5, 3.0)
    return np.clip(gc.astype(np.float32) * gain.reshape(-1, 1), 0, 255).astype(np.uint8)


def from_canonical(view: np.ndarray, nadir: Optional[str]) -> np.ndarray:
    k = _TO_TOP_K.get(nadir or "", 0)
    return np.ascontiguousarray(np.rot90(view, (4 - k) % 4)) if k else view


# ================================================================================ pipeline
class Canonicaliser:
    """Stage 1. Stateless; reuse one instance. ``process`` never raises on odd input — anything it
    cannot measure is reported as not measured."""

    @contextmanager
    def _t(self, name: str, sink: dict):
        t0 = time.perf_counter()
        yield
        sink[name] = sink.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0

    def process(self, image: np.ndarray, frame_id: str = "", nadir: Optional[str] = None) -> CanonicalFrame:
        t: dict[str, float] = {}
        with self._t("luminance", t):
            gray, palette = to_luminance(image)
        orient = resolve_orientation(frame_id, override=nadir, gray=gray)
        H, W = gray.shape[:2]
        cf = CanonicalFrame(orientation=orient, palette=palette, gray=gray, height=H, width=W, timings_ms=t)
        with self._t("orient", t):
            gc = canonical_view(gray, orient.nadir)
        if gc is None:
            return cf
        with self._t("bottom_track", t):
            line, alt, conf, step = bottom_track(gc)
        cf.bottom_line, cf.altitude_px, cf.track_conf, cf.step = line, alt, conf, step
        return cf

    def detector_input(self, image: np.ndarray, cf: CanonicalFrame, mode: str = "raw") -> np.ndarray:
        """The image the detector sees. ``raw`` (default: what the model was trained on) |
        ``gain`` (range-gain normalised luminance) | ``gray`` (luminance only)."""
        if mode == "raw" or image.ndim == 2 and mode == "gray":
            return image
        t = cf.timings_ms
        if mode == "gray":
            return cv2.cvtColor(cf.gray, cv2.COLOR_GRAY2BGR)
        if mode == "gain":
            with self._t("range_gain", t):
                gc = canonical_view(cf.gray, cf.orientation.nadir)
                if gc is None:
                    return image
                out = from_canonical(range_gain(gc, cf.altitude_px), cf.orientation.nadir)
            return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        raise ValueError(f"unknown detector_input mode: {mode}")

    def ground_view(self, cf: CanonicalFrame) -> Optional[np.ndarray]:
        """Ground-range corrected canonical view (display + benchmark); None if not measured."""
        if cf.bottom_line is None:
            return None
        with self._t("ground_remap", cf.timings_ms):
            gc = canonical_view(cf.gray, cf.orientation.nadir)
            return ground_range_remap(gc, cf.bottom_line)
