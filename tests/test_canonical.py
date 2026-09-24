"""
Tests for Stage-1 sonar canonicalisation (src/cv_pipeline/canonical.py) on synthetic sonograms with a
KNOWN altitude: ring-down band → dark water column → seabed from row ``alt``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv_pipeline.canonical import (Canonicaliser, bottom_track, canonical_view, range_gain,
                                       ground_range_remap, to_luminance)


def _sonogram(alt: int = 18, h: int = 640, w: int = 640, seed: int = 0, slope: float = 0.0) -> np.ndarray:
    """nadir at top: rows 0-5 bright ring-down, water column dark, seabed (speckle) from ``alt``
    (+ ``slope`` px per 100 pings), fading with range like real side-scan."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    img[:6] = 120
    img[6:] = 70
    for x in range(w):
        a = int(round(alt + slope * x / 100))
        img[a:, x] = 125 - 45 * np.arange(h - a) / h
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_bottom_track_recovers_known_altitude():
    for alt in (12, 18, 30):
        line, a, conf, step = bottom_track(_sonogram(alt))
        assert a is not None and abs(a - alt) <= 1.5, (alt, a)
        assert conf > 0.9 and step > 8


def test_bottom_track_follows_a_sloping_seabed():
    line, a, _, _ = bottom_track(_sonogram(14, slope=2.0))           # 14 → ~27 px across the frame
    assert line[20] < line[-20] - 8                                   # tracks the slope, not a constant


def test_no_water_column_means_not_measured():
    g = np.clip(np.random.default_rng(1).normal(100, 8, (640, 640)), 0, 255).astype(np.uint8)
    line, a, conf, _ = bottom_track(g)
    assert line is None and a is None


def test_orientation_rules_rotate_before_tracking():
    C = Canonicaliser()
    g = _sonogram(20)
    # a wcp filename → nadir top by source rule
    cf = C.process(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "Rec1_wcp_ss_port_00001")
    assert cf.orientation.nadir == "top" and abs(cf.altitude_px - 20) <= 1.5
    # the same frame rotated so the nadir is on the LEFT, with an explicit override → same altitude
    left = np.ascontiguousarray(np.rot90(g, 1))                     # CCW: top edge → left edge
    cf2 = C.process(cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), "upload", nadir="left")
    assert abs(cf2.altitude_px - 20) <= 1.5
    # unknown source → nothing measured, never guessed
    cf3 = C.process(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "mosaic_001")
    assert cf3.orientation.nadir is None and not cf3.measured and cf3.bottom_polyline() == []


def test_ground_range_and_water_column():
    C = Canonicaliser()
    cf = C.process(cv2.cvtColor(_sonogram(30), cv2.COLOR_GRAY2BGR), "Rec1_wcp_ss_star_00002")
    alt = cf.altitude_px
    # ground = sqrt(slant^2 - alt^2); far away it approaches the slant range
    a100 = cf.altitude_at(100)                                        # per-ping altitude
    assert abs(cf.ground_range_px(100, 50) - np.sqrt(50 ** 2 - a100 ** 2)) < 1e-6
    assert cf.ground_range_px(100, 400) > 398
    assert cf.in_water_column((100, 8, 120, 20)) is True              # above the seabed
    assert cf.in_water_column((100, 200, 120, 220)) is False
    # the polyline is in native pixels and lies on the tracked seabed
    pts = cf.bottom_polyline(16)
    assert len(pts) == 16 and all(abs(y - alt) <= 3 for _, y in pts)


def test_ground_remap_and_gain_shapes():
    g = _sonogram(25)
    line, a, _, _ = bottom_track(g)
    gv = ground_range_remap(g, line)
    assert gv.shape == g.shape and gv.dtype == np.uint8
    out = range_gain(g, a)
    assert out.shape == g.shape and out.dtype == np.uint8
    # gain equalises range fall-off: far-range median is lifted towards the near-range median
    before = np.median(g[500:]) / np.median(g[60:120])
    after = np.median(out[500:]) / np.median(out[60:120])
    assert after > before


def test_palette_detection():
    gray = cv2.cvtColor(_sonogram(), cv2.COLOR_GRAY2BGR)
    _, kind = to_luminance(gray)
    assert kind == "gray"
    orange = cv2.applyColorMap(_sonogram(), cv2.COLORMAP_HOT)
    lum, kind = to_luminance(orange)
    assert kind == "color" and lum.ndim == 2


def test_detector_input_modes():
    C = Canonicaliser()
    im = cv2.cvtColor(_sonogram(), cv2.COLOR_GRAY2BGR)
    cf = C.process(im, "Rec1_wcp_ss_port_00003")
    assert C.detector_input(im, cf, "raw") is im
    for m in ("gray", "gain"):
        x = C.detector_input(im, cf, m)
        assert x.shape == im.shape and x.dtype == np.uint8
    assert canonical_view(cf.gray, None) is None


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} canonical tests passed")


if __name__ == "__main__":
    _run_all()
