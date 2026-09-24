"""
Tests for the Prove-stage shadow measurement (src/agentic/shadow.py).

Pure OpenCV/NumPy — no model needed. Runs under pytest, or standalone: python tests/test_shadow.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic.shadow import ShadowProver, ShadowConfig, _rotate_point
from src.agentic.types import ShadowQuality


def _synthetic_object(nadir: str = "top", size: int = 200) -> tuple[np.ndarray, tuple]:
    """Grey seabed with a bright echo and a dark shadow on the far-range side. Returns (img, box)."""
    rng = np.random.default_rng(0)
    img = (90 + rng.normal(0, 4, (size, size))).clip(40, 140).astype(np.uint8)
    # build with far range = down, then rotate to the requested nadir
    cx, echo_y = 100, 55
    img[echo_y - 6:echo_y + 6, cx - 6:cx + 6] = 210          # bright echo
    img[echo_y + 8:echo_y + 40, cx - 4:cx + 5] = 18          # dark acoustic shadow (below)
    box = (cx - 8, echo_y - 8, cx + 8, echo_y + 8)
    # rotate the far-down base so its nadir lands on the requested edge (top -> X, CCW)
    r = {"top": 0, "left": 1, "bottom": 2, "right": 3}[nadir]
    if r:
        h, w = size, size
        img = np.rot90(img, r)
        (bx1, by1), (bx2, by2) = (_rotate_point(box[0], box[1], r, h, w),
                                  _rotate_point(box[2], box[3], r, h, w))
        box = (min(bx1, bx2), min(by1, by2), max(bx1, bx2), max(by1, by2))
    return img, box


def test_rotate_point_matches_numpy():
    """_rotate_point must agree with np.rot90 for every k."""
    h, w = 12, 20
    marker = np.zeros((h, w), np.uint8)
    x, y = 15, 3
    marker[y, x] = 255
    for k in range(4):
        rot = np.rot90(marker, k)
        ys, xs = np.where(rot == 255)
        got = _rotate_point(x, y, k, h, w)
        assert (int(xs[0]), int(ys[0])) == got, f"k={k}: numpy {(xs[0], ys[0])} != {got}"


def test_shadow_detected_all_orientations():
    """A real echo+shadow must be found regardless of which edge the nadir is on."""
    for nadir in ("top", "bottom", "left", "right"):
        img, box = _synthetic_object(nadir)
        proof = ShadowProver(ShadowConfig(nadir=nadir)).prove(img, box)
        assert proof.has_shadow, f"{nadir}: expected a shadow, got {proof.quality}"
        assert proof.quality is ShadowQuality.CLEAR, f"{nadir}: {proof.quality} contrast={proof.contrast}"
        assert proof.contrast > 0.3 and proof.run_px >= 6
        # no measured altitude => relative height only (never an assumed 10 m)
        assert proof.height_m is None and 0 < proof.height_rel < 1


def test_flat_seabed_has_no_shadow():
    """Featureless seabed must NOT hallucinate a shadow (honest NONE)."""
    rng = np.random.default_rng(1)
    img = (90 + rng.normal(0, 4, (200, 200))).clip(40, 140).astype(np.uint8)
    proof = ShadowProver(ShadowConfig(nadir="top")).prove(img, (96, 50, 112, 70))
    assert not proof.has_shadow
    assert proof.quality is ShadowQuality.NONE


def test_nadir_auto_calibration():
    """The darkest, flattest edge band is picked as the nadir."""
    rng = np.random.default_rng(2)
    img = (90 + rng.normal(0, 4, (200, 200))).clip(40, 140).astype(np.uint8)
    img[:24, :] = 5                                           # dark water-column band at top
    assert ShadowProver(ShadowConfig(nadir="auto")).calibrate_nadir(img) == "top"


def test_height_scales_with_shadow_length():
    """Longer shadow ⇒ taller inferred object (monotonic)."""
    prover = ShadowProver(ShadowConfig(nadir="top"))
    rng = np.random.default_rng(3)

    def height_for(run_len: int) -> float:
        img = (90 + rng.normal(0, 3, (240, 200))).clip(40, 140).astype(np.uint8)
        cx, ey = 100, 55
        img[ey - 6:ey + 6, cx - 6:cx + 6] = 210
        img[ey + 8:ey + 8 + run_len, cx - 4:cx + 5] = 18
        return prover.prove(img, (cx - 8, ey - 8, cx + 8, ey + 8)).height_rel

    assert height_for(40) > height_for(15) > 0


def test_metres_only_with_measured_altitude():
    img, box = _synthetic_object("top")
    rel = ShadowProver(ShadowConfig(nadir="top")).prove(img, box)
    met = ShadowProver(ShadowConfig(nadir="top")).prove(img, box, altitude_m=3.0)
    assert rel.height_m is None
    assert met.height_m is not None and abs(met.height_m - 3.0 * met.height_rel) < 0.02


def test_unknown_orientation_is_not_measured():
    """No nadir => nothing is measured (the old auto-guess was wrong 74% of the time)."""
    img, box = _synthetic_object("top")
    proof = ShadowProver().prove(img, box)                     # default config: nadir unknown
    assert proof.orientation_known is False and not proof.has_shadow


def test_thin_line_shadow_is_found_in_a_wide_box():
    """PINGMapper shadows are 2-4 px lines; a 3-px line under a wide box must still read CLEAR."""
    rng = np.random.default_rng(5)
    img = (90 + rng.normal(0, 4, (220, 200))).clip(40, 140).astype(np.uint8)
    cx, ey = 100, 55
    img[ey - 6:ey + 6, cx - 14:cx + 14] = 210                  # wide bright echo
    img[ey + 8:ey + 50, cx - 1:cx + 2] = 15                    # 3-px dark line shadow
    proof = ShadowProver(ShadowConfig(nadir="top")).prove(img, (cx - 16, ey - 8, cx + 16, ey + 8))
    assert proof.quality is ShadowQuality.CLEAR, (proof.quality, proof.contrast, proof.run_px)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} shadow tests passed")


if __name__ == "__main__":
    _run_all()
