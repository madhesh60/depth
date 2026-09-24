"""
Tests for source-rule orientation (src/cv_pipeline/orientation.py). Standalone: python tests/test_orientation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv_pipeline.orientation import resolve_orientation, auto_guess_nadir


def test_pingmapper_sonograms_are_top():
    for fid in ("Rec9_wcp_ss_port_00031_png_jpg.rf.abc", "crabpot_Rec09_Sensor_Depth_wcp_ss_star_00005",
                "Rec6_wcp_ss_starboard_00030"):
        o = resolve_orientation(fid)
        assert o.nadir == "top" and o.source == "source-rule" and o.known, fid


def test_unrecognised_sources_are_unknown_not_guessed():
    for fid in ("BC_POST_T2_00_00_1_7", "baycove_01_18", "TI0030", "Contact_203_sslo", "upload"):
        o = resolve_orientation(fid)
        assert o.nadir is None and o.label == "unknown" and not o.known, fid


def test_user_override_wins_and_auto_is_labelled():
    assert resolve_orientation("Rec9_wcp_ss_port_00031", override="left").source == "user"
    g = np.full((120, 120), 100, np.uint8)
    g[:12, :] = 5
    o = resolve_orientation("upload", override="auto", gray=g)
    assert o.nadir == "top" and o.source == "auto-guess" and "unreliable" in o.rule
    assert auto_guess_nadir(g) == "top"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} orientation tests passed")


if __name__ == "__main__":
    _run_all()
