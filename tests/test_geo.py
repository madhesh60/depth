"""
Tests for the Act-stage geotagging (src/agentic/geo.py). Pure math — no model.
Runs under pytest, or standalone: python tests/test_geo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic.geo import (offset_latlon, parse_side, range_px_from_nadir, geotag,
                             synthetic_track, haversine_m, PingFix)
from src.detection.infer import Detection
from src.agentic.types import Candidate


def test_offset_directions_and_distance():
    north = offset_latlon(0.0, 0.0, 111_000, 0.0)
    east = offset_latlon(0.0, 0.0, 111_000, 90.0)
    assert north[0] > 0 and abs(north[1]) < 1e-6         # north increases lat only
    assert east[1] > 0 and abs(east[0]) < 1e-6           # east increases lon only
    a = (37.98, -76.0)
    b = offset_latlon(*a, 250.0, 42.0)
    assert abs(haversine_m(a, b) - 250.0) < 1.0          # round-trips through haversine


def test_parse_side():
    assert parse_side("crabpot_train_Rec6_wcp_ss_port_00114") == "port"
    assert parse_side("crabpot_train_Rec6_wcp_ss_star_00029") == "starboard"
    assert parse_side("something_without_side") is None


def test_range_px_from_nadir():
    assert range_px_from_nadir((90, 100, 110, 140), "top", 640, 640) == 120     # cy
    assert range_px_from_nadir((90, 100, 110, 140), "bottom", 640, 640) == 520  # h - cy


def test_geotag_fills_coords_with_error():
    c = Candidate(bbox=(300, 400, 340, 460), cls_id=0, cls_name="fishing_gear", conf=0.5)
    fix = PingFix(lat=37.98, lon=-76.0, heading_deg=0.0)
    geotag(c, fix, nadir="top", w=640, h=640, frame_id="Rec6_wcp_ss_starboard_00030")
    assert c.lat is not None and c.lon is not None
    assert c.geo_error_m is not None and c.geo_error_m >= 3.0
    # starboard + heading 0 (north) ⇒ object to the east ⇒ lon greater than the boat
    assert c.lon > fix.lon


def test_synthetic_track_is_flagged_and_ordered():
    ids = ["Rec6_wcp_ss_port_00030", "Rec6_wcp_ss_port_00010", "Rec6_wcp_ss_port_00020"]
    tr = synthetic_track(ids)
    assert all(f.synthetic for f in tr.values())
    # positions advance with ping index (…_00010 is the start)
    assert haversine_m((tr[ids[1]].lat, tr[ids[1]].lon), (tr[ids[2]].lat, tr[ids[2]].lon)) > 0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} geo tests passed")


if __name__ == "__main__":
    _run_all()
