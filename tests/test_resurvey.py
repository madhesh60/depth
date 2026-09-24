"""
Tests for repeat-sighting merging and the opposite-side re-survey planner (src/agentic/resurvey.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic.geo import PingFix, offset_latlon, haversine_m
from src.agentic.resurvey import merge_repeat_sightings, plan_resurvey
from src.agentic.types import TrackedObject, Verdict

LAT0, LON0 = 37.80, -76.15


def _obj(oid, frame, lat, lon, verdict=Verdict.REVIEW, p=0.5, err=4.0):
    return TrackedObject(oid=oid, cls_name="fishing_gear", verdict=verdict, conf=0.3, evidence_score=0.3,
                         frame_id=frame, bbox=(0, 0, 10, 10), lat=lat, lon=lon, geo_error_m=err, p_pot=p)


def test_repeat_sightings_merge_only_overlapping_different_frames():
    a = _obj("H1", "RecA_wcp_ss_port_00001", LAT0, LON0, p=0.4)
    b_lat, b_lon = offset_latlon(LAT0, LON0, 1.5, 45)                     # 1.5 m away, other pass → same object
    b = _obj("H2", "RecB_wcp_ss_star_00007", b_lat, b_lon, verdict=Verdict.CONFIRMED, p=None)
    c_lat, c_lon = offset_latlon(LAT0, LON0, 12.0, 90)                    # 12 m away → a different pot in the string
    c = _obj("H3", "RecB_wcp_ss_star_00008", c_lat, c_lon)
    d = _obj("H4", "RecA_wcp_ss_port_00001", *offset_latlon(LAT0, LON0, 1.0, 0))   # same frame as H1
    out, merges = merge_repeat_sightings([a, b, c, d])
    ids = {t.oid for t in out}
    # H2 (other pass, CONFIRMED) absorbs its CLOSEST sighting from frame A (H4, ~1.06 m) — and H1, a
    # distinct detection in that same frame, must NOT fuse in transitively; H3 is another pot
    assert merges == 1 and ids == {"H1", "H2", "H3"}
    h2 = next(t for t in out if t.oid == "H2")
    assert h2.sightings == 2 and h2.also_in == ["RecA_wcp_ss_port_00001"]


def test_resurvey_line_is_on_the_opposite_side_at_mid_swath():
    heading = 0.0                                                         # boat going north
    boat = (LAT0, LON0)
    t_lat, t_lon = offset_latlon(*boat, 10.0, 90.0)                        # target 10 m to starboard (east)
    t = _obj("H1", "Rec1_wcp_ss_star_00001", t_lat, t_lon, p=0.5)
    track = {"Rec1_wcp_ss_star_00001": PingFix(*boat, heading_deg=heading, synthetic=True)}
    plan = plan_resurvey([t], track, swath_m=32.0)
    L = plan["lines"][0]
    mid_lat = (L["start"][0] + L["end"][0]) / 2; mid_lon = (L["start"][1] + L["end"][1]) / 2
    # the new pass runs 16 m EAST of the target (the far side), northbound, target on its PORT side
    assert abs(haversine_m((mid_lat, mid_lon), (t_lat, t_lon)) - 16.0) < 0.5
    assert mid_lon > t_lon and L["heading_deg"] == 0.0 and L["targets_on"] == "port"
    # falsifiable prediction: the shadow pointed east (away from the old boat); now it must point west
    pr = L["predictions"][0]
    assert pr["shadow_was"] == 90.0 and pr["shadow_must_point"] == 270.0
    assert L["synthetic"] and "not executed" in L["status"]


def test_grouping_and_boat_budget_prefer_uncertainty_per_metre():
    track, objs = {}, []
    heading = 20.0
    for i in range(6):                                                    # 6 uncertain targets along one line
        fid = f"Rec1_wcp_ss_port_{i:05d}"
        b = offset_latlon(LAT0, LON0, 12.0 * i, heading)
        track[fid] = PingFix(*b, heading_deg=heading)
        objs.append(_obj(f"H{i}", fid, *offset_latlon(*b, 8.0, heading - 90), p=0.5))
    far = "Rec2_wcp_ss_port_00001"                                        # one near-certain target far away
    fb = offset_latlon(LAT0, LON0, 900.0, 200.0)
    track[far] = PingFix(*fb, heading_deg=heading)
    objs.append(_obj("HX", far, *offset_latlon(*fb, 8.0, heading - 90), p=0.97))
    plan = plan_resurvey(objs, track)
    assert plan["lines"][0]["targets"] == [f"H{i}" for i in range(6)]    # one pass images all six
    assert plan["lines"][0]["voi_per_100m"] > plan["lines"][-1]["voi_per_100m"]
    tight = plan_resurvey(objs, track, boat_minutes=plan["lines"][0]["boat_min"])
    assert [L["targets"] for L in tight["lines"]] == [[f"H{i}" for i in range(6)]]   # the budget keeps the dense pass
    assert tight["voi_covered"] < tight["voi_total"]


def test_no_gps_means_no_plan():
    t = _obj("H1", "Rec1_wcp_ss_port_00001", None, None)
    assert plan_resurvey([t], {})["lines"] == []
