"""
Tests for the Act-stage mission plan + exporters (src/agentic/mission.py). Pure logic — no model.
Runs under pytest, or standalone: python tests/test_mission.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic.mission import build_mission, to_geojson, to_gpx, to_kml, to_csv, export
from src.agentic.types import TrackedObject, Verdict, SurveyResult, MissionPlan


def _obj(oid, verdict, lat=None, lon=None, frame="f1"):
    return TrackedObject(oid=oid, cls_name="fishing_gear", verdict=verdict, conf=0.5,
                         evidence_score=0.6, frame_id=frame, bbox=(1, 2, 3, 4),
                         lat=lat, lon=lon, geo_error_m=3.0 if lat else None,
                         height_m=0.3, shadow_quality="weak")


def _tracked_gps():
    return [_obj("H001", Verdict.CONFIRMED, 37.980, -76.000),
            _obj("H002", Verdict.CONFIRMED, 37.981, -76.001),
            _obj("H003", Verdict.CONFIRMED, 37.983, -76.003),
            _obj("H004", Verdict.REVIEW, 37.982, -76.002)]


def test_mission_with_gps_builds_route_and_resurvey():
    m = build_mission(_tracked_gps(), gps_available=True, gps_synthetic=True)
    assert set(m.recovery_route) == {"H001", "H002", "H003"}     # confirmed only
    assert m.resurvey == ["H004"]                                # review only
    assert m.route_length_m and m.route_length_m > 0
    assert m.human_approval_required is True
    assert m.counts["confirmed"] == 3 and m.counts["review"] == 1


def test_mission_without_gps_is_honest():
    tracked = [_obj("H001", Verdict.CONFIRMED), _obj("H002", Verdict.CONFIRMED)]
    m = build_mission(tracked, gps_available=False)
    assert m.gps_available is False
    assert m.route_length_m is None                              # no geometry without GPS
    assert set(m.recovery_route) == {"H001", "H002"}


def test_exporters_emit_valid_content():
    tracked = _tracked_gps()
    m = build_mission(tracked, gps_available=True, gps_synthetic=True)
    gj = json.loads(to_geojson(tracked, m))
    assert gj["type"] == "FeatureCollection"
    assert any(f["geometry"]["type"] == "LineString" for f in gj["features"])   # the route
    assert any(f["geometry"]["type"] == "Point" for f in gj["features"])
    gpx = to_gpx(tracked, m)
    assert "<wpt" in gpx and "<rte>" in gpx and "SYNTHETIC DEMO GPS" in gpx
    assert "<Placemark>" in to_kml(tracked, m)
    csv_txt = to_csv(tracked)
    assert csv_txt.splitlines()[0].startswith("id,class,verdict") and len(csv_txt.splitlines()) == 5


def test_export_dispatch_all_formats():
    tracked = _tracked_gps()
    m = build_mission(tracked, gps_available=True, gps_synthetic=True)
    survey = SurveyResult(survey_id="s", frames=[], tracked=tracked, mission=m)
    for fmt in ("geojson", "gpx", "kml", "csv", "json"):
        assert isinstance(export(fmt, survey), str) and export(fmt, survey)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} mission tests passed")


if __name__ == "__main__":
    _run_all()
