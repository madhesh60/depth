"""
Tests for STUDY-12 helpers (src/agentic/study_causal.py): the per-recording synthetic track and the
survey diff. No model needed.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic.geo import haversine_m
from src.agentic.study_causal import _kendall, diff, survey_track
from src.agentic.types import Candidate, FrameResult, MissionPlan, SurveyResult, TrackedObject, Verdict


def test_survey_track_separates_recordings_and_pairs_sides():
    ids = ["Rec6_wcp_ss_port_00004", "Rec6_wcp_ss_star_00004", "Rec6_wcp_ss_port_00005", "Rec19_wcp_ss_port_00004"]
    t = survey_track(ids)
    a, b = t["Rec6_wcp_ss_port_00004"], t["Rec6_wcp_ss_star_00004"]
    assert (a.lat, a.lon) == (b.lat, b.lon)                                  # port/star of one chunk share a fix
    other = t["Rec19_wcp_ss_port_00004"]
    assert haversine_m((a.lat, a.lon), (other.lat, other.lon)) > 400          # same chunk index, different line
    assert all(f.synthetic for f in t.values())


def _survey():
    tr = [TrackedObject(f"H{i}", "fishing_gear", Verdict.REVIEW, 0.5, 0.5, "F1", (i, 0, i + 5, 5),
                        lat=37.8 + i * 1e-4, lon=-76.15, geo_error_m=4.0, p_pot=0.9 - i / 10) for i in range(1, 5)]
    m = MissionPlan([], [t.oid for t in tr], None, True, review_queue=[t.oid for t in tr],
                    inspection_route=["H1", "H2"], inspection_length_m=11.1)
    m.budget = {"review_ids": ["H1", "H2"]}
    cands = [Candidate(t.bbox, 0, "fishing_gear", 0.5, verdict=Verdict.REVIEW) for t in tr]
    return SurveyResult("s", [FrameResult("F1", 64, 64, cands)], tr, m)


def test_diff_of_identical_surveys_is_zero_and_detects_moves():
    A = _survey()
    d = diff(A, copy.deepcopy(A))
    assert d["verdict_flips"] == 0 and d["queue_kendall_tau"] == 1.0 and d["budget_picks_changed"] == 0
    assert d["inspection_stops_changed"] == 0 and d["geotag_shift_m"]["max"] == 0.0
    B = copy.deepcopy(A)
    B.tracked[0].lat += 1e-4                                                 # ~11 m: outside a 4 m circle
    B.mission.inspection_route = ["H2", "H1"]
    d = diff(A, B)
    assert d["geotag_shift_m"]["outside_stated_error"] == 1 and d["inspection_stops_changed"] == 2


def test_kendall():
    assert _kendall([1, 2, 3], [1, 2, 3]) == 1.0 and _kendall([1, 2, 3], [3, 2, 1]) == -1.0
    assert _kendall([1], [1]) is None
