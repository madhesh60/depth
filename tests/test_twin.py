"""
Tests for the 3D digital twin payloads (src/agentic/twin.py). The key invariant: the twin must not
contradict the rest of the product — every hazard's geotag lies on its own frame's swath ribbon
(both come from the same ping / ground-range geometry), heights exist only where a real shadow was
measured, and frames without tracked geometry are honestly not drawn.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.infer import DEFAULT_ONNX
from src.dashboard import samples as samples_mod

_HAVE = DEFAULT_ONNX.exists() and len(samples_mod.list_samples()) >= 3


def _frames(n):
    out = []
    for s in samples_mod.list_samples()[:n]:
        p = samples_mod.sample_path(s["id"])
        out.append((p.stem, cv2.imread(str(p))))
    return out


def _inside(pt, quad, tol):
    """Point in the (convex) ribbon quad, allowing ``tol`` metres."""
    a, b, d, c = quad["p0g0"], quad["p1g0"], quad["p1g1"], quad["p0g1"]      # perimeter order
    poly = np.array([a, b, d, c], np.float32)
    dist = cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(pt[0]), float(pt[1])), True)
    return dist >= -tol


def test_frame_twin_geometry_and_heights():
    if not _HAVE:
        return
    from src.agentic.pipeline import AgenticPipeline
    from src.agentic.twin import frame_twin
    fid, img = _frames(2)[1]
    res = AgenticPipeline().run_frame(img, frame_id=fid)
    tw = frame_twin(img, res)
    assert tw["available"] and tw["n_pings"] == img.shape[1] and 0 < tw["n_ground"] <= img.shape[0]
    assert tw["texture"].startswith("data:image/jpeg")
    by_id = {i: c for i, c in enumerate(res.candidates)}
    for o in tw["objects"]:
        assert 0 <= o["ping"] <= tw["n_pings"] and 0 <= o["ground"] <= img.shape[0]
        sh = by_id[o["id"]].evidence.shadow
        assert (o["height_px"] is not None) == (sh.has_shadow and sh.height_rel > 0)     # height ⇔ real shadow


def test_frame_twin_refuses_unmeasured_geometry():
    from src.agentic.twin import frame_twin
    from src.agentic.types import FrameResult
    flat = np.full((200, 200, 3), 100, np.uint8)
    tw = frame_twin(flat, FrameResult(frame_id="mosaic_1", width=200, height=200, candidates=[]))
    assert tw["available"] is False and "not" in tw["reason"]


def test_survey_twin_hazards_lie_on_their_own_swath():
    if not _HAVE:
        return
    from src.agentic.pipeline import AgenticPipeline
    from src.agentic.geo import synthetic_track
    from src.agentic.twin import survey_twin
    frames = _frames(4)
    track = synthetic_track([f for f, _ in frames])
    res = AgenticPipeline().run_survey(frames, track=track, survey_id="t")
    tw = survey_twin(frames, res, track)
    assert tw["available"] and tw["synthetic"] and tw["ribbons"]
    ribbons = {r["frame_id"]: r for r in tw["ribbons"]}
    checked = 0
    for o in tw["objects"]:
        r = ribbons.get(o["frame_id"])
        if r is None:
            continue
        assert _inside(o["xy"], r["corners"], tol=1.0), (o["oid"], o["xy"], r["corners"])
        checked += 1
    assert checked > 0
    assert len(tw["track"]) == len([f for f, _ in frames if f in track])


def test_survey_twin_needs_a_track():
    from src.agentic.twin import survey_twin
    from src.agentic.types import SurveyResult, MissionPlan
    empty = SurveyResult("s", [], [], MissionPlan([], [], None, False))
    assert survey_twin([], empty, None)["available"] is False
