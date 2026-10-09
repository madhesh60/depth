"""STUDY-16 — the agent's sensor cross-check: OpenCV seabed track vs the depth sounder."""
import numpy as np
import pytest

from src.agentic import sensor_check as sc
from src.agentic.geo import PingFix, geotag
from src.agentic.types import Candidate, Verdict
from src.cv_pipeline.canonical import Canonicaliser, edge_picks, line_quality

M_PER_SAMPLE = 0.02


def sonogram(alt=150, H=400, W=200, decoy=None, seabed=True, seed=0):
    """Rows = range samples (nadir first), columns = pings: ring-down, dark water column, bright seabed
    from row ``alt``; ``decoy`` adds a strong bright layer at that row (what a tracker can lock onto)."""
    rng = np.random.default_rng(seed)
    im = rng.normal(25, 4, (H, W))
    im[:4] = 220
    if seabed:
        im[alt:] = rng.normal(130, 12, (H - alt, W))
    if decoy is not None:
        im[decoy:decoy + 6] = 200
    return np.clip(im, 0, 255).astype(np.uint8)


def steady(alt=150, W=200):
    return np.full(W, alt * M_PER_SAMPLE)


def jumping(W=200, seed=1):
    rng = np.random.default_rng(seed)
    return np.where(rng.random(W) < 0.5, 1.5, 6.0)


def test_edge_tools_find_the_seabed_in_the_window_they_are_given():
    son = sonogram(alt=150)
    rows, st = edge_picks(son, np.full(200, 100.0), np.full(200, 200.0))
    q = line_quality(rows, st)
    assert abs(np.nanmedian(rows) - 150) <= 2 and sc.coherent(q)
    rows, st = edge_picks(son, np.full(200, 250.0), np.full(200, 350.0))   # speckle only
    assert not sc.coherent(line_quality(rows, st))


def test_agree_when_tracker_and_steady_sounder_match():
    g = sc.check_chunk("port", 0, sonogram(alt=100), steady(100), M_PER_SAMPLE)
    assert g.status == "agree" and g.source == "tracker" and abs(np.median(g.line) - 100) <= 3
    assert g.altitude_m == pytest.approx(2.0)
    assert [s.tool for s in g.steps][:3] == ["bottom_track", "sounder_check", "crosscheck"]


def test_conflict_with_steady_sounder_triggers_a_guided_retrack():
    son = sonogram(alt=150, decoy=40)                    # the tracker locks onto the decoy layer
    g = sc.check_chunk("port", 0, son, steady(150), M_PER_SAMPLE)
    assert g.status in ("corrected", "recovered") and g.source == "sounder-guided re-track"
    assert abs(np.median(g.line) - 150) <= 3
    assert any(s.tool == "retrack_guided" and s.status == "done" for s in g.steps)


def test_lost_lock_sounder_defers_to_a_coherent_image_edge():
    g = sc.check_chunk("port", 0, sonogram(alt=60), jumping(), M_PER_SAMPLE)
    assert not g.sounder_steady and g.status == "image_trusted" and g.source == "tracker"
    assert g.altitude_m == pytest.approx(np.median(g.line) * M_PER_SAMPLE)       # metres from the image
    assert any(s.tool == "retrack_guided" and s.status == "skipped" for s in g.steps)


def test_nothing_coherent_is_never_forced():
    son = np.clip(np.random.default_rng(3).normal(90, 30, (400, 200)), 0, 255).astype(np.uint8)
    g = sc.check_chunk("port", 0, son, jumping(), M_PER_SAMPLE)
    assert g.status == "not_measured" and g.line is None and g.altitude_m is None
    assert g.steps[-1].tool == "resolve"


def test_the_other_channel_guides_a_last_retrack():
    # the seabed lies beyond Stage 1's search band (top 35% of range) -> the tracker finds nothing,
    # the sounder is unsteady, but the port channel's accepted line covers the same pings
    other = sc.ChunkGeometry("port", 0, n_pings=200, samples=400, status="image_trusted", line=np.full(200, 160.0))
    g = sc.check_chunk("starboard", 0, sonogram(alt=160, seed=5), jumping(), M_PER_SAMPLE, other=other)
    assert g.metrics["tracker"]["altitude_samples"] is None and abs(np.median(g.line) - 160) <= 3
    assert g.status == "image_trusted" and g.source == "cross-channel re-track"


def test_recording_loop_refits_the_scale_until_it_stops_moving():
    alts = [100, 120, 90, 110, 130]
    chunks = [(c, j, sonogram(alt=a, seed=j), np.full(200, a * M_PER_SAMPLE))
              for j, a in enumerate(alts) for c in ("port", "starboard")]
    res = sc.check_recording(chunks, m_per_sample=0.024)               # first estimate 20% off
    assert abs(res["scale"]["m_per_sample"] - M_PER_SAMPLE) / M_PER_SAMPLE < 0.03
    assert len(res["iterations"]) >= 2 and res["iterations"][0]["moved"] > sc.SCALE_CONVERGED
    s = sc.summary(res)
    assert s["trusted"] == len(chunks) and s["consistency"]["median_rel_diff"] < 0.02


def test_rejected_geometry_reaches_stage1_and_widens_the_error_radius():
    from src.cv_pipeline.orientation import Orientation
    cf = Canonicaliser().process(np.dstack([sonogram(alt=150)[:200]] * 3), "R1_ss_port_00000", nadir="top")
    assert cf.orientation.nadir == "top"
    Canonicaliser.apply_geometry(cf, "not_measured", None)
    assert cf.altitude_px is None and cf.bottom_line is None and cf.to_dict()["geometry_status"] == "not_measured"
    Canonicaliser.apply_geometry(cf, "corrected", [77.0] * cf.width)
    assert cf.altitude_px == pytest.approx(77.0)

    n = 100
    base = dict(lat=36.0, lon=-111.0, ping_lat=[36.0] * n, ping_lon=[-111.0] * n, ping_heading=[0.0] * n,
                range_m_per_px=0.05, range_scale_rel_unc=0.1, altitude_bound_m=6.0)
    errs = {}
    for status in ("agree", "not_measured"):
        c = Candidate((0, 0, 4, 4), 0, "ghost_gear", 0.5, verdict=Verdict.REVIEW)
        c.ping_px, c.n_pings, c.ground_range_px = 10.0, n, 200.0
        geotag(c, PingFix(geometry=status, **base), "top", 100, 640, "R1_ss_star_00000")
        errs[status] = c.geo_error_m
    assert errs["not_measured"] == pytest.approx(errs["agree"] + 6.0, abs=0.11)
