"""
Data dropouts in a raw recording (``humminbird.chunk_dropouts``): pings the sonar did not image are found
from the recording itself — checked here on injected dropouts — and the agent says so per frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic.geo import PingFix
from src.agentic.pipeline import _flag_dropouts
from src.agentic.types import Candidate, Evidence, FrameResult, RelookResult, ShadowProof, ShadowQuality, Verdict
from src.cv_pipeline.humminbird import Ping, chunk_dropouts

N, SAMPLES, DT = 500, 300, 40          # pings, samples per ping, ms between pings


def _pings(dt: list[int] | None = None) -> list[Ping]:
    t = np.cumsum([0] + (dt or [DT] * (N - 1)))
    return [Ping(record=i, time_ms=int(t[i]), lat=0.0, lon=0.0, heading_deg=0.0, speed_raw=0, depth_m=5.0,
                 beam=2, freq_hz=455000, offset=0, header_len=0, n_returns=SAMPLES) for i in range(N)]


def _speckle(seed: int = 0) -> np.ndarray:
    """Samples × pings: Rayleigh speckle over a seabed that fades with range (no dropout)."""
    rng = np.random.default_rng(seed)
    fade = np.linspace(1.0, 0.4, SAMPLES)[:, None]
    return np.clip(rng.rayleigh(40.0, (SAMPLES, N)) * fade, 0, 255).astype(np.uint8)


def _ref(son: np.ndarray) -> tuple[float, float]:
    return float(DT), float(np.median(son.astype(np.float32).mean(axis=0)))


def test_clean_speckle_has_no_dropouts():
    for seed in range(5):
        son = _speckle(seed)
        d = chunk_dropouts(_pings(), son, *_ref(son))
        assert d["dark"] == 0 and d["gap"] == 0 and d["runs"] == []


def test_blank_and_weak_pings_are_found():
    son = _speckle()
    ref = _ref(son)
    son[:, 100:105] = 0                                          # transducer out of the water: blank pings
    son[:, 300:303] = (son[:, 300:303] * 0.1).astype(np.uint8)   # aerated water: a weak return
    d = chunk_dropouts(_pings(), son, *ref)
    assert d["runs"] == [[100, 104], [300, 302]] and d["dark"] == 8 and d["gap"] == 0


def test_a_time_gap_marks_missing_pings():
    dt = [DT] * (N - 1)
    dt[249] = 5 * DT                                             # the ping after index 249 came 4 intervals late
    son = _speckle()
    d = chunk_dropouts(_pings(dt), son, *_ref(son))
    assert d["gap"] == 1 and d["runs"] == [[250, 250]] and d["missing_s"] == pytest.approx(4 * DT / 1000)


def test_a_dark_chunk_is_found_against_the_channel_median():
    son = _speckle()
    ref = _ref(son)
    dark = (son * 0.05).astype(np.uint8)                         # the whole chunk lost its returns
    d = chunk_dropouts(_pings(), dark, *ref)
    assert d["dark"] == N and d["runs"] == [[0, N - 1]]


def _cand(x1: int, x2: int) -> Candidate:
    proof = ShadowProof(ShadowQuality.NONE, 0.0, 0, 0.0, None, 0.0, 1.0, (0, 0), (0, 0, 0, 0), orientation_known=False)
    ev = Evidence(relook=RelookResult(found=False, conf=0.0, gain=0.0, scale=1.0), shadow=proof, echo_ratio=1.0,
                  evidence_score=0.3, notes=["detector 0.30"])
    return Candidate(bbox=(x1, 100, x2, 140), cls_id=0, cls_name="ghost_gear", conf=0.3, evidence=ev,
                     verdict=Verdict.REVIEW, trace=[])


def test_the_agent_files_an_incident_and_notes_touching_finds():
    fr = FrameResult("f", 640, 640, [_cand(90, 120), _cand(400, 430)])
    fix = PingFix(lat=0.0, lon=0.0, dropout_cols=[[100, 110]], dropout_share=0.02)
    _flag_dropouts(fr, fix)
    inc = [i for i in fr.incidents if i["code"] == "data_dropout"]
    assert len(inc) == 1 and inc[0]["affected"] == 1 and inc[0]["runs_px"] == [[100, 110]]
    assert fr.candidates[0].evidence.notes[0].startswith("touches a data dropout")
    assert not fr.candidates[1].evidence.notes[0].startswith("touches")
