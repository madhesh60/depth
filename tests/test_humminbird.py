"""
Tests for the raw Humminbird reader (src/cv_pipeline/humminbird.py): exact decoding of synthetic
files built byte by byte, defensive handling of corrupt / truncated / absurd input, the coordinate
formula, per-ping geotagging, and — when the PINGMapper sample is present — the physical checks.
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic.geo import PingFix, geotag
from src.agentic.types import Candidate, Verdict
from src.cv_pipeline import humminbird as hb

SAMPLE = hb.SAMPLE_DIR / "Test-Small-DS.DAT"


def _ping(rec: int, t_ms: int, e: int, n: int, heading10: int, speed10: int, depth_dm: int, beam: int,
          returns: bytes) -> bytes:
    """A 67-byte-header (9xx) ping, laid out exactly like a real one."""
    h = hb.START
    h += b"\x80" + struct.pack(">i", rec) + b"\x81" + struct.pack(">i", t_ms)
    h += b"\x82" + struct.pack(">i", e) + b"\x83" + struct.pack(">i", n)
    h += b"\x84" + struct.pack(">HH", 1, heading10) + b"\x85" + struct.pack(">HH", 1, speed10)
    h += b"\x87" + struct.pack(">i", depth_dm) + b"\x50" + bytes([beam]) + b"\x51\x0a"
    h += b"\x92" + struct.pack(">i", 455000) + b"\x53\x07\x54\x01" + b"\x95" + struct.pack(">i", 26) + b"\x56\x76\x57\x8b"
    h += b"\xa0" + struct.pack(">I", len(returns)) + b"\x21"
    assert len(h) == 67
    return h + returns


def _write_recording(root: Path, n_pings: int = 60, corrupt: int | None = None) -> Path:
    dat = bytearray(64)
    dat[0], dat[1] = 0xC1, 0
    dat[20:24] = struct.pack(">I", 1382657324)
    dat[24:28] = struct.pack(">i", -12413175); dat[28:32] = struct.pack(">i", 4396652)
    dat[32:42] = b"R00001.SON"
    dat[44:48] = struct.pack(">I", n_pings * 2); dat[48:52] = struct.pack(">I", n_pings * 40)
    (root / "R00001.DAT").write_bytes(bytes(dat))
    folder = root / "R00001"; folder.mkdir()
    rng = np.random.default_rng(0)
    for fname, beam in (("B002", 2), ("B003", 3)):
        son, idx = bytearray(), bytearray()
        for i in range(n_pings):
            ret = bytes(rng.integers(0, 255, 400, dtype=np.uint8))
            idx += struct.pack(">II", i * 40, len(son))
            p = _ping(i + 1, i * 40, -12413175 + i * 3, 4396652 - i * 2, 1977, 20, 42, beam, ret)
            if corrupt is not None and i == corrupt:
                p = b"\x00\x11" + p[2:]                     # break the start marker
            son += p
        (folder / f"{fname}.SON").write_bytes(bytes(son))
        (folder / f"{fname}.IDX").write_bytes(bytes(idx))
    return root / "R00001.DAT"


def test_decodes_every_field_exactly(tmp_path):
    rec = hb.read_recording(_write_recording(tmp_path))
    assert set(rec.channels) == {"port", "starboard"} and rec.name == "R00001"
    p = rec.channels["port"].pings[0]
    assert (p.record, p.time_ms, p.header_len, p.n_returns, p.freq_hz) == (1, 0, 67, 400, 455000)
    assert p.heading_deg == 197.7 and p.speed_raw == 20 and p.depth_m == 4.2 and p.beam == 2
    assert rec.dat["records"] == 120 and rec.dat["family"] == "9xx/11xx/helix"
    assert rec.channels["port"].returns(0, 5).shape == (5, 400)


def test_coordinate_formula_matches_the_real_first_fix():
    e = struct.unpack(">i", bytes.fromhex("ff429309"))[0]          # the sample's first ping, bytes as recorded
    n = struct.unpack(">i", bytes.fromhex("0043166c"))[0]
    lat, lon = hb.hum_latlon(e, n)
    assert abs(lat - 36.878808) < 1e-6 and abs(lon - (-111.514259)) < 1e-6


def test_corrupt_ping_is_skipped_not_fatal(tmp_path):
    rec = hb.read_recording(_write_recording(tmp_path, corrupt=10))
    ch = rec.channels["port"]
    assert len(ch.pings) == 59 and "malformed" in ch.issues[0]
    assert [p.record for p in ch.pings][9:11] == [10, 12]           # resynced on the next marker


def test_truncated_and_absurd_input(tmp_path):
    good = _ping(1, 0, 0, 0, 0, 0, 10, 2, b"\x01" * 100)
    (tmp_path / "t.SON").write_bytes(good + good[:40])              # second ping cut off
    ch = hb.read_son(tmp_path / "t.SON")
    assert len(ch.pings) == 1 and ch.issues
    bad = bytearray(good); bad[62:66] = struct.pack(">I", 10 ** 9)  # sample count out of range
    (tmp_path / "b.SON").write_bytes(bytes(bad))
    with pytest.raises(hb.HumError):
        hb.read_son(tmp_path / "b.SON")
    (tmp_path / "x.DAT").write_bytes(b"not a recording")
    with pytest.raises(hb.HumError):
        hb.read_dat(tmp_path / "x.DAT")


def test_frames_and_track_carry_real_per_ping_fixes(tmp_path):
    rec = hb.read_recording(_write_recording(tmp_path, n_pings=60))
    frames, track, meta = hb.frames_and_track(rec, nchunk=30, size=64)
    assert [f for f, _ in frames] == ["R00001_ss_port_00000", "R00001_ss_port_00001",
                                      "R00001_ss_star_00000", "R00001_ss_star_00001"]
    fx = track["R00001_ss_port_00000"]
    assert fx.synthetic is False and len(fx.ping_lat) == 64 and fx.depth_m == 4.2
    assert frames[0][1].shape == (64, 64, 3)


def test_geotag_uses_the_objects_own_ping_and_measured_scale():
    n = 100
    lat = [36.0 + i * 1e-6 for i in range(n)]
    fx = PingFix(lat=lat[50], lon=-111.0, heading_deg=0.0, ping_lat=lat, ping_lon=[-111.0] * n,
                 ping_heading=[0.0] * n, depth_m=5.0, range_m_per_px=0.05, range_scale_rel_unc=0.1)
    c = Candidate((0, 0, 4, 4), 0, "fishing_gear", 0.5, verdict=Verdict.REVIEW)
    c.ping_px, c.n_pings, c.ground_range_px = 10.0, n, 200.0              # 200 px × 0.05 = 10 m
    geotag(c, fx, "top", 100, 640, "R1_ss_star_00000")
    assert abs(c.lat - lat[10]) < 1e-9                                    # this ping, not the frame centre
    east_m = (c.lon - (-111.0)) * 111320 * math.cos(math.radians(36.0))
    assert abs(east_m - 10.0) < 0.05                                      # starboard of a north heading = east
    assert c.geo_error_m == round(3.0 + 10.0 * (0.1 + math.sin(math.radians(6.0))), 1)


@pytest.mark.skipif(not SAMPLE.exists(), reason="PINGMapper sample not fetched (python -m src.cv_pipeline.humminbird fetch)")
def test_real_sample_is_physically_consistent():
    rec = hb.read_recording(SAMPLE)
    v = hb.validate(rec)
    p = v["channels"]["port"]
    assert p["pings"] == 3453 and p["records_monotonic"] and p["time_monotonic"] and not p["issues"]
    assert 0.09 < p["gps_speed_per_speed_unit"] < 0.11 and p["speed_corr"] > 0.9      # speed unit = 0.1 m/s
    assert p["course_vs_heading_deg_median"] < 5                                        # heading unit = 0.1 deg
    frames, track, meta = hb.frames_and_track(rec)
    sc = meta["range_scale"]
    assert sc["m_per_sample"] and abs(sc["m_per_sample"] - sc["physics_m_per_sample"]) / sc["m_per_sample"] < 0.3


@pytest.mark.skipif(not SAMPLE.exists(), reason="PINGMapper sample not fetched")
def test_api_survey_on_the_real_recording():
    from fastapi.testclient import TestClient
    from src.dashboard import app as app_mod
    from src.detection.infer import DEFAULT_ONNX
    c = TestClient(app_mod.app)
    assert c.get("/api/health").json()["recording"]["available"] is True
    info = c.get("/api/recording").json()
    assert info["available"] and info["checks"]["speed_corr"] > 0.9 and info["range_scale"]["m_per_sample"]
    if not DEFAULT_ONNX.exists():
        pytest.skip("model absent")
    assert c.post("/api/survey?gps=recording").status_code == 413       # 14 frames: the sync endpoint refers to jobs
    import time
    job = c.post("/api/jobs/survey?gps=recording&budget_minutes=2").json()
    for _ in range(600):
        rec = c.get(f"/api/jobs/{job['job_id']}").json()
        if rec["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert rec["status"] == "done", rec.get("error")
    s = rec["result"]
    assert s["survey_id"].endswith("-rec") and s["mission"]["gps_synthetic"] is False
    assert s["recording"]["available"] and "MEASURED" in s["twin"]["note"]
    assert s["stage1_counterfactual"]["available"] is False            # the scale itself comes from Stage 1
    for t in s["tracked"]:
        assert 36.87 < t["lat"] < 36.89 and -111.52 < t["lon"] < -111.51 and t["geo_error_m"] >= 3.0
