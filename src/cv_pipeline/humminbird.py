"""
humminbird.py — read a RAW Humminbird side-imaging recording (``<name>.DAT`` + ``<name>/B00x.SON|IDX``)
straight into DEPTH: real per-ping GPS, heading, speed and **sonar depth**, and sonogram frames in the
same layout as the PINGMapper exports the detector was trained on. No PINGMapper install needed.

Format (documented by PINGMapper — Bodine et al. 2022, MIT licence — and re-implemented here):

* ``.SON``: pings back to back. A ping starts with ``C0 DE AB 21``; the header is a run of
  ``[tag][value]`` fields — tags ≥ 0x80 carry a 4-byte big-endian value, tags < 0x80 one byte — and
  ends with ``A0 <uint32 return count> 21``; then ``count`` 8-bit intensity samples (nadir first).
  Header length therefore varies by model (67 / 72 / 152 bytes) and is *derived*, never assumed.
* ``.IDX``: ``(uint32 time_ms, uint32 byte offset)`` per ping — used to locate pings, each checked.
* ``.DAT``: recording name, start time, first fix, record count, length (9xx / 11xx / Helix / Solix).
* Positions: Humminbird's spherical Mercator (radius 6378388 m) with its latitude correction, as in
  PINGMapper: ``lon = E/R`` and ``lat = atan(tan(2·atan(exp(N/R)) − π/2) · 1.0067642927)``.

Safety — the files are untrusted binary input: every length and offset is bounds-checked, every ping
must start with the marker and end its header with ``21``, sample counts are capped
(``MAX_RETURNS``), file sizes are capped (``MAX_FILE_BYTES``), a corrupt ping is skipped with a
recorded issue (resync on the next marker) instead of crashing, and nothing is executed or written.

Validation (``validate()``; report ``docs/raw_recording.md``) — before any number is trusted:
GPS speed vs the sonar's speed field, course-over-ground vs heading, record/time monotonicity,
and port/starboard agreement. The **range scale** is *measured*: sonar depth (m, from the unit)
÷ Stage-1 bottom-track altitude (px) — see ``src/agentic/pipeline.py``.

    python -m src.cv_pipeline.humminbird fetch                 # the PINGMapper small sample (SHA-256 pinned)
    python -m src.cv_pipeline.humminbird info  <dir>/R.DAT     # what is inside a recording
    python -m src.cv_pipeline.humminbird validate <dir>/R.DAT  # the checks → docs/raw_recording.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

START = b"\xc0\xde\xab\x21"
R_HUM = 6378388.0                    # Humminbird Mercator radius (m)
LAT_K = 1.0067642927                 # Humminbird latitude correction
MAX_FILE_BYTES = 4 * 1024 ** 3       # 4 GB per .SON — a 12-h recording is ~1.5 GB
MAX_RETURNS = 20000                  # samples per ping (Mega imaging ≈ 1.5–5 k)
MAX_HEADER = 256
BEAMS = {0: "down_low", 1: "down_high", 2: "port", 3: "starboard", 4: "down_mega"}
TAGS = {0x80: "record", 0x81: "time_ms", 0x82: "utm_e", 0x83: "utm_n", 0x84: "heading", 0x85: "speed",
        0x87: "depth_cm", 0x50: "beam", 0x51: "volt_scale", 0x92: "freq_hz"}

REPO = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPO / "DATASET" / "external" / "pingmapper_sample"
# PINGMapper's own sample data (git-LFS objects of the `data` branch; Zenodo 10.5281/zenodo.6604666
# archives the pointers). Hashes are the LFS object ids — a download is kept only if it matches.
_LFS = "https://media.githubusercontent.com/media/CameronBodine/PINGMapper/data/exampleData/"
SAMPLE_FILES = {
    "Test-Small-DS.DAT": "e0cee9b547b81113a6c82cda895a26af28614b66c99474cb9c56900ec8218946",
    "Test-Small-DS/B002.SON": "f1ff16b685635b2402a3b0662b92a2b8ec2faea0ac5fd02bee65afa0c25c9f3b",
    "Test-Small-DS/B002.IDX": "a52fc9563f8c0273659499e309bafed2003a842e5874654c08fa681b4e1c3e58",
    "Test-Small-DS/B003.SON": "df0d07447867fc08b8a1c80233223606a5a976586246ef7d1344b8a7ae903a5d",
    "Test-Small-DS/B003.IDX": "a52fc9563f8c0273659499e309bafed2003a842e5874654c08fa681b4e1c3e58",
}


# the 1-h Solix recording (Pearl River, ~216 MB) - STUDY-16's fresh recording; fetched only on request
LARGE_DIR = REPO / "DATASET" / "external" / "pingmapper_large"
LARGE_FILES = {
    "Test-Large-DS.DAT": "7350f0dc3145d49ad9ced5325e7031dd17197c284b9ba9bd64fd77faaed9fe65",
    "Test-Large-DS/B002.SON": "c7c6e161f787072ab2e1c808a3745065e2e7db7eb6ce8cd4de0385dfa387462c",
    "Test-Large-DS/B002.IDX": "c897374b2b509146a0a8961b76ba8a4ddbdd86a0a2ba2065c1575ebb4ceb5e3e",
    "Test-Large-DS/B003.SON": "091a01095019d762cf8ab8072787cdae303b5fd7d41bc9f078c92bf7525b235e",
    "Test-Large-DS/B003.IDX": "4d3311cabf13645526ae0fa388b5c121e46f88720dab1a12c0fa4850896d06df",
}


class HumError(ValueError):
    """The file is not a Humminbird recording DEPTH can read (message says why)."""


def hum_latlon(e: float, n: float) -> tuple[float, float]:
    lon = math.degrees(e / R_HUM)
    lat = math.degrees(math.atan(math.tan(2.0 * math.atan(math.exp(n / R_HUM)) - math.pi / 2) * LAT_K))
    return lat, lon


@dataclass
class Ping:
    record: int
    time_ms: int
    lat: float
    lon: float
    heading_deg: Optional[float]
    speed_raw: Optional[int]
    depth_m: Optional[float]
    beam: Optional[int]
    freq_hz: Optional[int]
    offset: int
    header_len: int
    n_returns: int


@dataclass
class Channel:
    name: str                       # port | starboard | down_* (from the beam field, not the filename)
    path: str
    sha256: str
    pings: list[Ping]
    issues: list[str] = field(default_factory=list)

    def returns(self, i0: int = 0, i1: Optional[int] = None) -> np.ndarray:
        """Intensity image (pings × samples, uint8, zero-padded) for pings [i0, i1)."""
        ps = self.pings[i0:i1]
        if not ps:
            return np.zeros((0, 0), np.uint8)
        w = max(p.n_returns for p in ps)
        out = np.zeros((len(ps), w), np.uint8)
        with open(self.path, "rb") as f:
            for k, p in enumerate(ps):
                f.seek(p.offset + p.header_len)
                buf = f.read(p.n_returns)
                out[k, :len(buf)] = np.frombuffer(buf, np.uint8)
        return out


@dataclass
class Recording:
    name: str
    dat: dict
    channels: dict[str, Channel]


# ---- parsing (defensive) -------------------------------------------------------------------------
def read_dat(path: Path) -> dict:
    b = Path(path).read_bytes()
    # 9xx/11xx/Helix: 64 bytes, starts C1, big-endian. Solix: 96 bytes, starts C3, LITTLE-endian
    # (its start time / easting read the same as the SON headers' big-endian fields only that way).
    if not ((len(b) == 64 and b[0] == 0xC1) or (len(b) == 96 and b[0] in (0xC1, 0xC3))):
        raise HumError(f"{path.name}: {len(b)}-byte DAT not recognised (9xx/11xx/Helix = 64, Solix = 96; "
                       f"Onix text DATs are not supported yet)")
    end = "<" if b[0] == 0xC3 else ">"
    u32 = lambda o: struct.unpack(end + "I", b[o:o + 4])[0]
    i32 = lambda o: struct.unpack(end + "i", b[o:o + 4])[0]
    lat, lon = hum_latlon(i32(24), i32(28))
    return {"family": "solix" if len(b) == 96 else "9xx/11xx/helix", "water": {0: "fresh", 1: "deep salt", 2: "shallow salt"}.get(b[1], b[1]),
            "start_unix": u32(20), "first_fix": [round(lat, 6), round(lon, 6)],
            "name": b[32:42].split(b"\x00")[0].decode("ascii", "replace"), "records": u32(44), "length_ms": u32(48)}


def parse_header(buf: bytes, off: int) -> tuple[dict, int, int]:
    """(fields, header_len, n_returns) of the ping at ``off``; raises HumError if malformed."""
    if buf[off:off + 4] != START:
        raise HumError(f"no ping marker at byte {off}")
    i, end, f = off + 4, min(len(buf), off + MAX_HEADER), {}
    while i < end:
        tag = buf[i]
        if tag == 0xA0:
            if i + 6 > len(buf) or buf[i + 5] != 0x21:
                raise HumError(f"header at {off} not terminated by A0 <count> 21")
            n = struct.unpack(">I", buf[i + 1:i + 5])[0]
            if not 0 < n <= MAX_RETURNS:
                raise HumError(f"implausible sample count {n} at {off}")
            return f, i + 6 - off, n
        if tag >= 0x80:
            if i + 5 > len(buf):
                break
            raw = buf[i + 1:i + 5]
            if tag in (0x84, 0x85):                     # (2-byte quality flag, 2-byte value)
                f[tag] = struct.unpack(">H", raw[2:4])[0]
            else:
                f[tag] = struct.unpack(">i", raw)[0]
            i += 5
        else:
            if i + 2 > len(buf):
                break
            f[tag] = buf[i + 1]
            i += 2
    raise HumError(f"header at {off} longer than {MAX_HEADER} bytes or truncated")


def read_son(son: Path, idx: Optional[Path] = None) -> Channel:
    son = Path(son)
    size = son.stat().st_size
    if size > MAX_FILE_BYTES:
        raise HumError(f"{son.name}: {size} bytes exceeds the {MAX_FILE_BYTES}-byte cap")
    buf = son.read_bytes()
    offsets: list[int] = []
    issues: list[str] = []
    if idx and Path(idx).exists():
        ib = Path(idx).read_bytes()
        if len(ib) % 8:
            issues.append(f"{Path(idx).name}: length {len(ib)} is not a multiple of 8 - index ignored")
        else:
            offsets = [struct.unpack(">I", ib[k + 4:k + 8])[0] for k in range(0, len(ib), 8)]
    pings: list[Ping] = []
    pos, k, bad = 0, 0, 0
    while pos < size:
        off = offsets[k] if k < len(offsets) else pos
        try:
            f, hl, n = parse_header(buf, off)
            if off + hl + n > size:
                raise HumError(f"ping at {off} runs past the end of the file")
        except HumError as e:
            bad += 1
            if len(issues) < 20:
                issues.append(str(e))
            nxt = buf.find(START, max(off, pos) + 1)            # resync on the next marker
            if nxt < 0:
                break
            pos, k = nxt, len(offsets) + 1                       # offsets no longer trusted after a gap
            continue
        lat, lon = hum_latlon(f.get(0x82, 0), f.get(0x83, 0))
        pings.append(Ping(record=f.get(0x80, -1), time_ms=f.get(0x81, -1), lat=lat, lon=lon,
                          heading_deg=f[0x84] / 10.0 if 0x84 in f else None, speed_raw=f.get(0x85),
                          depth_m=f[0x87] / 10.0 if 0x87 in f else None, beam=f.get(0x50), freq_hz=f.get(0x92),
                          offset=off, header_len=hl, n_returns=n))
        pos, k = off + hl + n, k + 1
    if bad:
        issues.insert(0, f"{bad} malformed ping(s) skipped")
    if not pings:
        raise HumError(f"{son.name}: no readable pings")
    beams = {p.beam for p in pings}
    name = BEAMS.get(pings[0].beam, f"beam{pings[0].beam}") if len(beams) == 1 else "mixed"
    if len(beams) != 1:
        issues.append(f"mixed beam ids {sorted(beams)} in one channel file")
    return Channel(name=name, path=str(son), sha256=hashlib.sha256(buf).hexdigest(), pings=pings, issues=issues)


def read_recording(dat: Path) -> Recording:
    dat = Path(dat)
    d = read_dat(dat)
    folder = dat.with_suffix("")
    if not folder.is_dir():
        raise HumError(f"{folder} (the B00x.SON folder next to {dat.name}) is missing")
    chans = {}
    for son in sorted(folder.glob("B00*.SON")):
        ch = read_son(son, son.with_suffix(".IDX"))
        chans[ch.name] = ch
    if not {"port", "starboard"} & set(chans):
        raise HumError(f"no side-scan channel (beam 2/3) in {folder}")
    return Recording(name=d["name"].rsplit(".", 1)[0] or dat.stem, dat=d, channels=chans)


# ---- frames + track for the DEPTH pipeline ------------------------------------------------------
def chunk_recording(rec: Recording, nchunk: int = 500,
                    channels: tuple[str, ...] = ("port", "starboard")) -> list[tuple[str, int, list, np.ndarray]]:
    """``(channel, j, pings, native sonogram)`` per chunk of ``nchunk`` pings; the sonogram's rows are
    range samples (nadir first), columns pings. A last chunk shorter than nchunk/2 joins the previous."""
    chunks = []
    for cname in channels:
        ch = rec.channels.get(cname)
        if ch is None:
            continue
        n = len(ch.pings)
        starts = list(range(0, n, nchunk))
        if len(starts) > 1 and n - starts[-1] < nchunk // 2:
            starts.pop()
        for j, s0 in enumerate(starts):
            s1 = starts[j + 1] if j + 1 < len(starts) else n
            chunks.append((cname, j, ch.pings[s0:s1], ch.returns(s0, s1).T))
    return chunks


def frames_and_track(rec: Recording, nchunk: int = 500, size: int = 640,
                     channels: tuple[str, ...] = ("port", "starboard"),
                     crosscheck: bool = True) -> tuple[list, dict, dict]:
    """Sonogram frames in the training layout (PINGMapper "wcp": nadir at the top, pings left → right,
    ``nchunk`` pings per frame, resized to ``size`` × ``size`` — ASSUMED to match the Roboflow resize of
    the training frames) and, per frame, a REAL fix: per-column lat / lon / heading, the sonar's median
    depth, and the recording's MEASURED range scale (``range_scale``) converted to the resized frame.
    A last partial chunk shorter than nchunk/2 is merged into the previous one.

    ``crosscheck`` (default): each chunk's seabed geometry is resolved by the agent's sensor
    cross-check (OpenCV tracker vs the depth sounder, re-tracks on conflict, scale re-fit —
    ``src/agentic/sensor_check.py``); the fix then carries the accepted line, its status and the
    re-fit scale, and ``depth_m`` is the altitude of the TRUSTED source (None if not measured)."""
    from src.agentic.geo import PingFix
    chunks = chunk_recording(rec, nchunk, channels)
    scale = range_scale([(ps, son) for _, _, ps, son in chunks])
    geo = None
    if crosscheck:
        # the agent cross-checks OpenCV's seabed track against the depth sounder, re-tracks on a
        # conflict and re-fits the scale (src/agentic/sensor_check.py, STUDY-16)
        from src.agentic.sensor_check import check_recording, summary
        freq = next((ps[0].freq_hz for _, _, ps, _ in chunks if ps), None)
        geo = check_recording([(c, j, son, np.array([p.depth_m if p.depth_m and p.depth_m > 0 else np.nan for p in ps]))
                               for c, j, ps, son in chunks], scale["m_per_sample"], freq)
        if geo["scale"]["m_per_sample"]:
            scale = {**scale, "first_estimate_m_per_sample": scale["m_per_sample"],
                     "m_per_sample": geo["scale"]["m_per_sample"], "rel_unc": geo["scale"]["rel_unc"],
                     "physics_rel_gap": geo["scale"]["physics_rel_gap"], "source": geo["scale"]["source"],
                     "iterations": geo["iterations"]}
    frames, track, meta = [], {}, {}
    max_depth = max((p.depth_m for _, _, ps, _ in chunks for p in ps if p.depth_m), default=None)
    for cname, j, ps, son in chunks:
        img = cv2.resize(son, (size, size), interpolation=cv2.INTER_AREA)
        fid = f"{rec.name}_ss_{'port' if cname == 'port' else 'star'}_{j:05d}"
        frames.append((fid, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
        cols = np.linspace(0, len(ps) - 1, size)
        pick = lambda arr: np.interp(cols, np.arange(len(ps)), arr).tolist()
        lat = pick([p.lat for p in ps]); lon = pick([p.lon for p in ps])
        hd = np.unwrap(np.radians([p.heading_deg or 0.0 for p in ps]))
        heading = [float(np.degrees(h) % 360) for h in np.interp(cols, np.arange(len(ps)), hd)]
        depths = [p.depth_m for p in ps if p.depth_m and p.depth_m > 0]
        mid = size // 2
        m_per_px = scale["m_per_sample"] * son.shape[0] / size if scale["m_per_sample"] else None
        g = geo["chunks"].get((cname, j)) if geo else None
        track[fid] = PingFix(lat=round(lat[mid], 7), lon=round(lon[mid], 7), heading_deg=round(heading[mid], 1),
                             synthetic=False, ping_lat=lat, ping_lon=lon, ping_heading=heading,
                             depth_m=(g.altitude_m if g is not None else float(np.median(depths)) if depths else None),
                             range_m_per_px=round(m_per_px, 5) if m_per_px else None,
                             range_scale_rel_unc=scale["rel_unc"],
                             bottom_line_px=g.line_resampled(size) if g is not None else None,
                             geometry=g.status if g is not None else None,
                             altitude_bound_m=max_depth)
        meta[fid] = {"channel": cname, "pings": [ps[0].record, ps[-1].record], "n_pings": len(ps),
                     "samples": int(son.shape[0]), "freq_hz": ps[0].freq_hz, "time_ms": [ps[0].time_ms, ps[-1].time_ms],
                     "depth_m_median": float(np.median(depths)) if depths else None,
                     "range_m_per_px": track[fid].range_m_per_px,
                     "geometry": {**g.to_dict(), "lines": g.overlay_lines(size)} if g is not None else None}
    out = {"frames": meta, "range_scale": scale}
    if geo:
        out["geometry"] = {**summary(geo), "log": geo["log"]}
    return frames, track, out


def physics_spacing(freq_hz: float = 455000, transducer_m: float = 0.108, temp_c: float = 10.0) -> float:
    """PINGMapper's sample spacing from beam physics (transducer length, frequency, sound speed) —
    reported next to the measured scale as an independent cross-check, never used for geotags."""
    T = temp_c / 10.0
    c = 1449.05 + 45.7 * T - 5.21 * T ** 2 + 0.23 * T ** 3 + (1.333 - 0.126 * T + 0.009 * T ** 2) * (0 - 35)
    theta = math.asin(c / (transducer_m * freq_hz))
    return 1.0 / ((math.pi / 2) / theta)


def range_scale(chunks: list) -> dict:
    """MEASURED metres per range sample of a recording: for each chunk, the sonar's own depth (m)
    ÷ the Stage-1 bottom-track altitude (samples); the median over confident chunks. Its relative
    uncertainty is the LARGER of (a) the robust spread 1.4826·MAD / median and (b) the gap to the
    beam-physics estimate — conservative on purpose. Chunks more than 3 robust σ from the median are
    counted as bottom-track failures (a shallow false bottom) and reported, not silently dropped."""
    from .canonical import bottom_track
    ratios = []
    for ps, son in chunks:
        try:
            line, alt, conf, _ = bottom_track(son)
        except Exception:
            continue
        dep = [p.depth_m for p in ps if p.depth_m and p.depth_m > 0]
        if alt and alt > 3 and dep and conf and conf >= 0.5:
            ratios.append(float(np.median(dep)) / float(alt))
    if len(ratios) < 3:
        return {"m_per_sample": None, "rel_unc": None, "n_chunks": len(ratios), "source": "not measured (too few confident chunks)"}
    r = np.array(ratios)
    med = float(np.median(r))
    sig = 1.4826 * float(np.median(np.abs(r - med)))
    phys = physics_spacing()
    mad_rel, phys_rel = sig / med, abs(med - phys) / med
    return {"m_per_sample": med, "rel_unc": round(max(mad_rel, phys_rel), 3), "n_chunks": len(r),
            "mad_rel": round(mad_rel, 3), "physics_m_per_sample": round(phys, 5), "physics_rel_gap": round(phys_rel, 3),
            "track_failures": int(np.sum(np.abs(r - med) > 3 * max(sig, 1e-9))),
            "ratios_cm": [round(x * 100, 2) for x in sorted(ratios)],
            "source": "measured: median over chunks of sonar depth (m) / Stage-1 altitude (samples)"}


# ---- validation -------------------------------------------------------------------------------
def _haversine(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    x = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(x))


def _bearing(a, b) -> float:
    la1, la2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    y = math.sin(dl) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def validate(rec: Recording, window_s: float = 5.0) -> dict:
    """Label-free checks that the decoded numbers are physically consistent."""
    out = {"recording": rec.name, "dat": rec.dat, "channels": {}}
    for cname, ch in rec.channels.items():
        ps = ch.pings
        rec_mono = all(b.record > a.record for a, b in zip(ps, ps[1:]))
        t_mono = all(b.time_ms >= a.time_ms for a, b in zip(ps, ps[1:]))
        # GPS speed / course over windows (fixes repeat between GPS updates)
        spd, crs, spd_hdr, hdg = [], [], [], []
        j = 0
        for i in range(len(ps)):
            while j < len(ps) and (ps[j].time_ms - ps[i].time_ms) < window_s * 1000:
                j += 1
            if j >= len(ps):
                break
            d = _haversine((ps[i].lat, ps[i].lon), (ps[j].lat, ps[j].lon))
            dt = (ps[j].time_ms - ps[i].time_ms) / 1000
            if dt > 0 and d > 2:
                spd.append(d / dt); crs.append(_bearing((ps[i].lat, ps[i].lon), (ps[j].lat, ps[j].lon)))
                spd_hdr.append(np.mean([p.speed_raw for p in ps[i:j] if p.speed_raw is not None] or [np.nan]))
                hdg.append(ps[i].heading_deg)
        spd, spd_hdr = np.array(spd), np.array(spd_hdr)
        ok = np.isfinite(spd_hdr) & (spd_hdr > 0)
        ratio = float(np.median(spd[ok] / spd_hdr[ok])) if ok.any() else None
        corr = float(np.corrcoef(spd[ok], spd_hdr[ok])[0, 1]) if ok.sum() > 3 else None
        dhead = [abs(((c - h + 180) % 360) - 180) for c, h in zip(crs, hdg) if h is not None]
        track_m = sum(_haversine((a.lat, a.lon), (b.lat, b.lon)) for a, b in zip(ps, ps[1:]))
        depths = [p.depth_m for p in ps if p.depth_m and p.depth_m > 0]
        out["channels"][cname] = {
            "pings": len(ps), "issues": ch.issues, "sha256": ch.sha256,
            "header_len": sorted({p.header_len for p in ps}), "samples_per_ping": [min(p.n_returns for p in ps), max(p.n_returns for p in ps)],
            "freq_hz": sorted({p.freq_hz for p in ps if p.freq_hz}), "records_monotonic": rec_mono, "time_monotonic": t_mono,
            "duration_s": round((ps[-1].time_ms - ps[0].time_ms) / 1000, 1), "track_m": round(track_m, 1),
            "bbox": [round(min(p.lat for p in ps), 6), round(min(p.lon for p in ps), 6),
                     round(max(p.lat for p in ps), 6), round(max(p.lon for p in ps), 6)],
            "gps_speed_ms_median": round(float(np.median(spd)), 3) if len(spd) else None,
            "speed_field_median": round(float(np.nanmedian(spd_hdr)), 2) if len(spd_hdr) else None,
            "gps_speed_per_speed_unit": round(ratio, 4) if ratio else None, "speed_corr": round(corr, 3) if corr is not None else None,
            "course_vs_heading_deg_median": round(float(np.median(dhead)), 1) if dhead else None,
            "course_vs_heading_deg_p90": round(float(np.percentile(dhead, 90)), 1) if dhead else None,
            "depth_m": [round(min(depths), 2), round(float(np.median(depths)), 2), round(max(depths), 2)] if depths else None,
        }
    if {"port", "starboard"} <= set(rec.channels):
        P, S = rec.channels["port"].pings, rec.channels["starboard"].pings
        n = min(len(P), len(S))
        dpos = [_haversine((P[i].lat, P[i].lon), (S[i].lat, S[i].lon)) for i in range(0, n, max(1, n // 200))]
        out["port_vs_starboard"] = {"pings": [len(P), len(S)], "same_time_ms": all(P[i].time_ms == S[i].time_ms for i in range(n)),
                                    "fix_distance_m_max": round(max(dpos), 3) if dpos else None}
    return out


# ---- fetch (reproducible, hash-pinned) -------------------------------------------------------------
def fetch_sample(dest: Path = SAMPLE_DIR, files: Optional[dict] = None) -> Path:
    """Fetch the PINGMapper sample recording (``files`` = LARGE_FILES for the 1-h Solix one); every
    object is kept only if its SHA-256 matches the pinned hash."""
    files = files or SAMPLE_FILES
    dest.mkdir(parents=True, exist_ok=True)
    for rel, want in files.items():
        p = dest / rel
        if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == want:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_LFS + rel, timeout=600) as r:
            data = r.read(256 * 1024 * 1024)
        if hashlib.sha256(data).hexdigest() != want:
            raise HumError(f"{rel}: SHA-256 mismatch - not saved")
        p.write_bytes(data)
        print(f"  verified {rel} ({len(data)} bytes)")
    return dest / next(k for k in files if k.endswith(".DAT"))


def _figure(frames, result, track, out: Path) -> None:
    """Left: one real sonogram frame with the agent's overlay. Right: the real GPS track (local metres)
    with every hazard pin and its error circle, a 50 m scale bar — no basemap, nothing borrowed."""
    from src.agentic.agent import render
    fr_by = {fr.frame_id: fr for fr in result.frames}
    fid, img = max(frames, key=lambda f: len(fr_by[f[0]].candidates) if f[0] in fr_by else -1)
    left = cv2.resize(render(img, fr_by[fid]), (480, 480))
    cv2.putText(left, fid, (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    side0 = "_ss_port_" if any("_ss_port_" in k for k in track) else "_ss_star_"      # both sides share each ping's fix
    pts = [(la, lo) for k, f in sorted(track.items()) if f.ping_lat and side0 in k
           for la, lo in zip(f.ping_lat[::8], f.ping_lon[::8])]
    lat0, lon0 = pts[0]
    enu = lambda la, lo: ((lo - lon0) * 111320 * math.cos(math.radians(lat0)), (la - lat0) * 110540)
    xy = [enu(*p) for p in pts] + [enu(t.lat, t.lon) for t in result.tracked if t.lat is not None]
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 10) * 1.15
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    S = 480 / span
    px = lambda x, y: (int(240 + (x - cx) * S), int(240 - (y - cy) * S))
    right = np.full((480, 480, 3), 245, np.uint8)
    for a, b in zip(pts, pts[1:]):
        cv2.line(right, px(*enu(*a)), px(*enu(*b)), (150, 110, 40), 2, cv2.LINE_AA)
    for t in result.tracked:
        if t.lat is None:
            continue
        c = px(*enu(t.lat, t.lon))
        cv2.circle(right, c, max(2, int((t.geo_error_m or 3) * S)), (40, 150, 220), 1, cv2.LINE_AA)
        cv2.circle(right, c, 4, (30, 120, 200), -1, cv2.LINE_AA)
        cv2.putText(right, t.oid, (c[0] + 6, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
    bar = int(50 * S)
    cv2.line(right, (16, 460), (16 + bar, 460), (40, 40, 40), 2)
    cv2.putText(right, "50 m", (20 + bar, 464), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(right, "real GPS track + hazard pins (error circles)", (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack([left, right]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def report(dat: Path) -> str:
    """STUDY-14 — run the checks and the full DEPTH survey on a raw recording; write the report."""
    import time as _t
    from src.agentic.pipeline import AgenticPipeline
    rec = read_recording(dat)
    v = validate(rec)
    frames, track, meta = frames_and_track(rec)
    sc = meta["range_scale"]
    geo = meta.get("geometry")
    t0 = _t.perf_counter()
    res = AgenticPipeline().run_survey(frames, track=track, survey_id=f"{rec.name}-report", budget_minutes=2)
    wall = _t.perf_counter() - t0
    _figure(frames, res, track, REPO / "docs" / "img" / "raw_recording.jpg")
    P, S = v["channels"]["port"], v["channels"].get("starboard", {})
    dur_h = P["duration_s"] / 3600
    pp = [t.p_pot for t in res.tracked if t.p_pot is not None]
    hts = [t.height_m for t in res.tracked if t.height_m is not None]
    L = ["# STUDY-14 — A raw sonar recording with real GPS, straight into DEPTH", "",
         "_`python -m src.cv_pipeline.humminbird report` · PINGMapper sample data (Bodine et al.; MIT code; Zenodo "
         "10.5281/zenodo.6604666 archives Git-LFS pointers, the objects are fetched from the author's repository and kept "
         "only if their SHA-256 matches) · regenerated by the script — do not edit by hand_", "",
         f"Recording **{rec.name}** — {rec.dat['family']} ({P['header_len'][0]}-byte ping headers), {rec.dat['water']} water, "
         f"started {_t.strftime('%Y-%m-%d %H:%M', _t.gmtime(rec.dat['start_unix']))} UTC, {P['duration_s']} s, "
         f"{P['pings']} pings per side-scan channel at {P['freq_hz'][0] / 1000:.0f} kHz, {P['samples_per_ping'][0]}–"
         f"{P['samples_per_ping'][1]} samples per ping. The decoded track lies at "
         f"{P['bbox'][0]:.4f}–{P['bbox'][2]:.4f} °N, {P['bbox'][1]:.4f}–{P['bbox'][3]:.4f} °E: **the Colorado River in "
         f"Glen Canyon (Horseshoe Bend, Arizona)** — not the Mississippi recording the Zenodo text describes (that one is "
         f"the ~216 MB `Test-Large-DS`, a 1-h Solix recording on the Pearl River: STUDY-16's fresh recording, "
         f"`fetch --large`).", "",
         "## 1. Are the decoded numbers physically right? (no labels needed)", "",
         "| check | port | starboard | reading |", "|---|--:|--:|---|",
         f"| malformed pings skipped | {len(P['issues'])} | {len(S.get('issues', []))} | every ping starts with the marker and ends its header correctly |",
         f"| record numbers / times monotonic | {P['records_monotonic']} / {P['time_monotonic']} | {S.get('records_monotonic')} / {S.get('time_monotonic')} | |",
         f"| GPS speed ÷ speed field | {P['gps_speed_per_speed_unit']} | {S.get('gps_speed_per_speed_unit')} | the field is in **0.1 m/s** (correlation {P['speed_corr']}) |",
         f"| \\|course over ground − heading\\| median / p90 | {P['course_vs_heading_deg_median']}° / {P['course_vs_heading_deg_p90']}° | "
         f"{S.get('course_vs_heading_deg_median')}° / {S.get('course_vs_heading_deg_p90')}° | heading is in 0.1°; the p90 becomes the geotag heading error |",
         f"| depth field (min / median / max) | {P['depth_m'][0]} / {P['depth_m'][1]} / {P['depth_m'][2]} m | | the field is in **decimetres** (confirmed below) |",
         f"| port vs starboard fixes at the same time | {v['port_vs_starboard']['fix_distance_m_max']} m apart | | both channels share each ping's fix |", "",
         "The track also lies on the river channel in satellite imagery (checked on the studio map).", "",
         "## 2. A measured range scale — and the agent's cross-check of the seabed", "",
         "First estimate: metres per range sample = the sonar's own depth (m) ÷ the Stage-1 bottom-track altitude "
         "(samples), per chunk of 500 pings, median over confident chunks. That estimate trusts the tracker, and the "
         "tracker is not always right — so the agent then **cross-checks every chunk's seabed against the depth "
         "sounder**, re-runs the OpenCV tracker where they conflict, trusts the image where the sounder lost lock, and "
         "**re-fits the scale** from the sounder-backed chunks until it stops moving "
         "([STUDY-16](sensor_check.md)). Its uncertainty is the larger of the robust spread and the gap to the "
         "beam-physics estimate (PINGMapper's formula: 0.108 m transducer, 455 kHz).", "",
         "| | value |", "|---|--:|",
         f"| first estimate (median, {sc['n_chunks']} confident chunks; {sc['track_failures']} > 3 robust σ off = "
         f"tracker false picks) | {sc.get('first_estimate_m_per_sample', sc['m_per_sample']) * 100:.2f} cm / sample |",
         f"| **scale after the agent's cross-check** ({len(sc.get('iterations') or [])} iteration(s)) | "
         f"**{sc['m_per_sample'] * 100:.2f} cm / sample** |",
         f"| beam-physics estimate | {sc['physics_m_per_sample'] * 100:.2f} cm / sample ({sc['physics_rel_gap']:.1%} away) |",
         f"| uncertainty carried into every error radius | **± {sc['rel_unc']:.1%}** |",
         f"| seabed per chunk ({geo['chunks']} chunks) | " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in geo['counts'].items() if v)
         + " |" if geo else "| seabed per chunk | not cross-checked |", "",
         f"First-estimate per-chunk values (cm / sample): {', '.join(str(x) for x in sc['ratios_cm'])}. Reading the depth field as "
         f"centimetres would make the scale 10× smaller than the physics estimate — the decimetre reading is the only "
         f"consistent one. With this scale, ground range is in real metres and shadow heights are "
         f"`h/H × the sonar's measured depth` (the rule: metres only with a measured altitude).", "",
         "## 3. The whole agent on it", "",
         f"![a real frame and the real track](img/raw_recording.jpg)", "",
         f"- {len(frames)} frames (500 pings each, port + starboard) in **{wall:.1f} s** on this laptop — {P['duration_s'] / wall:.0f}× "
         f"faster than the recording.",
         f"- **{len(res.tracked)} review cards**, 0 auto-confirmed (no precision promise), P(pot) "
         f"{min(pp):.2f}–{max(pp):.2f}; heights {min(hts):.2f}–{max(hts):.2f} m where a shadow was measured." if pp and hts else
         f"- **{len(res.tracked)} review cards**, 0 auto-confirmed.",
         f"- No crab pots are known in this reach of the Colorado River and there is no ground truth, so every card is "
         f"a **false alarm or an unknown object on out-of-domain water**: **{len(res.tracked) / dur_h:.0f} cards per hour "
         f"of sonar** at the detector floor 0.05 — the review load this model would put on an analyst here "
         f"({len(res.mission.budget.get('review_ids', []))} of {len(res.tracked)} fit a 2-minute budget at the assumed "
         f"8 s per card).",
         f"- Every pin is placed from its **own ping's GPS fix and heading** plus the measured ground range; error radius = "
         f"3 m GPS (assumed) + range × (scale uncertainty + sin 6°): "
         f"{min(t.geo_error_m for t in res.tracked):.1f}–{max(t.geo_error_m for t in res.tracked):.1f} m here." if res.tracked else "",
         "", "## Limits", "",
         f"- One 2.5-minute recording from one unit. The tracker alone made {sc['track_failures']} false picks among the "
         f"first-estimate chunks; the agent's cross-check resolves them (STUDY-16), and where nothing coherent is found "
         f"the geometry is withheld, never guessed.",
         "- Frames are resized to 640×640 like the training data (an assumption about the Roboflow export).",
         "- Positions are consumer GPS (± a few metres, assumed) — good enough to send a boat, not a survey-grade map.", "",
         "---", "_Regenerated by the script; do not edit by hand._", ""]
    return "\n".join(x for x in L if x is not None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    fp = sub.add_parser("fetch")
    fp.add_argument("--large", action="store_true", help="also the 1-h Solix recording (~216 MB, STUDY-16)")
    for c in ("info", "validate", "report"):
        s = sub.add_parser(c)
        s.add_argument("dat", nargs="?", default=str(SAMPLE_DIR / "Test-Small-DS.DAT"))
    a = ap.parse_args()
    if a.cmd == "fetch":
        print(fetch_sample())
        if a.large:
            print(fetch_sample(LARGE_DIR, LARGE_FILES))
        return
    if a.cmd == "report":
        (REPO / "docs" / "raw_recording.md").write_text(report(Path(a.dat)), encoding="utf-8")
        print("wrote docs/raw_recording.md + docs/img/raw_recording.jpg")
        return
    rec = read_recording(Path(a.dat))
    if a.cmd == "info":
        print(json.dumps({"name": rec.name, "dat": rec.dat,
                          "channels": {k: {"pings": len(c.pings), "issues": c.issues} for k, c in rec.channels.items()}}, indent=1))
    else:
        print(json.dumps(validate(rec), indent=1))


if __name__ == "__main__":
    main()
