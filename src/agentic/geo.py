"""
geo.py — the **Act** stage's honest geotagging.

Turns a detection's pixel position into a latitude/longitude *only when real per-ping GPS is
available* (a PINGMapper-style track: each frame → boat lat/lon + heading). A side-scan detection
sits at an across-track ground range from the boat track, on the port or starboard side, so:

    detection_latlon = offset(boat_latlon, ground_range_m, heading ± 90°)

Stage 1 (``src.cv_pipeline.canonical``) supplies the two measured inputs:

* **across-track ground range** — slant range corrected with the bottom-tracked altitude,
  ``ground = sqrt(slant^2 - altitude^2)`` (``Candidate.ground_range_px``), instead of the raw
  distance from the nadir edge;
* **along-track position** — the object's ping index within the frame (``Candidate.ping_px``). A
  frame's fix is its centre ping; each object is moved along the heading by its ping offset, so two
  objects in one frame no longer share one boat position (PINGMapper convention: ping order = time
  order, left to right in a sonogram).

Honesty (report §5.9): the Hugging Face crab-pot frames carry **no GPS**. In that case we attach no
coordinates and flag ``gps_available = False`` — we never stamp one fake lat/lon onto every object
(the exact bug called out in ``docs/WINNING_REPORT.md``). For demos a clearly-labelled *synthetic*
track can be generated (``synthetic_track``), but every object it produces is marked
``synthetic=True`` and surfaced as "SYNTHETIC DEMO GPS — not real coordinates" downstream.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from .types import Candidate

_EARTH_R = 6_371_000.0     # metres
DEFAULT_M_PER_PX = 0.05    # demo range scale: a 640px frame ≈ 32 m across-track (per-survey in reality)


@dataclass
class PingFix:
    """The boat's position/attitude for one frame (from a real track, or synthetic for demos)."""
    lat: float
    lon: float
    heading_deg: float = 0.0     # course over ground, degrees from north
    altitude_m: float = 10.0     # sonar height above seabed (feeds the shadow-height estimate)
    synthetic: bool = False


def offset_latlon(lat: float, lon: float, dist_m: float, bearing_deg: float) -> tuple[float, float]:
    """Offset a lat/lon by ``dist_m`` along ``bearing_deg`` (equirectangular — accurate at survey scale)."""
    b = math.radians(bearing_deg)
    dlat = (dist_m * math.cos(b)) / _EARTH_R
    dlon = (dist_m * math.sin(b)) / (_EARTH_R * math.cos(math.radians(lat)))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


def parse_side(frame_id: str) -> Optional[str]:
    """Read 'port'/'starboard' from a PINGMapper-style filename (…_ss_port_…). None if unknown."""
    m = re.search(r"_ss_(port|star|starboard)", frame_id.lower())
    if not m:
        return None
    return "port" if m.group(1) == "port" else "starboard"


def parse_recording(frame_id: str) -> Optional[str]:
    """Read the recording id (e.g. 'rec6') from a PINGMapper-style filename. None if unknown."""
    m = re.search(r"(rec\d+)", frame_id.lower())
    return m.group(1) if m else None


def parse_ping(frame_id: str) -> Optional[int]:
    """Read the trailing ping/chunk index from a PINGMapper-style filename. None if unknown."""
    m = re.search(r"_(\d{3,6})(?:_png)?", frame_id.lower())
    return int(m.group(1)) if m else None


def range_px_from_nadir(bbox, nadir: Optional[str], w: int, h: int) -> Optional[float]:
    """Across-track distance (pixels) of the box centre from the nadir edge; None if the frame's
    orientation is unknown (never guessed — see ``src.cv_pipeline.orientation``)."""
    cx, cy = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
    return {"top": cy, "bottom": h - cy, "left": cx, "right": w - cx}.get(nadir or "")


def geotag(cand: Candidate, fix: PingFix, nadir: str, w: int, h: int,
           frame_id: str = "", m_per_px: float = DEFAULT_M_PER_PX) -> Candidate:
    """Fill ``cand.lat/lon/geo_error_m`` from the boat fix + across-track range. Mutates and returns.

    Unknown orientation ⇒ the range can't be measured, so the object is placed at the boat fix with
    an error radius covering the whole swath (an honest "somewhere in this frame")."""
    range_px = cand.ground_range_px if cand.ground_range_px is not None \
        else range_px_from_nadir(cand.bbox, nadir, w, h)
    if range_px is None:
        cand.lat, cand.lon = round(fix.lat, 6), round(fix.lon, 6)
        cand.geo_error_m = round(max(3.0, max(w, h) * m_per_px), 1)
        return cand
    lat0, lon0 = fix.lat, fix.lon
    if cand.ping_px is not None and cand.n_pings:                 # along-track: this object's ping
        along_m = (cand.ping_px - 0.5 * cand.n_pings) * m_per_px
        lat0, lon0 = offset_latlon(lat0, lon0, along_m, fix.heading_deg)
    ground_range_m = range_px * m_per_px
    side = parse_side(frame_id) or "starboard"
    cross_bearing = fix.heading_deg + (90.0 if side == "starboard" else -90.0)
    lat, lon = offset_latlon(lat0, lon0, ground_range_m, cross_bearing)
    cand.lat, cand.lon = round(lat, 6), round(lon, 6)
    # honest, coarse error: at least 3 m, growing with range (slant/heading/scale uncertainty)
    cand.geo_error_m = round(max(3.0, 0.25 * ground_range_m), 1)
    return cand


def synthetic_track(frame_ids: list[str], start=(37.8000, -76.1500),   # open Chesapeake Bay water
                    heading_deg: float = 20.0,
                    frame_len_m: float = 640 * DEFAULT_M_PER_PX) -> dict[str, PingFix]:
    """Build a plausible straight boat track for a demo. **All fixes are ``synthetic=True``** and must
    be surfaced as demo-only. Frames are ordered by the trailing chunk index in their name; the fix
    of chunk ``k`` is its centre ping, ``k * frame_len_m`` along the track, so consecutive chunks
    tile the track (a stitched object lands in one place from both chunks)."""
    ordered = sorted(frame_ids, key=lambda fid: parse_ping(fid) or 0)
    lat, lon = start
    k0 = parse_ping(ordered[0]) or 0 if ordered else 0
    track: dict[str, PingFix] = {}
    for i, fid in enumerate(ordered):
        k = parse_ping(fid)
        pos = (k - k0) if k is not None else i
        la, lo = offset_latlon(lat, lon, pos * frame_len_m, heading_deg)
        track[fid] = PingFix(lat=round(la, 6), lon=round(lo, 6), heading_deg=heading_deg, synthetic=True)
    return track


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    x = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_R * math.asin(math.sqrt(x))


def stage1_counterfactual(frames, tracked, track: Optional[dict], m_per_px: float = DEFAULT_M_PER_PX) -> dict:
    """Where each hazard's pin would be WITHOUT Stage 1 — slant range read from the box and the frame's
    centre ping instead of the tracked ground range and the object's own ping (STUDY-12, live for this
    survey). Nothing is re-inferred: the same candidate is geotagged again with Stage-1 geometry removed.
    ``outside`` = the pin would leave its own stated error circle."""
    import copy
    import statistics
    if not track:
        return {"available": False, "reason": "no GPS track"}
    idx = {(fr.frame_id, tuple(c.bbox)): (fr, c) for fr in frames for c in fr.candidates}
    pins, shifts = {}, []
    for t in tracked:
        hit = idx.get((t.frame_id, tuple(t.bbox)))
        if t.lat is None or hit is None or t.frame_id not in track:
            continue
        fr, c = hit
        naive = copy.copy(c)
        naive.ping_px = naive.n_pings = naive.ground_range_px = None
        geotag(naive, track[t.frame_id], fr.nadir, fr.width, fr.height, fr.frame_id, m_per_px)
        if naive.lat is None:
            continue
        s = haversine_m((t.lat, t.lon), (naive.lat, naive.lon))
        pins[t.oid] = {"lat": naive.lat, "lon": naive.lon, "shift_m": round(s, 1),
                       "outside": bool(t.geo_error_m is not None and s > t.geo_error_m)}
        shifts.append(s)
    return {"available": bool(pins), "pins": pins, "n": len(pins),
            "outside": sum(p["outside"] for p in pins.values()),
            "median_shift_m": round(statistics.median(shifts), 1) if shifts else None,
            "max_shift_m": round(max(shifts), 1) if shifts else None,
            "note": "counterfactual: slant range + frame-centre ping instead of Stage-1 ground range + own ping "
                    "(tiers are unchanged - Stage 1 is not a gate)"}
