"""
geo.py — the **Act** stage's honest geotagging.

Turns a detection's pixel position into a latitude/longitude *only when real per-ping GPS is
available* (a PINGMapper-style track: each frame → boat lat/lon + heading). A side-scan detection
sits at an across-track ground range from the boat track, on the port or starboard side, so:

    detection_latlon = offset(boat_latlon, ground_range_m, heading ± 90°)

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


def range_px_from_nadir(bbox, nadir: str, w: int, h: int) -> float:
    """Across-track distance (pixels) of the box centre from the nadir edge."""
    cx, cy = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
    return {"top": cy, "bottom": h - cy, "left": cx, "right": w - cx}.get(nadir, cy)


def geotag(cand: Candidate, fix: PingFix, nadir: str, w: int, h: int,
           frame_id: str = "", m_per_px: float = DEFAULT_M_PER_PX) -> Candidate:
    """Fill ``cand.lat/lon/geo_error_m`` from the boat fix + across-track range. Mutates and returns."""
    ground_range_m = range_px_from_nadir(cand.bbox, nadir, w, h) * m_per_px
    side = parse_side(frame_id) or "starboard"
    cross_bearing = fix.heading_deg + (90.0 if side == "starboard" else -90.0)
    lat, lon = offset_latlon(fix.lat, fix.lon, ground_range_m, cross_bearing)
    cand.lat, cand.lon = round(lat, 6), round(lon, 6)
    # honest, coarse error: at least 3 m, growing with range (slant/heading/scale uncertainty)
    cand.geo_error_m = round(max(3.0, 0.25 * ground_range_m), 1)
    return cand


def synthetic_track(frame_ids: list[str], start=(37.9800, -76.0000),
                    heading_deg: float = 20.0, ping_spacing_m: float = 4.0) -> dict[str, PingFix]:
    """Build a plausible straight boat track for a demo. **All fixes are ``synthetic=True``** and must
    be surfaced as demo-only. Frames are ordered by the trailing ping index in their name when present."""
    def ping_index(fid: str) -> int:
        m = re.search(r"_(\d{3,6})(?:_png)?", fid)
        return int(m.group(1)) if m else 0

    ordered = sorted(frame_ids, key=ping_index)
    lat, lon = start
    track: dict[str, PingFix] = {}
    for i, fid in enumerate(ordered):
        la, lo = offset_latlon(lat, lon, i * ping_spacing_m, heading_deg)
        track[fid] = PingFix(lat=round(la, 6), lon=round(lo, 6), heading_deg=heading_deg, synthetic=True)
    return track


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    x = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_R * math.asin(math.sqrt(x))
