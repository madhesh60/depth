"""
twin.py — the 3D digital twin payloads (physics-grounded, not decoration).

Two scenes, both built only from measured geometry:

* :func:`frame_twin` — ONE sonar frame in true across-track **ground-range** geometry:
  the seabed texture is Stage 1's slant→ground remap (``cv2.remap``); the sonar path is the
  **tracked altitude per ping**; every find sits at its ping and ground range, and its height is the
  **shadow-derived** relative height × the tracked altitude (h = h/H · H, both in pixels — no scale
  assumption needed for the proportions). Finds without a measurable shadow get no height (drawn
  flat, labelled). For the selected find the viewer draws the acoustic ray triangle the height came
  from: sonar → object top → end of the shadow on the seabed.
* :func:`survey_twin` — a SURVEY on its track (local metres, east/north from the first fix): each
  frame as a swath ribbon on its side of the track (texture = its ground-range view), hazards at their
  geotags with shadow-derived heights, recovery / inspection routes, the opposite-side re-survey
  passes, and the track itself for the replay animation. Needs GPS (synthetic demo tracks are
  flagged). Absolute scale uses the same ``m_per_px`` as the geotag (a demo assumption, labelled).
"""
from __future__ import annotations

import base64
import math
from typing import Optional

import cv2
import numpy as np

from src.cv_pipeline.canonical import Canonicaliser, CanonicalFrame
from .geo import DEFAULT_M_PER_PX, PingFix, parse_side, _EARTH_R
from .types import FrameResult, SurveyResult


def _jpeg(img: np.ndarray, max_side: int, quality: int = 82) -> str:
    s = max_side / max(1, max(img.shape[:2]))
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _ground_texture(C: Canonicaliser, cf: CanonicalFrame) -> Optional[tuple[np.ndarray, int]]:
    """Ground-range view cropped to the rows that exist at every ping (rows = ground range px)."""
    gv = C.ground_view(cf)
    if gv is None:
        return None
    H = gv.shape[0]
    alt_max = float(np.max(cf.bottom_line))
    g_max = int(math.floor(math.sqrt(max(0.0, (H - 1) ** 2 - alt_max ** 2))))
    return gv[: max(2, g_max)], max(2, g_max)


def frame_twin(frame: np.ndarray, result: FrameResult, nadir: Optional[str] = None,
               C: Optional[Canonicaliser] = None, max_tex: int = 512) -> dict:
    C = C or Canonicaliser()
    cf = C.process(frame, result.frame_id, nadir=nadir)
    if not cf.measured:
        return {"available": False,
                "reason": "the seabed was not tracked in this frame (unknown orientation or no water-column "
                          "step) - the 3D twin needs measured geometry, so it is not drawn"}
    tex = _ground_texture(C, cf)
    if tex is None:
        return {"available": False, "reason": "no ground-range view"}
    gimg, G = tex
    n_pings = len(cf.bottom_line)
    idx = np.linspace(0, n_pings - 1, min(n_pings, 96)).round().astype(int)
    objects = []
    for i, c in enumerate(result.candidates):
        if c.ping_px is None or c.ground_range_px is None:
            continue
        x1, y1, x2, y2 = c.bbox
        corners = [cf.to_canonical(x, y) for x, y in ((x1, y1), (x2, y2))]
        pings = sorted(p for p, _ in corners)
        alt = cf.altitude_at(c.ping_px) or cf.altitude_px
        grounds = sorted(math.sqrt(max(s * s - alt * alt, 0.0)) for _, s in corners)
        ev = c.evidence
        h_px, shadow = None, None
        # height only from a real (CLEAR / WEAK) shadow - a NONE-quality run is speckle, not geometry
        if ev and ev.shadow.orientation_known and ev.shadow.has_shadow and ev.shadow.height_rel > 0:
            h_px = round(float(ev.shadow.height_rel) * alt, 2)
            sx1, sy1, sx2, sy2 = ev.shadow.strip
            s_c = [cf.to_canonical(x, y) for x, y in ((sx1, sy1), (sx2, sy2))]
            sg = sorted(math.sqrt(max(s * s - alt * alt, 0.0)) for _, s in s_c)
            shadow = {"g0": round(sg[0], 1), "g1": round(sg[1], 1), "quality": ev.shadow.quality.value}
        objects.append({
            "id": i, "ping": round(c.ping_px, 1), "ground": round(c.ground_range_px, 1),
            "along": [round(pings[0], 1), round(pings[1], 1)], "across": [round(grounds[0], 1), round(grounds[1], 1)],
            "height_px": h_px, "height_rel": ev.shadow.height_rel if ev else 0.0, "shadow": shadow,
            "altitude_px": round(alt, 2), "verdict": c.verdict.value if c.verdict else None,
            "conf": round(c.conf, 3), "p_pot": ev.p_pot if ev else None, "cls": c.cls_name,
            "in_water_column": c.in_water_column})
    return {
        "available": True, "units": "px", "n_pings": n_pings, "n_ground": G,
        "altitude": {"pings": idx.tolist(), "px": [round(float(cf.bottom_line[j]), 2) for j in idx]},
        "altitude_px": round(cf.altitude_px, 2),
        "texture": _jpeg(cv2.cvtColor(gimg, cv2.COLOR_GRAY2BGR), max_tex),
        "objects": objects,
        "note": "seabed in across-track GROUND range (Stage 1 remap); heights from the measured shadow x the "
                "tracked altitude; relief view = backscatter, not bathymetry; px along-track ~ px across-track assumed",
    }


# ======================================================================= survey twin
def _enu(lat, lon, lat0, lon0):
    return (math.radians(lon - lon0) * _EARTH_R * math.cos(math.radians(lat0)),
            math.radians(lat - lat0) * _EARTH_R)


def _unit(b):
    r = math.radians(b)
    return math.sin(r), math.cos(r)


def survey_twin(frames: list[tuple[str, np.ndarray]], result: SurveyResult, track: Optional[dict[str, PingFix]],
                m_per_px: float = DEFAULT_M_PER_PX, nadir: Optional[str] = None) -> dict:
    if not track:
        return {"available": False, "reason": "the survey twin needs a track (GPS) - run with a GPS source"}
    C = Canonicaliser()
    fixes = [track[f] for f, _ in frames if f in track]
    if not fixes:
        return {"available": False, "reason": "no frame has a fix"}
    lat0, lon0 = fixes[0].lat, fixes[0].lon
    tex_side = 256 if len(frames) <= 20 else 160
    ribbons, alts = [], {}
    fr_by_id = {fr.frame_id: fr for fr in result.frames}
    for fid, img in frames:
        fix, side = track.get(fid), parse_side(fid)
        if fix is None or side is None:
            continue
        cf = C.process(img, fid, nadir=nadir)
        if not cf.measured:
            continue
        tex = _ground_texture(C, cf)
        if tex is None:
            continue
        gimg, G = tex
        W = len(cf.bottom_line)
        alts[fid] = cf.altitude_px
        cx, cy = _enu(fix.lat, fix.lon, lat0, lon0)
        ux, uy = _unit(fix.heading_deg)
        vx, vy = _unit(fix.heading_deg + (90.0 if side == "starboard" else -90.0))

        def at(p, g):
            a, c = (p - W / 2) * m_per_px, g * m_per_px
            return [round(cx + ux * a + vx * c, 3), round(cy + uy * a + vy * c, 3)]
        ribbons.append({"frame_id": fid, "side": side, "heading": fix.heading_deg,
                        "corners": {"p0g0": at(0, 0), "p1g0": at(W, 0), "p0g1": at(0, G), "p1g1": at(W, G)},
                        "texture": _jpeg(cv2.cvtColor(gimg, cv2.COLOR_GRAY2BGR), tex_side, 75),
                        "altitude_m": round(cf.altitude_px * m_per_px, 3),
                        "counts": fr_by_id[fid].counts if fid in fr_by_id else {}})
    objs = []
    for t in result.tracked:
        if t.lat is None:
            continue
        x, y = _enu(t.lat, t.lon, lat0, lon0)
        alt_px = alts.get(t.frame_id)
        h = (round(t.height_rel * alt_px * m_per_px, 3)
             if (alt_px and t.height_rel and t.shadow_quality in ("clear", "weak")) else None)
        objs.append({"oid": t.oid, "xy": [round(x, 3), round(y, 3)], "height_m": h, "verdict": t.verdict.value,
                     "p_pot": t.p_pot, "conf": t.conf, "err_m": t.geo_error_m, "sightings": t.sightings,
                     "frame_id": t.frame_id})
    by = {o["oid"]: o["xy"] for o in objs}
    m = result.mission
    rs = []
    for L in (m.resurvey_plan or {}).get("lines", []):
        rs.append({"id": L["id"], "a": [round(v, 3) for v in _enu(*L["start"], lat0, lon0)],
                   "b": [round(v, 3) for v in _enu(*L["end"], lat0, lon0)], "targets": L["targets"],
                   "voi": L["voi"], "boat_min": L["boat_min"]})
    path = []
    for fid, fx in sorted(((f, track[f]) for f, _ in frames if f in track),
                          key=lambda kv: _along(kv[1], lat0, lon0, fixes[0].heading_deg)):
        cx, cy = _enu(fx.lat, fx.lon, lat0, lon0)
        path.append([round(cx, 3), round(cy, 3)])
    alt_m = float(np.median(list(alts.values()))) * m_per_px if alts else 1.0
    return {
        "available": True, "units": "m", "m_per_px": m_per_px, "synthetic": any(f.synthetic for f in fixes),
        "origin": [lat0, lon0], "ribbons": ribbons, "objects": objs, "track": path, "altitude_m": round(alt_m, 3),
        "routes": {"recovery": [by[i] for i in m.recovery_route if i in by],
                   "inspection": [by[i] for i in m.inspection_route if i in by]},
        "resurvey": rs,
        "note": ("SYNTHETIC demo track - positions are not real. " if any(f.synthetic for f in fixes) else "")
                + f"Scale: {m_per_px} m/px (demo assumption, same as the geotag); heights = shadow h/H x tracked altitude.",
    }


def _along(fix: PingFix, lat0, lon0, heading) -> float:
    x, y = _enu(fix.lat, fix.lon, lat0, lon0)
    ux, uy = _unit(heading)
    return x * ux + y * uy
