"""
mission.py — the **Act** stage: turn confirmed hazards into a human-approved cleanup plan.

* CONFIRMED objects → a **recovery route** (greedy nearest-neighbour ordering — "always go to the
  nearest next pot", which is enough for a small survey).
* REVIEW objects → a **resurvey list** for a human to revisit (nothing is dispatched from REVIEW).
* Exports the plan as GeoJSON / GPX / KML / CSV / JSON that a boat crew's GPS or a GIS can open.

Nothing is ever auto-dispatched: ``MissionPlan.human_approval_required`` is always True. Geometry
(route/waypoints) is only emitted when GPS is available; without it the plan degrades honestly to a
frame-ordered CSV and a "no GPS" flag. When the GPS is a demo track, exports carry a clear
``SYNTHETIC DEMO GPS`` marker.
"""
from __future__ import annotations

import csv
import io
import json
from typing import Optional

from .geo import haversine_m
from .types import MissionPlan, TrackedObject, Verdict, SurveyResult

_SYNTH_WARN = "SYNTHETIC DEMO GPS - not real coordinates"


def _nearest_neighbour(points: list[TrackedObject]) -> list[TrackedObject]:
    """Greedy NN route over geolocated objects, starting from the south-west-most."""
    pts = [p for p in points if p.lat is not None and p.lon is not None]
    if len(pts) <= 2:
        return pts
    remaining = pts[:]
    cur = min(remaining, key=lambda p: (p.lat, p.lon))
    remaining.remove(cur)
    route = [cur]
    while remaining:
        nxt = min(remaining, key=lambda p: haversine_m((cur.lat, cur.lon), (p.lat, p.lon)))
        remaining.remove(nxt)
        route.append(nxt)
        cur = nxt
    return route


def build_mission(tracked: list[TrackedObject], gps_available: bool,
                  gps_synthetic: bool = False) -> MissionPlan:
    confirmed = [t for t in tracked if t.verdict is Verdict.CONFIRMED]
    review = [t for t in tracked if t.verdict is Verdict.REVIEW]

    if gps_available:
        ordered = _nearest_neighbour(confirmed)
        route_ids = [t.oid for t in ordered]
        length = sum(haversine_m((a.lat, a.lon), (b.lat, b.lon))
                     for a, b in zip(ordered, ordered[1:])) if len(ordered) > 1 else 0.0
        length = round(length, 1)
    else:
        # no GPS → deterministic frame-ordered plan, no geometry
        route_ids = [t.oid for t in sorted(confirmed, key=lambda t: (t.frame_id, t.oid))]
        length = None

    counts = {v.value: sum(t.verdict is v for t in tracked) for v in Verdict}
    return MissionPlan(
        recovery_route=route_ids, resurvey=[t.oid for t in review], route_length_m=length,
        gps_available=gps_available, gps_synthetic=gps_synthetic,
        human_approval_required=True, counts=counts,
    )


# ---- exporters --------------------------------------------------------------------------------
def _by_id(tracked: list[TrackedObject]) -> dict[str, TrackedObject]:
    return {t.oid: t for t in tracked}


def to_geojson(tracked: list[TrackedObject], mission: MissionPlan) -> str:
    features = []
    for t in tracked:
        if t.lat is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [t.lon, t.lat]},
            "properties": {"id": t.oid, "class": t.cls_name, "verdict": t.verdict.value,
                           "conf": t.conf, "evidence_score": t.evidence_score,
                           "height_m": t.height_m, "height_rel_alt": t.height_rel,
                           "shadow": t.shadow_quality, "also_in": t.also_in,
                           "geo_error_m": t.geo_error_m, "frame": t.frame_id},
        })
    idx = _by_id(tracked)
    coords = [[idx[i].lon, idx[i].lat] for i in mission.recovery_route if idx.get(i) and idx[i].lat is not None]
    if len(coords) > 1:
        features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                         "properties": {"name": "recovery_route", "length_m": mission.route_length_m}})
    fc = {"type": "FeatureCollection",
          "properties": {"human_approval_required": True,
                         "note": _SYNTH_WARN if mission.gps_synthetic else "real GPS"},
          "features": features}
    return json.dumps(fc, indent=2)


def to_gpx(tracked: list[TrackedObject], mission: MissionPlan) -> str:
    idx = _by_id(tracked)
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<gpx version="1.1" creator="marine-debris-agent" xmlns="http://www.topografix.com/GPX/1/1">']
    if mission.gps_synthetic:
        out.append(f"  <metadata><desc>{_SYNTH_WARN}</desc></metadata>")
    for t in tracked:
        if t.lat is None:
            continue
        out.append(f'  <wpt lat="{t.lat}" lon="{t.lon}"><name>{t.oid}</name>'
                   f'<desc>{t.verdict.value} {t.cls_name} conf={t.conf:.2f}</desc></wpt>')
    route = [idx[i] for i in mission.recovery_route if idx.get(i) and idx[i].lat is not None]
    if len(route) > 1:
        out.append(f'  <rte><name>recovery_route ({mission.route_length_m} m)</name>')
        out += [f'    <rtept lat="{t.lat}" lon="{t.lon}"><name>{t.oid}</name></rtept>' for t in route]
        out.append("  </rte>")
    out.append("</gpx>")
    return "\n".join(out)


def to_kml(tracked: list[TrackedObject], mission: MissionPlan) -> str:
    idx = _by_id(tracked)
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           f"  <name>marine debris hazards{' (' + _SYNTH_WARN + ')' if mission.gps_synthetic else ''}</name>"]
    for t in tracked:
        if t.lat is None:
            continue
        out.append(f"  <Placemark><name>{t.oid}</name>"
                   f"<description>{t.verdict.value} {t.cls_name} conf={t.conf:.2f} "
                   f"height_rel_alt={t.height_rel} shadow={t.shadow_quality}</description>"
                   f"<Point><coordinates>{t.lon},{t.lat},0</coordinates></Point></Placemark>")
    route = [idx[i] for i in mission.recovery_route if idx.get(i) and idx[i].lat is not None]
    if len(route) > 1:
        line = " ".join(f"{t.lon},{t.lat},0" for t in route)
        out.append(f"  <Placemark><name>recovery_route</name><LineString>"
                   f"<coordinates>{line}</coordinates></LineString></Placemark>")
    out.append("</Document></kml>")
    return "\n".join(out)


def to_csv(tracked: list[TrackedObject]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "class", "verdict", "conf", "evidence_score", "frame",
                "lat", "lon", "geo_error_m", "height_m", "height_rel_alt", "shadow", "bbox", "also_in"])
    for t in tracked:
        w.writerow([t.oid, t.cls_name, t.verdict.value, f"{t.conf:.3f}", t.evidence_score, t.frame_id,
                    t.lat if t.lat is not None else "", t.lon if t.lon is not None else "",
                    t.geo_error_m if t.geo_error_m is not None else "",
                    t.height_m if t.height_m is not None else "", t.height_rel, t.shadow_quality,
                    " ".join(map(str, t.bbox)), " ".join(t.also_in)])
    return buf.getvalue()


def to_json(survey: SurveyResult) -> str:
    return json.dumps(survey.to_dict(), indent=2)


EXPORTERS = {"geojson": to_geojson, "gpx": to_gpx, "kml": to_kml}     # need tracked + mission


def export(fmt: str, survey: SurveyResult) -> str:
    """Render a survey's mission in ``fmt`` ∈ {geojson, gpx, kml, csv, json}."""
    fmt = fmt.lower()
    if fmt in EXPORTERS:
        return EXPORTERS[fmt](survey.tracked, survey.mission)
    if fmt == "csv":
        return to_csv(survey.tracked)
    if fmt == "json":
        return to_json(survey)
    raise ValueError(f"unknown format: {fmt}")
