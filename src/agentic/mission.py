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

import copy
import csv
import io
import json
from typing import Optional

from .geo import haversine_m
from .types import MissionPlan, TrackedObject, Verdict, SurveyResult

_SYNTH_WARN = "SYNTHETIC DEMO GPS - not real coordinates"
# Responsible use (docs/responsible_use.md): wrecks can be war graves / protected heritage sites.
# In a PUBLIC share their positions are generalised to ~1.1 km (0.01°) and re-survey passes that would
# reveal them are dropped; cleanup targets (ghost gear) keep full precision.
SENSITIVE_CLASSES = {"structural_fragment", "wreck_debris"}
_GENERAL_DEG = 0.01
_REDACT_NOTE = "public share: protected-site locations generalised to ~1.1 km (0.01 deg); full precision on request"
# Seconds an analyst spends per REVIEW card. ASSUMED until the timed user study replaces it
# (docs/user_study.md); every budget plan says so.
DEFAULT_SEC_PER_CARD = 8.0


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


def review_queue(tracked: list[TrackedObject]) -> list[TrackedObject]:
    """REVIEW cards, most-likely-real first: calibrated P(pot), then evidence score, then confidence."""
    rev = [t for t in tracked if t.verdict is Verdict.REVIEW]
    return sorted(rev, key=lambda t: (-(t.p_pot if t.p_pot is not None else -1.0),
                                      -t.evidence_score, -t.conf, t.oid))


def plan_inspection(mission: MissionPlan, tracked: list[TrackedObject], ids: list[str]) -> None:
    """Nearest-neighbour INSPECTION route through the REVIEW cards an analyst/boat should check first
    (the budgeted, most-likely-real ones). Distinct from the recovery route: nothing on it is a
    confirmed hazard — it is where a human (or a re-survey) must look. Needs GPS; mutates ``mission``."""
    if not mission.gps_available:
        mission.inspection_route, mission.inspection_length_m = list(ids), None
        return
    idx = _by_id(tracked)
    ordered = _nearest_neighbour([idx[i] for i in ids if i in idx])
    mission.inspection_route = [t.oid for t in ordered]
    mission.inspection_length_m = round(sum(haversine_m((a.lat, a.lon), (b.lat, b.lon))
                                            for a, b in zip(ordered, ordered[1:])), 1) if len(ordered) > 1 else 0.0


def plan_budget(queue: list[TrackedObject], minutes: float,
                sec_per_card: float = DEFAULT_SEC_PER_CARD, measured: bool = False) -> dict:
    """Budget mode: given the analyst's minutes, which cards to review first and what that buys.

    Expected real pots = Σ calibrated P(pot) over the cards reviewed. Returned figures are
    expectations under the calibration, not guarantees."""
    k = max(0, int(minutes * 60 // max(1e-6, sec_per_card)))
    ps = [t.p_pot for t in queue if t.p_pot is not None]
    in_budget = [t.p_pot for t in queue[:k] if t.p_pot is not None]
    total = float(sum(ps))
    got = float(sum(in_budget))
    return {
        "minutes": minutes, "sec_per_card": sec_per_card,
        "sec_per_card_source": "measured (user study)" if measured else "ASSUMED - replace with the timed user study",
        "cards_total": len(queue), "cards_affordable": min(k, len(queue)),
        "review_ids": [t.oid for t in queue[:k]],
        "expected_pots_in_budget": round(got, 2), "expected_pots_in_queue": round(total, 2),
        "share_of_expected_pots": round(got / total, 3) if total > 0 else None,
        "minutes_for_whole_queue": round(len(queue) * sec_per_card / 60, 1),
    }


# ---- exporters --------------------------------------------------------------------------------
def _by_id(tracked: list[TrackedObject]) -> dict[str, TrackedObject]:
    return {t.oid: t for t in tracked}


def redact_public(tracked: list[TrackedObject], mission: MissionPlan) -> tuple[list[TrackedObject], MissionPlan, int]:
    """Public-share copy: sensitive-class hazards get generalised coordinates; re-survey passes that
    target them are dropped (a pass line would point straight at the wreck)."""
    out, sens = [], set()
    for t in tracked:
        t2 = copy.deepcopy(t)
        if t2.cls_name in SENSITIVE_CLASSES and t2.lat is not None:
            t2.lat, t2.lon = round(round(t2.lat / _GENERAL_DEG) * _GENERAL_DEG, 4), round(round(t2.lon / _GENERAL_DEG) * _GENERAL_DEG, 4)
            t2.geo_error_m = max(t2.geo_error_m or 0.0, 1100.0)
            sens.add(t2.oid)
        out.append(t2)
    m2 = copy.deepcopy(mission)
    rp = dict(m2.resurvey_plan or {})
    if rp.get("lines"):
        rp["lines"] = [L for L in rp["lines"] if not (set(L["targets"]) & sens)]
        m2.resurvey_plan = rp
    return out, m2, len(sens)


def to_geojson(tracked: list[TrackedObject], mission: MissionPlan, prov: Optional[dict] = None,
               note: Optional[str] = None) -> str:
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
                           "shadow": t.shadow_quality, "also_in": t.also_in, "p_pot": t.p_pot,
                           "geo_error_m": t.geo_error_m, "frame": t.frame_id},
        })
    idx = _by_id(tracked)
    coords = [[idx[i].lon, idx[i].lat] for i in mission.recovery_route if idx.get(i) and idx[i].lat is not None]
    if len(coords) > 1:
        features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                         "properties": {"name": "recovery_route", "length_m": mission.route_length_m}})
    icoords = [[idx[i].lon, idx[i].lat] for i in mission.inspection_route if idx.get(i) and idx[i].lat is not None]
    if len(icoords) > 1:
        features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": icoords},
                         "properties": {"name": "inspection_route", "length_m": mission.inspection_length_m,
                                        "note": "REVIEW cards to inspect first - pending human approval"}})
    for L in (mission.resurvey_plan or {}).get("lines", []):
        features.append({"type": "Feature",
                         "geometry": {"type": "LineString", "coordinates": [L["start"][::-1], L["end"][::-1]]},
                         "properties": {"name": L["id"], "kind": "resurvey_plan", "status": L["status"],
                                        "heading_deg": L["heading_deg"], "targets": L["targets"],
                                        "targets_on": L["targets_on"], "voi": L["voi"], "boat_min": L["boat_min"],
                                        "predictions": L["predictions"], "why": L["why"]}})
    fc = {"type": "FeatureCollection",
          "properties": {"human_approval_required": True,
                         "note": _SYNTH_WARN if mission.gps_synthetic else "real GPS",
                         "guarantees": mission.guarantees, "provenance": prov, "redaction": note},
          "features": features}
    return json.dumps(fc, indent=2)


def to_gpx(tracked: list[TrackedObject], mission: MissionPlan, prov: Optional[dict] = None,
           note: Optional[str] = None) -> str:
    from .provenance import short
    idx = _by_id(tracked)
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<gpx version="1.1" creator="DEPTH" xmlns="http://www.topografix.com/GPX/1/1">']
    desc = " | ".join(x for x in ((_SYNTH_WARN if mission.gps_synthetic else ""), (short(prov) if prov else ""), note or "") if x)
    if desc:
        out.append(f"  <metadata><desc>{desc}</desc></metadata>")
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
    for L in (mission.resurvey_plan or {}).get("lines", []):
        out.append(f'  <rte><name>{L["id"]} re-survey (PLANNED, {L["boat_min"]} min)</name>'
                   f'<desc>{L["why"]}</desc>')
        out.append(f'    <rtept lat="{L["start"][0]}" lon="{L["start"][1]}"><name>{L["id"]}-start</name></rtept>')
        out.append(f'    <rtept lat="{L["end"][0]}" lon="{L["end"][1]}"><name>{L["id"]}-end</name></rtept>')
        out.append("  </rte>")
    out.append("</gpx>")
    return "\n".join(out)


def to_kml(tracked: list[TrackedObject], mission: MissionPlan, prov: Optional[dict] = None,
           note: Optional[str] = None) -> str:
    from .provenance import short
    idx = _by_id(tracked)
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           f"  <name>DEPTH hazards{' (' + _SYNTH_WARN + ')' if mission.gps_synthetic else ''}</name>",
           f"  <description>{' | '.join(x for x in ((short(prov) if prov else ''), note or '') if x)}</description>"]
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
    for L in (mission.resurvey_plan or {}).get("lines", []):
        out.append(f"  <Placemark><name>{L['id']} re-survey (planned)</name><description>{L['why']}</description>"
                   f"<LineString><coordinates>{L['start'][1]},{L['start'][0]},0 {L['end'][1]},{L['end'][0]},0"
                   f"</coordinates></LineString></Placemark>")
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


def to_json(survey: SurveyResult, prov: Optional[dict] = None, note: Optional[str] = None) -> str:
    d = survey.to_dict()
    d["provenance"], d["redaction"] = prov, note
    return json.dumps(d, indent=2)


def to_trace(survey: SurveyResult, prov: Optional[dict] = None) -> str:
    """The agent's flight recorder: one JSON line per event — provenance, every frame's Stage-1 record,
    every tool call of every candidate (inputs → rationale → outputs, timed), every verdict, and the
    mission decisions (budget, routes, re-survey plan, human gate). Replayable and diffable."""
    L = [{"type": "provenance", **(prov or {})},
         {"type": "survey", "survey_id": survey.survey_id, "frames": len(survey.frames),
          "gps_available": survey.mission.gps_available, "gps_synthetic": survey.mission.gps_synthetic,
          "guarantees": survey.mission.guarantees}]
    for fr in survey.frames:
        s1 = {k: v for k, v in (fr.stage1 or {}).items() if k != "bottom_line_native"}
        L.append({"type": "frame", "frame_id": fr.frame_id, "orientation": fr.orientation, "stage1": s1,
                  "stage_ms": {k: round(v, 2) for k, v in fr.stage_ms.items()}, "counts": fr.counts})
        for ci, c in enumerate(fr.candidates):
            for si, st in enumerate(c.trace):
                L.append({"type": "step", "frame_id": fr.frame_id, "cand": ci, "step": si, "cls": c.cls_name,
                          "bbox": list(c.bbox), **st.to_dict()})
            L.append({"type": "verdict", "frame_id": fr.frame_id, "cand": ci, "cls": c.cls_name, "conf": round(c.conf, 4),
                      "verdict": c.verdict.value if c.verdict else None,
                      "score": c.evidence.evidence_score if c.evidence else None,
                      "p_pot": c.evidence.p_pot if c.evidence else None, "lat": c.lat, "lon": c.lon})
    m = survey.mission
    L.append({"type": "mission", "counts": m.counts, "recovery_route": m.recovery_route, "review_queue": m.review_queue,
              "inspection_route": m.inspection_route, "budget": m.budget, "repeat_merges": m.repeat_merges,
              "resurvey": {k: v for k, v in (m.resurvey_plan or {}).items() if k != "lines"},
              "resurvey_lines": [{k: L_[k] for k in ("id", "targets", "voi", "boat_min", "status")}
                                 for L_ in (m.resurvey_plan or {}).get("lines", [])],
              "human_approval_required": m.human_approval_required})
    return "\n".join(json.dumps(x, default=str) for x in L) + "\n"


EXPORTERS = {"geojson": to_geojson, "gpx": to_gpx, "kml": to_kml}     # need tracked + mission


def export(fmt: str, survey: SurveyResult, public: bool = False, prov: Optional[dict] = None) -> str:
    """Render a survey's mission in ``fmt`` ∈ {geojson, gpx, kml, csv, json, trace}. ``public=True``
    generalises protected-site locations (``SENSITIVE_CLASSES``). Every format except CSV carries the
    provenance stamp (CSV stays a clean table)."""
    fmt = fmt.lower()
    if prov is None:
        from .provenance import stamp
        prov = stamp()
    tracked, mission, note = survey.tracked, survey.mission, None
    if public:
        tracked, mission, n = redact_public(tracked, mission)
        note = f"{_REDACT_NOTE} ({n} object(s) generalised)"
    if fmt in EXPORTERS:
        return EXPORTERS[fmt](tracked, mission, prov, note)
    if fmt == "csv":
        return to_csv(tracked)
    if fmt == "json":
        return to_json(SurveyResult(survey.survey_id, survey.frames, tracked, mission) if public else survey, prov, note)
    if fmt == "trace":
        if public:
            raise ValueError("the decision log carries exact positions - it is not shared publicly")
        return to_trace(survey, prov)
    raise ValueError(f"unknown format: {fmt}")
