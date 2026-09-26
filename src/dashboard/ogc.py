"""
ogc.py — DEPTH as an **OGC API – Features** service (OGC 17-069r4, Part 1: Core + GeoJSON).

GIS and operations consoles connect with no DEPTH-specific code: QGIS ("Add WFS / OGC API – Features
layer"), ArcGIS Pro ("OGC API server"), OpenLayers / Leaflet, GeoServer clients, and any HTTP client
that reads GeoJSON. Collections span every remembered survey (in memory and persisted), newest first:

* ``hazards``           — one Point per hazard: tier, calibrated P(pot), review rank, evidence summary,
  error radius, and the **approval status** of the latest request that names it;
* ``resurvey_passes``   — opposite-side second-look passes (LineString, PLANNED) with shadow predictions;
* ``routes``            — recovery + inspection routes (LineString, pending human approval);
* ``work_orders``       — requests a **person approved** (MultiPoint of the targets): what a console
  should act on. Nothing else in DEPTH is an instruction.

Query parameters (Part 1): ``limit`` (≤ 1000), ``offset``, ``bbox=minLon,minLat,maxLon,maxLat``; plus
``survey_id`` (an id or ``latest``) and ``public=true`` (protected-site / wreck positions generalised
to ~1.1 km — forced for every request when ``DEPTH_OGC_PUBLIC=1``). Synthetic demo positions carry
``gps_synthetic: true`` on every feature.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from src.agentic.mission import SENSITIVE_CLASSES

CONF = ["http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core",
        "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/geojson",
        "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/oas30"]
GEOJSON = "application/geo+json"
_GENERAL_DEG = 0.01
COLLECTIONS = {
    "hazards": ("Hazards", "Possible ghost gear / wreck debris: tier, calibrated P(pot), review rank, approval status."),
    "resurvey_passes": ("Re-survey passes", "Opposite-side second-look passes (PLANNED; need human approval)."),
    "routes": ("Routes", "Recovery and inspection routes (pending human approval)."),
    "work_orders": ("Approved work orders", "Requests a named person approved in the DEPTH studio."),
}


def _public_forced() -> bool:
    return os.environ.get("DEPTH_OGC_PUBLIC", "0") == "1"


def _gen(v: float) -> float:
    return round(round(v / _GENERAL_DEG) * _GENERAL_DEG, 4)


def _feature(fid: str, geom: dict, props: dict) -> dict:
    return {"type": "Feature", "id": fid, "geometry": geom, "properties": props}


def _approvals_by_target(approvals) -> dict:
    out: dict[tuple, dict] = {}
    for r in sorted(approvals.list(limit=10_000), key=lambda r: r["t"]):
        for t in r["targets"]:
            out[(r["survey_id"], t)] = r                      # latest request naming this target wins
    return out


def build_features(cid: str, surveys: list[dict], approvals, public: bool) -> list[dict]:
    """All features of one collection. ``surveys``: newest-first dicts with tracked / mission."""
    feats: list[dict] = []
    appr = _approvals_by_target(approvals) if cid in ("hazards", "work_orders") else {}
    for s in surveys:
        sid, m = s["survey_id"], s.get("mission") or {}
        synth = bool(m.get("gps_synthetic"))
        tracked = s.get("tracked") or []
        sens = {t["oid"] for t in tracked if t.get("cls_name") in SENSITIVE_CLASSES}
        pos = {}
        for t in tracked:
            if t.get("lat") is None:
                continue
            lat, lon = t["lat"], t["lon"]
            if public and t["oid"] in sens:
                lat, lon = _gen(lat), _gen(lon)
            pos[t["oid"]] = (lon, lat)
        if cid == "hazards":
            rank = {oid: i + 1 for i, oid in enumerate(m.get("review_queue") or [])}
            budget = set((m.get("budget") or {}).get("review_ids") or [])
            for t in tracked:
                if t["oid"] not in pos:
                    continue
                a = appr.get((sid, t["oid"]))
                feats.append(_feature(f"{sid}:{t['oid']}", {"type": "Point", "coordinates": list(pos[t["oid"]])}, {
                    "survey_id": sid, "hazard_id": t["oid"], "class": t.get("cls_name"), "tier": t.get("verdict"),
                    "p_pot": t.get("p_pot"), "confidence": t.get("conf"), "evidence_score": t.get("evidence_score"),
                    "shadow": t.get("shadow_quality"), "relative_height": t.get("height_rel"),
                    "sightings": t.get("sightings", 1), "review_rank": rank.get(t["oid"]),
                    "in_analyst_budget": t["oid"] in budget,
                    "error_m": 1100.0 if (public and t["oid"] in sens) else t.get("geo_error_m"),
                    "protected_site_generalised": bool(public and t["oid"] in sens),
                    "approval": {"id": a["id"], "action": a["action"], "status": a["status"]} if a else None,
                    "frame": t.get("frame_id"), "gps_synthetic": synth,
                    "human_approval_required": True}))
        elif cid == "resurvey_passes":
            for L in (m.get("resurvey_plan") or {}).get("lines") or []:
                if public and set(L.get("targets", [])) & sens:
                    continue                                   # a pass line would point at the wreck
                feats.append(_feature(f"{sid}:{L['id']}", {"type": "LineString",
                                                           "coordinates": [L["start"][::-1], L["end"][::-1]]}, {
                    "survey_id": sid, "pass_id": L["id"], "status": L.get("status"), "heading_deg": L.get("heading_deg"),
                    "length_m": L.get("length_m"), "boat_min": L.get("boat_min"), "targets": L.get("targets"),
                    "targets_on": L.get("targets_on"), "uncertainty_sum": L.get("voi"),
                    "shadow_predictions": L.get("predictions"), "why": L.get("why"), "gps_synthetic": synth}))
        elif cid == "routes":
            for kind, ids, length in (("recovery", m.get("recovery_route") or [], m.get("route_length_m")),
                                      ("inspection", m.get("inspection_route") or [], m.get("inspection_length_m"))):
                coords = [list(pos[i]) for i in ids if i in pos]
                if len(coords) > 1:
                    feats.append(_feature(f"{sid}:{kind}", {"type": "LineString", "coordinates": coords}, {
                        "survey_id": sid, "route": kind, "stops": ids, "length_m": length,
                        "status": "pending human approval", "gps_synthetic": synth}))
        elif cid == "work_orders":
            seen = set()
            for r in approvals.list(status="approved", survey_id=sid, limit=10_000):
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                pts = [list(pos[t]) for t in r["targets"] if t in pos]
                feats.append(_feature(r["id"], {"type": "MultiPoint", "coordinates": pts} if pts else None, {
                    "survey_id": sid, "request_id": r["id"], "action": r["action"], "targets": r["targets"],
                    "rationale": r["rationale"], "requested_by": r["requested_by"], "channel": r["channel"],
                    "approved_by": (r.get("decision") or {}).get("by"),
                    "approved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime((r.get("decision") or {}).get("t", r["t"]))),
                    "gps_synthetic": synth}))
    return feats


def _in_bbox(f: dict, bb: Optional[list[float]]) -> bool:
    if not bb or not f.get("geometry"):
        return bb is None
    g = f["geometry"]
    pts = [g["coordinates"]] if g["type"] == "Point" else g["coordinates"]
    return any(bb[0] <= p[0] <= bb[2] and bb[1] <= p[1] <= bb[3] for p in pts)


def _extent(feats: list[dict]) -> Optional[list[float]]:
    pts = []
    for f in feats:
        g = f.get("geometry")
        if g:
            pts += [g["coordinates"]] if g["type"] == "Point" else g["coordinates"]
    if not pts:
        return None
    return [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]


def make_router(surveys: Callable[[], list[dict]], approvals: Callable[[], object]) -> APIRouter:
    r = APIRouter(prefix="/ogc", tags=["OGC API - Features"])

    def base(req: Request) -> str:
        return str(req.base_url).rstrip("/") + "/ogc"

    def pick(survey_id: Optional[str]) -> list[dict]:
        ss = surveys()
        if survey_id == "latest":
            return ss[:1]
        if survey_id:
            ss = [s for s in ss if s["survey_id"] == survey_id]
            if not ss:
                raise HTTPException(404, f"unknown survey_id '{survey_id}'")
        return ss

    def coll_meta(req: Request, cid: str, feats: Optional[list[dict]] = None) -> dict:
        title, desc = COLLECTIONS[cid]
        b = base(req)
        out = {"id": cid, "title": title, "description": desc, "itemType": "feature",
               "crs": ["http://www.opengis.net/def/crs/OGC/1.3/CRS84"],
               "links": [{"href": f"{b}/collections/{cid}", "rel": "self", "type": "application/json"},
                         {"href": f"{b}/collections/{cid}/items", "rel": "items", "type": GEOJSON, "title": title}]}
        ext = _extent(feats or [])
        if ext:
            out["extent"] = {"spatial": {"bbox": [ext], "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}}
        return out

    @r.get("")
    @r.get("/")
    def landing(req: Request):
        b = base(req)
        return {"title": "DEPTH - marine-debris hazards (OGC API - Features)",
                "description": "Side-scan sonar hazards found by the DEPTH agent, with calibrated tiers, "
                               "re-survey passes, routes and human-approved work orders.",
                "links": [{"href": b, "rel": "self", "type": "application/json", "title": "this document"},
                          {"href": str(req.base_url) + "openapi.json", "rel": "service-desc",
                           "type": "application/vnd.oai.openapi+json;version=3.0", "title": "the API definition"},
                          {"href": str(req.base_url) + "docs", "rel": "service-doc", "type": "text/html"},
                          {"href": f"{b}/conformance", "rel": "conformance", "type": "application/json"},
                          {"href": f"{b}/collections", "rel": "data", "type": "application/json"}]}

    @r.get("/conformance")
    def conformance():
        return {"conformsTo": CONF}

    @r.get("/collections")
    def collections(req: Request):
        ss, ap = surveys(), approvals()
        pub = _public_forced()
        return {"links": [{"href": f"{base(req)}/collections", "rel": "self", "type": "application/json"}],
                "collections": [coll_meta(req, cid, build_features(cid, ss, ap, pub)) for cid in COLLECTIONS]}

    @r.get("/collections/{cid}")
    def collection(cid: str, req: Request):
        if cid not in COLLECTIONS:
            raise HTTPException(404, f"no collection '{cid}'")
        return coll_meta(req, cid, build_features(cid, surveys(), approvals(), _public_forced()))

    @r.get("/collections/{cid}/queryables")
    def queryables(cid: str, req: Request):
        """Part 3 queryables: the properties a console can filter on (JSON Schema)."""
        if cid not in COLLECTIONS:
            raise HTTPException(404, f"no collection '{cid}'")
        feats = build_features(cid, surveys()[:1], approvals(), _public_forced())
        jt = lambda v: ("boolean" if isinstance(v, bool) else "integer" if isinstance(v, int) else
                        "number" if isinstance(v, float) else "string" if isinstance(v, str) else
                        "array" if isinstance(v, list) else "object")
        props = {k: {"title": k, "type": jt(v)} for k, v in (feats[0]["properties"] if feats else {}).items() if v is not None}
        return JSONResponse({"$schema": "https://json-schema.org/draft/2020-12/schema",
                             "$id": f"{base(req)}/collections/{cid}/queryables", "type": "object",
                             "title": COLLECTIONS[cid][0], "properties": props}, media_type="application/schema+json")

    @r.get("/collections/{cid}/items")
    def items(cid: str, req: Request, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
              bbox: Optional[str] = Query(None), survey_id: Optional[str] = Query(None), public: bool = Query(False)):
        if cid not in COLLECTIONS:
            raise HTTPException(404, f"no collection '{cid}'")
        bb = None
        if bbox:
            try:
                bb = [float(x) for x in bbox.split(",")]
                assert len(bb) == 4 and bb[0] <= bb[2] and bb[1] <= bb[3]
            except (ValueError, AssertionError):
                raise HTTPException(400, "bbox must be minLon,minLat,maxLon,maxLat")
        feats = build_features(cid, pick(survey_id), approvals(), public or _public_forced())
        if bb:
            feats = [f for f in feats if _in_bbox(f, bb)]
        page = feats[offset:offset + limit]
        b = base(req)
        q = {k: v for k, v in req.query_params.items() if k not in ("offset",)}
        qs = lambda off: "&".join([f"{k}={v}" for k, v in q.items()] + [f"offset={off}"])
        links = [{"href": f"{b}/collections/{cid}/items?{qs(offset)}", "rel": "self", "type": GEOJSON},
                 {"href": f"{b}/collections/{cid}", "rel": "collection", "type": "application/json"}]
        if offset + limit < len(feats):
            links.append({"href": f"{b}/collections/{cid}/items?{qs(offset + limit)}", "rel": "next", "type": GEOJSON})
        return JSONResponse({"type": "FeatureCollection", "features": page, "numberMatched": len(feats),
                             "numberReturned": len(page), "timeStamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "links": links}, media_type=GEOJSON)

    @r.get("/collections/{cid}/items/{fid}")
    def item(cid: str, fid: str, req: Request, public: bool = Query(False)):
        if cid not in COLLECTIONS:
            raise HTTPException(404, f"no collection '{cid}'")
        sid = fid.split(":", 1)[0] if ":" in fid else None
        f = next((x for x in build_features(cid, pick(sid) if sid else surveys(), approvals(), public or _public_forced())
                  if x["id"] == fid), None)
        if f is None:
            raise HTTPException(404, f"no feature '{fid}'")
        f = {**f, "links": [{"href": f"{base(req)}/collections/{cid}/items/{fid}", "rel": "self", "type": GEOJSON},
                            {"href": f"{base(req)}/collections/{cid}", "rel": "collection", "type": "application/json"}]}
        return JSONResponse(f, media_type=GEOJSON)

    return r


class SurveyIndex:
    """Newest-first survey dicts for the OGC service: in-memory results first, then persisted job
    results (parsed once per file mtime)."""

    def __init__(self, memory: Callable[[], dict], persist_dir: Path, max_surveys: int = 50):
        self.memory, self.persist_dir, self.max = memory, Path(persist_dir), max_surveys
        self._cache: dict[str, tuple[float, dict]] = {}

    def __call__(self) -> list[dict]:
        out = {}
        for sid, s in list(self.memory().items()):
            out[sid] = {"survey_id": sid, "tracked": [t.to_dict() for t in s.tracked], "mission": s.mission.to_dict()}
        if self.persist_dir.exists():
            for d in sorted(self.persist_dir.iterdir(), reverse=True)[: self.max]:
                p = d / "result.json"
                if d.name in out or not p.exists():
                    continue
                mt = p.stat().st_mtime
                hit = self._cache.get(d.name)
                if not hit or hit[0] != mt:
                    try:
                        raw = json.loads(p.read_text(encoding="utf-8"))
                        hit = (mt, {"survey_id": raw.get("survey_id", d.name), "tracked": raw.get("tracked") or [],
                                    "mission": raw.get("mission") or {}})
                    except (OSError, json.JSONDecodeError):
                        continue
                    self._cache[d.name] = hit
                out[d.name] = hit[1]
        return [out[k] for k in sorted(out, reverse=True)][: self.max]
