"""
mcp_server.py — DEPTH as a Model Context Protocol server: any MCP client (Claude Desktop / Code, IDEs,
agent frameworks, a partner's own agent) can run the sonar agent and read its results.

Two transports, one tool set:

* **stdio** (local):   ``python -m src.dashboard.mcp_server``  — for Claude Desktop / Claude Code.
* **Streamable HTTP** (anywhere): mounted on the DEPTH server at ``/mcp`` (stateless, JSON responses,
  so it works behind CloudFront / load balancers). Protected by ``DEPTH_MCP_TOKEN`` (bearer) when set;
  DNS-rebinding protection with an allow-list (``DEPTH_MCP_ALLOWED_HOSTS``, comma-separated).

What an agent can do: list samples, analyse a frame (the full Stage 1 → See → Prove → Decide loop),
run a survey, read the review queue / a hazard's decision trace / the re-survey plan / the Stage-1
counterfactual, export reports, read the mission brief, and **ask** for human approval.

What an agent cannot do: approve, dispatch, label, or change a threshold. Approval is a person's
action in the DEPTH studio (``src/agentic/approvals.py``); tiers come only from ``calibration.json``.
Tools import the app lazily, so this module can be imported by ``app.py`` without a cycle.
"""
from __future__ import annotations

import base64
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP, Image
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

INSTRUCTIONS = """DEPTH finds ghost fishing gear and wreck debris in side-scan sonar and turns it into a
human-approved cleanup plan. Typical flow: depth_status -> run_survey (or analyze_frame) ->
get_review_queue -> get_hazard (evidence + decision trace) -> get_mission_brief -> request_human_approval.
Tiers carry calibrated promises (see depth_status). DEPTH never dispatches anything: approval is made by
a person in the DEPTH studio. Coordinates from a synthetic demo track are labelled SYNTHETIC - never
present them as real positions."""

_READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
_RUN = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
MAX_MCP_FRAMES = int(os.environ.get("DEPTH_MCP_MAX_FRAMES", "24"))


def _security() -> TransportSecuritySettings:
    extra = [h.strip() for h in os.environ.get("DEPTH_MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    if "*" in extra:                               # explicitly opened (e.g. behind CloudFront + a token)
        if os.environ.get("DEPTH_MCP_TOKEN"):
            return TransportSecuritySettings(enable_dns_rebinding_protection=False)
        import logging
        logging.getLogger(__name__).warning("DEPTH_MCP_ALLOWED_HOSTS=* ignored: set DEPTH_MCP_TOKEN first")
        extra = [h for h in extra if h != "*"]
    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*", "127.0.0.1", "localhost"] + extra
    origins = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"] + [f"https://{h}" for h in extra]
    return TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=origins)


mcp = FastMCP("DEPTH", instructions=INSTRUCTIONS, website_url="https://github.com/madhesh60/depth",
              stateless_http=True, json_response=True, transport_security=_security())


def _app():
    from . import app as A                          # lazy: app.py imports this module to mount /mcp
    return A


def _survey(survey_id: str):
    A = _app()
    s = A._SURVEYS.get(survey_id)
    if s is None:
        raise ValueError(f"unknown survey_id '{survey_id}' - run_survey first (results are kept for the "
                         f"last {A.MAX_SURVEYS} surveys)")
    return s


def _cand_summary(c, i: int) -> dict:
    ev = c.evidence
    return {"index": i, "class": c.cls_name, "verdict": c.verdict.value if c.verdict else None,
            "detector_conf": round(c.conf, 3), "score": ev.evidence_score if ev else None,
            "p_pot": ev.p_pot if ev else None, "bbox_px": list(c.bbox),
            "shadow": ev.shadow.quality.value if ev else None,
            "relative_height": ev.shadow.height_rel if ev and ev.shadow.has_shadow else None,
            "in_water_column": c.in_water_column, "notes": list(ev.notes) if ev else [],
            "why": c.trace[-1].rationale if c.trace else None}


# ---- tools -------------------------------------------------------------------------------------
@mcp.tool(annotations=_READ)
def depth_status() -> dict:
    """Model, calibrated promises, OpenCV build (COOL or stock) and limits of this DEPTH server."""
    A = _app()
    h = A.health()
    return {"model": h["model"], "model_loaded": h["model_loaded"], "opencv": h["opencv"],
            "cool": h["is_cool_path"], "calibration": h["calibration"], "limits": h["limits"],
            "approvals": A._APPROVALS.counts(),
            "policy": "tiers from calibration.json; nothing is dispatched without a person's approval"}


@mcp.tool(annotations=_READ)
def list_samples() -> dict:
    """The sonar frames shipped with DEPTH (CC-BY-SA crab-pot sonograms) - usable without uploads."""
    return {"samples": _app().samples_mod.list_samples()}


@mcp.tool(annotations=_RUN)
def analyze_frame(sample_id: Optional[str] = None, image_base64: Optional[str] = None,
                  frame_id: str = "upload", nadir: Optional[str] = None) -> dict:
    """Run the full agent on ONE sonar frame: Stage 1 (seabed track -> altitude, ground range), See
    (YOLO11 via OpenCV 5 cv2.dnn), Prove (shadow, relative height, water column), Decide (calibrated
    tier + P(pot)). Give a sample_id from list_samples, or a base64 PNG/JPEG. Returns every candidate
    with its tier, evidence and the agent's reason."""
    import cv2
    import numpy as np
    A = _app()
    if sample_id:
        p = A.samples_mod.sample_path(sample_id)
        if not p or not p.exists():
            raise ValueError(f"unknown sample_id '{sample_id}' - see list_samples")
        frame, fid = cv2.imread(str(p)), p.stem
    elif image_base64:
        raw = base64.b64decode(image_base64.split(",")[-1], validate=False)
        if len(raw) > A.MAX_UPLOAD_MB * 1024 * 1024:
            raise ValueError(f"image larger than {A.MAX_UPLOAD_MB} MB")
        frame, fid = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), frame_id[:80]
    else:
        raise ValueError("give sample_id or image_base64")
    if frame is None:
        raise ValueError("could not decode the image")
    pipe = A.get_pipeline()
    with A._INFER_LOCK:
        r = pipe.run_frame(frame, frame_id=fid, nadir=nadir)
    A._METRICS.record("mcp_analyze", fid, r.stage_ms, None, r.counts)
    return {"frame_id": fid, "size": [r.width, r.height], "counts": r.counts,
            "stage1": {k: r.stage1.get(k) for k in ("altitude_px", "measured", "track_conf", "palette")},
            "orientation": r.orientation, "candidates": [_cand_summary(c, i) for i, c in enumerate(r.candidates)],
            "stage_ms": {k: round(v, 1) for k, v in r.stage_ms.items()}, "guarantees": pipe.guarantees(),
            "provenance": A.provenance_stamp()}


@mcp.tool(annotations=_READ)
def frame_overlay(sample_id: str) -> Image:
    """The analysed sample frame as an image: seabed track, boxes coloured by tier."""
    import cv2
    A = _app()
    p = A.samples_mod.sample_path(sample_id)
    if not p or not p.exists():
        raise ValueError(f"unknown sample_id '{sample_id}'")
    frame = cv2.imread(str(p))
    pipe = A.get_pipeline()
    with A._INFER_LOCK:
        r = pipe.run_frame(frame, frame_id=p.stem)
    ok, buf = cv2.imencode(".jpg", A.render(frame, r), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Image(data=buf.tobytes(), format="jpeg")


@mcp.tool(annotations=_RUN)
def run_survey(use_samples: bool = True, images_base64: Optional[list[str]] = None,
               frame_ids: Optional[list[str]] = None, budget_minutes: float = 5.0,
               boat_minutes: Optional[float] = None, gps: str = "synthetic") -> dict:
    """Run a whole survey: every frame through the agent, then chunk stitching, repeat-sighting merge,
    the review queue ordered by calibrated P(pot), the analyst budget, inspection/recovery routes,
    opposite-side re-survey passes and the Stage-1 counterfactual. gps='synthetic' lays a labelled demo
    track (the public frames carry no GPS); gps='none' gives a table-only plan. Returns a summary and a
    survey_id for the other tools."""
    import time
    import cv2
    import numpy as np
    A = _app()
    if images_base64:
        if len(images_base64) > MAX_MCP_FRAMES:
            raise ValueError(f"at most {MAX_MCP_FRAMES} frames per MCP survey (use the jobs API for more)")
        ids = list(frame_ids or []) + [f"upload_{i:04d}" for i in range(len(images_base64))]
        frames = []
        for i, b in enumerate(images_base64):
            img = cv2.imdecode(np.frombuffer(base64.b64decode(b.split(",")[-1]), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"image {i} could not be decoded")
            frames.append((str(ids[i])[:80], img))
    elif use_samples:
        frames = A._sample_frames()
    else:
        raise ValueError("use_samples=true or give images_base64")
    if gps not in ("synthetic", "none"):
        raise ValueError("gps must be 'synthetic' or 'none'")
    sid = f"survey-{time.strftime('%Y%m%d-%H%M%S')}-{len(frames)}f-mcp"
    d = A._run_survey(frames, gps, budget_minutes, None, sid, None, boat_minutes)
    m = d["mission"]
    rp = m.get("resurvey_plan") or {}
    cf = d.get("stage1_counterfactual") or {}
    return {"survey_id": d["survey_id"], "frames": len(frames), "counts": m["counts"],
            "gps": "none" if not m["gps_available"] else ("SYNTHETIC demo track" if m["gps_synthetic"] else "real"),
            "guarantees": {k: m["guarantees"].get(k) for k in ("recall_promise", "precision_promise", "verified_on_test")},
            "budget": m.get("budget"), "review_queue_top": m["review_queue"][:10],
            "inspection_route": {"stops": m["inspection_route"], "length_m": m["inspection_length_m"]},
            "recovery_route": {"stops": m["recovery_route"], "length_m": m["route_length_m"]},
            "resurvey": {"passes": len(rp.get("lines", [])), "boat_minutes": rp.get("boat_minutes_planned"),
                         "targets": f"{rp.get('targets_covered')}/{rp.get('targets_total')}"},
            "stage1_counterfactual": {k: cf.get(k) for k in ("n", "outside", "median_shift_m")} if cf.get("available") else None,
            "human_approval_required": True, "provenance": d.get("provenance")}


@mcp.tool(annotations=_READ)
def get_review_queue(survey_id: str, limit: int = 10) -> dict:
    """The REVIEW cards of a survey, most-likely-real first (calibrated P(pot)), with their evidence."""
    s = _survey(survey_id)
    idx = {t.oid: t for t in s.tracked}
    out = []
    for rank, oid in enumerate(s.mission.review_queue[:max(1, min(limit, 200))], 1):
        t = idx[oid]
        out.append({"rank": rank, "id": oid, "class": t.cls_name, "p_pot": t.p_pot, "conf": t.conf,
                    "shadow": t.shadow_quality, "sightings": t.sightings, "frame": t.frame_id,
                    "lat": t.lat, "lon": t.lon, "error_m": t.geo_error_m,
                    "in_budget": oid in (s.mission.budget or {}).get("review_ids", [])})
    return {"survey_id": survey_id, "queue_length": len(s.mission.review_queue), "cards": out,
            "gps_synthetic": s.mission.gps_synthetic}


@mcp.tool(annotations=_READ)
def get_hazard(survey_id: str, hazard_id: str) -> dict:
    """One hazard in full: evidence, position with error radius, and the agent's decision trace (every
    tool call with its rationale) - the audit trail behind its tier."""
    s = _survey(survey_id)
    t = next((x for x in s.tracked if x.oid == hazard_id), None)
    if t is None:
        raise ValueError(f"unknown hazard_id '{hazard_id}'")
    fr = next(f for f in s.frames if f.frame_id == t.frame_id)
    c = next((c for c in fr.candidates if tuple(c.bbox) == tuple(t.bbox)), None)
    passes = [L["id"] for L in (s.mission.resurvey_plan or {}).get("lines", []) if hazard_id in L["targets"]]
    return {**t.to_dict(), "resurvey_passes": passes,
            "evidence_notes": list(c.evidence.notes) if c and c.evidence else [],
            "trace": [{"tool": st.tool, "why": st.rationale, "ms": st.latency_ms} for st in (c.trace if c else [])],
            "gps_synthetic": s.mission.gps_synthetic}


@mcp.tool(annotations=_READ)
def get_resurvey_plan(survey_id: str) -> dict:
    """Opposite-side second-look passes: each re-images uncertain targets from the far side at
    mid-swath; a real object's shadow must flip to the predicted bearing. Status is always PLANNED."""
    rp = _survey(survey_id).mission.resurvey_plan or {}
    return {**{k: v for k, v in rp.items() if k != "lines"},
            "passes": [{k: L[k] for k in ("id", "targets", "targets_on", "heading_deg", "length_m", "boat_min", "voi",
                                          "predictions", "status", "why")} for L in rp.get("lines", [])]}


@mcp.tool(annotations=_READ)
def get_stage1_counterfactual(survey_id: str) -> dict:
    """Where each pin would land if the agent ignored Stage 1's bottom track (STUDY-12, live)."""
    A = _app()
    s = _survey(survey_id)
    from src.agentic.geo import stage1_counterfactual, synthetic_track
    track = synthetic_track([f.frame_id for f in s.frames]) if s.mission.gps_synthetic else None
    return stage1_counterfactual(s.frames, s.tracked, track, A.get_pipeline().m_per_px)


@mcp.tool(annotations=_READ)
def get_mission_brief(survey_id: str, public: bool = False) -> dict:
    """The one-page hand-over for the crew. Every number in it is checked against the survey."""
    out = _app().mission_brief(survey_id=survey_id, public=public, writer="template")
    return {k: out[k] for k in ("text", "writer", "grounding")}


@mcp.tool(annotations=_READ)
def export_report(survey_id: str, fmt: str = "geojson", public: bool = False) -> str:
    """Export a survey: geojson | gpx | kml | csv | json | brief | trace. public=true generalises
    protected-site (wreck) positions; the decision log (trace) is never public."""
    from src.agentic.mission import export
    if public and fmt == "trace":
        raise ValueError("the decision log carries exact positions - it is not shared publicly")
    return export(fmt, _survey(survey_id), public=public)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
def request_human_approval(survey_id: str, action: str, targets: list[str], rationale: str,
                           requested_by: str = "mcp-agent") -> dict:
    """ASK a person to approve an action: inspect | recover | resurvey | share_public | other, on the
    given hazard / pass IDs, with your rationale. This does NOT approve or dispatch anything - the
    request waits for a named person in the DEPTH studio. Poll get_approval_status."""
    A = _app()
    s = _survey(survey_id)
    known = {t.oid for t in s.tracked} | {L["id"] for L in (s.mission.resurvey_plan or {}).get("lines", [])}
    bad = [t for t in targets if t not in known]
    if bad:
        raise ValueError(f"unknown target IDs for this survey: {bad}")
    r = A._APPROVALS.request(survey_id, action, targets, rationale, requested_by=requested_by, channel="mcp")
    A._notify("approval.requested", r)
    return r


@mcp.tool(annotations=_READ)
def get_approval_status(request_id: Optional[str] = None, status: Optional[str] = None) -> dict:
    """One approval request by id, or the list (optionally filtered: pending | approved | declined)."""
    A = _app()
    if request_id:
        r = A._APPROVALS.get(request_id)
        if r is None:
            raise ValueError(f"unknown request_id '{request_id}'")
        return {"requests": [r]}
    return {"requests": A._APPROVALS.list(status=status, limit=50), "counts": A._APPROVALS.counts()}


# ---- resources + a prompt ------------------------------------------------------------------------
@mcp.resource("depth://guarantees", mime_type="application/json")
def guarantees_resource() -> dict:
    """The calibrated promises the tiers carry, and how they were verified."""
    return _app().get_pipeline().guarantees()


@mcp.resource("depth://survey/{survey_id}/brief", mime_type="text/markdown")
def brief_resource(survey_id: str) -> str:
    """The mission brief of a survey (markdown)."""
    return get_mission_brief(survey_id)["text"]


@mcp.prompt()
def triage_survey(survey_id: str) -> str:
    """Walk a survey's results with DEPTH's tools and prepare approval requests for a person."""
    return (f"Triage DEPTH survey {survey_id}. 1) get_mission_brief. 2) get_review_queue (limit 10). "
            f"3) For the top cards, get_hazard and read the evidence and the decision trace. "
            f"4) get_resurvey_plan. 5) If an inspection or a re-survey pass is justified, call "
            f"request_human_approval with the IDs and a short rationale that cites the evidence. "
            f"Never state that anything was dispatched; say which positions are SYNTHETIC.")


# ---- HTTP mounting (used by app.py) ---------------------------------------------------------------
class MCPEndpoint:
    """ASGI endpoint for ``/mcp``: bearer-token check, then the current session manager. A fresh
    stateless manager is created per app lifespan (a manager can only ``run()`` once)."""

    def __init__(self):
        self.manager = None

    def new_manager(self):
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        self.manager = StreamableHTTPSessionManager(app=mcp._mcp_server, json_response=True, stateless=True,
                                                    security_settings=mcp.settings.transport_security)
        return self.manager

    async def __call__(self, scope, receive, send):
        token = os.environ.get("DEPTH_MCP_TOKEN")
        if token:
            auth = dict(scope.get("headers") or []).get(b"authorization", b"").decode()
            if auth != f"Bearer {token}":
                await _plain(send, 401, b'{"error":"missing or wrong bearer token (DEPTH_MCP_TOKEN)"}',
                             [(b"www-authenticate", b'Bearer realm="depth-mcp"')])
                return
        if self.manager is None:
            await _plain(send, 503, b'{"error":"MCP transport not running (app lifespan not started)"}')
            return
        await self.manager.handle_request(scope, receive, send)


async def _plain(send, status: int, body: bytes, headers=()):
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), *headers]})
    await send({"type": "http.response.body", "body": body})


ENDPOINT = MCPEndpoint()


if __name__ == "__main__":                         # stdio: Claude Desktop / Claude Code / IDEs
    os.environ.setdefault("DEPTH_LAZY_MODEL", "1")
    # Import the app BEFORE the stdio loop starts: importing it inside the first tool call, while the
    # transport's reader thread is blocked on stdin, deadlocks on Windows (found by tests/test_mcp.py).
    from src.dashboard import app as _preloaded  # noqa: F401
    mcp.run()
