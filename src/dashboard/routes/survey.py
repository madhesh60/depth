"""
routes/survey.py — surveys (sync and background jobs), the raw recording, exports, the brief, a person's
decisions on hazards and the survey's audit log.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import cv2
from fastapi import APIRouter, Body, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from ..jobs import PUBLIC_FORMATS, REPORT_FORMATS
from ..ratelimit import client_of
from src.agentic import brief as brief_mod
from src.agentic.mission import export
from src.agentic.provenance import stamp as provenance_stamp
from src.agentic.types import SurveyResult
from .. import service as S

router = APIRouter()


@router.post("/api/jobs/survey")
def submit_survey(use_samples: bool = Query(False), gps: str = Query("synthetic"),
                  budget_minutes: Optional[float] = Query(None, ge=0, le=600),
                  recall: Optional[float] = Query(None, ge=0.5, le=0.99, description="recall promise to keep (the goal)"),
                  compute_s: Optional[float] = Query(None, ge=1, le=3600, description="compute budget for perception, s"),
                  nadir: Optional[str] = Query(None),
                  boat_minutes: Optional[float] = Query(None, ge=0, le=1440),
                  files: Optional[list[UploadFile]] = File(None), simulate: Optional[str] = Query(None),
                  active: bool = Query(False, description="active vision even if not validated (EXPERIMENTAL)")):
    faults = S._drills(simulate) | ({"opt:active"} if active is True else set())
    if "model_unavailable" in faults:
        S._pipeline_or_503(faults)
    busy = S._JOBS.counts()
    if busy["queued"] + busy["running"] >= int(os.environ.get("DEPTH_MAX_QUEUED_JOBS", "4")):
        raise HTTPException(503, "the survey queue is full - try again in a minute")
    track = None
    if gps == "recording":
        if not S._recording_summary().get("available"):
            raise HTTPException(404, "no raw recording on this server (python -m src.cv_pipeline.humminbird fetch)")
        frames, track, _, _ = S._recording()
    else:
        problems: list = []
        frames = S._collect_frames(use_samples, files, problems)
    survey_id = f"survey-{time.strftime('%Y%m%d-%H%M%S')}-{len(frames)}f{'-rec' if track else ''}"
    probs = problems if gps != "recording" else []
    jid = S._JOBS.submit(lambda progress: S._run_survey(frames, gps, budget_minutes, nadir, survey_id, progress,
                                                        boat_minutes, track, frozenset(faults), probs, recall, compute_s),
                         total=len(frames))
    return {"job_id": jid, "survey_id": survey_id, "frames": len(frames)}


@router.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    rec = S._JOBS.get(job_id)
    if rec is None:
        raise HTTPException(404, "unknown job_id")
    return rec


@router.post("/api/survey")
def survey(use_samples: bool = Query(False), gps: str = Query("synthetic"),
           budget_minutes: Optional[float] = Query(None, ge=0, le=600),
           recall: Optional[float] = Query(None, ge=0.5, le=0.99, description="recall promise to keep (the goal)"),
           compute_s: Optional[float] = Query(None, ge=1, le=3600, description="compute budget for perception, s"),
           nadir: Optional[str] = Query(None),
           files: Optional[list[UploadFile]] = File(None), simulate: Optional[str] = Query(None),
           boat_minutes: Optional[float] = Query(None, ge=0, le=1440),
           active: bool = Query(False, description="active vision even if not validated (EXPERIMENTAL)")):
    faults = S._drills(simulate) | ({"opt:active"} if active is True else set())
    track, problems = None, []
    if gps == "recording":
        if not S._recording_summary().get("available"):
            raise HTTPException(404, "no raw recording on this server (python -m src.cv_pipeline.humminbird fetch)")
        frames, track, _, _ = S._recording()
    else:
        frames = S._collect_frames(use_samples, files, problems)
    if len(frames) > S.MAX_SYNC_FRAMES:
        raise HTTPException(413, f"{len(frames)} frames: use POST /api/jobs/survey for more than "
                                 f"{S.MAX_SYNC_FRAMES} (runs in the background, poll /api/jobs/<id>)")
    survey_id = f"survey-{time.strftime('%Y%m%d-%H%M%S')}-{len(frames)}f{'-rec' if track else ''}"
    return JSONResponse(S._run_survey(frames, gps, budget_minutes, nadir, survey_id, track=track,
                                      boat_minutes=boat_minutes, faults=frozenset(faults), problems=problems,
                                      recall=recall, compute_s=compute_s))


@router.get("/api/recording")
def recording_info():
    """The raw recording available as a survey source (real per-ping GPS) and its validation."""
    return S._recording_summary()


@router.get("/api/recording/geometry/{frame_id}.jpg")
def recording_geometry_overlay(frame_id: str):
    """One chunk of the recording with the agent's seabed cross-check drawn by OpenCV: red = the
    tracker alone, blue = the depth sounder, green = what the agent accepted (none = not measured)."""
    if not S._recording_summary().get("available"):
        raise HTTPException(404, "no raw recording on this server")
    frames, _, meta, _ = S._recording()
    img = dict(frames).get(frame_id)
    geo = (meta["frames"].get(frame_id) or {}).get("geometry")
    if img is None or not geo:
        raise HTTPException(404, f"unknown recording frame {frame_id}")
    vis = img.copy()
    for key, col in (("sounder", (255, 150, 40)), ("tracker", (60, 60, 235)), ("accepted", (60, 210, 90))):
        pts = [(int(x), int(y)) for x, y in (geo["lines"].get(key) or []) if y is not None]
        for a, b in zip(pts, pts[1:]):
            if key == "sounder":
                cv2.circle(vis, a, 2, col, -1, cv2.LINE_AA)
            else:
                cv2.line(vis, a, b, col, 2, cv2.LINE_AA)
    ys = [y for k in ("sounder", "tracker", "accepted") for _, y in (geo["lines"].get(k) or []) if y is not None]
    h = int(min(vis.shape[0], max(vis.shape[0] * 0.3, 1.8 * max(ys)) if ys else vis.shape[0]))
    vis = cv2.resize(vis[:h], (640, max(1, int(round(640 * h / vis.shape[1])))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


_MEDIA = {"geojson": "application/geo+json", "gpx": "application/gpx+xml",
          "kml": "application/vnd.google-earth.kml+xml", "csv": "text/csv", "json": "application/json", "trace": "application/x-ndjson",
          "brief": "text/markdown; charset=utf-8"}


@router.get("/api/report/{fmt}")
def report(fmt: str, survey_id: str = Query(...), public: bool = Query(False)):
    """Mission exports. ``public=1`` generalises protected-site (wreck) locations; ``trace`` is the agent
    decision log (JSONL) and is never public."""
    if fmt not in _MEDIA:
        raise HTTPException(400, f"format must be one of {list(_MEDIA)}")
    if public and fmt == "trace":
        raise HTTPException(403, "the decision log carries exact positions - it is not shared publicly")
    key = f"public.{fmt}" if public else fmt
    body = (export(fmt, S._SURVEYS[survey_id], public=public) if survey_id in S._SURVEYS
            else S._JOBS.report(survey_id, key))
    if body is None:
        raise HTTPException(404, "unknown survey_id (run a survey first)")
    ext = {"trace": "trace.jsonl", "brief": "brief.md"}.get(fmt, fmt)
    return Response(body, media_type=_MEDIA[fmt],
                    headers={"Content-Disposition": f'attachment; filename="mission{".public" if public else ""}.{ext}"'})


@router.get("/api/brief")
def mission_brief(survey_id: str = Query(...), public: bool = Query(False),
                  writer: str = Query("auto", pattern="^(auto|template|llm)$"),
                  simulate: Optional[str] = Query(None)):
    """The mission brief + how it was written. ``auto`` uses Claude on Amazon Bedrock when
    ``DEPTH_BRIEF_LLM=bedrock`` (it writes, never decides; every number must be a survey fact or the
    deterministic template is served instead — ``writer`` / ``fallback_reason`` / ``grounding`` say which)."""
    if survey_id in S._SURVEYS:
        s = S._SURVEYS[survey_id]
        d, prov = s.to_dict(), provenance_stamp()
        if public:
            from src.agentic.mission import redact_public
            tr, mi, _ = redact_public(s.tracked, s.mission)
            d = SurveyResult(s.survey_id, s.frames, tr, mi).to_dict()
    else:
        raw = S._JOBS.report(survey_id, "public.json" if public else "json")
        if raw is None:
            raise HTTPException(404, "unknown survey_id (run a survey first)")
        d = json.loads(raw)
        prov = d.get("provenance")
    faults = S._drills(simulate)
    out = brief_mod.write(d, prov, public=public, writer=writer, simulate_failure="llm_failed" in faults)
    out["llm_enabled"] = brief_mod.llm_enabled()
    if survey_id in S._SURVEYS:
        S._SURVEYS[survey_id].plan_args.setdefault("events", []).append(
            {"t": time.time(), "actor": "llm" if out["writer"] == "llm" else "system", "kind": "brief",
             "title": f"Mission brief written by the {out['writer']}",
             "text": out.get("fallback_reason") or out["authority"], "incident": out.get("incident")})
    return out


# ---- a person's decision on a hazard -> the agent re-plans (routes, queue, budget, passes) ----------
@router.post("/api/survey/{survey_id}/decide")
def survey_decide(survey_id: str, request: Request, body: dict = Body(...)):
    """confirm / reject a card, mark a route stop recovered / not_found, or undo. Made by a named
    person in the studio (``DEPTH_APPROVER_PIN`` when set). The agent's own verdict is never
    overwritten; the plan is rebuilt around the decision and every decision is logged."""
    try:
        S._PIN_GUARD.check(client_of(request.scope), body.get("pin"))
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e
    s = S._SURVEYS.get(survey_id)
    if s is None:
        raise HTTPException(409, "this survey is no longer in memory - re-run it to decide on its hazards")
    from src.agentic.mission import apply_human
    try:
        t = apply_human(s, str(body.get("hazard_id", "")), str(body.get("decision", "")), str(body.get("by", "")),
                        str(body.get("note", "")))
    except KeyError:
        raise HTTPException(404, "unknown hazard_id") from None
    except ValueError as e:
        msg = str(e)
        raise HTTPException(409 if any(k in msg for k in ("already", "not on the", "no decision", "no priority")) else 400, msg) from e
    label = _survey_label(s, t, str(body.get("decision", "")))
    replan = s.mission.replans[-1] if s.mission.replans else None
    requests = S._sync_agent_requests(s, reason=f"re-planned after {t.human_by or body.get('by')} chose "
                                              f"'{body.get('decision')}' on {t.oid}")
    if replan is not None:
        replan["requests"] = requests
    d = s.to_dict()
    d.pop("frames", None)
    try:                                              # the persisted copy follows (OGC, reports)
        prov = provenance_stamp()
        old = json.loads(S._JOBS.report(survey_id, "json") or "{}")
        keep = {k: old[k] for k in ("twin", "frame_refs", "stage1_counterfactual", "opencv_counterfactual") if k in old}
        S._JOBS.persist(survey_id, {**d, **keep, "provenance": prov},
                        {fmt: export(fmt, s, prov=prov) for fmt in REPORT_FORMATS} |
                        {f"public.{fmt}": export(fmt, s, public=True, prov=prov) for fmt in PUBLIC_FORMATS})
    except Exception:
        S.log.exception("persist after decision failed")
    S._notify("survey.replanned", {"survey_id": survey_id, "hazard_id": t.oid, "decision": body.get("decision"),
                                   "by": t.human_by, "recovery_route": s.mission.recovery_route,
                                   "route_length_m": s.mission.route_length_m, "impact": s.mission.impact})
    return {"hazard": t.to_dict(), "tracked": d["tracked"], "mission": d["mission"], "replan": replan,
            "label": label, "dispatch": S._dispatch(s)}


def _survey_label(s: SurveyResult, t, decision: str) -> Optional[dict]:
    """A person's confirm / reject on a survey hazard is also a training label (feedback.py)."""
    dec = {"confirm": "confirm", "reject": "reject"}.get(decision)
    if dec is None:
        return None
    ref = S._frame_ref(t.frame_id)
    if not ref:
        return None
    try:
        return S._FEEDBACK.add({"frame_id": t.frame_id, "frame_ref": ref, "bbox": list(t.bbox), "cls_name": t.cls_name,
                                "decision": dec, "verdict_before": t.verdict.value, "conf": t.conf, "p_pot": t.p_pot,
                                "source": "survey", "annotator": t.human_by or "anon"})
    except Exception:
        S.log.exception("survey label not saved")
        return None


# ---- the audit log: one timeline of who did what, and why -------------------------------------------
@router.get("/api/survey/{survey_id}/log")
def survey_log(survey_id: str):
    """Agent plan steps, incidents, approval requests / decisions / withdrawals, people's decisions,
    re-plans and brief writers - merged, oldest first. Actors: agent | person | system | llm."""
    s = S._SURVEYS.get(survey_id)
    if s is None:
        raise HTTPException(404, "unknown survey_id (this log lives with the survey in memory - re-run it)")
    t0, m, E = s.plan_args.get("t0") or 0.0, s.mission, []
    for fr in s.frames:
        for inc in fr.incidents:
            E.append({"t": t0, "actor": "system", "kind": "incident", "title": inc["title"], "text": inc["message"],
                      "severity": inc["severity"], "incident": inc})
    for inc in m.incidents:
        E.append({"t": t0, "actor": "system", "kind": "incident", "title": inc["title"], "text": inc["message"],
                  "severity": inc["severity"], "incident": inc})
    for st in m.agent_log:
        E.append({"t": t0, "actor": "agent", "kind": st["tool"], "title": st["tool"].replace("_", " "),
                  "text": st["rationale"], "status": st["status"]})
    for r in S._APPROVALS.list(survey_id=survey_id, limit=500):
        E.append({"t": r["t"], "actor": "agent" if r.get("channel") == "agent" else "system", "kind": "approval_request",
                  "title": f"Approval requested: {r['action']} ({len(r['targets'])})", "text": r["rationale"], "id": r["id"],
                  "by": r.get("requested_by")})
        if r.get("decision"):
            dd = r["decision"]
            E.append({"t": dd["t"], "actor": "agent" if r["status"] == "withdrawn" else "person",
                      "kind": "approval_" + r["status"], "title": f"{r['action']} {r['status']}",
                      "text": dd.get("note") or "", "id": r["id"], "by": dd.get("by")})
    for h in m.human_log:
        E.append({"t": h["t"], "actor": "person", "kind": "decision", "title": f"{h['decision']} {h['hazard']}",
                  "text": h.get("note") or f"{h.get('before') or 'open'} -> {h.get('after') or 'open'}"
                          + (" (priority)" if h["decision"].endswith("priority") or h["decision"].endswith("prioritize") else ""),
                  "by": h.get("by")})
    for r in m.replans:
        E.append({"t": r["t"] + 0.01, "actor": "agent", "kind": "replan", "title": "Re-planned", "text": r["rationale"]})
    for ev in s.plan_args.get("events", []):
        E.append(dict(ev))
    E.sort(key=lambda e: e["t"])
    return {"survey_id": survey_id, "entries": E, "dispatch": S._dispatch(s),
            "llm_authority": "writer only - the LLM never decides, approves or dispatches"}
