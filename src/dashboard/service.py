"""
service.py — the studio server's shared state and the work behind its routes.

One process serves the studio: one ``cv2.dnn`` network (loaded and warmed in the background; every
inference holds ``_INFER_LOCK``, so ``/api/health`` answers while a survey runs), the analyze queue the
agent prices its look against (``_QUEUE``), the bounded survey memory and background jobs, the stores
for labels, approvals and webhooks, and the helpers the routes share: frame decoding and payloads, the
raw recording, the survey run (``_run_survey``: the core loop, then the plan, the exports and the
approval requests), and the dispatch gate (the agent files requests; a person decides).

The routes live in ``routes/`` and reach this module as ``S.<name>`` at call time, so a test can swap
a store (``service._APPROVALS = ...``) and every route sees it. ``app.py`` builds the FastAPI app.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from src.agentic.agent import render
from src.agentic.approvals import ApprovalStore, PinGuard
from src.agentic.core import Goal, SurveyAgent
from src.agentic.feedback import FeedbackStore
from src.agentic.geo import stage1_counterfactual, synthetic_track
from src.agentic.incidents import DRILLS, incident, parse_simulate
from src.agentic.mission import export
from src.agentic.perception import Perceptor
from src.agentic.pipeline import AgenticPipeline
from src.agentic.provenance import stamp as provenance_stamp
from src.agentic.shadow import ShadowProver
from src.agentic.twin import survey_twin
from src.agentic.types import SurveyResult
from src.detection.infer import configure_runtime
from . import samples as samples_mod
from .detector_queue import DetectorQueue
from .integrations import Webhooks
from .jobs import JobStore, PUBLIC_FORMATS, REPORT_FORMATS
from .metrics import Metrics

log = logging.getLogger("depth.api")
REPO = Path(__file__).resolve().parents[2]
_WEBUI = REPO / "webui"
WEBUI_DIST = _WEBUI / "dist" if (_WEBUI / "dist").exists() else _WEBUI

MAX_UPLOAD_MB = float(os.environ.get("DEPTH_MAX_UPLOAD_MB", "20"))
MAX_FRAMES = int(os.environ.get("DEPTH_MAX_FRAMES", "50"))
MAX_SYNC_FRAMES = int(os.environ.get("DEPTH_MAX_SYNC_FRAMES", "12"))
MAX_SURVEYS = int(os.environ.get("DEPTH_MAX_SURVEYS", "24"))
INTERACTIVE_S = float(os.environ.get("DEPTH_INTERACTIVE_S", "2.0"))   # compute budget of one /api/analyze call
_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp", "application/octet-stream", ""}

# ---- shared state ----------------------------------------------------------------------------
_PIPELINE: Optional[AgenticPipeline] = None
_MODEL_STATE = {"loaded": False, "loading": False, "error": None, "warmup_ms": None}
_INFER_LOCK = threading.Lock()                  # one inference at a time on the shared cv2.dnn net
_QUEUE = DetectorQueue()                        # expected compute of the analyze calls in flight
_LOAD_LOCK = threading.Lock()
_PROVER = ShadowProver()
_SURVEYS: "OrderedDict[str, SurveyResult]" = OrderedDict()   # bounded LRU (report downloads)
_JOBS = JobStore(max_jobs=50, workers=1)
configure_runtime()                             # $DEPTH_THREADS
_METRICS = Metrics()
_FEEDBACK = FeedbackStore()
_APPROVALS = ApprovalStore()
_PIN_GUARD = PinGuard()                         # wrong-PIN lockout per client
_WEBHOOKS = Webhooks()
PUBLIC_URL = os.environ.get("DEPTH_PUBLIC_URL", "").rstrip("/")     # e.g. https://<cloudfront-domain>
_STUDY: "OrderedDict[str, dict]" = OrderedDict()          # active study sessions (plan + the cards shown)
_UPLOADS: "OrderedDict[str, np.ndarray]" = OrderedDict()     # recent uploaded frames (feedback persists them)
MAX_UPLOAD_CACHE = 64


def get_pipeline() -> AgenticPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        with _LOAD_LOCK:
            if _PIPELINE is None:
                _MODEL_STATE["loading"] = True
                try:
                    pipe = AgenticPipeline()
                    t0 = time.perf_counter()
                    pipe.seed_pace(pipe.agent.perceptor.warmup())      # the agent prices its modes with it
                    _MODEL_STATE["warmup_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    _PIPELINE = pipe
                    _MODEL_STATE["loaded"] = True
                except Exception as e:                  # surfaced by /api/health, never crashes
                    _MODEL_STATE["error"] = f"{type(e).__name__}: {e}"
                    raise
                finally:
                    _MODEL_STATE["loading"] = False
    return _PIPELINE


def _warm_in_background() -> None:
    try:
        get_pipeline()
    except Exception:
        log.exception("model failed to load at startup")


def _notify(event: str, payload: dict) -> None:
    """DEPTH events -> signed webhooks (src/dashboard/integrations.py). Approval events carry the
    targets' positions so a console can act on an approved work order without calling back."""
    data = dict(payload)
    if event.startswith("approval.") and data.get("survey_id") in _SURVEYS:
        s = _SURVEYS[data["survey_id"]]
        idx = {t.oid: t for t in s.tracked}
        data["targets_detail"] = [{"id": i, "class": idx[i].cls_name, "lat": idx[i].lat, "lon": idx[i].lon,
                                   "error_m": idx[i].geo_error_m, "p_pot": idx[i].p_pot} for i in data.get("targets", []) if i in idx]
        data["gps_synthetic"] = s.mission.gps_synthetic
    data["links"] = {"approvals": f"{PUBLIC_URL}/api/approvals",
                     "hazards": f"{PUBLIC_URL}/ogc/collections/hazards/items?survey_id={data.get('survey_id', 'latest')}",
                     **(data.get("links") or {})}
    try:
        n = _WEBHOOKS.emit(event, data)
        log.info("event %s %s -> %d webhook(s)", event, payload.get("id") or payload.get("survey_id"), n)
    except Exception:                                   # integrations never break the product
        log.exception("webhook emit failed")


def _survey_event(d: dict) -> dict:
    m, sid = d["mission"], d["survey_id"]
    rp = m.get("resurvey_plan") or {}
    idx = {t["oid"]: t for t in d["tracked"]}
    return {"survey_id": sid, "frames": len(d.get("frames") or []), "counts": m["counts"],
            "gps": "none" if not m["gps_available"] else ("synthetic" if m["gps_synthetic"] else "real"),
            "recall_promise": (m.get("guarantees") or {}).get("recall_promise"),
            "review_top": [{"id": i, "p_pot": idx[i].get("p_pot"), "lat": idx[i].get("lat"), "lon": idx[i].get("lon")}
                           for i in (m.get("review_queue") or [])[:5] if i in idx],
            "resurvey_passes": len(rp.get("lines") or []), "boat_minutes": rp.get("boat_minutes_planned"),
            "human_approval_required": True,
            "links": {"brief": f"{PUBLIC_URL}/api/brief?survey_id={sid}",
                      "geojson": f"{PUBLIC_URL}/api/report/geojson?survey_id={sid}",
                      "ogc_hazards": f"{PUBLIC_URL}/ogc/collections/hazards/items?survey_id={sid}"}}


# CORS origins for the API: off unless DEPTH_CORS_ORIGINS is set (the studio is same-origin)
_cors = [o.strip() for o in os.environ.get("DEPTH_CORS_ORIGINS", "").split(",") if o.strip()]


# ---- helpers ----------------------------------------------------------------------------------
def _fail(status: int, inc: dict) -> HTTPException:
    """An API error that carries the incident (what happened, the fallback, what a person should do)."""
    return HTTPException(status, detail={"message": inc["message"], "incident": inc})


def _drills(simulate: Optional[str]) -> set:
    if not isinstance(simulate, str):                 # called as a function (MCP): the Query default
        return set()
    want = parse_simulate(simulate)
    bad = want - set(DRILLS)
    if bad:
        raise HTTPException(400, f"unknown drill(s) {sorted(bad)} - choose from {list(DRILLS)}")
    return want


def _pipeline_or_503(faults: set = frozenset()) -> AgenticPipeline:
    """The loaded pipeline, or a 503 ``model_unavailable`` incident (never a bare 500)."""
    if "model_unavailable" in faults:
        raise _fail(503, incident("model_unavailable", "detector unavailable (failure drill)", simulated=True))
    try:
        return get_pipeline()
    except Exception as e:
        raise _fail(503, incident("model_unavailable", f"the detector could not be loaded: {type(e).__name__}: {e}"[:300])) from e


def _b64(img: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


def _evidence_crop(frame: np.ndarray, cand, target: int = 300) -> str:
    """Zoomed crop around a candidate with the shadow/echo overlay — the evidence-card image."""
    vis = _PROVER.overlay(frame, cand.bbox, cand.evidence.shadow) if cand.evidence else frame
    x1, y1, x2, y2 = cand.bbox
    bw, bh = x2 - x1, y2 - y1
    pad = int(max(bw, bh) * 1.6) + 12
    H, W = frame.shape[:2]
    crop = vis[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
    if crop.size == 0:
        return ""
    scale = target / max(1, max(crop.shape[:2]))
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return _b64(crop)


def _cache_upload(data: bytes, img: np.ndarray) -> str:
    """Keep a recently uploaded frame so a later correction can persist it as a training image."""
    sha = hashlib.sha256(data).hexdigest()[:32]
    _UPLOADS[sha] = img
    _UPLOADS.move_to_end(sha)
    while len(_UPLOADS) > MAX_UPLOAD_CACHE:
        _UPLOADS.popitem(last=False)
    return sha


_UPLOAD_SHA: "OrderedDict[str, str]" = OrderedDict()   # frame_id → sha of the most recent uploads (bounded)


def _decode_upload(f: UploadFile) -> tuple[str, np.ndarray]:
    """Validate type + size, then decode. Raises HTTPException with a clear message."""
    if (f.content_type or "").lower() not in _IMAGE_TYPES and not (f.content_type or "").startswith("image/"):
        raise HTTPException(415, f"{f.filename}: only images are accepted (got {f.content_type})")
    data = f.file.read(int(MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"{f.filename}: larger than {MAX_UPLOAD_MB:g} MB")
    try:          # OpenCV 5 raises on an empty buffer (it used to return None) - judge audit 2026-10-08
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    except cv2.error:
        img = None
    if img is None:
        raise HTTPException(400, f"{f.filename}: could not decode image")
    stem = Path(f.filename or "upload").stem
    _UPLOAD_SHA[stem] = _cache_upload(data, img)
    _UPLOAD_SHA.move_to_end(stem)
    while len(_UPLOAD_SHA) > 4 * MAX_UPLOAD_CACHE:          # a long-running public server never grows it
        _UPLOAD_SHA.popitem(last=False)
    return stem, img


def _frame_ref(frame_id: str) -> dict:
    """How a correction on this frame finds its image again: a shipped sample or a cached upload."""
    sid = samples_mod.sample_for_frame(frame_id)
    if sid:
        return {"sample": sid}
    sha = _UPLOAD_SHA.get(frame_id)
    return {"upload": sha} if sha else {}


def _frame_payload(frame: np.ndarray, result, full: bool = False) -> dict:
    """The frame result + its images (JPEG data URLs, despite the ``_png`` names). The studio draws the
    boxes itself and shows the CLAHE re-look, so by default the server-rendered overlay and the plain zoom
    are left out (~1.18 MB -> ~0.78 MB on a 7-find frame); ``full=True`` (``?images=full``) adds them."""
    d = result.to_dict()
    d["frame_png"] = _b64(frame)
    if full:
        d["overlay_png"] = _b64(render(frame, result))
    for cand, cd in zip(result.candidates, d["candidates"]):
        cd["crop_png"] = _evidence_crop(frame, cand)
        rv = Perceptor.relook_view(frame, cand.bbox)      # the agent's-eye zoom (display-only)
        if rv:
            cd["relook_view"] = {"enhanced_png": _b64(rv["enhanced"]), "scale": round(rv["scale"], 2),
                                 "obj_box": rv["obj_box"], **({"zoom_png": _b64(rv["zoom"])} if full else {})}
    return d


def _sample_frames() -> list[tuple[str, np.ndarray]]:
    out = []
    for s in samples_mod.list_samples():
        p = samples_mod.sample_path(s["id"])
        img = cv2.imread(str(p)) if p and p.exists() else None
        if img is not None:
            out.append((p.stem, img))
    return out


# ---- a RAW sonar recording with real per-ping GPS (src/cv_pipeline/humminbird.py) --------------------
RECORDING_DAT = Path(os.environ.get("DEPTH_RECORDING", str(REPO / "DATASET" / "external" / "pingmapper_sample" / "Test-Small-DS.DAT")))
_REC_CACHE: dict = {}
_REC_LOCK = threading.Lock()


def _recording():
    """(frames, track, meta, validation) of the configured recording; cached after the first load."""
    with _REC_LOCK:
        if "frames" not in _REC_CACHE:
            from src.cv_pipeline.humminbird import read_recording, frames_and_track, validate
            rec = read_recording(RECORDING_DAT)
            frames, track, meta = frames_and_track(rec)
            _REC_CACHE.update(frames=frames, track=track, meta=meta, validation=validate(rec), name=rec.name)
        return _REC_CACHE["frames"], _REC_CACHE["track"], _REC_CACHE["meta"], _REC_CACHE["validation"]


def _recording_summary() -> dict:
    if not RECORDING_DAT.exists():
        return {"available": False, "how": "python -m src.cv_pipeline.humminbird fetch"}
    try:
        frames, track, meta, val = _recording()
    except Exception as e:                                   # a bad file never breaks the app
        return {"available": False, "reason": f"{type(e).__name__}: {e}"[:200]}
    ch = val["channels"].get("port") or next(iter(val["channels"].values()))
    return {"available": True, "name": _REC_CACHE.get("name"), "frames": len(frames), "dat": val["dat"],
            "duration_s": ch["duration_s"], "track_m": ch["track_m"], "bbox": ch["bbox"],
            "range_scale": meta["range_scale"],
            "geometry": _geometry_summary(meta),
            "checks": {"speed_gps_per_unit": ch["gps_speed_per_speed_unit"], "speed_corr": ch["speed_corr"],
                       "course_vs_heading_deg_median": ch["course_vs_heading_deg_median"],
                       "course_vs_heading_deg_p90": ch["course_vs_heading_deg_p90"],
                       "records_monotonic": ch["records_monotonic"], "issues": ch["issues"]},
            "source": "PINGMapper sample data (Bodine et al., MIT; Zenodo 10.5281/zenodo.6604666) - real Humminbird "
                      "recording, real per-ping GPS; downloaded hash-checked, not redistributed"}


def _geometry_summary(meta: dict) -> Optional[dict]:
    """The agent's sensor cross-check of the recording (STUDY-16): the summary, the scale iterations
    and one entry per chunk (verdict, reason, the agent's steps) - the overlay is /api/recording/geometry/<frame>.jpg."""
    g = meta.get("geometry")
    if not g:
        return None
    chunks = [{"frame_id": fid, **{k: v for k, v in (m.get("geometry") or {}).items() if k != "lines"}}
              for fid, m in meta["frames"].items() if m.get("geometry")]
    return {**g, "chunks": chunks, "report": "docs/reports/sensor_check.md"}


def _collect_frames(use_samples: bool, files: Optional[list[UploadFile]],
                    problems: Optional[list] = None) -> list[tuple[str, np.ndarray]]:
    """Frames for a survey. An undecodable upload is reported in ``problems`` (a ``corrupt_frame``
    incident) and skipped, so one bad file does not sink the survey; type / size limits still refuse."""
    if use_samples:
        frames = _sample_frames()
    else:
        files = files or []
        if len(files) > MAX_FRAMES:
            raise HTTPException(413, f"at most {MAX_FRAMES} frames per survey (got {len(files)})")
        frames = []
        for f in files:
            try:
                frames.append(_decode_upload(f))
            except HTTPException as e:
                if e.status_code != 400 or problems is None:
                    raise
                problems.append(incident("corrupt_frame", str(e.detail), frame_id=Path(f.filename or "upload").stem))
    if not frames:
        raise HTTPException(400, "no readable frames (use ?use_samples=1 or upload files)")
    return frames


def _remember(result: SurveyResult) -> None:
    _SURVEYS[result.survey_id] = result
    _SURVEYS.move_to_end(result.survey_id)
    while len(_SURVEYS) > MAX_SURVEYS:
        _SURVEYS.popitem(last=False)


def _run_survey(frames, gps: str, budget_minutes: Optional[float], nadir: Optional[str],
                survey_id: str, progress: Optional[Callable[[dict], None]] = None,
                boat_minutes: Optional[float] = None, track: Optional[dict] = None,
                faults: frozenset = frozenset(), problems: Optional[list] = None,
                recall: Optional[float] = None, compute_s: Optional[float] = None) -> dict:
    faults = set(faults)
    if "corrupt_frame" in faults:                         # failure drill: an unreadable frame in the batch
        frames = list(frames) + [("drill_corrupt_frame", None)]
    if "missing_gps" in faults:                           # failure drill: the survey has no GPS
        gps, track = "none", None
    if track is None:
        track = synthetic_track([fid for fid, _ in frames]) if gps == "synthetic" else None
    pipe = _pipeline_or_503(faults)
    total = len(frames)
    done = {"n": 0}

    def cb(stage: str, payload: dict) -> None:
        if stage == "frame_done":
            done["n"] += 1
        if progress:
            progress({"done": done["n"], "total": total, "stage": stage,
                      "frame": payload.get("frame_id")})

    goal = Goal(recall=recall, analyst_minutes=budget_minutes, boat_minutes=boat_minutes, compute_s=compute_s)
    geometry = (_recording()[2].get("geometry") if gps == "recording" else None)
    result = SurveyAgent(pipe).run(frames, goal=goal, track=track, survey_id=survey_id, nadir=nadir,
                                   progress_cb=cb, frame_lock=_INFER_LOCK,   # per frame: /api/analyze can interleave
                                   faults=faults, incidents=problems, geometry=geometry)
    for inc in result.mission.incidents:
        if inc["code"] == "missing_gps" and "missing_gps" in faults:
            inc["simulated"] = True
    result.plan_args["t0"] = time.time()
    _sync_agent_requests(result, reason="survey planned")
    _remember(result)
    for fr in result.frames:
        _METRICS.record("survey", fr.frame_id, fr.stage_ms, counts=fr.counts)
    frame_map = {fid: im for fid, im in frames if im is not None}
    d = result.to_dict()
    for fr, fd in zip(result.frames, d["frames"]):          # light thumbnails, not full-res base64
        thumb = cv2.resize(render(frame_map[fr.frame_id], fr), (200, 200), interpolation=cv2.INTER_AREA)
        fd["thumb_png"] = _b64(thumb, 70)
    d["frame_refs"] = {fid: _frame_ref(fid) for fid, _ in frames}
    try:
        d["twin"] = survey_twin([(f, im) for f, im in frames if im is not None], result, track, nadir=nadir)
    except Exception as e:
        log.exception("survey twin failed")
        d["twin"] = {"available": False, "reason": f"twin error: {type(e).__name__}"}
        _add_incident(result, d, incident("view_failed", f"3D survey twin: {type(e).__name__}"))
    if gps == "recording":
        d["recording"] = _recording_summary()
    try:                                                    # STUDY-12, live: where pins go without Stage 1
        d["stage1_counterfactual"] = stage1_counterfactual(result.frames, result.tracked, track, pipe.m_per_px)
    except Exception as e:
        log.exception("stage-1 counterfactual failed")
        d["stage1_counterfactual"] = {"available": False, "reason": f"error: {type(e).__name__}"}
        _add_incident(result, d, incident("view_failed", f"Stage-1 counterfactual: {type(e).__name__}"))
    try:                                                    # detector only vs + Stage 1 + OpenCV evidence
        from src.agentic.counterfactual import without_opencv
        d["opencv_counterfactual"] = without_opencv(result, d.get("stage1_counterfactual"))
    except Exception as e:
        log.exception("opencv counterfactual failed")
        d["opencv_counterfactual"] = {"available": False, "reason": f"error: {type(e).__name__}"}
    prov = provenance_stamp()
    d["provenance"] = prov
    reports = {fmt: export(fmt, result, prov=prov) for fmt in REPORT_FORMATS}
    reports.update({f"public.{fmt}": export(fmt, result, public=True, prov=prov) for fmt in PUBLIC_FORMATS})
    _JOBS.persist(result.survey_id, {k: v for k, v in d.items() if k != "frames"}, reports,
                  simulate_storage_failure="storage_failed" in faults)
    d["storage"] = _JOBS.storage.get(result.survey_id, {})
    if d["storage"].get("s3") == "failed":
        _add_incident(result, d, incident("storage_failed", d["storage"].get("error") or "S3 upload failed",
                                          simulated="storage_failed" in faults))
    d["dispatch"] = _dispatch(result)
    _notify("survey.completed", _survey_event(d))
    return d


def _add_incident(result: SurveyResult, d: dict, inc: dict) -> None:
    result.mission.incidents.append(inc)
    d["mission"]["incidents"] = list(result.mission.incidents)


# ---- the dispatch gate: the agent files approval requests, a person decides -------------------------
_AGENT = "DEPTH agent (planner)"


def _wanted_requests(s: SurveyResult) -> dict[str, tuple[list, str]]:
    """What the agent needs a person to approve for the CURRENT plan: action -> (targets, rationale)."""
    m, out = s.mission, {}
    gps = "real GPS" if m.gps_available and not m.gps_synthetic else ("SYNTHETIC demo GPS" if m.gps_available else "no GPS")
    if m.inspection_route:
        out["inspect"] = (list(m.inspection_route),
                          f"inspect {len(m.inspection_route)} REVIEW card(s) first - the analyst budget's best "
                          f"expected yield ({(m.budget or {}).get('expected_pots_in_budget', '?')} expected pots)"
                          + (f", route {m.inspection_length_m} m" if m.inspection_length_m else "") + f"; {gps}")
    if m.recovery_route:
        out["recover"] = (list(m.recovery_route),
                          f"recover {len(m.recovery_route)} confirmed hazard(s)"
                          + (f", route {m.route_length_m} m" if m.route_length_m else "") + f"; {gps}")
    wd = (m.guarantees or {}).get("promise_withdrawn")
    if wd:                                                         # perception said the promise does not apply here
        fids = [f.frame_id for f in s.frames]
        out["full_review"] = (fids, f"Agent recommends: review all {len(fids)} frame(s) by eye. Reason: {wd['why']}. "
                                    f"The recall promise (>= {round(100 * (wd.get('was') or 0))}%) was fit on other "
                                    "recordings and is withdrawn for this survey; the cards are hints, not a promise")
    for L in (m.resurvey_plan or {}).get("lines") or []:          # one request per pass: each can be declined
        out["resurvey:" + ",".join(L["targets"])] = (
            list(L["targets"]),
            f"Agent recommends: {L.get('recommendation', 'Perform opposite-side resurvey')} {L['id']} over "
            f"{', '.join(L['targets'][:6])}. Reason: {L.get('reason', 'uncertain visual evidence')}. "
            f"{L['boat_min']} boat-min, {L['voi']} {L.get('info_unit', '')} expected; a real object's shadow must "
            f"flip. If declined, the agent re-plans; {gps}")
    return out


def _sync_agent_requests(s: SurveyResult, reason: str) -> list[dict]:
    """Keep the agent's pending requests in step with the plan: file what is needed, withdraw what a
    re-plan made stale. Returns what changed. Nothing here executes anything."""
    have = s.plan_args.setdefault("requests", {})
    changes = []
    want = _wanted_requests(s)
    for key in sorted(set(have) | set(want) | {"inspect", "recover"}):
        action = key.split(":")[0]
        cur = _APPROVALS.get(have[key]) if key in have else None
        targets, why = want.get(key, ([], ""))
        if cur is not None and cur["status"] == "declined" and action == "resurvey":
            continue                                    # a person's decline stands; the plan already moved on
        if cur is not None and cur["targets"] == targets[:200]:
            continue
        if cur is not None and cur["status"] == "pending":
            try:
                _APPROVALS.withdraw(cur["id"], f"superseded - {reason}", by=_AGENT)
                changes.append({"action": action, "withdrawn": cur["id"]})
            except Exception:
                log.exception("withdraw failed")
        have.pop(key, None)
        if targets:
            try:
                r = _APPROVALS.request(s.survey_id, action, targets, why, requested_by=_AGENT, channel="agent")
                have[key] = r["id"]
                changes.append({"action": action, "requested": r["id"], "targets": len(targets)})
                _notify("approval.requested", r)
            except Exception as e:                       # the gate stays closed; a person is told
                log.exception("approval request failed")
                s.mission.incidents.append(incident("tool_failed", f"could not file the '{action}' approval request: "
                                                    f"{type(e).__name__}", fallback="nothing dispatched; the plan "
                                                    "stays visible", human_action="file the request from the Approvals tab"))
    for c in changes:
        s.mission.agent_log.append({"tool": "request_approval", "status": "done",
                                    "rationale": (f"asked a person to approve '{c['action']}' ({c['requested']}, "
                                                  f"{c['targets']} target(s)) - {reason}") if "requested" in c else
                                                 f"withdrew stale request {c['withdrawn']} ('{c['action']}') - {reason}",
                                    "detail": c})
    return changes


def _dispatch(s: SurveyResult) -> dict:
    """Per action: the agent's request and where a person has got to with it."""
    out = {}
    for action, rid in (s.plan_args.get("requests") or {}).items():
        r = _APPROVALS.get(rid)
        if r:
            out[action] = {"id": rid, "status": r["status"], "targets": len(r["targets"]),
                           "by": (r.get("decision") or {}).get("by")}
    return out


# ---- OGC API - Features: any GIS / operations console reads hazards, passes, routes, work orders -----


# ---- MCP (Streamable HTTP) - the same tools as `python -m src.dashboard.mcp_server` (stdio) -------
