"""
app.py — FastAPI service exposing the See → Prove → Decide → Act loop to the web dashboard.

Endpoints
    GET  /api/health                 — liveness, OpenCV version + WHERE cv2 was loaded from (COOL
                                       provenance), runtime, model state, calibration promises, jobs
    GET  /api/samples                — shipped sample frames (webui/samples, CC-BY-SA, attributed)
    POST /api/analyze?sample=ID      — See→Prove→Decide on a sample or ONE uploaded frame → evidence
         (multipart file=…)            cards + traces   [?nadir=top|bottom|left|right overrides]
    POST /api/jobs/survey            — queue a survey (samples or ≤ MAX_FRAMES uploads) → {job_id}
    GET  /api/jobs/{job_id}          — status + progress; the survey result when done
    POST /api/survey?use_samples=1   — synchronous survey for small runs (≤ MAX_SYNC_FRAMES)
    GET  /api/report/{fmt}?survey_id — geojson | gpx | kml | csv | json (memory, else disk)
    GET  /api/metrics                — rolling per-stage p50/p95 on THIS host + arch / EC2 type / COOL

Hardening (review §3.8 / I-9):
* endpoints are plain ``def`` (FastAPI runs them in a thread pool) and every inference is guarded by
  one lock — the shared ``cv2.dnn`` network is not thread-safe — so ``/api/health`` always answers
  while a survey runs;
* long surveys are background **jobs** (no 30 s proxy timeouts), bounded and persisted to disk
  (``jobs.py``) so report downloads survive a restart;
* uploads are limited (``DEPTH_MAX_UPLOAD_MB`` = 20 MB/file, ``DEPTH_MAX_FRAMES`` = 50, images only);
* recent surveys are kept in a bounded LRU; CORS is **off** unless ``DEPTH_CORS_ORIGINS`` is set
  (the UI is same-origin);
* the model is loaded + warmed up at startup in the background (``/api/health`` shows when ready).

Run:  uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import logging
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src.agentic.pipeline import AgenticPipeline
from src.agentic.agent import render
from src.agentic.perception import Perceptor
from src.agentic.shadow import ShadowProver
from src.agentic.geo import synthetic_track
from src.agentic.mission import export
from src.agentic.types import SurveyResult
from src.detection.calibration import load_calibration
from src.detection.infer import configure_runtime, DEFAULT_ONNX
from . import samples as samples_mod
from .jobs import JobStore, REPORT_FORMATS
from .metrics import Metrics

log = logging.getLogger("depth.api")
REPO = Path(__file__).resolve().parents[2]
_WEBUI = REPO / "webui"
WEBUI_DIST = _WEBUI / "dist" if (_WEBUI / "dist").exists() else _WEBUI

MAX_UPLOAD_MB = float(os.environ.get("DEPTH_MAX_UPLOAD_MB", "20"))
MAX_FRAMES = int(os.environ.get("DEPTH_MAX_FRAMES", "50"))
MAX_SYNC_FRAMES = int(os.environ.get("DEPTH_MAX_SYNC_FRAMES", "12"))
MAX_SURVEYS = int(os.environ.get("DEPTH_MAX_SURVEYS", "24"))
_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp", "application/octet-stream", ""}

# ---- shared state ----------------------------------------------------------------------------
_PIPELINE: Optional[AgenticPipeline] = None
_MODEL_STATE = {"loaded": False, "loading": False, "error": None, "warmup_ms": None}
_INFER_LOCK = threading.Lock()                  # one inference at a time on the shared cv2.dnn net
_LOAD_LOCK = threading.Lock()
_PROVER = ShadowProver()
_SURVEYS: "OrderedDict[str, SurveyResult]" = OrderedDict()   # bounded LRU (report downloads)
_JOBS = JobStore(max_jobs=50, workers=1)
_RUNTIME = configure_runtime()
_METRICS = Metrics()


def get_pipeline() -> AgenticPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        with _LOAD_LOCK:
            if _PIPELINE is None:
                _MODEL_STATE["loading"] = True
                try:
                    pipe = AgenticPipeline()
                    t0 = time.perf_counter()
                    pipe.agent.perceptor.warmup()
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("DEPTH_LAZY_MODEL", "0") != "1":
        threading.Thread(target=_warm_in_background, name="depth-warmup", daemon=True).start()
    yield


app = FastAPI(title="DEPTH — See→Prove→Decide→Act", version="2.0.0", lifespan=lifespan)
_cors = [o.strip() for o in os.environ.get("DEPTH_CORS_ORIGINS", "").split(",") if o.strip()]
if _cors:
    app.add_middleware(CORSMiddleware, allow_origins=_cors, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def _revalidate_static(request, call_next):
    """Browsers must revalidate the zero-build UI on every load (ETag makes it cheap), so a judge
    never sees a stale app.js/styles.css after a deploy."""
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-cache")
    return resp


# ---- helpers ----------------------------------------------------------------------------------
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


def _decode_upload(f: UploadFile) -> tuple[str, np.ndarray]:
    """Validate type + size, then decode. Raises HTTPException with a clear message."""
    if (f.content_type or "").lower() not in _IMAGE_TYPES and not (f.content_type or "").startswith("image/"):
        raise HTTPException(415, f"{f.filename}: only images are accepted (got {f.content_type})")
    data = f.file.read(int(MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"{f.filename}: larger than {MAX_UPLOAD_MB:g} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"{f.filename}: could not decode image")
    return Path(f.filename or "upload").stem, img


def _frame_payload(frame: np.ndarray, result) -> dict:
    d = result.to_dict()
    d["frame_png"] = _b64(frame)
    d["overlay_png"] = _b64(render(frame, result))
    for cand, cd in zip(result.candidates, d["candidates"]):
        cd["crop_png"] = _evidence_crop(frame, cand)
        rv = Perceptor.relook_view(frame, cand.bbox)      # the agent's-eye zoom (display-only)
        if rv:
            cd["relook_view"] = {
                "zoom_png": _b64(rv["zoom"]), "enhanced_png": _b64(rv["enhanced"]),
                "scale": round(rv["scale"], 2), "obj_box": rv["obj_box"],
            }
    return d


def _sample_frames() -> list[tuple[str, np.ndarray]]:
    out = []
    for s in samples_mod.list_samples():
        p = samples_mod.sample_path(s["id"])
        img = cv2.imread(str(p)) if p and p.exists() else None
        if img is not None:
            out.append((p.stem, img))
    return out


def _collect_frames(use_samples: bool, files: Optional[list[UploadFile]]) -> list[tuple[str, np.ndarray]]:
    if use_samples:
        frames = _sample_frames()
    else:
        files = files or []
        if len(files) > MAX_FRAMES:
            raise HTTPException(413, f"at most {MAX_FRAMES} frames per survey (got {len(files)})")
        frames = [_decode_upload(f) for f in files]
    if not frames:
        raise HTTPException(400, "no frames (use ?use_samples=1 or upload files)")
    return frames


def _remember(result: SurveyResult) -> None:
    _SURVEYS[result.survey_id] = result
    _SURVEYS.move_to_end(result.survey_id)
    while len(_SURVEYS) > MAX_SURVEYS:
        _SURVEYS.popitem(last=False)


def _run_survey(frames, gps: str, budget_minutes: Optional[float], nadir: Optional[str],
                survey_id: str, progress: Optional[Callable[[dict], None]] = None) -> dict:
    track = synthetic_track([fid for fid, _ in frames]) if gps == "synthetic" else None
    pipe = get_pipeline()
    total = len(frames)
    done = {"n": 0}

    def cb(stage: str, payload: dict) -> None:
        if stage == "frame_done":
            done["n"] += 1
        if progress:
            progress({"done": done["n"], "total": total, "stage": stage,
                      "frame": payload.get("frame_id")})

    result = pipe.run_survey(frames, track=track, survey_id=survey_id, nadir=nadir,
                             budget_minutes=budget_minutes, progress_cb=cb,
                             frame_lock=_INFER_LOCK)      # per frame: /api/analyze can interleave
    _remember(result)
    for fr in result.frames:
        _METRICS.record("survey", fr.frame_id, fr.stage_ms, counts=fr.counts)
    frame_map = dict(frames)
    d = result.to_dict()
    for fr, fd in zip(result.frames, d["frames"]):          # light thumbnails, not full-res base64
        thumb = cv2.resize(render(frame_map[fr.frame_id], fr), (200, 200), interpolation=cv2.INTER_AREA)
        fd["thumb_png"] = _b64(thumb, 70)
    reports = {fmt: export(fmt, result) for fmt in REPORT_FORMATS}
    _JOBS.persist(result.survey_id, {k: v for k, v in d.items() if k != "frames"}, reports)
    return d


# ---- API ---------------------------------------------------------------------------------------
@app.get("/api/health")
def health():
    cal = load_calibration()
    summary = cal.summary()
    summary["detector_floor"] = cal.conf.get(cal.guaranteed_class)
    cv2_file = getattr(cv2, "__file__", "") or ""
    return {
        "status": "ok", "opencv": cv2.__version__,
        "cv2_file": cv2_file, "is_cool_path": "/opt/cool" in cv2_file,
        "runtime": _RUNTIME, "model": DEFAULT_ONNX.parent.parent.name,
        "model_loaded": _MODEL_STATE["loaded"], "model_loading": _MODEL_STATE["loading"],
        "model_error": _MODEL_STATE["error"], "warmup_ms": _MODEL_STATE["warmup_ms"],
        "samples": len(samples_mod.list_samples()), "calibration": summary, "jobs": _JOBS.counts(),
        "limits": {"max_upload_mb": MAX_UPLOAD_MB, "max_frames": MAX_FRAMES, "max_sync_frames": MAX_SYNC_FRAMES},
    }


@app.get("/api/metrics")
def metrics():
    """Rolling per-stage latency on this server + host / OpenCV (COOL) provenance."""
    out = _METRICS.summary()
    out["runtime"] = _RUNTIME
    return out


@app.get("/api/samples")
def samples():
    return {"samples": samples_mod.list_samples()}


@app.get("/api/sample_thumb/{sample_id}")
def sample_thumb(sample_id: str, size: int = Query(220, ge=32, le=640)):
    """Downscaled JPEG of a sample frame — for the gallery cards (real sonar, not a placeholder)."""
    p = samples_mod.sample_path(sample_id)
    if not p or not p.exists():
        raise HTTPException(404, f"sample not found: {sample_id}")
    img = cv2.imread(str(p))
    if img is None:
        raise HTTPException(400, "could not read sample")
    scale = size / max(1, max(img.shape[:2]))
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise HTTPException(500, "encode failed")
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.post("/api/analyze")
def analyze(sample: Optional[str] = Query(None), nadir: Optional[str] = Query(None),
            file: Optional[UploadFile] = File(None)):
    if file is not None:
        frame_id, frame = _decode_upload(file)
    elif sample:
        p = samples_mod.sample_path(sample)
        if not p or not p.exists():
            raise HTTPException(404, f"sample not found: {sample}")
        frame, frame_id = cv2.imread(str(p)), p.stem
        if frame is None:
            raise HTTPException(400, "could not decode image")
    else:
        raise HTTPException(400, "provide ?sample=ID or a multipart file")
    if nadir and nadir not in ("top", "bottom", "left", "right", "auto"):
        raise HTTPException(400, "nadir must be top|bottom|left|right|auto")

    t0 = time.perf_counter()
    pipe = get_pipeline()
    with _INFER_LOCK:
        result = pipe.run_frame(frame, frame_id=frame_id, nadir=nadir)
    payload = _frame_payload(frame, result)
    payload["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    payload["guarantees"] = pipe.guarantees()
    _METRICS.record("analyze", frame_id, result.stage_ms, payload["wall_ms"], result.counts)
    return JSONResponse(payload)


@app.post("/api/jobs/survey")
def submit_survey(use_samples: bool = Query(False), gps: str = Query("synthetic"),
                  budget_minutes: Optional[float] = Query(None, ge=0, le=600),
                  nadir: Optional[str] = Query(None),
                  files: Optional[list[UploadFile]] = File(None)):
    frames = _collect_frames(use_samples, files)
    survey_id = f"survey-{time.strftime('%Y%m%d-%H%M%S')}-{len(frames)}f"
    jid = _JOBS.submit(lambda progress: _run_survey(frames, gps, budget_minutes, nadir, survey_id, progress),
                       total=len(frames))
    return {"job_id": jid, "survey_id": survey_id, "frames": len(frames)}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    rec = _JOBS.get(job_id)
    if rec is None:
        raise HTTPException(404, "unknown job_id")
    return rec


@app.post("/api/survey")
def survey(use_samples: bool = Query(False), gps: str = Query("synthetic"),
           budget_minutes: Optional[float] = Query(None, ge=0, le=600),
           nadir: Optional[str] = Query(None),
           files: Optional[list[UploadFile]] = File(None)):
    frames = _collect_frames(use_samples, files)
    if len(frames) > MAX_SYNC_FRAMES:
        raise HTTPException(413, f"{len(frames)} frames: use POST /api/jobs/survey for more than "
                                 f"{MAX_SYNC_FRAMES} (runs in the background, poll /api/jobs/<id>)")
    survey_id = f"survey-{time.strftime('%Y%m%d-%H%M%S')}-{len(frames)}f"
    return JSONResponse(_run_survey(frames, gps, budget_minutes, nadir, survey_id))


_MEDIA = {"geojson": "application/geo+json", "gpx": "application/gpx+xml",
          "kml": "application/vnd.google-earth.kml+xml", "csv": "text/csv", "json": "application/json"}


@app.get("/api/report/{fmt}")
def report(fmt: str, survey_id: str = Query(...)):
    if fmt not in _MEDIA:
        raise HTTPException(400, f"format must be one of {list(_MEDIA)}")
    body = export(fmt, _SURVEYS[survey_id]) if survey_id in _SURVEYS else _JOBS.report(survey_id, fmt)
    if body is None:
        raise HTTPException(404, "unknown survey_id (run a survey first)")
    return Response(body, media_type=_MEDIA[fmt],
                    headers={"Content-Disposition": f'attachment; filename="mission.{fmt}"'})


# ---- serve the zero-build frontend -------------------------------------------------------------
if WEBUI_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEBUI_DIST), html=True), name="webui")
