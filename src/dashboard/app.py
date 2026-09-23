"""
app.py — FastAPI service exposing the See → Prove → Decide → Act loop to the web dashboard.

Endpoints
    GET  /api/health                 — liveness + OpenCV version + model status (shown in the footer)
    GET  /api/samples                — curated sample frames (one-click demo, no sonar files needed)
    POST /api/analyze?sample=ID      — run See→Prove→Decide on a sample or an uploaded frame; returns
         (multipart file=…)            the annotated frame, per-candidate evidence cards + traces
    POST /api/survey?use_samples=1   — run the full loop over many frames → hazards + mission plan
    GET  /api/report/{fmt}?survey_id — download the mission as geojson | gpx | kml | csv | json

The built React app in ``webui/dist`` is served at ``/`` when present. The model (ONNX) is loaded
lazily on the first analyze so ``/api/health`` stays instant.

Run:  uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src.agentic.pipeline import AgenticPipeline
from src.agentic.agent import render
from src.agentic.shadow import ShadowProver
from src.agentic.geo import synthetic_track
from src.agentic.mission import export
from src.agentic.types import SurveyResult
from . import samples as samples_mod

REPO = Path(__file__).resolve().parents[2]
WEBUI_DIST = REPO / "webui" / "dist"

app = FastAPI(title="Marine Debris — See→Prove→Decide→Act", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_PIPELINE: Optional[AgenticPipeline] = None
_PROVER = ShadowProver()
_SURVEYS: dict[str, SurveyResult] = {}          # in-memory cache so report downloads work
_MODEL_OK = False


def get_pipeline() -> AgenticPipeline:
    global _PIPELINE, _MODEL_OK
    if _PIPELINE is None:
        _PIPELINE = AgenticPipeline()
        _MODEL_OK = True
    return _PIPELINE


# ---- helpers ----------------------------------------------------------------------------------
def _b64(img: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


def _evidence_crop(frame: np.ndarray, cand, target: int = 240) -> str:
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
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return _b64(crop)


def _read_upload(data: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _frame_payload(frame: np.ndarray, result) -> dict:
    d = result.to_dict()
    d["frame_png"] = _b64(frame)
    d["overlay_png"] = _b64(render(frame, result))
    for cand, cd in zip(result.candidates, d["candidates"]):
        cd["crop_png"] = _evidence_crop(frame, cand)
    return d


# ---- API ---------------------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "opencv": cv2.__version__, "model_loaded": _MODEL_OK,
            "samples": len(samples_mod.list_samples())}


@app.get("/api/samples")
def samples():
    return {"samples": samples_mod.list_samples()}


@app.post("/api/analyze")
async def analyze(sample: Optional[str] = Query(None), file: Optional[UploadFile] = File(None)):
    if file is not None:
        frame = _read_upload(await file.read())
        frame_id = Path(file.filename or "upload").stem
    elif sample:
        p = samples_mod.sample_path(sample)
        if not p or not p.exists():
            raise HTTPException(404, f"sample not found: {sample}")
        frame = cv2.imread(str(p)); frame_id = p.stem
    else:
        raise HTTPException(400, "provide ?sample=ID or a multipart file")
    if frame is None:
        raise HTTPException(400, "could not decode image")

    t0 = time.perf_counter()
    result = get_pipeline().run_frame(frame, frame_id=frame_id)
    payload = _frame_payload(frame, result)
    payload["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return JSONResponse(payload)


@app.post("/api/survey")
async def survey(use_samples: bool = Query(False), gps: str = Query("synthetic"),
                 files: Optional[list[UploadFile]] = File(None)):
    frames: list[tuple[str, np.ndarray]] = []
    if use_samples:
        for s in samples_mod.list_samples():
            p = samples_mod.sample_path(s["id"])
            if p and p.exists():
                frames.append((p.stem, cv2.imread(str(p))))
    elif files:
        for f in files:
            img = _read_upload(await f.read())
            if img is not None:
                frames.append((Path(f.filename or "frame").stem, img))
    frames = [(fid, im) for fid, im in frames if im is not None]
    if not frames:
        raise HTTPException(400, "no frames (use ?use_samples=1 or upload files)")

    track = synthetic_track([fid for fid, _ in frames]) if gps == "synthetic" else None
    survey_id = f"survey-{int(time.time())}"
    result = get_pipeline().run_survey(frames, track=track, survey_id=survey_id)
    _SURVEYS[survey_id] = result

    frame_map = dict(frames)
    d = result.to_dict()
    # light per-frame thumbnails for the gallery (not the full-res base64)
    for fr, fd in zip(result.frames, d["frames"]):
        thumb = cv2.resize(render(frame_map[fr.frame_id], fr), (200, 200), interpolation=cv2.INTER_AREA)
        fd["thumb_png"] = _b64(thumb, 70)
    return JSONResponse(d)


_MEDIA = {"geojson": "application/geo+json", "gpx": "application/gpx+xml",
          "kml": "application/vnd.google-earth.kml+xml", "csv": "text/csv", "json": "application/json"}


@app.get("/api/report/{fmt}")
def report(fmt: str, survey_id: str = Query(...)):
    if survey_id not in _SURVEYS:
        raise HTTPException(404, "unknown survey_id (run /api/survey first)")
    if fmt not in _MEDIA:
        raise HTTPException(400, f"format must be one of {list(_MEDIA)}")
    body = export(fmt, _SURVEYS[survey_id])
    ext = fmt                                  # geojson→.geojson, json→.json, etc.
    return Response(body, media_type=_MEDIA[fmt],
                    headers={"Content-Disposition": f'attachment; filename="mission.{ext}"'})


# ---- serve the built frontend (if present) -----------------------------------------------------
if WEBUI_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEBUI_DIST), html=True), name="webui")
else:
    @app.get("/")
    def _root():
        return {"status": "ok", "note": "frontend not built yet — run `npm run build` in webui/",
                "api": ["/api/health", "/api/samples", "/api/analyze", "/api/survey", "/api/report/{fmt}"]}
