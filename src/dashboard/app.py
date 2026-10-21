"""
app.py — the FastAPI app that serves the DEPTH studio: middleware, the routers, OGC API, MCP and the UI.

Endpoints (one router per area, ``routes/``; shared state and helpers in ``service.py``)
    system      GET  /api/health · /api/cool (the AWS-signed COOL receipt) · /api/metrics · /api/samples ·
                     /api/sample_thumb/{id} · /api/incidents (the failure catalogue + drills)
    analyze     POST /api/analyze?sample=ID | multipart file — See → Prove → Decide on one frame, with the
                     agent's perception mode priced against the server queue
    survey      POST /api/jobs/survey (background) · GET /api/jobs/{id} · POST /api/survey (≤ MAX_SYNC_FRAMES) ·
                     GET /api/recording · /api/report/{fmt} · /api/brief · POST /api/survey/{id}/decide ·
                     GET /api/survey/{id}/log
    people      /api/feedback · /api/study/* · /api/effort · /api/audit/*
    approvals   GET/POST /api/approvals · POST /api/approvals/{id}/decide (a person, PIN + lockout)
    connect     GET /api/integrations · /api/integrations/webhooks (admin token)
    + /ogc (OGC API – Features, ``ogc.py``) · /mcp (MCP over Streamable HTTP, ``mcp_server.py``) · / (the UI)

Failure handling (``src/agentic/incidents.py``): every failure is an incident with a safe fallback and a
human action; a failed tool sends a find to a person, never auto-confirms; a bad frame is skipped, not
fatal; a missing / implausible position is withheld; an unavailable model answers 503 with the reason.
``?simulate=<code>`` on analyze / survey / brief runs a failure drill so the fallback can be watched.

Hardening:
* endpoints are plain ``def`` (FastAPI runs them in a thread pool) and every inference holds one lock —
  the shared ``cv2.dnn`` network is not thread-safe — so ``/api/health`` always answers while a survey runs;
* long surveys are background **jobs**, bounded and persisted to disk (``jobs.py``);
* uploads are limited (``DEPTH_MAX_UPLOAD_MB`` = 20 MB/file, ``DEPTH_MAX_FRAMES`` = 50, images only);
* per-client rate limits, the origin guard behind CloudFront, gzip for JSON (never for event streams);
  CORS is **off** unless ``DEPTH_CORS_ORIGINS`` is set (the UI is same-origin);
* a server in public mode refuses to start without its tokens (``origin_guard.public_config_errors``);
* the model is loaded and warmed up at startup in the background (``/api/health`` shows when ready).

Run:  uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.routing import Route

from . import ogc as ogc_mod
from . import service as S
from .mcp_server import ENDPOINT as MCP_ENDPOINT
from .origin_guard import OriginGuardMiddleware, public_config_errors
from .ratelimit import RateLimitMiddleware
from .routes import analyze, approvals, connect, people, survey, system


@asynccontextmanager
async def lifespan(_app: FastAPI):
    errs = public_config_errors()                       # a public server never runs with an open door
    if errs:
        raise RuntimeError("DEPTH refuses to start in public mode: " + "; ".join(errs)
                           + " - infra/setup_cool_instance.sh generates them (DEPTH_PUBLIC=0 for a local proxy test)")
    if os.environ.get("DEPTH_LAZY_MODEL", "0") != "1":
        threading.Thread(target=S._warm_in_background, name="depth-warmup", daemon=True).start()
    async with MCP_ENDPOINT.new_manager().run():       # /mcp: a fresh stateless manager per lifespan
        yield
    MCP_ENDPOINT.manager = None


app = FastAPI(title="DEPTH — See→Prove→Decide→Act", version="2.0.0", lifespan=lifespan)
# gzip the JSON (an analyze response carries base64 JPEGs: ~25% smaller); event streams (/mcp) are never
# buffered - Starlette excludes text/event-stream - and CloudFront passes an encoded response through
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
app.add_middleware(RateLimitMiddleware)                # per-client limits on the expensive endpoints
if S._cors:
    app.add_middleware(CORSMiddleware, allow_origins=S._cors, allow_methods=["GET", "POST"], allow_headers=["*"])
app.add_middleware(OriginGuardMiddleware)              # behind CloudFront: only our distribution gets in


@app.middleware("http")
async def _revalidate_static(request, call_next):
    """Browsers must revalidate the zero-build UI on every load (ETag makes it cheap), so a judge
    never sees a stale app.js/styles.css after a deploy."""
    resp = await call_next(request)
    path = request.url.path
    if path.startswith(("/vendor/", "/img/")):          # pinned third-party files + static art: cache a week
        resp.headers.setdefault("Cache-Control", "public, max-age=604800")
    elif not path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-cache")
    return resp


for _r in (system, analyze, survey, people, approvals, connect):
    app.include_router(_r.router)

# ---- OGC API - Features: any GIS / operations console reads hazards, passes, routes, work orders -----
_OGC_SURVEYS = ogc_mod.SurveyIndex(lambda: S._SURVEYS, S._JOBS.persist_dir)
app.include_router(ogc_mod.make_router(_OGC_SURVEYS, lambda: S._APPROVALS))

# ---- MCP (Streamable HTTP) - the same tools as `python -m src.dashboard.mcp_server` (stdio) -------
app.router.routes.append(Route("/mcp", endpoint=MCP_ENDPOINT))
app.router.routes.append(Route("/mcp/", endpoint=MCP_ENDPOINT))

# ---- serve the zero-build frontend -------------------------------------------------------------
if S.WEBUI_DIST.exists():
    app.mount("/", StaticFiles(directory=str(S.WEBUI_DIST), html=True), name="webui")
