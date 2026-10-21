"""
routes/connect.py — the Connect page's overview and signed-webhook management (admin token).
"""
from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, Body, HTTPException, Request

from ..integrations import EVENTS as WEBHOOK_EVENTS, WebhookError
from .. import service as S

router = APIRouter()


# ---- integrations: signed webhooks + the Connect page's overview -----------------------------------
def _admin(request: Request) -> None:
    """Webhook management needs ``X-DEPTH-Admin: $DEPTH_ADMIN_TOKEN`` when that token is set."""
    want = os.environ.get("DEPTH_ADMIN_TOKEN")
    if want and not hmac.compare_digest(request.headers.get("x-depth-admin", ""), want):
        raise HTTPException(403, "admin token required (X-DEPTH-Admin)")


@router.get("/api/integrations")
def integrations(request: Request):
    base = S.PUBLIC_URL or str(request.base_url).rstrip("/")
    return {"base_url": base,
            "mcp": {"url": f"{base}/mcp", "transport": "streamable-http", "token_required": bool(os.environ.get("DEPTH_MCP_TOKEN")),
                    "stdio": "python -m src.dashboard.mcp_server"},
            "ogc": {"landing": f"{base}/ogc", "collections": ["hazards", "resurvey_passes", "routes", "work_orders"],
                    "public_forced": os.environ.get("DEPTH_OGC_PUBLIC", "0") == "1"},
            "webhooks": {"events": list(WEBHOOK_EVENTS), "count": len(S._WEBHOOKS.list()),
                         "admin_token_required": bool(os.environ.get("DEPTH_ADMIN_TOKEN"))},
            "embed": {"script": f"{base}/embed/depth-embed.js"},
            "openapi": f"{base}/openapi.json", "docs": f"{base}/docs",
            "cors_origins": S._cors}


@router.get("/api/integrations/webhooks")
def webhooks_list(request: Request):
    _admin(request)
    return {"webhooks": S._WEBHOOKS.list(), "deliveries": list(S._WEBHOOKS.deliveries)[:30], "events": list(WEBHOOK_EVENTS)}


@router.post("/api/integrations/webhooks")
def webhooks_add(request: Request, body: dict = Body(...)):
    _admin(request)
    try:
        return S._WEBHOOKS.add(body.get("url", ""), body.get("events"), body.get("description", ""))
    except WebhookError as e:
        raise HTTPException(400, str(e)) from e


@router.delete("/api/integrations/webhooks/{hid}")
def webhooks_remove(hid: str, request: Request):
    _admin(request)
    if not S._WEBHOOKS.remove(hid):
        raise HTTPException(404, "unknown webhook")
    return {"removed": hid}


@router.post("/api/integrations/webhooks/{hid}/test")
def webhooks_test(hid: str, request: Request):
    _admin(request)
    try:
        return S._WEBHOOKS.send_now(hid)
    except KeyError:
        raise HTTPException(404, "unknown webhook") from None
