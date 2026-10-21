"""
origin_guard.py — only OUR CloudFront distribution may reach the origin.

The security group admits CloudFront's origin-facing address ranges only, but those ranges are shared by
every CloudFront distribution in the world: anyone could point their own distribution at this server.
CloudFront therefore adds a secret header to every origin request (``X-Origin-Verify``, set in
``infra/cfn/depth.yaml``) and this middleware refuses any request without it — the pattern AWS documents
for restricting access to custom origins.

``DEPTH_ORIGIN_SECRET`` unset (local use, tests) ⇒ the guard is off. Loopback requests (the setup script's
health wait, on-box checks over SSM) are always allowed.

A public server also refuses to START without its tokens (:func:`public_config_errors`): locally every
token is optional and its feature is open (fail-open is fine on a laptop); on a public address an unset
token would leave ``/mcp``, webhook management or the approvals open to anyone, so a missing one is a
configuration error, not a default. Public = ``DEPTH_PUBLIC=1``, or behind the proxy
(``DEPTH_TRUST_PROXY=1``, set by ``infra/depth.service``), or an origin secret is set; ``DEPTH_PUBLIC=0``
overrides (a local proxy test).
"""
from __future__ import annotations

import hmac
import os
from typing import Mapping

HEADER = b"x-origin-verify"
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def public_mode(env: Mapping[str, str] = os.environ) -> bool:
    if env.get("DEPTH_PUBLIC") in ("0", "1"):
        return env["DEPTH_PUBLIC"] == "1"
    return env.get("DEPTH_TRUST_PROXY") == "1" or bool(env.get("DEPTH_ORIGIN_SECRET"))


def public_config_errors(env: Mapping[str, str] = os.environ) -> list[str]:
    """What a public server is missing (empty = fine, or not public)."""
    if not public_mode(env):
        return []
    errs = []
    if not env.get("DEPTH_MCP_TOKEN"):
        errs.append("DEPTH_MCP_TOKEN is not set (the /mcp bearer token)")
    if not env.get("DEPTH_ADMIN_TOKEN"):
        errs.append("DEPTH_ADMIN_TOKEN is not set (webhook management would be open)")
    if not env.get("DEPTH_APPROVER_PIN") and env.get("DEPTH_APPROVALS_OPEN") != "1":
        errs.append("DEPTH_APPROVER_PIN is not set (set it, or DEPTH_APPROVALS_OPEN=1 to open approvals on purpose)")
    return errs


class OriginGuardMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        secret = os.environ.get("DEPTH_ORIGIN_SECRET", "")
        if scope["type"] != "http" or not secret:
            return await self.app(scope, receive, send)
        peer = (scope.get("client") or ("",))[0]
        got = dict(scope.get("headers") or []).get(HEADER, b"").decode(errors="replace")
        if peer in _LOOPBACK or hmac.compare_digest(got, secret):
            return await self.app(scope, receive, send)
        await send({"type": "http.response.start", "status": 403,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body",
                    "body": b'{"detail":"direct origin access is not allowed - use the public HTTPS address"}'})
