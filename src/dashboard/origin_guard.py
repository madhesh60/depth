"""
origin_guard.py — only OUR CloudFront distribution may reach the origin.

The security group admits CloudFront's origin-facing address ranges only, but those ranges are shared by
every CloudFront distribution in the world: anyone could point their own distribution at this server.
CloudFront therefore adds a secret header to every origin request (``X-Origin-Verify``, set in
``infra/cfn/depth.yaml``) and this middleware refuses any request without it — the pattern AWS documents
for restricting access to custom origins.

``DEPTH_ORIGIN_SECRET`` unset (local use, tests) ⇒ the guard is off. Loopback requests (the setup script's
health wait, on-box checks over SSM) are always allowed.
"""
from __future__ import annotations

import hmac
import os

HEADER = b"x-origin-verify"
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


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
