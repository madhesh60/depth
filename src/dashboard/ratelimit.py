"""
ratelimit.py — per-client limits on the expensive endpoints, so one visitor cannot slow the public
demo for everyone else (judges included).

Sliding one-minute windows per (client, bucket). Buckets and default limits (requests / minute / client):

    infer    POST /api/analyze · /api/study/…            DEPTH_RATE_INFER   (30)
    survey   POST /api/survey · /api/jobs/survey          DEPTH_RATE_SURVEY  (6)
    decide   POST /api/survey/*/decide · /api/approvals*  DEPTH_RATE_DECIDE  (60)
    mcp      POST /mcp                                     DEPTH_RATE_MCP     (60)
    write    other POST / DELETE                           DEPTH_RATE_WRITE   (60)

GET requests (pages, reports, OGC, health) are not limited. Over the limit → 429 with Retry-After.
The client is the socket peer, or — only when ``DEPTH_TRUST_PROXY=1`` (behind CloudFront) — the first
``X-Forwarded-For`` address. ``DEPTH_RATE_LIMITS=0`` switches limiting off (tests, local use).
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

DEFAULTS = {"infer": 30, "survey": 6, "decide": 60, "mcp": 60, "write": 60}
WINDOW_S = 60.0


def bucket_for(method: str, path: str):
    if method not in ("POST", "PUT", "PATCH", "DELETE"):
        return None
    if path.startswith("/api/analyze") or path.startswith("/api/study/"):
        return "infer"
    if path in ("/api/survey", "/api/jobs/survey"):
        return "survey"
    if path.startswith("/api/approvals") or (path.startswith("/api/survey/") and path.endswith("/decide")):
        return "decide"
    if path.rstrip("/") == "/mcp":
        return "mcp"
    return "write"


def limit_for(bucket: str) -> int:
    return int(os.environ.get(f"DEPTH_RATE_{bucket.upper()}", DEFAULTS[bucket]))


class RateLimiter:
    def __init__(self):
        self._hits: dict[tuple, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client: str, bucket: str, now: float | None = None) -> float:
        """0 if allowed (and counted); otherwise seconds until the next request would be allowed."""
        now = time.monotonic() if now is None else now
        lim = limit_for(bucket)
        with self._lock:
            q = self._hits[(client, bucket)]
            while q and now - q[0] >= WINDOW_S:
                q.popleft()
            if len(q) >= lim:
                return max(0.1, WINDOW_S - (now - q[0]))
            q.append(now)
            if len(self._hits) > 20_000:                       # bounded memory under a flood of clients
                for k in [k for k, v in self._hits.items() if not v][:10_000]:
                    self._hits.pop(k, None)
            return 0.0


def client_of(scope: dict) -> str:
    if os.environ.get("DEPTH_TRUST_PROXY") == "1":
        xff = dict(scope.get("headers") or []).get(b"x-forwarded-for", b"").decode()
        if xff:
            return xff.split(",")[0].strip()
    peer = scope.get("client")
    return peer[0] if peer else "unknown"


class RateLimitMiddleware:
    """Pure ASGI middleware (works for FastAPI routes and the mounted /mcp endpoint alike)."""

    def __init__(self, app, limiter: RateLimiter | None = None):
        self.app, self.limiter = app, limiter or RateLimiter()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or os.environ.get("DEPTH_RATE_LIMITS", "1") == "0":
            return await self.app(scope, receive, send)
        b = bucket_for(scope.get("method", "GET"), scope.get("path", ""))
        if b is None:
            return await self.app(scope, receive, send)
        wait = self.limiter.check(client_of(scope), b)
        if wait:
            body = (f'{{"detail":"rate limit: at most {limit_for(b)} {b} requests per minute per client - '
                    f'retry in {int(wait) + 1} s"}}').encode()
            await send({"type": "http.response.start", "status": 429,
                        "headers": [(b"content-type", b"application/json"), (b"retry-after", str(int(wait) + 1).encode())]})
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)
