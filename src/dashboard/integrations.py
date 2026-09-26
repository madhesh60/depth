"""
integrations.py — signed webhooks: DEPTH pushes events to a partner's operations console / ERP /
ticketing system (an ocean institute's mission control, a cleanup NGO's dispatch board).

Events: ``survey.completed`` · ``approval.requested`` · ``approval.decided`` (an approved decision is a
work order: targets with positions) · ``ping``.

Every delivery is a JSON POST signed like Stripe / GitHub webhooks, so the receiver can prove it came
from this DEPTH server and was not replayed:

    X-DEPTH-Event: approval.decided
    X-DEPTH-Delivery: <uuid>
    X-DEPTH-Signature: t=<unix seconds>,v1=<hex HMAC-SHA256(secret, f"{t}.{raw body}")>

Deliveries run on a background pool with 3 attempts (1 s, 4 s back-off) and are logged (last 200).
Safety: only http(s); hosts that resolve to private / loopback / link-local / reserved addresses (e.g.
the EC2 metadata service) are refused at registration AND at delivery, unless
``DEPTH_WEBHOOK_ALLOW_PRIVATE=1`` (on-premise consoles, tests). Secrets are shown once, stored in
``runs/integrations/webhooks.json`` (git-ignored).
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import os
import secrets
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

log = logging.getLogger("depth.integrations")
REPO = Path(__file__).resolve().parents[2]
EVENTS = ("survey.completed", "approval.requested", "approval.decided", "ping")
BACKOFF_S = (1.0, 4.0)


class WebhookError(ValueError):
    pass


def check_url(url: str) -> None:
    """Refuse non-http(s) URLs and hosts that resolve to non-public addresses (SSRF guard)."""
    u = urlparse(url or "")
    if u.scheme not in ("http", "https") or not u.hostname:
        raise WebhookError("webhook URL must be http(s)://host/…")
    if os.environ.get("DEPTH_WEBHOOK_ALLOW_PRIVATE") == "1":
        return
    try:
        addrs = {ai[4][0] for ai in socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80))}
    except socket.gaierror:
        raise WebhookError(f"cannot resolve {u.hostname}")
    for a in addrs:
        ip = ipaddress.ip_address(a.split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise WebhookError(f"{u.hostname} resolves to a non-public address ({ip}); set "
                               f"DEPTH_WEBHOOK_ALLOW_PRIVATE=1 for on-premise consoles")


def sign(secret: str, t: int, body: bytes) -> str:
    return hmac.new(secret.encode(), f"{t}.".encode() + body, hashlib.sha256).hexdigest()


def verify(secret: str, header: str, body: bytes, tolerance_s: int = 300) -> bool:
    """What a receiver runs: check the signature and reject stale (replayed) deliveries."""
    try:
        parts = dict(p.split("=", 1) for p in header.split(","))
        t = int(parts["t"])
    except (ValueError, KeyError):
        return False
    return abs(time.time() - t) <= tolerance_s and hmac.compare_digest(sign(secret, t, body), parts.get("v1", ""))


class Webhooks:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or os.environ.get("DEPTH_INTEGRATIONS_DIR") or REPO / "runs" / "integrations")
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "webhooks.json"
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="depth-webhook")
        self.deliveries: deque = deque(maxlen=200)

    # -- registry ---------------------------------------------------------------------------------
    def _load(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def _save(self, hooks: list[dict]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(hooks, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def list(self) -> list[dict]:
        return [{**{k: v for k, v in h.items() if k != "secret"}, "secret": h["secret"][:6] + "…"} for h in self._load()]

    def add(self, url: str, events: Optional[list[str]] = None, description: str = "") -> dict:
        check_url(url)
        events = list(events or ["*"])
        bad = [e for e in events if e != "*" and e not in EVENTS]
        if bad:
            raise WebhookError(f"unknown events {bad}; use {list(EVENTS)} or '*'")
        hook = {"id": f"wh_{uuid.uuid4().hex[:10]}", "url": url[:500], "events": events,
                "description": (description or "")[:200], "secret": "whsec_" + secrets.token_hex(24),
                "active": True, "created": time.time()}
        with self._lock:
            hooks = self._load()
            if len(hooks) >= 25:
                raise WebhookError("at most 25 webhooks")
            hooks.append(hook)
            self._save(hooks)
        return hook                                    # the only time the full secret is returned

    def remove(self, hid: str) -> bool:
        with self._lock:
            hooks = self._load()
            keep = [h for h in hooks if h["id"] != hid]
            self._save(keep)
        return len(keep) != len(hooks)

    # -- delivery ---------------------------------------------------------------------------------
    def emit(self, event: str, data: dict) -> int:
        """Queue a delivery to every active webhook subscribed to ``event``; returns how many."""
        hooks = [h for h in self._load() if h.get("active") and ("*" in h["events"] or event in h["events"])]
        for h in hooks:
            self._pool.submit(self._deliver, h, event, data)
        return len(hooks)

    def send_now(self, hid: str, event: str = "ping", data: Optional[dict] = None) -> dict:
        h = next((x for x in self._load() if x["id"] == hid), None)
        if h is None:
            raise KeyError(hid)
        return self._deliver(h, event, data or {"message": "DEPTH webhook test"}, retries=False)

    def _deliver(self, h: dict, event: str, data: dict, retries: bool = True) -> dict:
        did = str(uuid.uuid4())
        body = json.dumps({"id": did, "event": event, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "data": data}, default=str).encode()
        rec = {"delivery": did, "hook": h["id"], "event": event, "url": h["url"], "t": time.time()}
        attempts = (0.0,) + (BACKOFF_S if retries else ())
        for i, wait in enumerate(attempts, 1):
            if wait:
                time.sleep(wait)
            t0 = time.perf_counter()
            try:
                check_url(h["url"])                    # re-checked at delivery (DNS may have changed)
                t = int(time.time())
                req = urllib.request.Request(h["url"], data=body, method="POST", headers={
                    "Content-Type": "application/json", "User-Agent": "DEPTH-webhooks/1",
                    "X-DEPTH-Event": event, "X-DEPTH-Delivery": did,
                    "X-DEPTH-Signature": f"t={t},v1={sign(h['secret'], t, body)}"})
                with urllib.request.urlopen(req, timeout=6) as resp:
                    code = resp.status
                rec.update(status="delivered" if 200 <= code < 300 else "failed", code=code, attempt=i,
                           ms=round((time.perf_counter() - t0) * 1000, 1))
                if 200 <= code < 300:
                    break
            except urllib.error.HTTPError as e:
                rec.update(status="failed", code=e.code, attempt=i, error=f"HTTP {e.code}")
                if 400 <= e.code < 500 and e.code != 429:
                    break                              # the receiver rejected it; retrying will not help
            except (WebhookError, urllib.error.URLError, OSError) as e:
                rec.update(status="failed", code=None, attempt=i, error=f"{type(e).__name__}: {e}"[:200])
        self.deliveries.appendleft(rec)
        log.info("webhook %s %s -> %s", event, h["id"], rec.get("status"))
        return rec
