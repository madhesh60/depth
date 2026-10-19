"""
Tests for the per-client rate limits (src/dashboard/ratelimit.py) and the survey-queue cap.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from src.dashboard import app as app_mod
from src.dashboard.ratelimit import RateLimiter, bucket_for, client_of


def test_buckets_cover_the_expensive_endpoints():
    assert bucket_for("GET", "/api/analyze") is None and bucket_for("GET", "/ogc/collections") is None
    assert bucket_for("POST", "/api/analyze") == "infer"
    assert bucket_for("POST", "/api/jobs/survey") == "survey" and bucket_for("POST", "/api/survey") == "survey"
    assert bucket_for("POST", "/api/survey/s-1/decide") == "decide" and bucket_for("POST", "/api/approvals") == "decide"
    assert bucket_for("POST", "/mcp") == "mcp" and bucket_for("DELETE", "/api/integrations/webhooks/x") == "write"


def test_sliding_window(monkeypatch):
    monkeypatch.setenv("DEPTH_RATE_SURVEY", "2")
    rl = RateLimiter()
    assert rl.check("a", "survey", 0.0) == 0 and rl.check("a", "survey", 1.0) == 0
    wait = rl.check("a", "survey", 2.0)
    assert 57 < wait <= 58                                   # the first hit leaves the window at t = 60
    assert rl.check("b", "survey", 2.0) == 0                 # another client is unaffected
    assert rl.check("a", "survey", 60.5) == 0                # the window slid


def test_proxy_header_only_when_trusted(monkeypatch):
    # what the origin receives behind CloudFront: the viewer's own (forgeable) X-Forwarded-For value,
    # then the address CloudFront saw, appended by CloudFront
    scope = {"client": ("10.0.0.5", 1234), "headers": [(b"x-forwarded-for", b"203.0.113.7, 198.51.100.9")]}
    monkeypatch.delenv("DEPTH_TRUST_PROXY", raising=False)
    assert client_of(scope) == "10.0.0.5"                    # a spoofed header is ignored by default
    monkeypatch.setenv("DEPTH_TRUST_PROXY", "1")
    assert client_of(scope) == "198.51.100.9"                # the right-most hop: the one CloudFront added


def test_forged_first_hop_cannot_escape_the_limit(monkeypatch):
    monkeypatch.setenv("DEPTH_TRUST_PROXY", "1")
    monkeypatch.setenv("DEPTH_RATE_SURVEY", "2")
    rl = RateLimiter()
    waits = [rl.check(client_of({"client": ("130.176.1.1", 1), "headers": [
        (b"x-forwarded-for", f"10.9.9.{i}, 198.51.100.9".encode())]}), "survey", float(i)) for i in range(3)]
    assert waits[0] == 0 and waits[1] == 0 and waits[2] > 0  # a new fake first hop each time does not help


def test_operator_token_skips_the_limit(monkeypatch):
    from src.dashboard.ratelimit import is_operator
    monkeypatch.delenv("DEPTH_ADMIN_TOKEN", raising=False)
    assert not is_operator({"headers": [(b"x-depth-admin", b"")]})   # no token configured: nobody is operator
    monkeypatch.setenv("DEPTH_ADMIN_TOKEN", "s3cret")
    assert is_operator({"headers": [(b"x-depth-admin", b"s3cret")]})
    assert not is_operator({"headers": [(b"x-depth-admin", b"wrong")]})
    monkeypatch.setenv("DEPTH_RATE_LIMITS", "1")
    monkeypatch.setenv("DEPTH_RATE_DECIDE", "1")
    c = TestClient(app_mod.app)
    codes = [c.post("/api/approvals", json={}, headers={"X-DEPTH-Admin": "s3cret"}).status_code for _ in range(3)]
    assert 429 not in codes


def test_middleware_returns_429_with_retry_after(monkeypatch):
    monkeypatch.setenv("DEPTH_RATE_LIMITS", "1")
    monkeypatch.setenv("DEPTH_RATE_DECIDE", "2")
    c = TestClient(app_mod.app)
    codes = [c.post("/api/approvals", json={}).status_code for _ in range(3)]
    assert codes[:2] == [400, 400] and codes[2] == 429      # 400 = reached the handler (empty body)
    r = c.post("/api/approvals", json={})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert c.get("/api/health").status_code == 200           # reads are never limited


def test_survey_queue_cap(monkeypatch):
    monkeypatch.setenv("DEPTH_MAX_QUEUED_JOBS", "0")
    c = TestClient(app_mod.app)
    r = c.post("/api/jobs/survey?use_samples=1")
    assert r.status_code == 503 and "queue is full" in r.json()["detail"]
