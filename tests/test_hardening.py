"""
Public-deployment hardening: the wrong-PIN lockout, and a public server that refuses to start without
its tokens (judge audit 2026-10-20, §5 "Security").
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic.approvals import PinGuard
from src.dashboard.origin_guard import public_config_errors, public_mode

_TOKENS = {"DEPTH_MCP_TOKEN": "m", "DEPTH_ADMIN_TOKEN": "a", "DEPTH_APPROVER_PIN": "123456"}


def test_five_wrong_pins_lock_the_client_out_even_with_the_right_pin(monkeypatch):
    monkeypatch.setenv("DEPTH_APPROVER_PIN", "123456")
    g = PinGuard(max_fails=5, lock_s=900)
    for i in range(5):
        with pytest.raises(PermissionError, match="PIN required"):
            g.check("1.2.3.4", "000000", now=100.0 + i)
    with pytest.raises(PermissionError, match="locked"):
        g.check("1.2.3.4", "123456", now=110.0)
    g.check("5.6.7.8", "123456", now=110.0)                       # another client is not affected
    g.check("1.2.3.4", "123456", now=110.0 + 901)                 # the lock expires


def test_a_right_pin_clears_the_count(monkeypatch):
    monkeypatch.setenv("DEPTH_APPROVER_PIN", "123456")
    g = PinGuard(max_fails=3, lock_s=900)
    for t in (1.0, 2.0):
        with pytest.raises(PermissionError):
            g.check("c", "x", now=t)
    g.check("c", "123456", now=3.0)
    for t in (4.0, 5.0):
        with pytest.raises(PermissionError, match="PIN required"):
            g.check("c", "x", now=t)                              # not locked: the count restarted


def test_old_failures_age_out(monkeypatch):
    monkeypatch.setenv("DEPTH_APPROVER_PIN", "123456")
    g = PinGuard(max_fails=2, lock_s=60)
    with pytest.raises(PermissionError):
        g.check("c", "x", now=0.0)
    with pytest.raises(PermissionError, match="PIN required"):
        g.check("c", "x", now=100.0)                              # the first failure is older than the window
    g.check("c", "123456", now=101.0)


def test_no_pin_configured_means_nothing_to_guard(monkeypatch):
    monkeypatch.delenv("DEPTH_APPROVER_PIN", raising=False)
    PinGuard(max_fails=1).check("c", None)


def test_local_use_needs_no_tokens():
    assert not public_mode({}) and public_config_errors({}) == []


def test_a_public_server_needs_every_token():
    env = {"DEPTH_TRUST_PROXY": "1"}
    assert public_mode(env)
    errs = public_config_errors(env)
    assert len(errs) == 3 and any("MCP" in e for e in errs) and any("ADMIN" in e for e in errs)
    assert public_config_errors({**env, **_TOKENS}) == []


def test_approvals_can_be_opened_on_purpose_only():
    env = {"DEPTH_PUBLIC": "1", "DEPTH_MCP_TOKEN": "m", "DEPTH_ADMIN_TOKEN": "a"}
    assert any("APPROVER_PIN" in e for e in public_config_errors(env))
    assert public_config_errors({**env, "DEPTH_APPROVALS_OPEN": "1"}) == []


def test_public_mode_can_be_overridden_for_a_local_proxy_test():
    assert not public_mode({"DEPTH_TRUST_PROXY": "1", "DEPTH_PUBLIC": "0"})
    assert public_mode({"DEPTH_ORIGIN_SECRET": "s"})


def test_the_app_refuses_to_start_in_public_mode_without_tokens(monkeypatch):
    from fastapi.testclient import TestClient
    import src.dashboard.app as app_mod
    monkeypatch.setenv("DEPTH_PUBLIC", "1")
    for k in _TOKENS:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="refuses to start in public mode"):
        with TestClient(app_mod.app):
            pass


def test_the_api_locks_out_after_wrong_pins(monkeypatch):
    from fastapi.testclient import TestClient
    import src.dashboard.app as app_mod
    monkeypatch.setenv("DEPTH_APPROVER_PIN", "246810")
    monkeypatch.setattr(app_mod, "_PIN_GUARD", PinGuard(max_fails=2, lock_s=900))
    c = TestClient(app_mod.app)
    body = {"decision": "approved", "by": "A. Person", "pin": "1"}
    assert [c.post("/api/approvals/AR-none/decide", json=body).status_code for _ in range(2)] == [403, 403]
    r = c.post("/api/approvals/AR-none/decide", json={**body, "pin": "246810"})
    assert r.status_code == 403 and "locked" in r.json()["detail"]
