"""
approvals.py — human-approval requests: agents may *ask*, only people *decide*.

Any agent (the DEPTH studio, an MCP client such as Claude, or a partner console through the API) can
file a request — "send the boat to these REVIEW cards", "run re-survey pass RS2" — with its rationale.
The request is ``pending`` until a named person approves or declines it in the DEPTH studio. Nothing in
DEPTH executes an action; an approved request is a signed-off work order that downstream consoles
receive (webhook ``approval.decided``).

Storage is an append-only JSONL event log (``runs/approvals/events.jsonl``): every request and every
decision is kept, state is rebuilt by replay — auditable, diffable, never edited in place. A request is
decided once: deciding or withdrawing checks and writes under one lock, so two people deciding the same
request at the same moment get one success and one "already decided" (409), and on replay the first
decision stands — a later one never silently replaces it.

``DEPTH_APPROVER_PIN`` (optional): when set, deciding requires it, so an automated client holding the
API or MCP token still cannot approve its own request. :class:`PinGuard` locks a client out after
``DEPTH_PIN_MAX_FAILS`` (5) wrong PINs within ``DEPTH_PIN_LOCK_S`` (900 s) — a 6-digit PIN then takes
years to guess from one address, where the per-minute rate limit alone allowed ~11 days.
"""
from __future__ import annotations

import hmac
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

ACTIONS = ("inspect", "recover", "resurvey", "full_review", "share_public", "other")
DECISIONS = ("approved", "declined")
STATUSES = ("pending", "approved", "declined", "withdrawn")
REPO = Path(__file__).resolve().parents[2]


class ApprovalError(ValueError):
    pass


def _already(r: dict) -> str:
    """Why a request cannot be decided again: its status and who closed it."""
    by = (r.get("decision") or {}).get("by")
    return f"{r['id']} is already {r['status']}" + (f" by {by}" if by else "")


def pin_required() -> bool:
    return bool(os.environ.get("DEPTH_APPROVER_PIN"))


def pin_ok(pin: Optional[str]) -> bool:
    """True when no PIN is configured, or ``pin`` matches it (constant-time compare)."""
    want = os.environ.get("DEPTH_APPROVER_PIN")
    return not want or hmac.compare_digest(str(pin or "").encode(), want.encode())


class PinGuard:
    """Wrong-PIN lockout per client: ``max_fails`` wrong PINs within ``lock_s`` lock that client out for
    ``lock_s``, even with the right PIN; a right PIN clears the count. No PIN configured = nothing to guard."""

    def __init__(self, max_fails: Optional[int] = None, lock_s: Optional[float] = None):
        self.max_fails = int(os.environ.get("DEPTH_PIN_MAX_FAILS", "5")) if max_fails is None else max_fails
        self.lock_s = float(os.environ.get("DEPTH_PIN_LOCK_S", "900")) if lock_s is None else lock_s
        self._fails: dict[str, list[float]] = {}
        self._locked: dict[str, float] = {}
        self._lock = threading.Lock()

    def check(self, client: str, pin: Optional[str], now: Optional[float] = None) -> None:
        """Raise :class:`PermissionError` when the client is locked out or the PIN is wrong."""
        if not pin_required():
            return
        now = time.monotonic() if now is None else now
        with self._lock:
            until = self._locked.get(client, 0.0)
            if until > now:
                raise PermissionError(f"too many wrong approver PINs from this address - locked for another "
                                      f"{int(until - now) + 1} s")
            if pin_ok(pin):
                self._fails.pop(client, None)
                return
            q = [t for t in self._fails.get(client, []) if now - t < self.lock_s] + [now]
            self._fails[client] = q
            if len(q) >= self.max_fails:
                self._locked[client], self._fails[client] = now + self.lock_s, []
            if len(self._fails) + len(self._locked) > 10_000:   # bounded memory under a flood of clients
                for k in list(self._fails)[:5_000]:
                    self._fails.pop(k, None)
                for k in [k for k, t in self._locked.items() if t <= now]:
                    self._locked.pop(k, None)
        raise PermissionError("approver PIN required - approvals are made by a person in the DEPTH studio")


class ApprovalStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or os.environ.get("DEPTH_APPROVALS_DIR") or REPO / "runs" / "approvals")
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "events.jsonl"
        self._lock = threading.RLock()            # held from the pending check to the write (decide / withdraw)

    # -- write -----------------------------------------------------------------------------------
    def _append(self, ev: dict) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, default=str) + "\n")

    def request(self, survey_id: str, action: str, targets: list[str], rationale: str,
                requested_by: str = "agent", channel: str = "api") -> dict:
        if action not in ACTIONS:
            raise ApprovalError(f"action must be one of {list(ACTIONS)}")
        rationale = (rationale or "").strip()
        if len(rationale) < 8:
            raise ApprovalError("a rationale is required (why should a person approve this?)")
        targets = [str(t)[:24] for t in (targets or [])][:200]
        rid = f"AR-{uuid.uuid4().hex[:8]}"
        ev = {"type": "request", "id": rid, "t": time.time(),
              "survey_id": str(survey_id)[:80], "action": action, "targets": targets,
              "rationale": rationale[:2000], "requested_by": str(requested_by)[:60], "channel": str(channel)[:20]}
        self._append(ev)
        return self._current(rid)

    def decide(self, rid: str, decision: str, by: str, note: str = "", pin: Optional[str] = None) -> dict:
        if not pin_ok(pin):
            raise PermissionError("approver PIN required - approvals are made by a person in the DEPTH studio")
        if decision not in DECISIONS:
            raise ApprovalError(f"decision must be one of {list(DECISIONS)}")
        by = (by or "").strip()
        if not by:
            raise ApprovalError("the approver's name is required")
        return self._close(rid, "decision", {"decision": decision, "by": by[:60], "note": (note or "")[:1000]})

    def withdraw(self, rid: str, reason: str, by: str = "DEPTH agent") -> dict:
        """The requester retracts a PENDING request (e.g. the agent re-planned and the route it asked
        for no longer exists). Never touches a request a person already decided."""
        return self._close(rid, "withdrawn", {"by": str(by)[:60], "note": (reason or "")[:1000]})

    def _close(self, rid: str, kind: str, fields: dict) -> dict:
        """Decide or withdraw a PENDING request: the check and the write are one step under the lock, so
        of two at the same moment one succeeds and the other gets "already ..." (never both)."""
        with self._lock:
            cur = self.get(rid)
            if cur is None:
                raise KeyError(rid)
            if cur["status"] != "pending":
                raise ApprovalError(_already(cur))
            t = time.time()
            self._append({"type": kind, "id": rid, "t": t, **fields})
            r = self._current(rid)
        if (r.get("decision") or {}).get("t") != t:              # another process wrote first: the first stands
            raise ApprovalError(_already(r))
        return r

    def _current(self, rid: str) -> dict:
        """The state of a request this store has just written (it is always there: the log is append-only)."""
        r = self.get(rid)
        if r is None:                                             # pragma: no cover - a write that did not land
            raise KeyError(rid)
        return r

    # -- read ------------------------------------------------------------------------------------
    def _replay(self) -> dict[str, dict]:
        state: dict[str, dict] = {}
        if not self.path.exists():
            return state
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "request":
                state[ev["id"]] = {**{k: v for k, v in ev.items() if k != "type"}, "status": "pending", "decision": None}
                continue
            r = state.get(ev.get("id"))
            if r is None or r["status"] != "pending":
                continue                                # decided once: a later decision never replaces the first
            if ev.get("type") == "decision":
                r["status"] = ev["decision"]
                r["decision"] = {"by": ev["by"], "note": ev.get("note", ""), "t": ev["t"]}
            elif ev.get("type") == "withdrawn":
                r["status"] = "withdrawn"
                r["decision"] = {"by": ev.get("by", "DEPTH agent"), "note": ev.get("note", ""), "t": ev["t"]}
        return state

    def get(self, rid: str) -> Optional[dict]:
        return self._replay().get(rid)

    def list(self, status: Optional[str] = None, survey_id: Optional[str] = None, limit: int = 100) -> list[dict]:
        rs = sorted(self._replay().values(), key=lambda r: -r["t"])
        if status:
            rs = [r for r in rs if r["status"] == status]
        if survey_id:
            rs = [r for r in rs if r["survey_id"] == survey_id]
        return rs[:limit]

    def counts(self) -> dict:
        out = {"pending": 0, "approved": 0, "declined": 0, "withdrawn": 0}
        for r in self._replay().values():
            out[r["status"]] = out.get(r["status"], 0) + 1
        return out
