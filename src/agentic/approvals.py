"""
approvals.py — human-approval requests: agents may *ask*, only people *decide*.

Any agent (the DEPTH studio, an MCP client such as Claude, or a partner console through the API) can
file a request — "send the boat to these REVIEW cards", "run re-survey pass RS2" — with its rationale.
The request is ``pending`` until a named person approves or declines it in the DEPTH studio. Nothing in
DEPTH executes an action; an approved request is a signed-off work order that downstream consoles
receive (webhook ``approval.decided``).

Storage is an append-only JSONL event log (``runs/approvals/events.jsonl``): every request and every
decision is kept, state is rebuilt by replay — auditable, diffable, never edited in place.

``DEPTH_APPROVER_PIN`` (optional): when set, deciding requires it, so an automated client holding the
API or MCP token still cannot approve its own request.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

ACTIONS = ("inspect", "recover", "resurvey", "share_public", "other")
DECISIONS = ("approved", "declined")
REPO = Path(__file__).resolve().parents[2]


class ApprovalError(ValueError):
    pass


class ApprovalStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or os.environ.get("DEPTH_APPROVALS_DIR") or REPO / "runs" / "approvals")
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "events.jsonl"
        self._lock = threading.Lock()

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
        ev = {"type": "request", "id": f"AR-{uuid.uuid4().hex[:8]}", "t": time.time(),
              "survey_id": str(survey_id)[:80], "action": action, "targets": targets,
              "rationale": rationale[:2000], "requested_by": str(requested_by)[:60], "channel": str(channel)[:20]}
        self._append(ev)
        return self.get(ev["id"])

    def decide(self, rid: str, decision: str, by: str, note: str = "", pin: Optional[str] = None) -> dict:
        want = os.environ.get("DEPTH_APPROVER_PIN")
        if want and pin != want:
            raise PermissionError("approver PIN required - approvals are made by a person in the DEPTH studio")
        if decision not in DECISIONS:
            raise ApprovalError(f"decision must be one of {list(DECISIONS)}")
        by = (by or "").strip()
        if not by:
            raise ApprovalError("the approver's name is required")
        cur = self.get(rid)
        if cur is None:
            raise KeyError(rid)
        if cur["status"] != "pending":
            raise ApprovalError(f"{rid} is already {cur['status']}")
        self._append({"type": "decision", "id": rid, "t": time.time(), "decision": decision,
                      "by": by[:60], "note": (note or "")[:1000]})
        return self.get(rid)

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
            elif ev.get("type") == "decision" and ev.get("id") in state:
                r = state[ev["id"]]
                r["status"] = ev["decision"]
                r["decision"] = {"by": ev["by"], "note": ev.get("note", ""), "t": ev["t"]}
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
        out = {"pending": 0, "approved": 0, "declined": 0}
        for r in self._replay().values():
            out[r["status"]] = out.get(r["status"], 0) + 1
        return out
