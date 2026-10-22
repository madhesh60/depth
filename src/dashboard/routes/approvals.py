"""
routes/approvals.py — approval requests: agents ask, a named person decides (PIN + lockout); approvals re-
plan or issue work orders.
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request

from ..ratelimit import client_of
from src.agentic.approvals import ApprovalError
from src.agentic.types import SurveyResult
from .. import service as S

router = APIRouter()


# ---- human-approval requests: agents ask, people decide (src/agentic/approvals.py) ---------------
@router.get("/api/approvals")
def approvals(status: Optional[str] = Query(None, pattern="^(pending|approved|declined|withdrawn)$"),
              survey_id: Optional[str] = Query(None)):
    return {"requests": S._APPROVALS.list(status=status, survey_id=survey_id), "counts": S._APPROVALS.counts()}


@router.post("/api/approvals")
def approval_request(body: dict = Body(...)):
    """File a request (from the studio, a partner console, or an agent). It only ASKS."""
    try:
        r = S._APPROVALS.request(body.get("survey_id", ""), body.get("action", ""), body.get("targets") or [],
                                 body.get("rationale", ""), requested_by=body.get("requested_by", "api"), channel="api")
    except ApprovalError as e:
        raise HTTPException(400, str(e)) from e
    S._notify("approval.requested", r)
    return r


@router.post("/api/approvals/{rid}/decide")
def approval_decide(rid: str, request: Request, body: dict = Body(...)):
    """A person's decision, made in the DEPTH studio. Needs ``DEPTH_APPROVER_PIN`` when that is set (with a
    wrong-PIN lockout per client). A request is decided once: of two decisions at the same moment one is
    accepted and the other gets 409 with who decided first."""
    try:
        S._PIN_GUARD.check(client_of(request.scope), body.get("pin"))
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e
    with S._DECIDE_LOCK:                  # the decision and the re-plan it causes are one step
        return _decide(rid, body)


def _decide(rid: str, body: dict) -> dict:
    try:
        r = S._APPROVALS.decide(rid, body.get("decision", ""), body.get("by", ""), body.get("note", ""), body.get("pin"))
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e
    except KeyError:
        raise HTTPException(404, "unknown request id") from None
    except ApprovalError as e:
        raise HTTPException(409 if "already" in str(e) else 400, str(e)) from e
    S._notify("approval.decided", r)
    s = S._SURVEYS.get(r.get("survey_id"))
    if s is not None and rid in (s.plan_args.get("requests") or {}).values():
        if r["status"] == "declined" and r["action"] == "resurvey":
            # human control: a declined pass makes the agent change its plan, visibly
            from src.agentic.mission import decline_resurvey
            by = (r.get("decision") or {}).get("by") or body.get("by", "")
            entry = decline_resurvey(s, r["targets"], by, body.get("note", ""), rid)
            entry["requests"] = S._sync_agent_requests(s, reason=f"{by} declined the pass over {', '.join(r['targets'][:4])}")
            d = s.to_dict()
            d.pop("frames", None)
            r = {**r, "replan": entry, "tracked": d["tracked"], "mission": d["mission"]}
        elif r["status"] == "approved":
            r = {**r, "work_order": _work_order_issued(s, r)}
        r = {**r, "dispatch": S._dispatch(s)}
    return r


def _work_order_issued(s: SurveyResult, r: dict) -> dict:
    """An approval is a signed-off WORK ORDER, not a re-plan: DEPTH never drives the boat. The plan
    stands as approved; it changes when the crew reports back (recovered / not found on the route
    stops) or when the pass's sonar data is surveyed. Logged so the timeline says exactly that."""
    by = (r.get("decision") or {}).get("by") or "?"
    next_step = {"resurvey": "upload the pass's sonar recording as a new survey; the agent then re-scores these targets",
                 "inspect": "mark each stop confirmed / rejected in Hazards; the agent re-plans after each",
                 "recover": "mark each stop recovered / not found; the agent re-plans after each"}.get(
        r["action"], "report the outcome in the studio; the agent re-plans from it")
    wo = {"request": r["id"], "action": r["action"], "targets": len(r["targets"]), "approved_by": by,
          "sent_to": "webhook approval.decided" + (" + OGC work_orders" if r["action"] != "share_public" else ""),
          "plan_changed": False, "next": next_step}
    s.mission.agent_log.append({"tool": "work_order", "status": "done",
                                "rationale": f"{by} approved '{r['action']}' ({r['id']}, {len(r['targets'])} target(s)) "
                                             f"-> work order issued; the plan stands as approved. Next: {next_step}",
                                "detail": wo})
    s.plan_args.setdefault("events", []).append(
        {"t": time.time(), "actor": "person", "kind": "work_order", "by": by,
         "title": f"{r['action']} approved - work order issued",
         "text": f"DEPTH does not dispatch the boat; consoles receive the signed work order. Next: {next_step}"})
    return wo
