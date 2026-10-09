"""
mission.py — the **Act** stage: turn confirmed hazards into a human-approved cleanup plan.

* CONFIRMED objects → a **recovery route** (greedy nearest-neighbour ordering — "always go to the
  nearest next pot", which is enough for a small survey).
* REVIEW objects → a **resurvey list** for a human to revisit (nothing is dispatched from REVIEW).
* Exports the plan as GeoJSON / GPX / KML / CSV / JSON that a boat crew's GPS or a GIS can open.

Nothing is ever auto-dispatched: ``MissionPlan.human_approval_required`` is always True.

The planner is an agent too, and it keeps a decision log (``MissionPlan.agent_log``): triage → "can
another observation change these cards?" → analyst-budget check + stop rule → person overrides →
inspection / recovery route → re-survey passes chosen *and skipped* against the boat budget → the
dispatch gate. After every person's decision it re-plans and records what changed
(``MissionPlan.replans``: queue, routes, passes — before → after). Geometry
(route/waypoints) is only emitted when GPS is available; without it the plan degrades honestly to a
frame-ordered CSV and a "no GPS" flag. When the GPS is a demo track, exports carry a clear
``SYNTHETIC DEMO GPS`` marker.
"""
from __future__ import annotations

import copy
import csv
import io
import json
from typing import Optional

from .geo import haversine_m
from .types import MissionPlan, TrackedObject, Verdict, SurveyResult

_SYNTH_WARN = "SYNTHETIC DEMO GPS - not real coordinates"
# Responsible use (docs/responsible_use.md): wrecks can be war graves / protected heritage sites.
# In a PUBLIC share their positions are generalised to ~1.1 km (0.01°) and re-survey passes that would
# reveal them are dropped; cleanup targets (ghost gear) keep full precision.
SENSITIVE_CLASSES = {"structural_fragment", "wreck_debris"}
_GENERAL_DEG = 0.01
_REDACT_NOTE = "public share: protected-site locations generalised to ~1.1 km (0.01 deg); full precision on request"
# Seconds an analyst spends per REVIEW card. ASSUMED until the timed user study replaces it
# (docs/user_study.md); every budget plan says so.
DEFAULT_SEC_PER_CARD = 8.0


def _nearest_neighbour(points: list[TrackedObject]) -> list[TrackedObject]:
    """Greedy NN route over geolocated objects, starting from the south-west-most."""
    pts = [p for p in points if p.lat is not None and p.lon is not None]
    if len(pts) <= 2:
        return pts
    remaining = pts[:]
    cur = min(remaining, key=lambda p: (p.lat, p.lon))
    remaining.remove(cur)
    route = [cur]
    while remaining:
        nxt = min(remaining, key=lambda p: haversine_m((cur.lat, cur.lon), (p.lat, p.lon)))
        remaining.remove(nxt)
        route.append(nxt)
        cur = nxt
    return route


HUMAN_DECISIONS = ("confirm", "reject", "recovered", "not_found", "undo",
                   "prioritize", "deprioritize", "reset_priority")    # the last three override the agent's order
_PRIORITY_RANK = {"high": 0, None: 1, "low": 2}
_ACTION_RANK = {"accept": 0, "review": 1, None: 1, "watch": 2}
_CLOSED = ("rejected", "not_found", "recovered")         # a person closed it: off every route and queue


def to_recover(t: TrackedObject) -> bool:
    """On the recovery route: auto-CONFIRMED under the precision promise, or confirmed by a person —
    and not yet recovered / rejected / not found."""
    return t.human not in _CLOSED and (t.verdict is Verdict.CONFIRMED or t.human == "confirmed")


def undecided_review(t: TrackedObject) -> bool:
    return t.verdict is Verdict.REVIEW and t.human is None


def build_mission(tracked: list[TrackedObject], gps_available: bool,
                  gps_synthetic: bool = False) -> MissionPlan:
    confirmed = [t for t in tracked if to_recover(t)]
    review = [t for t in tracked if undecided_review(t)]

    if gps_available:
        ordered = _nearest_neighbour(confirmed)
        route_ids = [t.oid for t in ordered]
        length = sum(haversine_m((a.lat, a.lon), (b.lat, b.lon))
                     for a, b in zip(ordered, ordered[1:])) if len(ordered) > 1 else 0.0
        length = round(length, 1)
    else:
        # no GPS → deterministic frame-ordered plan, no geometry
        route_ids = [t.oid for t in sorted(confirmed, key=lambda t: (t.frame_id, t.oid))]
        length = None

    counts = {v.value: sum(t.verdict is v for t in tracked) for v in Verdict}
    return MissionPlan(
        recovery_route=route_ids, resurvey=[t.oid for t in review], route_length_m=length,
        gps_available=gps_available, gps_synthetic=gps_synthetic,
        human_approval_required=True, counts=counts,
    )


def review_queue(tracked: list[TrackedObject]) -> list[TrackedObject]:
    """Undecided REVIEW cards, most-likely-real first: calibrated P(pot), then evidence score, then
    confidence. Cards a person already decided leave the queue."""
    rev = [t for t in tracked if undecided_review(t)]
    return sorted(rev, key=lambda t: (_PRIORITY_RANK.get(t.human_priority, 1),        # a person's override first
                                      0 if t.fallback == "inspect" else 1,           # a declined pass -> look here
                                      _ACTION_RANK.get(t.action, 1),                  # the agent's evidence action
                                      -(t.p_evidence if t.p_evidence is not None else
                                        (t.p_pot if t.p_pot is not None else -1.0)),
                                      -t.evidence_score, -t.conf, t.oid))


def plan_inspection(mission: MissionPlan, tracked: list[TrackedObject], ids: list[str]) -> None:
    """Nearest-neighbour INSPECTION route through the REVIEW cards an analyst/boat should check first
    (the budgeted, most-likely-real ones). Distinct from the recovery route: nothing on it is a
    confirmed hazard — it is where a human (or a re-survey) must look. Needs GPS; mutates ``mission``."""
    if not mission.gps_available:
        mission.inspection_route, mission.inspection_length_m = list(ids), None
        return
    idx = _by_id(tracked)
    ordered = _nearest_neighbour([idx[i] for i in ids if i in idx])
    mission.inspection_route = [t.oid for t in ordered]
    mission.inspection_length_m = round(sum(haversine_m((a.lat, a.lon), (b.lat, b.lon))
                                            for a, b in zip(ordered, ordered[1:])), 1) if len(ordered) > 1 else 0.0


def plan_budget(queue: list[TrackedObject], minutes: float,
                sec_per_card: float = DEFAULT_SEC_PER_CARD, measured: bool = False) -> dict:
    """Budget mode: given the analyst's minutes, which cards to review first and what that buys.

    Expected real pots = Σ calibrated P(pot) over the cards reviewed. Returned figures are
    expectations under the calibration, not guarantees."""
    k = max(0, int(minutes * 60 // max(1e-6, sec_per_card)))
    ps = [t.p_pot for t in queue if t.p_pot is not None]
    in_budget = [t.p_pot for t in queue[:k] if t.p_pot is not None]
    total = float(sum(ps))
    got = float(sum(in_budget))
    return {
        "minutes": minutes, "sec_per_card": sec_per_card,
        "sec_per_card_source": "measured (user study)" if measured else "ASSUMED - replace with the timed user study",
        "cards_total": len(queue), "cards_affordable": min(k, len(queue)),
        "review_ids": [t.oid for t in queue[:k]],
        "expected_pots_in_budget": round(got, 2), "expected_pots_in_queue": round(total, 2),
        "share_of_expected_pots": round(got / total, 3) if total > 0 else None,
        "minutes_for_whole_queue": round(len(queue) * sec_per_card / 60, 1),
    }


def impact_ledger(tracked: list[TrackedObject], log: list) -> dict:
    """What people decided and what crews did — and the precision of the agent's REVIEW queue as
    measured by those decisions (Clopper-Pearson 95% interval: a live, audited number)."""
    from .guarantees import cp_lower, cp_upper
    n = {k: sum(t.human == k for t in tracked) for k in ("confirmed", "rejected", "recovered", "not_found")}
    real = n["confirmed"] + n["recovered"]
    judged = real + n["rejected"] + n["not_found"]
    return {**{f"person_{k}": v for k, v in n.items()},
            "open_review": sum(undecided_review(t) for t in tracked),
            "on_recovery_route": sum(to_recover(t) for t in tracked),
            "decisions": len(log),
            "reviewed_precision": round(real / judged, 3) if judged else None,
            "reviewed_precision_ci95": [round(cp_lower(real, judged), 3), round(cp_upper(real, judged), 3)] if judged else None,
            "note": "precision of the cards people have decided so far - not a promise for the rest of the queue"}


def plan_mission(tracked: list[TrackedObject], track: Optional[dict], guarantees: dict,
                 budget_minutes: Optional[float] = None, sec_per_card: float = DEFAULT_SEC_PER_CARD,
                 boat_minutes: Optional[float] = None, repeat_merges: int = 0,
                 human_log: Optional[list] = None, replans: Optional[list] = None,
                 incidents: Optional[list] = None, declined: Optional[list] = None) -> MissionPlan:
    """The whole Act plan from the current state of the hazards — used for the first plan AND after
    every person's decision (the agent re-plans; it never overrides a person)."""
    from .resurvey import plan_resurvey
    declined_ids = {oid for d in (declined or []) for oid in d.get("targets", [])}
    for t in tracked:                       # a person declined the pass over it -> inspect it instead
        t.fallback = "inspect" if (t.oid in declined_ids and t.human is None) else None
    gps_available = bool(track)
    gps_synthetic = gps_available and any(f.synthetic for f in track.values())
    mission = build_mission(tracked, gps_available, gps_synthetic)
    mission.repeat_merges = repeat_merges
    queue = review_queue(tracked)
    mission.review_queue = [t.oid for t in queue]
    mission.guarantees = guarantees
    if budget_minutes is not None:
        mission.budget = plan_budget(queue, budget_minutes, sec_per_card)
    plan_inspection(mission, tracked, mission.budget.get("review_ids") if mission.budget else mission.review_queue)
    if gps_available:                        # the physical second look: opposite side, mid-swath
        mission.resurvey_plan = plan_resurvey([t for t in tracked if t.human is None], track, boat_minutes,
                                              exclude=declined_ids)
        mission.resurvey_plan["declined"] = list(declined or [])
    mission.human_log = list(human_log or [])
    mission.replans = list(replans or [])
    mission.impact = impact_ledger(tracked, mission.human_log)
    mission.agent_log = geometry_log(track, tracked) + planner_log(tracked, mission, queue, budget_minutes, boat_minutes)
    mission.incidents = [i for i in (incidents or []) if i.get("code") != "missing_gps"]
    if not gps_available:
        from .incidents import incident
        mission.incidents.append(incident("missing_gps", "no GPS fix for any frame of this survey"))
    return mission


def _step(tool: str, why: str, status: str = "done", **detail) -> dict:
    return {"tool": tool, "status": status, "rationale": why, "detail": detail}


def geometry_log(track: Optional[dict], tracked: list[TrackedObject]) -> list[dict]:
    """A raw recording's seabed geometry, as the agent's sensor cross-check resolved it per chunk
    (``sensor_check.py``), and what that changed for the hazards: ground range and metres from the
    trusted source, or a widened error radius where nothing coherent was found."""
    st = {fid: f.geometry for fid, f in (track or {}).items() if getattr(f, "geometry", None)}
    if not st:
        return []
    n = {k: sum(v == k for v in st.values()) for k in ("agree", "corrected", "recovered", "image_trusted", "not_measured")}
    by = {}
    for t in tracked:
        by.setdefault(st.get(t.frame_id), []).append(t.oid)
    changed = by.get("corrected", []) + by.get("recovered", []) + by.get("image_trusted", [])
    why = (f"{len(st)} chunk(s): OpenCV seabed track and depth sounder agree on {n['agree']}; "
           f"{n['corrected'] + n['recovered']} re-tracked where the steady sounder pointed; {n['image_trusted']} "
           f"trusted the image where the sounder lost lock; {n['not_measured']} not measured")
    out = [_step("sensor_crosscheck", why, **n)]
    if changed:
        out.append(_step("geometry_update", f"{len(changed)} hazard(s) placed with geometry the cross-check corrected "
                                            f"({', '.join(changed[:6])}{'…' if len(changed) > 6 else ''}) - ground range and "
                                            f"heights follow the trusted source, not the tracker's first pick",
                         hazards=changed))
    if by.get("not_measured"):
        out.append(_step("geometry_withheld", f"{len(by['not_measured'])} hazard(s) on chunks with no coherent seabed "
                                              f"({', '.join(by['not_measured'][:6])}): slant range, error radius widened by the "
                                              f"altitude bound, no heights in metres", hazards=by["not_measured"]))
    return out


def planner_log(tracked: list[TrackedObject], m: MissionPlan, queue: list[TrackedObject],
                budget_minutes: Optional[float], boat_minutes: Optional[float]) -> list[dict]:
    """The Act-stage agent's decisions, in order, each with its reason (shown in the studio's Agent tab
    and written to the decision log). Deterministic: the same survey gives the same log."""
    L = []
    c = m.counts
    L.append(_step("triage", f"{len(tracked)} hazard(s): {c.get('confirmed', 0)} confirmed, {c.get('review', 0)} for "
                             f"review, {c.get('low_risk', 0)} low-risk kept for audit",
                   counts=dict(c)))
    ps = [t.p_pot for t in queue if t.p_pot is not None]
    unc = round(sum(p * (1 - p) for p in ps), 2)
    L.append(_step("voi_check", (f"can another observation change these {len(queue)} card(s)? yes - open "
                                 f"uncertainty sum p(1-p) = {unc}; options: a person's card "
                                 f"(~{(m.budget or {}).get('sec_per_card', DEFAULT_SEC_PER_CARD):g} s each) or an "
                                 f"opposite-side sonar pass (boat time)") if queue else
                   "no open REVIEW cards - nothing left that another observation could change",
                   status="done", open_uncertainty=unc, cards=len(queue)))
    b = m.budget or {}
    if b:
        k, n = b["cards_affordable"], b["cards_total"]
        nxt = queue[k] if k < len(queue) else None
        why = (f"analyst budget {b['minutes']:g} min / {b['sec_per_card']:g} s per card -> {k} of {n} card(s) fit; "
               f"expected {b['expected_pots_in_budget']} of {b['expected_pots_in_queue']} real pots in the queue")
        if b.get("share_of_expected_pots") is not None:
            why += f" ({b['share_of_expected_pots']:.0%})"
        L.append(_step("budget_check", why, cards_affordable=k, cards_total=n,
                       sec_per_card_source=b.get("sec_per_card_source")))
        if nxt is not None and nxt.p_pot is not None:
            L.append(_step("stop_rule", f"stop at card {k}: the next card ({nxt.oid}) is worth ~{nxt.p_pot:.2f} expected pots "
                                        f"and does not fit the budget; {n - k} card(s) deferred, lowest P(pot) first",
                           deferred=n - k, next_p_pot=nxt.p_pot))
        else:
            L.append(_step("stop_rule", "the whole queue fits the analyst budget - nothing deferred", deferred=0))
    else:
        L.append(_step("budget_check", "no analyst budget given -> the whole queue is the plan", status="skipped"))
    hi = [t.oid for t in tracked if t.human_priority == "high"]
    lo = [t.oid for t in tracked if t.human_priority == "low"]
    if hi or lo:
        L.append(_step("human_override", f"a person's order is kept: {len(hi)} prioritised"
                                         + (f" ({', '.join(hi[:6])})" if hi else "")
                                         + f", {len(lo)} deprioritised - the agent never re-orders them",
                       prioritized=hi, deprioritized=lo))
    if m.gps_available:
        L.append(_step("route_update", f"inspection route through {len(m.inspection_route)} card(s), "
                                       f"{m.inspection_length_m or 0:g} m, nearest-neighbour order",
                       stops=len(m.inspection_route), length_m=m.inspection_length_m))
    else:
        L.append(_step("route_update", "no GPS -> no route drawn; the cards are listed in frame order (positions withheld)",
                       status="skipped", stops=len(m.inspection_route)))
    if m.recovery_route:
        L.append(_step("recovery_route", f"{len(m.recovery_route)} confirmed stop(s), "
                                         f"{m.route_length_m if m.route_length_m is not None else '?'} m",
                       stops=len(m.recovery_route)))
    else:
        L.append(_step("recovery_route", "empty - nothing is confirmed (no precision promise for this model, and no "
                                         "person has confirmed a card yet)", status="skipped", stops=0))
    rp = m.resurvey_plan or {}
    if m.gps_available:
        lines, skipped = rp.get("lines") or [], rp.get("skipped") or []
        if lines:
            L.append(_step("resurvey_plan", f"{len(lines)} opposite-side pass(es) planned, "
                                            f"{rp.get('boat_minutes_planned')} boat-min"
                                            + (f" of {boat_minutes:g}" if boat_minutes is not None else "")
                                            + f"; they resolve {rp.get('voi_covered')} of {rp.get('voi_total')} open "
                                              f"uncertainty (the shadow must flip on a real object)",
                           passes=[x["id"] for x in lines]))
        elif not skipped:
            L.append(_step("resurvey_plan", rp.get("note") or "no pass needed", status="skipped"))
        for sk in skipped:
            L.append(_step("resurvey_plan", f"pass over {', '.join(sk['targets'][:4])} not planned - {sk['reason']}",
                           status="skipped", targets=sk["targets"], voi=sk["voi"]))
    for d in rp.get("declined") or []:
        moved = [oid for oid in d.get("targets", []) if oid in m.review_queue]
        n_left = len(rp.get("lines") or [])
        L.append(_step("human_declined", f"{d.get('by', 'a person')} declined the opposite-side pass over "
                                         f"{', '.join(d.get('targets', [])[:5])} -> the agent dropped it (never proposed "
                                         f"again), re-ranked the boat time over the remaining {n_left} pass(es) and moved "
                                         f"{len(moved)} target(s) to the front of the inspection queue - a person looks "
                                         f"there instead",
                       targets=d.get("targets", []), by=d.get("by")))
    lines = rp.get("lines") or []
    if lines and rp.get("ranking"):
        L.append(_step("resurvey_rank", rp["ranking"], status="done"))
    n_pass = len(lines)
    L.append(_step("dispatch_gate", f"nothing dispatched: the inspection route ({len(m.inspection_route)} stops), "
                                    f"the recovery route ({len(m.recovery_route)} stops) and {n_pass} re-survey "
                                    f"pass(es) wait for a named person's approval", status="done",
                   awaiting_approval=True))
    return L


def plan_summary(m: MissionPlan) -> dict:
    """The figures a re-plan compares (before -> after)."""
    rp = m.resurvey_plan or {}
    return {"queue": len(m.review_queue), "inspection_stops": len(m.inspection_route),
            "inspection_m": m.inspection_length_m, "recovery_stops": len(m.recovery_route),
            "recovery_m": m.route_length_m, "passes": len(rp.get("lines") or []),
            "cards_affordable": (m.budget or {}).get("cards_affordable")}


def apply_human(survey: SurveyResult, oid: str, decision: str, by: str, note: str = "") -> TrackedObject:
    """Record a PERSON's decision on one hazard and re-plan. The agent's verdict is kept as it was;
    the decision is its own field and its own log line. Rules: confirm / reject an undecided card
    (or an auto-CONFIRMED one); recovered / not_found only after it was on the recovery route; undo
    re-opens it."""
    import time as _t
    if decision not in HUMAN_DECISIONS:
        raise ValueError(f"decision must be one of {list(HUMAN_DECISIONS)}")
    by = (by or "").strip()
    if not by:
        raise ValueError("the person's name is required")
    t = next((x for x in survey.tracked if x.oid == oid), None)
    if t is None:
        raise KeyError(oid)
    if decision in ("confirm", "reject") and t.human is not None:
        raise ValueError(f"{oid} is already {t.human} - undo first")
    if decision in ("recovered", "not_found") and not to_recover(t):
        raise ValueError(f"{oid} is not on the recovery route (confirm it first)")
    if decision == "undo" and t.human is None:
        raise ValueError(f"{oid} has no decision to undo")
    if decision in ("prioritize", "deprioritize") and t.human is not None:
        raise ValueError(f"{oid} is already {t.human} - it is not in the queue")
    if decision == "reset_priority" and t.human_priority is None:
        raise ValueError(f"{oid} has no priority override to reset")
    old = survey.mission
    if decision in ("prioritize", "deprioritize", "reset_priority"):
        before = t.human_priority
        t.human_priority = {"prioritize": "high", "deprioritize": "low", "reset_priority": None}[decision]
        after = t.human_priority
    else:
        before = t.human
        t.human = {"confirm": "confirmed", "reject": "rejected", "recovered": "recovered",
                   "not_found": "not_found", "undo": None}[decision]
        t.human_by = by[:60] if t.human else None
        after = t.human
    now = round(_t.time(), 1)
    log = list(old.human_log) + [{"t": now, "hazard": oid, "decision": decision,
                                  "before": before, "after": after, "by": by[:60], "note": (note or "")[:300],
                                  "agent_verdict": t.verdict.value, "p_pot": t.p_pot}]
    _replan(survey, log, oid, decision, by, now)
    return t


def decline_resurvey(survey: SurveyResult, targets: list[str], by: str, note: str = "", request_id: str = "") -> dict:
    """A PERSON declined the agent's opposite-side pass over ``targets``. The agent must change its
    plan: the pass is never proposed again, the boat time goes to the next most informative pass,
    and the declined targets move to the front of the inspection queue (a person looks instead)."""
    import time as _t
    by = (by or "").strip()
    if not by:
        raise ValueError("the person's name is required")
    a = survey.plan_args
    a.setdefault("declined_resurvey", []).append({"targets": list(targets), "by": by[:60], "request": request_id,
                                                  "note": (note or "")[:300]})
    now = round(_t.time(), 1)
    log = list(survey.mission.human_log) + [{"t": now, "hazard": ",".join(targets[:6]), "decision": "decline_resurvey",
                                             "before": "pass planned", "after": "declined", "by": by[:60],
                                             "note": (note or "")[:300], "agent_verdict": "review", "p_pot": None}]
    return _replan(survey, log, ",".join(targets[:6]), "decline_resurvey", by, now)


def _replan(survey: SurveyResult, log: list, oid: str, decision: str, by: str, now: float) -> dict:
    old = survey.mission
    a = survey.plan_args or {}
    new = plan_mission(survey.tracked, survey.track, old.guarantees,
                       a.get("budget_minutes"), a.get("sec_per_card", DEFAULT_SEC_PER_CARD),
                       a.get("boat_minutes"), old.repeat_merges, log, old.replans, old.incidents,
                       declined=a.get("declined_resurvey"))
    b0, b1 = plan_summary(old), plan_summary(new)
    changed = {k: [b0[k], b1[k]] for k in b0 if b0[k] != b1[k]}
    words = {"queue": "queue", "inspection_stops": "inspection stops", "inspection_m": "inspection m",
             "recovery_stops": "recovery stops", "recovery_m": "recovery m", "passes": "re-survey passes",
             "cards_affordable": "cards in budget"}
    why = (f"re-planned after {by[:60]} chose '{decision}' on {oid}: "
           + ("; ".join(f"{words[k]} {v[0] if v[0] is not None else '-'} -> {v[1] if v[1] is not None else '-'}"
                        for k, v in changed.items()) if changed else "no route or queue change"))
    entry = {"t": now, "hazard": oid, "decision": decision, "by": by[:60], "changed": changed,
             "rationale": why, "inspection_route": list(new.inspection_route),
             "recovery_route": list(new.recovery_route),
             "passes": [{"id": L["id"], "targets": L["targets"]} for L in (new.resurvey_plan or {}).get("lines") or []],
             "queue_head": list(new.review_queue[:5])}
    new.replans.append(entry)
    survey.mission = new
    return entry


# ---- exporters --------------------------------------------------------------------------------
def _by_id(tracked: list[TrackedObject]) -> dict[str, TrackedObject]:
    return {t.oid: t for t in tracked}


def redact_public(tracked: list[TrackedObject], mission: MissionPlan) -> tuple[list[TrackedObject], MissionPlan, int]:
    """Public-share copy: sensitive-class hazards get generalised coordinates; re-survey passes that
    target them are dropped (a pass line would point straight at the wreck)."""
    out, sens = [], set()
    for t in tracked:
        t2 = copy.deepcopy(t)
        if t2.cls_name in SENSITIVE_CLASSES and t2.lat is not None:
            t2.lat, t2.lon = round(round(t2.lat / _GENERAL_DEG) * _GENERAL_DEG, 4), round(round(t2.lon / _GENERAL_DEG) * _GENERAL_DEG, 4)
            t2.geo_error_m = max(t2.geo_error_m or 0.0, 1100.0)
            sens.add(t2.oid)
        out.append(t2)
    m2 = copy.deepcopy(mission)
    rp = dict(m2.resurvey_plan or {})
    if rp.get("lines"):
        rp["lines"] = [L for L in rp["lines"] if not (set(L["targets"]) & sens)]
        m2.resurvey_plan = rp
    return out, m2, len(sens)


def to_geojson(tracked: list[TrackedObject], mission: MissionPlan, prov: Optional[dict] = None,
               note: Optional[str] = None) -> str:
    features = []
    for t in tracked:
        if t.lat is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [t.lon, t.lat]},
            "properties": {"id": t.oid, "class": t.cls_name, "verdict": t.verdict.value,
                           "conf": t.conf, "evidence_score": t.evidence_score,
                           "height_m": t.height_m, "height_rel_alt": t.height_rel,
                           "shadow": t.shadow_quality, "also_in": t.also_in, "p_pot": t.p_pot,
                           "geo_error_m": t.geo_error_m, "frame": t.frame_id,
                           "person_decision": t.human, "decided_by": t.human_by},
        })
    idx = _by_id(tracked)
    coords = [[idx[i].lon, idx[i].lat] for i in mission.recovery_route if idx.get(i) and idx[i].lat is not None]
    if len(coords) > 1:
        features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                         "properties": {"name": "recovery_route", "length_m": mission.route_length_m}})
    icoords = [[idx[i].lon, idx[i].lat] for i in mission.inspection_route if idx.get(i) and idx[i].lat is not None]
    if len(icoords) > 1:
        features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": icoords},
                         "properties": {"name": "inspection_route", "length_m": mission.inspection_length_m,
                                        "note": "REVIEW cards to inspect first - pending human approval"}})
    for L in (mission.resurvey_plan or {}).get("lines", []):
        features.append({"type": "Feature",
                         "geometry": {"type": "LineString", "coordinates": [L["start"][::-1], L["end"][::-1]]},
                         "properties": {"name": L["id"], "kind": "resurvey_plan", "status": L["status"],
                                        "heading_deg": L["heading_deg"], "targets": L["targets"],
                                        "targets_on": L["targets_on"], "voi": L["voi"], "boat_min": L["boat_min"],
                                        "predictions": L["predictions"], "why": L["why"]}})
    fc = {"type": "FeatureCollection",
          "properties": {"human_approval_required": True,
                         "note": _SYNTH_WARN if mission.gps_synthetic else "real GPS",
                         "guarantees": mission.guarantees, "provenance": prov, "redaction": note},
          "features": features}
    return json.dumps(fc, indent=2)


def to_gpx(tracked: list[TrackedObject], mission: MissionPlan, prov: Optional[dict] = None,
           note: Optional[str] = None) -> str:
    from .provenance import short
    idx = _by_id(tracked)
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<gpx version="1.1" creator="DEPTH" xmlns="http://www.topografix.com/GPX/1/1">']
    desc = " | ".join(x for x in ((_SYNTH_WARN if mission.gps_synthetic else ""), (short(prov) if prov else ""), note or "") if x)
    if desc:
        out.append(f"  <metadata><desc>{desc}</desc></metadata>")
    for t in tracked:
        if t.lat is None:
            continue
        out.append(f'  <wpt lat="{t.lat}" lon="{t.lon}"><name>{t.oid}</name>'
                   f'<desc>{t.verdict.value} {t.cls_name} conf={t.conf:.2f}</desc></wpt>')
    route = [idx[i] for i in mission.recovery_route if idx.get(i) and idx[i].lat is not None]
    if len(route) > 1:
        out.append(f'  <rte><name>recovery_route ({mission.route_length_m} m)</name>')
        out += [f'    <rtept lat="{t.lat}" lon="{t.lon}"><name>{t.oid}</name></rtept>' for t in route]
        out.append("  </rte>")
    for L in (mission.resurvey_plan or {}).get("lines", []):
        out.append(f'  <rte><name>{L["id"]} re-survey (PLANNED, {L["boat_min"]} min)</name>'
                   f'<desc>{L["why"]}</desc>')
        out.append(f'    <rtept lat="{L["start"][0]}" lon="{L["start"][1]}"><name>{L["id"]}-start</name></rtept>')
        out.append(f'    <rtept lat="{L["end"][0]}" lon="{L["end"][1]}"><name>{L["id"]}-end</name></rtept>')
        out.append("  </rte>")
    out.append("</gpx>")
    return "\n".join(out)


def to_kml(tracked: list[TrackedObject], mission: MissionPlan, prov: Optional[dict] = None,
           note: Optional[str] = None) -> str:
    from .provenance import short
    idx = _by_id(tracked)
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           f"  <name>DEPTH hazards{' (' + _SYNTH_WARN + ')' if mission.gps_synthetic else ''}</name>",
           f"  <description>{' | '.join(x for x in ((short(prov) if prov else ''), note or '') if x)}</description>"]
    for t in tracked:
        if t.lat is None:
            continue
        out.append(f"  <Placemark><name>{t.oid}</name>"
                   f"<description>{t.verdict.value} {t.cls_name} conf={t.conf:.2f} "
                   f"height_rel_alt={t.height_rel} shadow={t.shadow_quality}</description>"
                   f"<Point><coordinates>{t.lon},{t.lat},0</coordinates></Point></Placemark>")
    route = [idx[i] for i in mission.recovery_route if idx.get(i) and idx[i].lat is not None]
    if len(route) > 1:
        line = " ".join(f"{t.lon},{t.lat},0" for t in route)
        out.append(f"  <Placemark><name>recovery_route</name><LineString>"
                   f"<coordinates>{line}</coordinates></LineString></Placemark>")
    for L in (mission.resurvey_plan or {}).get("lines", []):
        out.append(f"  <Placemark><name>{L['id']} re-survey (planned)</name><description>{L['why']}</description>"
                   f"<LineString><coordinates>{L['start'][1]},{L['start'][0]},0 {L['end'][1]},{L['end'][0]},0"
                   f"</coordinates></LineString></Placemark>")
    out.append("</Document></kml>")
    return "\n".join(out)


def to_csv(tracked: list[TrackedObject]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "class", "verdict", "conf", "evidence_score", "frame",
                "lat", "lon", "geo_error_m", "height_m", "height_rel_alt", "shadow", "bbox", "also_in"])
    for t in tracked:
        w.writerow([t.oid, t.cls_name, t.verdict.value, f"{t.conf:.3f}", t.evidence_score, t.frame_id,
                    t.lat if t.lat is not None else "", t.lon if t.lon is not None else "",
                    t.geo_error_m if t.geo_error_m is not None else "",
                    t.height_m if t.height_m is not None else "", t.height_rel, t.shadow_quality,
                    " ".join(map(str, t.bbox)), " ".join(t.also_in)])
    return buf.getvalue()


def to_json(survey: SurveyResult, prov: Optional[dict] = None, note: Optional[str] = None) -> str:
    d = survey.to_dict()
    d["provenance"], d["redaction"] = prov, note
    return json.dumps(d, indent=2)


def to_trace(survey: SurveyResult, prov: Optional[dict] = None) -> str:
    """The agent's flight recorder: one JSON line per event — provenance, every frame's Stage-1 record,
    every tool call of every candidate (inputs → rationale → outputs, timed), every verdict, and the
    mission decisions (budget, routes, re-survey plan, human gate). Replayable and diffable."""
    L = [{"type": "provenance", **(prov or {})},
         {"type": "survey", "survey_id": survey.survey_id, "frames": len(survey.frames),
          "gps_available": survey.mission.gps_available, "gps_synthetic": survey.mission.gps_synthetic,
          "guarantees": survey.mission.guarantees}]
    for fr in survey.frames:
        s1 = {k: v for k, v in (fr.stage1 or {}).items() if k != "bottom_line_native"}
        L.append({"type": "frame", "frame_id": fr.frame_id, "orientation": fr.orientation, "stage1": s1,
                  "stage_ms": {k: round(v, 2) for k, v in fr.stage_ms.items()}, "counts": fr.counts})
        for ci, c in enumerate(fr.candidates):
            for si, st in enumerate(c.trace):
                L.append({"type": "step", "frame_id": fr.frame_id, "cand": ci, "step": si, "cls": c.cls_name,
                          "bbox": list(c.bbox), **st.to_dict()})
            L.append({"type": "verdict", "frame_id": fr.frame_id, "cand": ci, "cls": c.cls_name, "conf": round(c.conf, 4),
                      "verdict": c.verdict.value if c.verdict else None,
                      "score": c.evidence.evidence_score if c.evidence else None,
                      "p_pot": c.evidence.p_pot if c.evidence else None, "lat": c.lat, "lon": c.lon})
    m = survey.mission
    L.append({"type": "mission", "counts": m.counts, "recovery_route": m.recovery_route, "review_queue": m.review_queue,
              "inspection_route": m.inspection_route, "budget": m.budget, "repeat_merges": m.repeat_merges,
              "resurvey": {k: v for k, v in (m.resurvey_plan or {}).items() if k != "lines"},
              "resurvey_lines": [{k: L_[k] for k in ("id", "targets", "voi", "boat_min", "status")}
                                 for L_ in (m.resurvey_plan or {}).get("lines", [])],
              "human_approval_required": m.human_approval_required, "impact": m.impact})
    for st in m.agent_log:
        L.append({"type": "plan", **st})
    for inc in m.incidents + [i for fr in survey.frames for i in fr.incidents]:
        L.append({"type": "incident", **inc})
    for h in m.human_log:
        L.append({"type": "human", **h})
    for r in m.replans:
        L.append({"type": "replan", **{k: v for k, v in r.items() if k not in ("inspection_route", "recovery_route")}})
    return "\n".join(json.dumps(x, default=str) for x in L) + "\n"


EXPORTERS = {"geojson": to_geojson, "gpx": to_gpx, "kml": to_kml}     # need tracked + mission


def export(fmt: str, survey: SurveyResult, public: bool = False, prov: Optional[dict] = None) -> str:
    """Render a survey's mission in ``fmt`` ∈ {geojson, gpx, kml, csv, json, trace, brief}. ``public=True``
    generalises protected-site locations (``SENSITIVE_CLASSES``). Every format except CSV carries the
    provenance stamp (CSV stays a clean table). ``brief`` is the deterministic template mission brief
    (markdown); the optional LLM-written brief is served by ``/api/brief``, never baked into a report."""
    fmt = fmt.lower()
    if prov is None:
        from .provenance import stamp
        prov = stamp()
    tracked, mission, note = survey.tracked, survey.mission, None
    if public:
        tracked, mission, n = redact_public(tracked, mission)
        note = f"{_REDACT_NOTE} ({n} object(s) generalised)"
    if fmt in EXPORTERS:
        return EXPORTERS[fmt](tracked, mission, prov, note)
    if fmt == "csv":
        return to_csv(tracked)
    if fmt == "json":
        return to_json(SurveyResult(survey.survey_id, survey.frames, tracked, mission) if public else survey, prov, note)
    if fmt == "brief":
        from .brief import facts, template_brief
        s = SurveyResult(survey.survey_id, survey.frames, tracked, mission) if public else survey
        return template_brief(facts(s.to_dict(), prov, public))
    if fmt == "trace":
        if public:
            raise ValueError("the decision log carries exact positions - it is not shared publicly")
        return to_trace(survey, prov)
    raise ValueError(f"unknown format: {fmt}")
