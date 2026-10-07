"""
active.py — the agent's **active-vision policy** (pure, deterministic; shared by the live agent and
the offline replay in ``evidence_model.py`` so the two can never drift).

The agent holds a belief P(pot) for each candidate and chooses its next observation by **value of
information**: an OpenCV tool is run only if one of its possible outcomes would change the
candidate's ACTION. Cheapest informative tool first; stop when nothing left can change the action.

Actions (what happens to a candidate next — the calibrated tier never changes, so the recall promise
in ``calibration.json`` is untouched: every candidate ≥ τ_review still reaches a person):

* ``accept``  — P ≥ ``accept``: a priority **inspection target** (front of the queue, on the
  inspection route). Dispatch still needs a named person's approval.
* ``review``  — a normal evidence card for a person.
* ``watch``   — P < ``watch``: a low-priority card at the end of the queue (still shown to a person).

Observations (each a real OpenCV 5 measurement; likelihood ratios fit on validation):

* ``geometry_check`` — Stage-1 bottom track: is the box above the seabed (water column)? Already
  measured by Stage 1, so it costs nothing. A seabed object cannot sit in the water column: a
  physical rule, not a fitted number — such a find is never accepted.
* ``shadow_check``   — thin-line acoustic shadow behind the echo (~1 ms, no inference).
* ``zoom_relook``    — crop + upscale + re-detect with ``cv2.dnn`` (one inference, ~0.4 s on CPU).
* ``opposite_side_pass`` — a second sonar pass from the other side (boat time; needs a person's
  approval). Modelled as a fresh shadow observation — an ASSUMPTION (no paired passes exist in the
  data to measure it).

**Evidence conflict:** the detector alone points to one action and an OpenCV measurement to
another (or the find sits in the water column). The agent then asks for another observation: the
cheapest in-frame tool that can still change the action, else an opposite-side pass, else a person.
"""
from __future__ import annotations

import math
from typing import Optional

ACTION_RANK = {"accept": 0, "review": 1, "watch": 2}
IN_FRAME_TOOLS = ("shadow_check", "zoom_relook")


def action_of(p: Optional[float], thresholds: dict, water_column: Optional[bool] = None) -> str:
    if p is None:
        return "review"
    if water_column:
        return "review" if p >= thresholds["watch"] else "watch"     # never accepted off the seabed
    if p >= thresholds["accept"]:
        return "accept"
    return "watch" if p < thresholds["watch"] else "review"


def _disagrees(detector_action: str, evidence_action: str) -> bool:
    """An OpenCV measurement contradicts the detector: it pushes a find the detector rated
    accept/review DOWN, or lifts one the detector rated watch UP. (Review -> accept is agreement.)"""
    d, e = ACTION_RANK[detector_action], ACTION_RANK[evidence_action]
    return e > d or (detector_action == "watch" and e < d)


def odds(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return p / (1 - p)


def prob(o: float) -> float:
    return o / (1 + o)


def entropy(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def outcomes(model, p: float, conf: float, tool: str) -> list[tuple[str, float, float]]:
    """[(outcome, probability under the current belief, P(pot) after it)] for one observation."""
    kind = "shadow" if tool in ("shadow_check", "opposite_side_pass") else "relook"
    lr = model.lr_for(kind, conf)
    if lr is None:
        return []
    out = []
    for o in ("present", "absent"):
        p_o = p * lr[o + "_tp"] + (1 - p) * lr[o + "_fp"]           # P(outcome) under the belief
        out.append((o, p_o, prob(odds(p) * lr[o])))
    return out


def voi(model, p: float, conf: float, tool: str, thresholds: dict, water_column=None) -> dict:
    """Value of information of one observation: P(it changes the action) and the expected entropy
    drop (bits). Zero when no outcome can change what happens to the candidate."""
    now = action_of(p, thresholds, water_column)
    outs = outcomes(model, p, conf, tool)
    flip = sum(po for _, po, pa in outs if action_of(pa, thresholds, water_column) != now)
    ig = entropy(p) - sum(po * entropy(pa) for _, po, pa in outs) if outs else 0.0
    return {"tool": tool, "p_flip": round(flip, 4), "info_bits": round(max(ig, 0.0), 4),
            "outcomes": [{"outcome": o, "prob": round(po, 3), "p_after": round(pa, 3),
                          "action_after": action_of(pa, thresholds, water_column)} for o, po, pa in outs]}


def next_tool(model, p: float, conf: float, done: set, thresholds: dict, water_column=None,
              available: tuple = IN_FRAME_TOOLS) -> tuple[Optional[str], list[dict]]:
    """The cheapest in-frame tool whose result could change the action (None = stop). Returns the
    choice and every option's value of information (for the trace)."""
    opts = [voi(model, p, conf, t, thresholds, water_column) for t in available if t not in done]
    useful = [o for o in opts if o["p_flip"] > 0]
    if not useful:
        return None, opts
    best = min(useful, key=lambda o: (model.cost_ms(o["tool"]), -o["p_flip"]))
    return best["tool"], opts


def update(model, p: float, conf: float, kind: str, present: bool) -> float:
    lr = model.lr_for(kind, conf)
    if lr is None:
        return p
    return prob(odds(p) * lr["present" if present else "absent"])


def resurvey_gain(model, p: float, conf: float, thresholds: dict) -> dict:
    """What an opposite-side pass is expected to buy for this target (information + action flips)."""
    return voi(model, p, conf, "opposite_side_pass", thresholds)


def run_policy(model, conf: float, p0: float, observe, thresholds: Optional[dict] = None,
               water_column: Optional[bool] = None, available: tuple = IN_FRAME_TOOLS,
               on_choose=None, on_update=None) -> dict:
    """Run the active loop for one candidate. ``observe(tool) -> bool | None`` performs (or, offline,
    looks up) the measurement and returns present/absent (None = failed / not measurable).

    Returns the belief path, the tools run and skipped (with reasons), the conflict record and the
    final action — the same dict the live agent turns into its trace."""
    th = thresholds or model.thresholds
    p, done, steps = p0, set(), []
    a0 = action_of(p0, th, None)
    conflict = None
    if water_column:
        conflict = {"between": "detector vs geometry", "why": "the box sits above the tracked seabed "
                    "(water column) - a seabed object cannot be there"}
    while True:
        tool, opts = next_tool(model, p, conf, done, th, water_column, available)
        if tool is None:
            steps.append({"tool": "stop", "p": round(p, 4), "options": opts,
                          "why": "no remaining tool can change the action"})
            break
        before_action = action_of(p, th, water_column)
        if on_choose:
            on_choose(tool, opts, p, before_action)
        res = observe(tool)
        done.add(tool)
        kind = "shadow" if tool == "shadow_check" else "relook"
        if res is None:
            steps.append({"tool": tool, "status": "failed", "p_before": round(p, 4), "p_after": round(p, 4),
                          "options": opts})
            continue
        p_new = update(model, p, conf, kind, bool(res))
        after_action = action_of(p_new, th, water_column)
        st = {"tool": tool, "status": "done", "result": "present" if res else "absent",
              "p_before": round(p, 4), "p_after": round(p_new, 4), "action_before": before_action,
              "action_after": after_action, "options": opts}
        if conflict is None and _disagrees(a0, after_action):
            conflict = {"between": f"detector vs {tool}", "why": (
                f"the detector alone says {a0.upper()} (P {p0:.2f}) but the {tool.replace('_', ' ')} "
                f"says {after_action.upper()} (P {p_new:.2f})")}
            st["conflict"] = True
        steps.append(st)
        if on_update:
            on_update(st, conflict if st.get("conflict") else None)
        p = p_new
    final = action_of(p, th, water_column)
    rs = resurvey_gain(model, p, conf, th)
    resolved = conflict is not None and final == a0
    want_pass = (conflict is not None and not resolved and rs["p_flip"] > 0) or (
        final != "accept" and rs["p_flip"] >= th.get("resurvey_flip", 0.25))
    return {"p0": round(p0, 4), "p": round(p, 4), "action0": a0, "action": final, "steps": steps,
            "tools_run": [s["tool"] for s in steps if s.get("status") == "done"],
            "conflict": conflict, "conflict_resolved": resolved if conflict else None,
            "resurvey": rs, "request_resurvey": bool(want_pass)}
