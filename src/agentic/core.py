"""
core.py — the DEPTH agent: a goal, OpenCV tools, and a loop that re-plans on what it observes.

A survey is not a fixed pipeline. The operator gives a **goal** — the recall promise to keep, the
analyst minutes and boat minutes it may spend, how long it may compute — and the agent decides how to
meet it, step by step, logging every option it weighed:

1. **inspect** the data: which frames are usable (corrupt / blank frames are skipped and reported);
   then **check that the promise applies** (STUDY-19, ``applicability.py``): OpenCV statistics of every
   frame against the calibration recordings. If the recording does not look like them (another sonar,
   another rendering), the agent withdraws the recall promise for this survey, runs the cheapest look
   (no promise is left to buy) and asks a person to approve a full review;
2. **choose how hard to look**: among the calibrated perception modes (``calibration.json``: the
   standard full-frame look, and the look-harder tile pass if STUDY-18 shipped it), the cheapest whose
   promise meets the goal and whose cost fits the budgets — or, with no promise asked for, the strongest
   promise the budgets allow. A goal no mode can meet is said so, with the best available;
3. **perceive** every frame with OpenCV 5 (Stage 1, ``cv2.dnn``, evidence), **watching its compute
   budget and its tools**: if the measured pace says the budget will run out, or the closer look fails or
   overruns its deadline on a frame, it re-plans mid-run — the standard look for the remaining frames,
   the earlier ones re-tiered at the standard threshold. That keeps the standard promise: the extra tile
   finds can only add candidates, so recall at a fixed threshold cannot fall. A frame whose detector call
   finds the network still busy with an overrun call is retried once the network is free, not skipped;
4. **plan** for people and the boat: merge repeat sightings, the review queue and budget, the inspection
   and recovery routes, and the opposite-side re-survey passes (``pipeline.assemble`` → ``mission``);
5. **hand off**: nothing is dispatched; the server files one approval request per action and a named
   person decides; each decision makes the agent re-plan (``mission.apply_human`` / ``decline_resurvey``).

It stops when every frame is perceived and the plan is built — what remains needs a person. Every step
lands in ``SurveyResult.agent_run`` (and the studio's Agent panel): what it did, why, what it saw, and
what changed. The decision rules are deterministic and auditable; an LLM never decides.
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from src.detection.calibration import load_calibration
from .applicability import Reference, frame_stats
from .incidents import incident
from .mission import DEFAULT_SEC_PER_CARD
from .pipeline import AgenticPipeline, validate_frame
from .tools import ToolTimeout, default_timeout_s
from .types import AgentStep, FrameResult, SurveyResult, Verdict

_FLAKY = {"tool_failed", "tool_timeout", "invalid_geometry", "no_detection"}   # drills: first frame only


@dataclass
class Goal:
    """What the operator asks for. ``None`` = no constraint."""
    recall: Optional[float] = None            # the promise to keep: share of pots that must reach a person
    analyst_minutes: Optional[float] = None   # review time available: orders the queue (budget mode)
    boat_minutes: Optional[float] = None      # boat time available for re-survey passes
    compute_s: Optional[float] = None         # wall-clock budget for perception on this run
    sec_per_card: float = DEFAULT_SEC_PER_CARD

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Mode:
    """A calibrated perception mode: its promise and what it costs."""
    name: str
    label: str
    promise: float                       # recall promise (95% confidence), fit on calibration recordings
    tau: float                           # its review threshold
    cards_per_frame: Optional[float]     # review cards per frame on the calibration recordings
    passes: int                          # cv2.dnn forward passes per frame
    verified: dict = field(default_factory=dict)
    report: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def calibrated_modes(cal=None) -> list[Mode]:
    """The modes ``calibration.json`` carries (a mode is there only if it passed its gates)."""
    cal = cal or load_calibration()
    t = cal.tiers
    if t.get("tau_review") is None:
        return []
    g = t.get("guarantees") or {}
    bins, frames = t.get("p_pot_bins") or [], (cal.data.get("fit") or {}).get("frames") or 0
    out = [Mode("standard", "standard look", float(g.get("recall_promise") or 0.0), float(t["tau_review"]),
                round(sum(b["n"] for b in bins) / frames, 3) if bins and frames else None, 1,
                dict(g.get("verified_on_test") or {}), g.get("report"))]
    for name, m in cal.modes.items():
        if m.get("tau_review") is not None and m.get("recall_promise") is not None:
            out.append(Mode(name, m.get("label", name), float(m["recall_promise"]), float(m["tau_review"]),
                            m.get("cards_per_frame_cal"), int(m.get("forward_passes_per_frame", 1)),
                            dict(m.get("verified_on_test") or {}), m.get("report")))
    return out


def choose_mode(goal: Goal, modes: list[Mode], n_frames: int,
                ms_per_pass: Optional[float]) -> tuple[Optional[Mode], dict]:
    """Pick the perception mode for a goal. Returns the mode and the decision record: every option with
    its promise, expected review cards / minutes and compute, and why it was or was not chosen.

    The recall goal and the compute budget constrain the choice. The analyst budget does not: it orders
    the queue (budget mode reviews the likeliest cards first), so it is reported as the share of the
    queue it covers. With no recall asked for, the agent keeps the strongest promise the compute budget
    allows — a missed ghost pot keeps fishing, so recall is what the promise protects."""
    opts = []
    for m in modes:
        cards = None if m.cards_per_frame is None else m.cards_per_frame * n_frames
        minutes = None if cards is None else cards * goal.sec_per_card / 60.0
        comp = None if ms_per_pass is None else m.passes * n_frames * ms_per_pass / 1000.0
        meets = goal.recall is None or m.promise + 1e-9 >= goal.recall
        fits = goal.compute_s is None or comp is None or comp <= goal.compute_s
        why = []
        if not meets:
            why.append(f"its promise (>= {m.promise:.0%}) is below the goal (>= {goal.recall:.0%})")
        if not fits:
            why.append(f"~{comp:.1f} s of compute > the {goal.compute_s:g}-s budget")
        covers = (None if goal.analyst_minutes is None or not minutes else round(min(1.0, goal.analyst_minutes / minutes), 3))
        opts.append({"mode": m.name, "label": m.label, "promise": m.promise, "passes_per_frame": m.passes,
                     "expected_cards": None if cards is None else round(cards, 1),
                     "expected_analyst_min": None if minutes is None else round(minutes, 1),
                     "analyst_budget_covers": covers,
                     "expected_compute_s": None if comp is None else round(comp, 1),
                     "meets_goal": meets, "fits_budgets": fits, "why_not": why})
    if not opts:
        return None, {"options": [], "why": "no calibrated mode - legacy rules, no promise", "goal_met": False}
    by = {m.name: m for m in modes}
    cost = lambda o: (o["expected_compute_s"] or 0.0, o["expected_analyst_min"] or 0.0, o["passes_per_frame"])
    ok = [o for o in opts if o["meets_goal"] and o["fits_budgets"]]
    goal_met = bool(ok)
    if goal.recall is not None and ok:
        pick = min(ok, key=cost)
        why = (f"the cheapest mode whose promise (>= {pick['promise']:.0%}) meets the goal (>= {goal.recall:.0%})"
               + (f" within the {goal.compute_s:g}-s compute budget" if goal.compute_s else ""))
    elif ok:
        pick = max(ok, key=lambda o: (o["promise"], tuple(-c for c in cost(o))))
        why = (f"no promise asked for: the strongest promise (>= {pick['promise']:.0%})"
               + (f" that fits the {goal.compute_s:g}-s compute budget" if goal.compute_s else " available"))
    else:
        fit = [o for o in opts if o["fits_budgets"]]
        pick = max(fit, key=lambda o: o["promise"]) if fit else min(opts, key=cost)
        if not any(o["meets_goal"] for o in opts):
            why = (f"the goal cannot be met: no calibrated mode can promise >= {goal.recall:.0%} - the strongest"
                   + (" that fits the compute budget" if goal.compute_s else "")
                   + f" is >= {pick['promise']:.0%} ({pick['label']}); a person must accept it or change the goal")
        else:
            why = (f"the goal cannot be met: the modes that keep it do not fit the {goal.compute_s:g}-s compute budget "
                   f"- running >= {pick['promise']:.0%} ({pick['label']}); a person must accept it or raise the budget")
    if pick.get("analyst_budget_covers") is not None and pick["analyst_budget_covers"] < 1:
        why += (f"; the {goal.analyst_minutes:g}-min analyst budget covers ~{pick['analyst_budget_covers']:.0%} of the "
                f"~{pick['expected_analyst_min']:g} min queue, so budget mode puts the likeliest cards first")
    for o in opts:
        o["chosen"] = o is pick
    return by[pick["mode"]], {"options": opts, "why": why, "goal_met": goal_met}


class AgentRun:
    """The core loop's record: one entry per decision, in order (what, why, what it saw, what changed)."""

    def __init__(self, goal: Goal):
        self.goal, self.steps, self.t0 = goal, [], time.perf_counter()
        self.replans: list[dict] = []

    def log(self, act: str, why: str, observed: Optional[dict] = None, decided: Optional[str] = None,
            options: Optional[list] = None, status: str = "done") -> dict:
        s = {"n": len(self.steps) + 1, "act": act, "why": why, "status": status,
             "t_s": round(time.perf_counter() - self.t0, 2)}
        if observed:
            s["observed"] = observed
        if decided:
            s["decided"] = decided
        if options:
            s["options"] = options
        self.steps.append(s)
        return s

    def to_dict(self, mode: Optional[Mode], stop: str, extra: Optional[dict] = None) -> dict:
        return {"goal": self.goal.to_dict(), "mode": mode.to_dict() if mode else None, "steps": self.steps,
                "replans": self.replans, "stop": stop, "elapsed_s": round(time.perf_counter() - self.t0, 2),
                **(extra or {})}


def retier(fr: FrameResult, tiers) -> int:
    """Re-decide a frame's guaranteed-class finds at another mode's threshold (after a mid-run
    downgrade). Returns how many verdicts changed; each find's trace says why."""
    changed = 0
    for c in fr.candidates:
        if c.cls_name != tiers.guaranteed_class or c.evidence is None or c.verdict is None:
            continue
        v = tiers.tier(c.conf, c.evidence.evidence_score)
        p = tiers.p_pot(c.evidence.evidence_score) if v is Verdict.REVIEW else None
        if v is not c.verdict:
            changed += 1
        c.trace.append(AgentStep(
            tool="replan", latency_ms=0.0, conf_before=round(c.conf, 4), status="done",
            rationale=(f"the run fell back to the standard promise -> re-tiered at tau_review "
                       f"{tiers.tau_review:.3f}: {c.verdict.value} -> {v.value}"),
            detail={"from": c.verdict.value, "to": v.value, "tau_review": tiers.tau_review}))
        c.verdict, c.evidence.p_pot = v, p
    fr.perception["retiered_to"] = "standard"
    return changed


class SurveyAgent:
    """Runs a survey toward a :class:`Goal` (see the module docstring)."""

    def __init__(self, pipeline: AgenticPipeline, modes: Optional[list[Mode]] = None,
                 reference: Optional[Reference] = None, check_applicability: bool = True):
        self.pipe = pipeline
        self.modes = calibrated_modes() if modes is None else modes
        self._check_on = check_applicability
        self.retry_wait_s = max(30.0, 2 * default_timeout_s())
        self.reference = (reference or Reference.load()) if check_applicability else None
        self._applicability: Optional[dict] = None

    def run(self, frames: list[tuple[str, object]], goal: Optional[Goal] = None, track: Optional[dict] = None,
            survey_id: str = "survey", nadir: Optional[str] = None,
            progress_cb: Optional[Callable[[str, dict], None]] = None, frame_lock=None,
            faults: Optional[set] = None, incidents: Optional[list] = None,
            geometry: Optional[dict] = None) -> SurveyResult:
        """``geometry``: the sensor cross-check summary of a raw recording (logged as the agent's own
        measurement step; its per-chunk result already sits in ``track``)."""
        goal = goal or Goal()
        faults = set(faults or ())
        run = AgentRun(goal)
        survey_incidents = list(incidents or [])

        # 1. inspect the data
        bad = {fid: why for fid, im in frames if (why := validate_frame(im))}
        usable = len(frames) - len(bad)
        run.log("inspect_data", f"{len(frames)} frame(s): {usable} usable sonograms"
                + (f", {len(bad)} skipped ({', '.join(sorted(set(bad.values())))})" if bad else ""),
                observed={"frames": len(frames), "usable": usable, "skipped": sorted(bad)},
                decided="perceive the usable frames; report the rest")
        if geometry:
            c = geometry.get("counts") or {}
            run.log("measure_geometry", "raw recording: the OpenCV seabed track was cross-checked against the depth "
                    "sounder per chunk, re-tracked where they disagreed, and the range scale re-fit (STUDY-16)",
                    observed={"chunks": geometry.get("chunks"), "counts": c,
                              "scale_cm_per_sample": round(100 * (geometry.get("scale") or {}).get("m_per_sample", 0), 3),
                              "iterations": geometry.get("iterations")},
                    decided=f"{geometry.get('trusted', 0)} of {geometry.get('chunks', 0)} chunks trusted for metres and "
                            f"ground range; {c.get('not_measured', 0)} withheld (slant range, widened error radius)")
            d = geometry.get("dropouts") or {}
            if d.get("pings"):
                run.log("check_dropouts", f"{d['pings']} of {d['total_pings']} pings ({d['share']:.2%}) were not imaged "
                        f"({d['dark']} dark, {d['gap']} after a time gap, {d['missing_s']} s missing) in "
                        f"{d['frames_with']} frame(s) - a pot there cannot be seen",
                        observed={k: d.get(k) for k in ("pings", "share", "dark", "gap", "missing_s", "frames_with")},
                        decided="flag the dropped stretches per frame (data_dropout incident); the promise covers imaged "
                                "seabed only")

        applies = self._check_promise(run, frames, bad, survey_incidents)

        # 2. choose how hard to look
        if applies:
            mode, choice = choose_mode(goal, self.modes, usable, self.pipe.ms_per_pass())
        else:                                             # no promise left to keep: the cheapest look
            mode, choice = choose_mode(Goal(analyst_minutes=goal.analyst_minutes, compute_s=goal.compute_s,
                                            sec_per_card=goal.sec_per_card), self.modes[:1], usable,
                                       self.pipe.ms_per_pass())
            choice["why"] = ("the promise does not apply to this recording, so no mode can buy one - the cheapest "
                             "look (standard) serves the full human review")
        run.log("choose_mode", choice["why"], decided=(mode.label if mode else "legacy rules"),
                options=choice["options"], observed={"usable_frames": usable,
                                                     "ms_per_pass": self.pipe.ms_per_pass()})
        if applies and not choice["goal_met"] and mode is not None:
            survey_incidents.append(incident("goal_not_met", f"goal {goal.to_dict()} cannot be met within the budgets; "
                                             f"running {mode.label} (>= {mode.promise:.0%})"))
        name = mode.name if mode else "standard"
        agent = self.pipe.agent_for(name)

        # 3. perceive, watching the compute budget
        results: list[FrameResult] = []
        t_run, downgraded_at = time.perf_counter(), None
        for k, (fid, im) in enumerate(frames):
            if fid in bad:
                survey_incidents.append(incident(bad[fid], f"{fid}: skipped", frame_id=fid))
                if progress_cb:
                    progress_cb("frame_done", {"frame_id": fid, "counts": {}})
                continue
            fr_faults = faults if k == 0 else faults - _FLAKY
            run_one = lambda a: self._run_frame(a, im, fid, track, nadir, fr_faults, frame_lock)
            try:
                try:
                    fr = run_one(agent)
                except ToolTimeout as e:                    # the network is still busy with an overrun call
                    free = agent.tools.wait_idle(self.retry_wait_s)
                    run.log("retry", f"{fid}: {e} -> waited for the detector ({'free again' if free else 'still busy'}) "
                            "and ran the frame again", observed={"frame": fid, "detector_free": free})
                    fr = run_one(agent)
            except Exception as e:                          # one bad frame never sinks the survey
                survey_incidents.append(incident("tool_failed", f"{fid}: processing failed ({type(e).__name__}: "
                                                 f"{str(e)[:120]}) - frame skipped", frame_id=fid))
                if progress_cb:
                    progress_cb("frame_done", {"frame_id": fid, "counts": {}})
                continue
            results.append(fr)
            if progress_cb:
                progress_cb("frame_done", {"frame_id": fid, "counts": fr.counts})
            left = sum(1 for f, _ in frames[k + 1:] if f not in bad)
            if fr.perception.get("fallback") and mode is not None and mode.passes > 1 and downgraded_at is None:
                std = next((m for m in self.modes if m.name == "standard"), None)
                downgraded_at = fid
                why = next((s.get("rationale", "") for s in reversed(fr.perception.get("steps") or [])
                            if s.get("status") == "failed"), "")
                run.log("replan", f"the closer look failed on {fid} ({why[:140]}) -> this frame does not carry the "
                        f"look-harder promise, so the survey keeps the standard promise"
                        + (f" (>= {std.promise:.0%})" if std else "") + ": standard look for the remaining "
                        f"{left} frame(s), earlier frames re-tiered", observed={"frame": fid},
                        decided="downgrade to the standard look")
                run.replans.append({"at": fid, "why": "closer look failed", "from": mode.name, "to": "standard"})
                mode, agent = std, self.pipe.agent_for("standard")
            if goal.compute_s and mode is not None and mode.passes > 1 and left and downgraded_at is None:
                spent = time.perf_counter() - t_run
                projected = spent + left * spent / len(results)
                if projected > goal.compute_s:
                    std = next((m for m in self.modes if m.name == "standard"), None)
                    downgraded_at = fid
                    msg = (f"after {len(results)} frame(s) the pace projects {projected:.1f} s > the {goal.compute_s:g}-s "
                           f"compute budget -> standard look for the remaining {left} frame(s); the earlier frames are "
                           f"re-tiered at the standard threshold, so the survey keeps the standard promise"
                           + (f" (>= {std.promise:.0%})" if std else ""))
                    run.log("replan", msg, observed={"spent_s": round(spent, 2), "projected_s": round(projected, 1)},
                            decided="downgrade to the standard look")
                    run.replans.append({"at": fid, "why": "compute budget", "from": mode.name, "to": "standard"})
                    mode, agent = std, self.pipe.agent_for("standard")
        if downgraded_at is not None:
            n = sum(retier(fr, agent.cfg.tiers) for fr in results if fr.perception.get("mode") != "standard")
            run.log("retier", f"{n} verdict(s) changed when the look-harder frames were re-tiered at the standard "
                    "threshold", observed={"changed": n})
        n_new = sum(int(fr.perception.get("new") or 0) for fr in results)
        n_up = sum(int(fr.perception.get("raised") or 0) for fr in results)
        cards = sum(fr.counts.get("review", 0) + fr.counts.get("confirmed", 0) for fr in results)
        run.log("perceive", f"{len(results)} frame(s) perceived with OpenCV 5 (Stage 1 + cv2.dnn + evidence)"
                + (f"; the closer tile look added {n_new} candidate(s) and raised {n_up}" if n_new or n_up else ""),
                observed={"frames": len(results), "cards": cards, "tile_new": n_new, "tile_raised": n_up,
                          "seconds": round(time.perf_counter() - t_run, 2)})

        # 4. plan for people and the boat
        g = self.pipe.guarantees(agent)
        g["perception_mode"] = mode.name if mode else "legacy"
        if not applies:
            a = self._applicability or {}
            g["promise_withdrawn"] = {"was": g.get("recall_promise"), "why": a.get("why"),
                                      "check": {k: a.get(k) for k in
                                                ("atypical", "frames", "share", "share_lcb", "share_limit", "drivers")}}
            g["recall_promise"] = None
        survey = self.pipe.assemble(results, track, survey_id, goal.analyst_minutes, goal.sec_per_card,
                                    goal.boat_minutes, survey_incidents, progress_cb, g)
        m = survey.mission
        run.log("plan", f"{len(survey.tracked)} hazard(s): review queue {len(m.review_queue)}, inspection route "
                f"{len(m.inspection_route)} stop(s), recovery route {len(m.recovery_route)}, re-survey passes "
                f"{len((m.resurvey_plan or {}).get('lines') or [])}",
                observed={"hazards": len(survey.tracked), "repeat_merges": m.repeat_merges},
                decided="file one approval request per action; a named person decides")
        stop = ("plan built; every dispatch waits for a named person - each decision re-plans" if applies else
                "plan built WITHOUT a recall promise (it does not apply to this recording); a full review by a person "
                "is requested")
        survey.agent_run = run.to_dict(mode, stop, {"downgraded_at": downgraded_at,
                                                    "promise_applies": applies, "applicability": self._applicability})
        return survey

    def _run_frame(self, agent, im, fid: str, track, nadir, faults: set, frame_lock) -> FrameResult:
        with (frame_lock or nullcontext()):
            return self.pipe.run_frame(im, frame_id=fid, fix=track.get(fid) if track else None, nadir=nadir,
                                       faults=faults, agent=agent)

    def _check_promise(self, run: AgentRun, frames, bad: dict, survey_incidents: list) -> bool:
        """Does the calibrated promise apply to these frames? (STUDY-19) Logs the check and, on a failure,
        files the incident. Returns True when the promise may be kept."""
        self._applicability = None
        if self.reference is None:
            run.log("check_promise", ("the applicability check is off for this run" if not self._check_on else
                                      "no applicability reference for this model (python -m "
                                      "src.agentic.study_applicability)") + " - the promise is kept unchecked",
                    status="skipped")
            return True
        stats = [frame_stats(im) for fid, im in frames if fid not in bad and im is not None]
        chk = self.reference.check(stats)
        self._applicability = {k: v for k, v in chk.items() if k != "p_values"}
        run.log("check_promise", chk["why"],
                observed={k: chk[k] for k in ("atypical", "frames", "share", "share_lcb", "share_limit", "drivers")},
                decided=("keep the calibrated promise" if chk["applies"] else
                         "withdraw the recall promise for this survey; ask a person to approve a full review"))
        if not chk["applies"]:
            survey_incidents.append(incident("promise_not_applicable", chk["why"], atypical=chk["atypical"],
                                             frames=chk["frames"]))
        return bool(chk["applies"])
