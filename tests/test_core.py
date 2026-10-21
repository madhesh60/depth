"""
Tests for the core agent loop (src/agentic/core.py): the perception-mode choice for a goal, the
mid-run re-plan when the compute budget runs out (downgrade + re-tier), and the run record. A fake
pipeline stands in for the detector, so these run without the model; the modes come from the shipped
calibration.json (standard + look_harder, STUDY-18).
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from src.agentic.agent import _no_proof
from src.agentic.core import Goal, Mode, SurveyAgent, calibrated_modes, choose_mode, retier
from src.agentic.pipeline import AgenticPipeline
from src.agentic.policy import GuaranteedTiers
from src.agentic.tools import ToolTimeout
from src.agentic.types import Candidate, Evidence, FrameResult, RelookResult, Verdict
from src.detection.calibration import load_calibration

STD_PROMISE = float(load_calibration().guarantees["recall_promise"])   # the standard look's promise

STD = Mode("standard", "standard look", 0.79, 0.06, 3.5, 1)
LH = Mode("look_harder", "look harder", 0.84, 0.055, 4.9, 5)


# ---------------------------------------------------------------------------------- choose_mode
def test_no_goal_keeps_the_strongest_promise():
    m, rec = choose_mode(Goal(), [STD, LH], 10, 200.0)
    assert m.name == "look_harder" and rec["goal_met"]
    assert "strongest promise" in rec["why"]
    assert [o["chosen"] for o in rec["options"]] == [False, True]


def test_a_reachable_goal_takes_the_cheapest_mode_that_meets_it():
    m, rec = choose_mode(Goal(recall=0.79), [STD, LH], 10, 200.0)
    assert m.name == "standard" and rec["goal_met"]
    m, _ = choose_mode(Goal(recall=0.82), [STD, LH], 10, 200.0)
    assert m.name == "look_harder"


def test_an_unreachable_goal_is_said_so_with_the_best_available():
    m, rec = choose_mode(Goal(recall=0.90), [STD, LH], 10, 200.0)
    assert m.name == "look_harder" and not rec["goal_met"]
    assert "cannot be met" in rec["why"]
    assert all(o["why_not"] for o in rec["options"])


def test_the_compute_budget_rules_out_the_expensive_look():
    # 10 frames x 5 passes x 200 ms = 10 s > 4 s; the standard look needs 2 s
    m, rec = choose_mode(Goal(compute_s=4.0), [STD, LH], 10, 200.0)
    assert m.name == "standard" and rec["goal_met"]
    lh = next(o for o in rec["options"] if o["mode"] == "look_harder")
    assert not lh["fits_budgets"] and "compute" in lh["why_not"][0]


def test_the_analyst_budget_orders_the_queue_instead_of_blocking_a_mode():
    m, rec = choose_mode(Goal(analyst_minutes=1.0), [STD, LH], 10, 200.0)
    assert m.name == "look_harder"                              # not blocked by a small review budget
    chosen = next(o for o in rec["options"] if o["chosen"])
    assert 0 < chosen["analyst_budget_covers"] < 1 and "likeliest cards first" in rec["why"]


def test_no_calibrated_mode_means_no_promise():
    m, rec = choose_mode(Goal(), [], 3, None)
    assert m is None and not rec["goal_met"]


def test_the_shipped_calibration_carries_both_modes():
    modes = {m.name: m for m in calibrated_modes()}
    assert set(modes) >= {"standard", "look_harder"}
    assert modes["look_harder"].promise > modes["standard"].promise
    assert modes["look_harder"].passes == 5 and modes["standard"].passes == 1
    assert modes["look_harder"].verified.get("promise_held") is True


# ---------------------------------------------------------------------------------- re-tier + the loop
def _cand(conf: float, verdict: Verdict, cls: str) -> Candidate:
    ev = Evidence(relook=RelookResult(False, 0.0, 0.0, 1.0), shadow=_no_proof((0, 0, 10, 10)), echo_ratio=1.0,
                  evidence_score=conf)
    return Candidate(bbox=(0, 0, 10, 10), cls_id=0, cls_name=cls, conf=conf, evidence=ev, verdict=verdict)


def test_retier_moves_a_find_below_the_standard_threshold_to_low_risk():
    cal = load_calibration()
    std = GuaranteedTiers.from_calibration(cal, "standard")
    lh = GuaranteedTiers.from_calibration(cal, "look_harder")
    conf = (std.tau_review + lh.tau_review) / 2                 # REVIEW under look_harder, not under standard
    fr = FrameResult("f", 64, 64, [_cand(conf, Verdict.REVIEW, std.guaranteed_class),
                                   _cand(0.9, Verdict.REVIEW, std.guaranteed_class)])
    assert retier(fr, std) == 1
    assert fr.candidates[0].verdict is Verdict.LOW_RISK and fr.candidates[1].verdict is Verdict.REVIEW
    assert fr.candidates[0].trace[-1].tool == "replan" and fr.perception["retiered_to"] == "standard"


class _FakeTools:
    def __init__(self):
        self.waited = 0

    def wait_idle(self, timeout_s):
        self.waited += 1
        return True


class _FakeAgent:
    def __init__(self, mode: str):
        self.cfg = type("Cfg", (), {"tiers": GuaranteedTiers.from_calibration(load_calibration(), mode),
                                    "perception_mode": mode})()
        self.tools = _FakeTools()


class _FakePipe(AgenticPipeline):
    """The pipeline's planning half, with a detector stub: every frame yields one find just above the
    look-harder threshold, after a fixed delay (so the compute budget is predictable)."""

    def __init__(self, delay_s: float = 0.02, busy_once: tuple = (), tile_fails: tuple = ()):
        self._agents, self._ms_pass, self.m_per_px, self.delay = {}, None, 0.05, delay_s
        self.modes_run: list[str] = []
        self.busy_once, self.tile_fails = set(busy_once), set(tile_fails)

    def agent_for(self, mode):
        return self._agents.setdefault(mode, _FakeAgent(mode))

    def guarantees(self, agent=None):
        t = (agent or self.agent_for("standard")).cfg.tiers
        return {**t.guarantees, "mode": "calibrated", "tau_review": t.tau_review, "policy": t.policy}

    def run_frame(self, frame, frame_id="f", fix=None, nadir=None, progress_cb=None, faults=None, agent=None):
        time.sleep(self.delay)
        if frame_id in self.busy_once:                      # the network is still busy with an overrun call
            self.busy_once.discard(frame_id)
            raise ToolTimeout("detect not started: the detector is still busy with a call that timed out")
        mode = agent.cfg.perception_mode
        self.modes_run.append(mode)
        t = agent.cfg.tiers
        conf = t.tau_review + 0.001
        fr = FrameResult(frame_id, 64, 64, [_cand(conf, Verdict.REVIEW, t.guaranteed_class)])
        fr.perception = {"mode": mode, "passes": 4 if mode != "standard" else 0}
        if mode != "standard" and frame_id in self.tile_fails:
            fr.perception.update(passes=0, fallback="first look only", steps=[
                {"tool": "look_harder", "status": "failed", "rationale": "look_harder failed: ToolTimeout"}])
        return fr


def _frames(n: int):
    rng = np.random.default_rng(0)
    return [(f"f{i}", rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)) for i in range(n)]


def test_the_loop_runs_the_chosen_mode_and_records_every_decision():
    pipe = _FakePipe()
    s = SurveyAgent(pipe, check_applicability=False).run(_frames(3), goal=Goal())
    run = s.agent_run
    assert pipe.modes_run == ["look_harder"] * 3
    assert [st["act"] for st in run["steps"]] == ["inspect_data", "check_promise", "choose_mode", "perceive", "plan"]
    assert run["steps"][1]["status"] == "skipped"                # no reference given: kept unchecked, and said so
    assert run["mode"]["name"] == "look_harder" and run["downgraded_at"] is None
    assert s.mission.guarantees["recall_promise"] == pytest.approx(0.84)
    assert s.to_dict()["agent_run"]["goal"]["recall"] is None


def test_the_agent_replans_mid_run_when_the_compute_budget_runs_out():
    pipe = _FakePipe(delay_s=0.03)
    s = SurveyAgent(pipe, check_applicability=False).run(_frames(4), goal=Goal(compute_s=0.05))
    run = s.agent_run
    assert pipe.modes_run[0] == "look_harder" and pipe.modes_run[-1] == "standard"
    assert run["downgraded_at"] == "f0" and run["replans"][0]["to"] == "standard"
    acts = [st["act"] for st in run["steps"]]
    assert "replan" in acts and "retier" in acts
    # the look-harder frame's find sat just above the look-harder threshold: re-tiered to LOW-RISK
    assert s.frames[0].candidates[0].verdict is Verdict.LOW_RISK
    assert s.mission.guarantees["recall_promise"] == pytest.approx(STD_PROMISE)


def test_unusable_frames_are_skipped_and_reported():
    frames = _frames(2) + [("blank", np.zeros((64, 64, 3), np.uint8)), ("broken", None)]
    s = SurveyAgent(_FakePipe(), check_applicability=False).run(frames, goal=Goal(recall=STD_PROMISE))
    first = s.agent_run["steps"][0]
    assert first["act"] == "inspect_data" and set(first["observed"]["skipped"]) == {"blank", "broken"}
    codes = [i["code"] for i in s.mission.incidents]
    assert codes.count("invalid_frame") == 1 and codes.count("corrupt_frame") == 1
    assert len(s.frames) == 2


def test_a_busy_detector_is_waited_for_and_the_frame_retried_not_skipped():
    pipe = _FakePipe(busy_once=("f1",))
    s = SurveyAgent(pipe, check_applicability=False).run(_frames(3), goal=Goal(recall=STD_PROMISE))
    assert [f.frame_id for f in s.frames] == ["f0", "f1", "f2"]          # nothing skipped
    assert pipe.agent_for("standard").tools.waited == 1
    assert any(st["act"] == "retry" and st["observed"]["frame"] == "f1" for st in s.agent_run["steps"])
    assert not any(i["code"] == "tool_failed" for i in s.mission.incidents)


def test_a_failed_closer_look_downgrades_the_promise_honestly():
    pipe = _FakePipe(tile_fails=("f1",))
    s = SurveyAgent(pipe, check_applicability=False).run(_frames(4), goal=Goal())
    run = s.agent_run
    assert pipe.modes_run == ["look_harder", "look_harder", "standard", "standard"]
    assert run["downgraded_at"] == "f1" and run["replans"][0]["why"] == "closer look failed"
    assert s.mission.guarantees["recall_promise"] == pytest.approx(STD_PROMISE)   # not the look-harder 84%
    assert all(c.verdict is Verdict.LOW_RISK for f in s.frames[:2] for c in f.candidates)   # re-tiered
