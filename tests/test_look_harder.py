"""
Tests for the look-harder perception mode (STUDY-18): the tile grid, the merge rule the study scored,
the frame agent's tile pass (trace + perception record), its failure fallback, and the calibrated
mode's tiers. The perceptor is stubbed (no ONNX needed).
"""
from __future__ import annotations

import numpy as np

from src.agentic.agent import AgentConfig, ReLookAgent
from src.agentic.perception import merge_candidates, tile_windows
from src.agentic.policy import GuaranteedTiers
from src.agentic.shadow import ShadowConfig, ShadowProver
from src.agentic.types import RelookResult, Verdict
from src.detection.calibration import load_calibration
from src.detection.infer import Detection

G = "ghost_gear"


def _d(conf, box, cls=G):
    return Detection(bbox=box, cls_id=0, cls_name=cls, conf=conf)


def test_tile_windows_match_the_training_tiles():
    w = tile_windows(640, 480, 0.6)
    assert w == [(0, 0, 384, 288), (256, 0, 384, 288), (0, 192, 384, 288), (256, 192, 384, 288)]
    xs = {x for x, _, _, _ in w}
    assert max(xs) + 384 == 640                                  # the grid covers the frame


def test_merge_raises_a_seen_object_and_adds_a_new_one():
    base = [_d(0.10, (10, 10, 30, 30)), _d(0.40, (100, 100, 120, 120))]
    extra = [_d(0.30, (11, 11, 31, 31)),                         # same object, higher score -> raised
             _d(0.20, (101, 101, 121, 121)),                     # same object, lower score -> unchanged
             _d(0.08, (200, 200, 215, 215)),                     # only the tile look saw it -> new
             _d(0.50, (12, 12, 30, 30), cls="wreck_debris")]      # other class -> never merged into a pot
    out, src = merge_candidates(base, extra, 0.5)
    assert [round(d.conf, 2) for d in out] == [0.30, 0.40, 0.50, 0.08]
    assert src == ["full+tiles", "full", "tiles", "tiles"]
    assert base[0].conf == 0.10                                  # inputs are not mutated


class _Stub:
    """A perceptor with a fixed first look and a fixed tile look."""

    guaranteed_class = G

    def __init__(self, dets, tile_dets, fail=False):
        self.dets, self.tile_dets, self.fail, self.tiles_run = dets, tile_dets, fail, 0

    def perceive(self, frame):
        return list(self.dets)

    def look_harder(self, frame, tile_frac, cls_name=None):
        self.tiles_run += 1
        if self.fail:
            raise RuntimeError("tile pass backend down")
        return list(self.tile_dets)

    def zoom_relook(self, frame, bbox, cls_name=G, enhance=False):
        return RelookResult(False, 0.0, 0.0, 1.0)

    def relook_batch(self, frame, items, enhance=False):
        return [RelookResult(False, 0.0, 0.0, 1.0) for _ in items]


def _agent(stub, look_harder: bool):
    tiers = GuaranteedTiers(tau_review=0.055, tau_confirm=None, relook_mode=None, guaranteed_class=G,
                            p_pot_bins=[{"lo": 0.05, "hi": 1.0, "p_pot": 0.3, "n": 10}],
                            guarantees={"recall_promise": 0.84})
    cfg = AgentConfig(tiers=tiers, perception_mode="look_harder" if look_harder else "standard",
                      look_harder={"tile_frac": 0.6, "merge_iou": 0.5} if look_harder else None)
    return ReLookAgent(perceptor=stub, shadow_prover=ShadowProver(ShadowConfig(nadir="top")), cfg=cfg)


def _frame():
    return (90 + np.random.default_rng(1).normal(0, 4, (200, 200, 3))).clip(0, 255).astype(np.uint8)


def test_look_harder_mode_adds_the_tile_finds_and_says_so():
    stub = _Stub([_d(0.30, (20, 20, 40, 40))], [_d(0.09, (150, 150, 165, 165)), _d(0.45, (21, 21, 41, 41))])
    r = _agent(stub, True).run_frame(_frame(), frame_id="f")
    assert stub.tiles_run == 1
    assert r.perception["mode"] == "look_harder" and r.perception["new"] == 1 and r.perception["raised"] == 1
    assert r.stage_ms["inferences"] == 5.0
    assert [s["tool"] for s in r.perception["steps"]] == ["detect", "look_harder"]
    new = next(c for c in r.candidates if c.bbox == (150, 150, 165, 165))
    assert new.verdict is Verdict.REVIEW and new.trace[0].tool == "look_harder"
    assert "only the closer 2×2 tile look proposed" in new.trace[0].rationale
    raised = next(c for c in r.candidates if c.bbox == (20, 20, 40, 40))
    assert raised.conf == 0.45 and "raised" in raised.trace[0].rationale


def test_standard_mode_skips_the_tile_pass_with_a_reason():
    stub = _Stub([_d(0.30, (20, 20, 40, 40))], [_d(0.09, (150, 150, 165, 165))])
    r = _agent(stub, False).run_frame(_frame())
    assert stub.tiles_run == 0 and len(r.candidates) == 1
    skip = r.perception["steps"][-1]
    assert skip["tool"] == "look_harder" and skip["status"] == "skipped" and "standard mode" in skip["rationale"]


def test_a_failed_tile_pass_falls_back_to_the_first_look_and_is_reported():
    stub = _Stub([_d(0.30, (20, 20, 40, 40))], [], fail=True)
    r = _agent(stub, True).run_frame(_frame(), frame_id="f9")
    assert len(r.candidates) == 1 and r.perception["fallback"]
    assert r.perception["steps"][-1]["status"] == "failed"
    assert any(i["code"] == "tool_failed" and "tile pass" in i["message"] for i in r.incidents)


def test_the_calibrated_mode_carries_its_own_promise_and_threshold():
    cal = load_calibration()
    std, lh = GuaranteedTiers.from_calibration(cal), GuaranteedTiers.from_calibration(cal, "look_harder")
    assert lh.policy == "look_harder" and lh.tau_confirm is None
    assert lh.recall_promise > std.recall_promise and lh.tau_review <= std.tau_review
    assert lh.p_pot_bins and lh.guarantees["verified_on_test"]["recall_promise_held"] is True
    assert GuaranteedTiers.from_calibration(cal, "no_such_mode") is None
    cfg = AgentConfig.from_calibration("look_harder")
    assert cfg.look_harder == {"tile_frac": 0.6, "merge_iou": 0.5} and cfg.perception_mode == "look_harder"
