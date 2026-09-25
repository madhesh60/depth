"""
Tests for the mission brief (src/agentic/brief.py): the deterministic template, the grounding check
that gates any LLM-written brief, and the fall-back contract. No network: the Bedrock client is faked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic import brief
from src.agentic.mission import export
from src.agentic.types import MissionPlan, SurveyResult, TrackedObject, Verdict

PROV = {"model": "EXP-001", "code_commit": "abc1234", "opencv": {"version": "5.0.0", "is_cool_path": True}}


def _survey() -> SurveyResult:
    pots = [TrackedObject(f"H{i:03d}", "fishing_gear", Verdict.REVIEW, 0.5, 0.5, "F1", (0, 0, 9, 9),
                          lat=37.80 + i * 1e-4, lon=-76.15, p_pot=0.7 - i * 0.1, shadow_quality="clear")
            for i in range(1, 4)]
    wreck = TrackedObject("H004", "wreck_debris", Verdict.REVIEW, 0.4, 0.4, "F1", (0, 0, 9, 9),
                          lat=37.81, lon=-76.15, p_pot=0.2)
    m = MissionPlan([], ["H001", "H002", "H003", "H004"], 0.0, True, gps_synthetic=True,
                    counts={"confirmed": 0, "review": 4, "low_risk": 0},
                    review_queue=["H001", "H002", "H003", "H004"], inspection_route=["H001", "H002"],
                    inspection_length_m=31.4)
    m.guarantees = {"recall_promise": 0.65, "precision_promise": None, "tau_confirm": None,
                    "verified_on_test": {"recall": 0.8623, "recall_promise_held": True}}
    m.budget = {"minutes": 0.5, "sec_per_card": 8.0, "sec_per_card_source": "ASSUMED - replace with the timed user study",
                "cards_total": 4, "cards_affordable": 2, "review_ids": ["H001", "H002"],
                "expected_pots_in_budget": 1.2, "expected_pots_in_queue": 1.6, "share_of_expected_pots": 0.75,
                "minutes_for_whole_queue": 0.5}
    m.resurvey_plan = {"lines": [{"id": "RS1", "targets": ["H001", "H002"], "targets_on": "port", "heading_deg": 20.0,
                                  "boat_min": 2.3, "voi": 0.4, "start": [37.8, -76.15], "end": [37.81, -76.15],
                                  "predictions": [{"id": "H001", "p_pot": 0.6, "shadow_was": 110.0, "shadow_must_point": 290.0}]},
                                 {"id": "RS2", "targets": ["H004"], "targets_on": "port", "heading_deg": 20.0,
                                  "boat_min": 1.9, "voi": 0.16, "start": [37.81, -76.15], "end": [37.82, -76.15],
                                  "predictions": [{"id": "H004", "p_pot": 0.2, "shadow_was": 110.0, "shadow_must_point": 290.0}]}],
                       "boat_minutes_planned": 4.2, "targets_covered": 3, "targets_total": 4}
    return SurveyResult("s-test", [], pots + [wreck], m)


def _facts(public=False):
    return brief.facts(_survey().to_dict(), PROV, public)


def test_template_is_grounded_and_carries_the_caveats():
    F = _facts()
    t = brief.template_brief(F)
    g = brief.check(t, F)
    assert g["ok"], g
    assert g["numbers_checked"] >= 15
    low = t.lower()
    assert "synthetic" in low and "assumed" in low and "human approval" in low
    assert "86%" in t and "65%" in t and "RS1" in t and "H001" in t
    assert "37.8" not in t                           # no coordinates in the brief


def test_check_rejects_invented_numbers_ids_and_missing_caveats():
    F = _facts()
    t = brief.template_brief(F)
    assert not brief.check(t.replace("**4 possible hazards**", "**5 possible hazards**"), F)["ok"]
    assert "5" in brief.check(t.replace("**4 possible hazards**", "**5 possible hazards**"), F)["unsupported_numbers"]
    assert brief.check(t.replace("H001", "H777"), F)["unknown_ids"] == ["H777"]
    no_synth = t.replace("synthetic", "demo").replace("SYNTHETIC", "DEMO")
    assert "synthetic GPS" in brief.check(no_synth, F)["missing_caveats"]
    assert "twelve" not in t and not brief.check(t + "\nAbout twelve more.\n", F)["ok"]      # number words count
    assert not brief.check(t + " word" * brief.MAX_WORDS, F)["ok"]                         # word cap


def test_check_accepts_percentages_rounding_and_identifiers():
    F = _facts()
    ok = ("Bottom line: 4 finds, all for a human; nothing is dispatched without human approval. "
          "At least 65% of pots reach a human (86% on test). Review 2 of 4 cards: 75% of the expected pots. "
          "Model EXP-001, OpenCV 5.0.0. Positions are a synthetic demo track; card time is assumed. RS1 first.")
    g = brief.check(ok, F)
    assert g["ok"], g


class _FakeClient:
    def __init__(self, text=None, stop="end_turn", exc=None):
        self.text, self.stop, self.exc, self.calls = text, stop, exc, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(stop_reason=self.stop, content=[SimpleNamespace(type="text", text=self.text)])


def test_llm_brief_used_only_when_grounded():
    d = _survey().to_dict()
    good = ("## Bottom line\n4 possible hazards, 4 for review. Nothing is dispatched without human approval.\n"
            "## Caveats\nSynthetic demo GPS. The 8 s card time is assumed.\n")
    c = _FakeClient(good)
    r = brief.write(d, PROV, writer="llm", client=c)
    assert r["writer"] == "llm" and r["grounding"]["ok"] and r["text"].startswith("## Bottom line")
    kw = c.calls[0]
    assert kw["model"].startswith("anthropic.") and "temperature" not in kw
    assert '"survey_id": "s-test"' in kw["messages"][0]["content"]
    assert "37.8" not in kw["messages"][0]["content"]         # the model never sees coordinates

    bad = _FakeClient(good.replace("4 possible", "9 possible"))
    r = brief.write(d, PROV, writer="llm", client=bad)
    assert r["writer"] == "template" and "grounding" in r["fallback_reason"] and r["rejected"]["grounding"]["unsupported_numbers"] == ["9"]


def test_llm_failures_fall_back_to_the_template():
    d = _survey().to_dict()
    for c in (_FakeClient("x", stop="refusal"), _FakeClient("x", stop="max_tokens"),
              _FakeClient(exc=ConnectionError("no route to bedrock"))):
        r = brief.write(d, PROV, writer="llm", client=c)
        assert r["writer"] == "template" and r["fallback_reason"] and brief.check(r["text"], r["facts"])["ok"]
    # auto without DEPTH_BRIEF_LLM and without a client → template, no fallback noise
    r = brief.write(d, PROV, writer="auto")
    assert r["writer"] == "template" and r["fallback_reason"] is None


def test_llm_without_sdk_falls_back(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_anthropic(name, *a, **k):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_anthropic)
    r = brief.write(_survey().to_dict(), PROV, writer="llm")
    assert r["writer"] == "template" and "ImportError" in r["fallback_reason"]


def test_brief_export_and_public_share():
    s = _survey()
    priv = export("brief", s, prov=PROV)
    pub = export("brief", s, public=True, prov=PROV)
    assert priv.startswith("# Mission brief") and "generalise their positions before sharing" in priv
    assert "generalised in this public share" in pub
    assert "2 passes" in priv and "1 pass," in pub          # RS2 would point at the wreck → dropped publicly


def _run_all():
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))


if __name__ == "__main__":
    _run_all()
