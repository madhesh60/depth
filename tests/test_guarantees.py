"""
Tests for the statistical-promise machinery (src/agentic/guarantees.py). Standalone: python tests/test_guarantees.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic.guarantees import (cp_upper, cp_lower, tau_review, tau_confirm, achievable_recall,
                                    best_recall_guarantee)


def test_clopper_pearson_matches_reference_values():
    # reference values from scipy.stats.beta.ppf (one-sided, delta = 0.05)
    assert abs(cp_upper(0, 10) - 0.25887) < 1e-4
    assert abs(cp_upper(40, 135) - 0.36762) < 1e-4
    assert abs(cp_lower(20, 20) - 0.86089) < 1e-4
    assert abs(cp_lower(18, 20) - 0.71738) < 1e-4
    assert cp_upper(5, 5) == 1.0 and cp_lower(0, 7) == 0.0


def test_tau_review_is_highest_threshold_keeping_the_promise():
    pots = np.r_[np.full(95, 0.9), np.full(5, 0.02)]        # 95% of pots proposed with high score
    t = tau_review(pots, alpha=0.15, delta=0.05)
    assert t is not None and 0.5 < t <= 0.9                  # anything above 0.9 would miss 100%
    assert tau_review(np.r_[np.full(60, 0.9), np.zeros(40)], alpha=0.10) is None   # ceiling 60%


def test_best_recall_guarantee_reports_what_is_achievable():
    pots = np.r_[np.full(70, 0.5), np.zeros(30)]              # detector never proposes 30%
    promise, t = best_recall_guarantee(pots, floor=0.05)
    assert 0.5 < promise < 0.70 and t is not None and t >= 0.05
    assert achievable_recall(pots, t) >= promise - 1e-9


def test_tau_confirm_needs_enough_evidence_and_precision():
    s = np.r_[np.full(30, 0.9), np.full(30, 0.3)]
    tp = np.r_[np.ones(30, bool), np.zeros(30, bool)]
    t = tau_confirm(s, tp, target=0.85, min_n=15)
    assert t is not None and t > 0.3                          # only the clean high-score block qualifies
    assert tau_confirm(s[:10], tp[:10], target=0.85, min_n=15) is None       # too few to promise


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} guarantee tests passed")


if __name__ == "__main__":
    _run_all()
