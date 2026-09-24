"""
guarantees.py — turn tier thresholds into statistical PROMISES (conformal risk control / LTT).

DEPTH's detector is imperfect (EXP-001: AP ~0.47 on crab pots, never proposes ~27% of them). What an
NGO actually needs is two promises about the tiers the agent produces:

1. **Recall promise** — "at least 1-α of pots reach a human (REVIEW or CONFIRMED)".
2. **Precision promise** — "at least 1-ε of CONFIRMED finds are real".

Both are fit on a **calibration set of held-out frames** and hold with probability 1-δ, using exact
**Clopper-Pearson** binomial bounds. Thresholds are scanned in a **fixed order** and the scan stops at
the first failure (Learn-Then-Test, Angelopoulos et al. 2021-22), so trying many thresholds does not
break the guarantee.

Honesty built in: if a promise is not achievable with the model, the fitter returns ``None`` and
:func:`achievable_recall` reports what *is* achievable — e.g. "guaranteed recall with EXP-001: 62%",
which is exactly the scientific argument for EXP-002.

Caveat stated openly: frames from one recording are correlated, so the i.i.d. assumption behind the
bound is optimistic; ``calibrate.py`` therefore also reports a recording/frame-group bootstrap and a
verification on a *separate* test split.

Pure Python (``math.lgamma``) — no SciPy at runtime.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np


# ---- exact binomial tail + Clopper-Pearson bounds ----------------------------------------------
def _binom_cdf(k: int, n: int, p: float) -> float:
    """P[X ≤ k] for X ~ Bin(n, p), summed in log-space (stable for n in the thousands)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    lg_n1 = math.lgamma(n + 1)
    terms = [lg_n1 - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(k + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def _bisect(f, lo: float = 0.0, hi: float = 1.0, iters: int = 60) -> float:
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def cp_upper(k: int, n: int, delta: float = 0.05) -> float:
    """One-sided (1-δ) Clopper-Pearson UPPER bound on a rate observed as k/n."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    # smallest p with P[X ≤ k | p] ≤ δ
    return _bisect(lambda p: _binom_cdf(k, n, p) <= delta)


def cp_lower(k: int, n: int, delta: float = 0.05) -> float:
    """One-sided (1-δ) Clopper-Pearson LOWER bound on a rate observed as k/n."""
    if n <= 0 or k <= 0:
        return 0.0
    # largest p with P[X ≥ k | p] ≤ δ  ⇔  1 - P[X ≤ k-1 | p] ≤ δ
    return _bisect(lambda p: 1.0 - _binom_cdf(k - 1, n, p) > delta)


# ---- threshold fitting (fixed-sequence / Learn-Then-Test) --------------------------------------
def default_grid(lo: float = 0.0, hi: float = 1.0, n: int = 201) -> np.ndarray:
    return np.round(np.linspace(lo, hi, n), 4)


def tau_review(pot_scores: Sequence[float], alpha: float = 0.10, delta: float = 0.05,
               grid: Optional[Iterable[float]] = None) -> Optional[float]:
    """Highest threshold that still guarantees miss-rate ≤ α (scan low → high, stop at first failure).

    ``pot_scores[i]`` = best score of any candidate that overlaps true pot *i* (0 if never proposed).
    A pot is *missed* at threshold t when its score < t. Returns None when even the lowest threshold
    fails — the detector cannot support that recall."""
    s = np.asarray(pot_scores, float)
    n = len(s)
    best = None
    for t in (default_grid() if grid is None else grid):
        if cp_upper(int((s < t).sum()), n, delta) <= alpha:
            best = float(t)
        else:
            break
    return best


def tau_confirm(cand_scores: Sequence[float], cand_tp: Sequence[bool], target: float = 0.85,
                delta: float = 0.05, min_n: int = 20,
                grid: Optional[Iterable[float]] = None) -> Optional[float]:
    """Lowest threshold whose precision LOWER bound stays ≥ target (scan high → low, stop at the first
    failure once at least ``min_n`` candidates are selected). None ⇒ no auto-confirm can be promised."""
    s, tp = np.asarray(cand_scores, float), np.asarray(cand_tp, bool)
    best = None
    for t in (default_grid()[::-1] if grid is None else grid):
        sel = s >= t
        n = int(sel.sum())
        if n < min_n:
            continue
        if cp_lower(int(tp[sel].sum()), n, delta) >= target:
            best = float(t)
        else:
            break
    return best


def achievable_recall(pot_scores: Sequence[float], tau: float, delta: float = 0.05) -> float:
    """The recall that CAN be promised at threshold ``tau``: 1 - CP upper bound on the miss rate."""
    s = np.asarray(pot_scores, float)
    return 1.0 - cp_upper(int((s < tau).sum()), len(s), delta)


def best_recall_guarantee(pot_scores: Sequence[float], floor: float, delta: float = 0.05,
                          step: float = 0.01) -> tuple[float, Optional[float]]:
    """Honest fallback: the smallest α (on a ``step`` grid) whose promise holds at the lowest
    threshold ``floor``, and the highest threshold that still keeps it. Returns (1-α, τ_review)."""
    rec = achievable_recall(pot_scores, floor, delta)
    alpha = math.ceil((1.0 - rec) / step - 1e-9) * step
    return round(1.0 - alpha, 4), tau_review(pot_scores, alpha=alpha + 1e-12, delta=delta,
                                             grid=default_grid(floor, 1.0, int(round((1 - floor) / 0.005)) + 1))


def precision_at(cand_scores, cand_tp, tau: float) -> tuple[int, int, float]:
    s, tp = np.asarray(cand_scores, float), np.asarray(cand_tp, bool)
    sel = s >= tau
    n, k = int(sel.sum()), int(tp[sel].sum())
    return k, n, (k / n if n else float("nan"))
