"""
cards.py — one review card per object (STUDY-23).

The detector often puts two or three boxes on one object: a tight box and a looser one around it, and in the
look-harder mode the tile pass's own box beside the first look's. Each box used to be its own review card, so
a person judged the same pot two or three times (calibration split: 3.5 boxes per frame for the standard look,
4.9 looking harder).

**Rule.** Within one frame, two boxes of the same class that both go to a person are **one card** when at least
``ios`` of the smaller box lies inside the other (intersection over the smaller box), joined transitively. The
card's **head** is its strongest box: it gives the card's tier, P(pot), position and place in the queue. Every
other box stays on the card — drawn, listed in the trace and kept for audit. No box is dropped, so the pots that
reach a person, and with them the recall promise, cannot change; STUDY-23 checks that on both splits and
measures the card load (``study_cards.py``). ``ios`` comes from ``calibration.json`` (``cards.ios``); without it
every box stays its own card.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .types import AgentStep, Candidate, Verdict

_CARD_VERDICTS = (Verdict.REVIEW, Verdict.CONFIRMED)


def ios(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over the smaller box's area (1.0 = one box lies inside the other). Boxes are x1, y1, x2, y2."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small if small > 0 else 0.0


def group(boxes: Sequence[Sequence[float]], classes: Sequence[str], strength: Sequence, ios_min: float) -> list[list[int]]:
    """One frame's cards as lists of box indices, the head (strongest box) first; cards ordered by their head's
    index. Two boxes join when they share a class and ``ios >= ios_min``; joins are transitive (union-find)."""
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if classes[i] == classes[j] and ios(boxes[i], boxes[j]) >= ios_min:
                parent[find(j)] = find(i)
    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)
    cards = [sorted(m, key=lambda k: (strength[k], -k), reverse=True) for m in members.values()]
    return sorted(cards, key=lambda m: m[0])


def _strength(c: Candidate) -> tuple:
    ev = c.evidence
    return (c.verdict is Verdict.CONFIRMED, (ev.p_pot or 0.0) if ev else 0.0,
            ev.evidence_score if ev else c.conf, c.conf)


def fold(cands: list[Candidate], ios_min: Optional[float]) -> int:
    """Mark one card per object on a frame's candidates, in place. A box that goes to a person and is not its
    card's head gets ``card_of`` = the head's index in ``cands``; the head lists it in ``card_members``. Both get
    a trace step, so every box still traces to its card. LOW-RISK boxes are not cards and are left alone.
    Returns how many boxes were folded into another box's card."""
    for c in cands:
        c.card_of, c.card_members = None, []
    idx = [i for i, c in enumerate(cands) if c.verdict in _CARD_VERDICTS]
    if not ios_min or len(idx) < 2:
        return 0
    folded = 0
    for g in group([cands[i].bbox for i in idx], [cands[i].cls_name for i in idx],
                   [_strength(cands[i]) for i in idx], ios_min):
        if len(g) < 2:
            continue
        head = idx[g[0]]
        h = cands[head]
        for k in g[1:]:
            i = idx[k]
            m = cands[i]
            m.card_of = head
            h.card_members.append(i)
            m.trace.append(AgentStep(
                tool="one_card", latency_ms=0.0, conf_before=round(m.conf, 4), status="done",
                rationale=(f"same object as find #{head} ({h.cls_name} {h.conf:.2f}): at least {ios_min:.0%} of the "
                           "smaller box lies inside the other -> shown on that card, not as a card of its own; "
                           "the box is kept (STUDY-23)"),
                detail={"card_of": head, "ios_min": ios_min}))
            folded += 1
        h.trace.append(AgentStep(
            tool="one_card", latency_ms=0.0, conf_before=round(h.conf, 4), status="done",
            rationale=(f"{len(g) - 1} more {h.cls_name} box(es) on this object (at least {ios_min:.0%} of the smaller "
                       "box inside the other) are shown on this card -> one decision per object; no box dropped, "
                       "so the recall promise is unchanged (STUDY-23)"),
            detail={"members": list(h.card_members), "ios_min": ios_min}))
    return folded


def card_count(cands: Sequence[Candidate]) -> int:
    """Review cards a person sees on this frame (heads and single boxes; folded boxes are on their head's card)."""
    return sum(1 for c in cands if c.verdict in _CARD_VERDICTS and c.card_of is None)
