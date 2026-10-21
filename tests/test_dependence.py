"""
STUDY-21 (does the recall promise survive correlated frames?): blocks, the block bootstrap and rule R1.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic.study_dependence import apply_r1, block_bootstrap, blocks_of, cp_lcb, pot_rows


def _frame(rec: str, side: str, idx: int, hits: list[bool]) -> dict:
    """A frame in calibrate.py's format: one candidate on each pot it 'reached' (conf 0.5)."""
    gt = [[10 + 40 * i, 10, 40 + 40 * i, 40] for i in range(len(hits))]
    cands = [{"bbox": g, "conf": 0.5} for g, h in zip(gt, hits) if h]
    return {"name": f"{rec}_wcp_ss_{side}_{idx:05d}.jpg", "frame": f"{rec}_wcp_ss_{side}_{idx:05d}",
            "recording": rec, "gt": gt, "cands": cands}


def test_port_and_starboard_of_the_same_pings_share_a_block():
    rows = pot_rows([_frame("RecA", "port", 3, [True]), _frame("RecA", "star", 3, [False]),
                     _frame("RecA", "port", 9, [True])])
    b1 = blocks_of(rows, 0.1, 1)["RecA"]
    assert len(b1) == 2 and sorted(map(tuple, b1.tolist())) == [(1.0, 1.0), (1.0, 2.0)]
    assert len(blocks_of(rows, 0.1, 8)["RecA"]) == 2                 # 3 // 8 = 0, 9 // 8 = 1


def test_never_proposed_pots_count_as_misses():
    rows = pot_rows([_frame("RecA", "port", 1, [True, False])])
    assert rows[0]["scores"] == [0.5, 0.0]


def _split(clustered: bool, n_blocks: int = 60) -> list[dict]:
    """160 pots in 80% recall: either spread evenly (independent) or all-or-nothing per block of 8 frames."""
    frames = []
    for blk in range(n_blocks):
        for k in range(8):
            idx = blk * 8 + k
            hit = (blk % 5 != 0) if clustered else (idx % 5 != 0)
            frames.append(_frame("RecA" if blk % 2 else "RecB", "port", idx, [hit]))
    return frames


def test_clustered_misses_widen_the_bound():
    ind, clu = _split(False), _split(True)
    lo = {}
    for name, fr in (("independent", ind), ("clustered", clu)):
        r = block_bootstrap(blocks_of(pot_rows(fr), 0.1, 8), reps=4000)
        lo[name] = float(sorted(r)[int(0.05 * len(r))])
    assert lo["clustered"] < lo["independent"] - 0.03


def test_rule_r1_restates_only_when_the_bound_falls_short():
    ind, clu = _split(False), _split(True)
    n = 480
    k = int(0.8 * n)
    promise = round(int(cp_lcb(k, n) * 100) / 100, 2)               # what Clopper-Pearson alone supports
    stands = apply_r1(promise, ind, 0.1)
    assert stands["recall_promise"] == promise and "recall_promise_cp" not in stands
    restated = apply_r1(promise, clu, 0.1)
    assert restated["recall_promise"] < promise and restated["recall_promise_cp"] == promise
    assert restated["recall_promise"] == int(restated["dependence_lcb"] * 100) / 100
