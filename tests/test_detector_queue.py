"""
The analyze call prices its look against the server's queue: a busy detector leaves less of the
interactive budget, so the agent takes the standard look instead of a 5-pass look behind the queue.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic.core import Mode, choose_mode
from src.dashboard.detector_queue import DetectorQueue, interactive_budget

_MODES = [Mode("standard", "standard look", 0.79, 0.06, 2.7, 1),
          Mode("look_harder", "look harder", 0.84, 0.055, 3.9, 5)]


def test_the_queue_counts_calls_in_flight():
    q = DetectorQueue()
    assert q.ahead() == (0, 0.0)
    with q.call(1.5):
        with q.call(0.3):
            n, s = q.ahead()
            assert n == 2 and abs(s - 1.8) < 1e-9
        assert q.ahead()[0] == 1
    assert q.ahead() == (0, 0.0)


def test_the_queue_is_released_when_the_call_fails():
    q = DetectorQueue()
    try:
        with q.call(2.0):
            raise RuntimeError("detector failed")
    except RuntimeError:
        pass
    assert q.ahead() == (0, 0.0)


def test_the_queue_is_thread_safe():
    q, go = DetectorQueue(), threading.Event()

    def worker():
        with q.call(0.1):
            go.wait(2)
    ts = [threading.Thread(target=worker) for _ in range(8)]
    for t in ts:
        t.start()
    while q.ahead()[0] < 8:
        pass
    go.set()
    for t in ts:
        t.join()
    assert q.ahead() == (0, 0.0)


def test_a_running_survey_adds_one_pass_of_waiting():
    b = interactive_budget(2.0, 0, 0.0, True, 300.0)
    assert b["calls_ahead"] == 1 and b["wait_s"] == 0.3 and b["left_s"] == 1.7


def test_an_idle_server_looks_harder_and_a_busy_one_does_not():
    ms = 300.0                                                     # 5 passes = 1.5 s
    idle = interactive_budget(2.0, 0, 0.0, False, ms)
    mode, _ = choose_mode(_goal(idle["left_s"]), _MODES, 1, ms)
    assert mode.name == "look_harder"
    busy = interactive_budget(2.0, 1, 1.5, False, ms)              # one look-harder call ahead
    mode, choice = choose_mode(_goal(busy["left_s"]), _MODES, 1, ms)
    assert busy["left_s"] == 0.5 and mode.name == "standard"
    assert any(o["mode"] == "look_harder" and not o["fits_budgets"] for o in choice["options"])


def _goal(left_s: float):
    from src.agentic.core import Goal
    return Goal(compute_s=max(0.2, left_s))


def test_the_rest_of_an_answer_comes_off_the_budget():
    # 5 passes at 300 ms (1.5 s) fit a 2-s answer, but not once 0.6 s of Stage 1, images and provenance is paid
    b = interactive_budget(2.0, 0, 0.0, False, 300.0, overhead_s=0.6)
    assert b["overhead_s"] == 0.6 and b["left_s"] == 1.4
    mode, _ = choose_mode(_goal(b["left_s"]), _MODES, 1, 300.0)
    assert mode.name == "standard"


def test_the_queue_measures_the_rest_of_an_answer():
    q = DetectorQueue()
    assert q.overhead_s() == 0.0                                   # nothing measured yet: nothing taken off
    q.note_overhead(500.0)
    assert q.overhead_s() == 0.5
    q.note_overhead(0.0)                                           # a moving average: one fast answer
    assert abs(q.overhead_s() - 0.4) < 1e-9                        # does not erase it
    q.note_overhead(-5.0)                                          # clock noise never enlarges the budget
    assert q.overhead_s() <= 0.4
