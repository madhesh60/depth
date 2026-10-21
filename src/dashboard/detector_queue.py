"""
detector_queue.py — how much compute is already waiting for the one ``cv2.dnn`` network.

The server shares one network (``app._INFER_LOCK``): ``/api/analyze`` calls run one at a time, and a survey
job takes the lock frame by frame between them. An analyze call answers within an interactive budget
(``DEPTH_INTERACTIVE_S``, 2 s), and the agent picks its perception mode against that budget — the look-harder
mode costs 5 forward passes, the standard look 1. On an idle server the budget is all compute; on a busy one
part of it is spent waiting. So before it chooses, the agent asks this queue how much expected compute is
ahead of it, prices its modes against the time that is left, and logs the load it saw: a busy server gets
the standard look and an answer on time instead of a 5-pass look behind a queue.
"""
from __future__ import annotations

import itertools
import threading
from contextlib import contextmanager
from typing import Iterator


class DetectorQueue:
    """Expected seconds of every analyze call in flight (registered with :meth:`call`)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._calls: dict[int, float] = {}
        self._ids = itertools.count()

    def ahead(self) -> tuple[int, float]:
        """(calls in flight, their expected compute in seconds)."""
        with self._lock:
            return len(self._calls), sum(self._calls.values())

    @contextmanager
    def call(self, expected_s: float) -> Iterator[None]:
        """Register one call's expected compute while it waits for and uses the network."""
        k = next(self._ids)
        with self._lock:
            self._calls[k] = max(0.0, float(expected_s))
        try:
            yield
        finally:
            with self._lock:
                self._calls.pop(k, None)


def interactive_budget(budget_s: float, calls_ahead: int, wait_s: float, survey_running: bool,
                       ms_per_pass) -> dict:
    """The compute left for this call once the work ahead of it is done. A running survey holds the
    network for one frame at a time, so it adds one forward pass of waiting."""
    if survey_running:
        calls_ahead += 1
        wait_s += (ms_per_pass or 0.0) / 1000.0
    return {"budget_s": budget_s, "calls_ahead": calls_ahead, "wait_s": round(wait_s, 2),
            "left_s": round(max(0.0, budget_s - wait_s), 2), "survey_running": survey_running}
