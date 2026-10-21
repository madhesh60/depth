"""
detector_queue.py — how much compute is already waiting for the one ``cv2.dnn`` network.

The server shares one network (``app._INFER_LOCK``): ``/api/analyze`` calls run one at a time, and a survey
job takes the lock frame by frame between them. An analyze call answers within an interactive budget
(``DEPTH_INTERACTIVE_S``, 2 s), and the agent picks its perception mode against that budget — the look-harder
mode costs 5 forward passes, the standard look 1. On an idle server the budget is all compute; on a busy one
part of it is spent waiting. So before it chooses, the agent asks this queue how much expected compute is
ahead of it, prices its modes against the time that is left, and logs the load it saw: a busy server gets
the standard look and an answer on time instead of a 5-pass look behind a queue. The rest of an answer
(Stage 1, evidence, images, provenance, the 3D view) is measured per call and comes off the budget too, so
the detector passes are priced against the time they really have.
"""
from __future__ import annotations

import itertools
import threading
from contextlib import contextmanager
from typing import Iterator, Optional


class DetectorQueue:
    """Expected seconds of every analyze call in flight (registered with :meth:`call`), and the measured rest
    of an answer (:meth:`note_overhead`)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[int, float] = {}
        self._ids = itertools.count()
        self._overhead_ms: Optional[float] = None

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

    def note_overhead(self, ms: float) -> None:
        """Record the part of one answer that is not detector passes or waiting — a moving average, like the
        pass time the modes are priced with."""
        ms = max(0.0, float(ms))
        with self._lock:
            x = self._overhead_ms
            self._overhead_ms = ms if x is None else 0.8 * x + 0.2 * ms

    def overhead_s(self) -> float:
        """The measured rest of an answer, in seconds (0 until an answer has been measured)."""
        with self._lock:
            return 0.0 if self._overhead_ms is None else self._overhead_ms / 1000.0


def interactive_budget(budget_s: float, calls_ahead: int, wait_s: float, survey_running: bool,
                       ms_per_pass: Optional[float], overhead_s: float = 0.0) -> dict:
    """The compute left for this call's detector passes once the work ahead of it is done and the rest of
    the answer is paid for. A running survey holds the network for one frame at a time, so it adds one forward
    pass of waiting."""
    if survey_running:
        calls_ahead += 1
        wait_s += (ms_per_pass or 0.0) / 1000.0
    overhead_s = max(0.0, float(overhead_s or 0.0))
    return {"budget_s": budget_s, "calls_ahead": calls_ahead, "wait_s": round(wait_s, 2),
            "overhead_s": round(overhead_s, 2), "left_s": round(max(0.0, budget_s - wait_s - overhead_s), 2),
            "survey_running": survey_running}
