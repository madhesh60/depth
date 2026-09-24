"""
metrics.py — live per-stage latency on THIS server + where it runs (COOL provenance), for the UI's
Runtime panel, ``/api/metrics`` and CloudWatch.

Every analysed frame (single analyze or survey frame) records its ``stage_ms`` (stage1 · see ·
prove_decide) and wall time into a bounded window; ``/api/metrics`` returns rolling p50/p95 per
stage plus the host fingerprint (architecture, vCPUs, EC2 instance type when on EC2, OpenCV
version and whether ``cv2`` is loaded from ``/opt/cool``). With ``$DEPTH_LOG_JSON=1`` each frame also
emits one JSON log line (``{"evt": "frame", ...}``) — journald → CloudWatch agent → a metric filter
gives per-stage latency dashboards without any extra code on the hot path.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

log = logging.getLogger("depth.metrics")
_JSON = os.environ.get("DEPTH_LOG_JSON", "0") == "1"
STAGES = ("stage1", "see", "prove_decide", "wall")


class Metrics:
    def __init__(self, window: int = 500):
        self._win: deque = deque(maxlen=window)
        self._lock = threading.Lock()
        self.started = time.time()
        self.frames_total = 0
        self._host: Optional[dict] = None

    def record(self, kind: str, frame_id: str, stage_ms: dict, wall_ms: Optional[float] = None,
               counts: Optional[dict] = None) -> None:
        row = {k: float(stage_ms[k]) for k in ("stage1", "see", "prove_decide") if k in stage_ms}
        if wall_ms is not None:
            row["wall"] = float(wall_ms)
        with self._lock:
            self._win.append(row)
            self.frames_total += 1
        if _JSON:
            print(json.dumps({"evt": "frame", "kind": kind, "frame": frame_id,
                              "ms": {k: round(v, 1) for k, v in row.items()}, "counts": counts or {}}),
                  flush=True)

    def host(self) -> dict:
        """Fingerprint once (EC2 IMDS lookup has a short timeout; cached)."""
        if self._host is None:
            from src.bench.fingerprint import host, opencv_provenance, ec2_identity
            self._host = {"host": host(), "opencv": opencv_provenance(), "ec2": ec2_identity()}
        return self._host

    def summary(self) -> dict:
        with self._lock:
            rows = list(self._win)
        stages = {}
        for k in STAGES:
            v = [r[k] for r in rows if k in r]
            if v:
                a = np.asarray(v)
                stages[k] = {"p50": round(float(np.percentile(a, 50)), 1),
                             "p95": round(float(np.percentile(a, 95)), 1), "n": len(v)}
        return {"window": len(rows), "frames_total": self.frames_total,
                "uptime_s": round(time.time() - self.started, 1), "stages_ms": stages, **self.host()}
