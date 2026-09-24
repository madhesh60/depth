"""
jobs.py — a small, bounded background-job queue for long survey runs.

Why (review §3.8, failure scenarios 10/13): a 20-50 frame survey takes tens of seconds on CPU — longer
than a CloudFront origin timeout (30 s) — and used to run inside one HTTP request that blocked the
server. Now ``POST /api/jobs/survey`` returns a job id immediately, the survey runs on a single worker
thread (the ``cv2.dnn`` network is shared and guarded by a lock), and the UI polls
``GET /api/jobs/{id}`` for progress.

Bounded + restart-safe:
* at most ``max_jobs`` job records are kept in memory (oldest finished jobs are evicted first);
* each finished survey's result JSON and all its report files are written to ``persist_dir``
  (``$DEPTH_JOBS_DIR``, default ``runs/jobs``) so report downloads survive a server restart;
* if ``$DEPTH_S3_BUCKET`` is set and ``boto3`` is importable, the same files are uploaded to
  ``s3://$DEPTH_S3_BUCKET/results/<survey_id>/`` (best-effort; failures are logged, never fatal).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("depth.jobs")
REPO = Path(__file__).resolve().parents[2]
REPORT_FORMATS = ("geojson", "gpx", "kml", "csv", "json")


class JobStore:
    def __init__(self, max_jobs: int = 50, workers: int = 1, persist_dir: Optional[Path] = None):
        self.max_jobs = max_jobs
        self._jobs: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="depth-job")
        self.persist_dir = Path(persist_dir or os.environ.get("DEPTH_JOBS_DIR", REPO / "runs" / "jobs"))
        self.bucket = os.environ.get("DEPTH_S3_BUCKET") or None

    # -- lifecycle -----------------------------------------------------------------------
    def submit(self, fn: Callable[[Callable[[dict], None]], dict], total: int, kind: str = "survey") -> str:
        """Queue ``fn(progress_cb) -> result``. Returns the job id immediately."""
        jid = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        rec = {"id": jid, "kind": kind, "status": "queued", "created": time.time(),
               "progress": {"done": 0, "total": total, "stage": "queued"}, "result": None, "error": None}
        with self._lock:
            self._jobs[jid] = rec
            self._evict()

        def progress(p: dict) -> None:
            with self._lock:
                rec["progress"].update(p)

        def run():
            rec["status"] = "running"
            rec["started"] = time.time()
            try:
                rec["result"] = fn(progress)
                rec["status"] = "done"
            except Exception as e:                      # surfaced to the client, never crashes the server
                log.exception("job %s failed", jid)
                rec["status"], rec["error"] = "error", f"{type(e).__name__}: {e}"
            rec["finished"] = time.time()

        self._pool.submit(run)
        return jid

    def get(self, jid: str) -> Optional[dict]:
        with self._lock:
            rec = self._jobs.get(jid)
            return None if rec is None else dict(rec, progress=dict(rec["progress"]))

    def counts(self) -> dict:
        with self._lock:
            out = {"queued": 0, "running": 0, "done": 0, "error": 0}
            for r in self._jobs.values():
                out[r["status"]] = out.get(r["status"], 0) + 1
            return out

    def _evict(self) -> None:
        while len(self._jobs) > self.max_jobs:
            victim = next((k for k, r in self._jobs.items() if r["status"] in ("done", "error")), None)
            if victim is None:
                break
            self._jobs.pop(victim)

    # -- persistence -----------------------------------------------------------------------
    def persist(self, survey_id: str, result_json: dict, reports: dict[str, str]) -> Path:
        d = self.persist_dir / survey_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(result_json), encoding="utf-8")
        for fmt, body in reports.items():
            (d / f"mission.{fmt}").write_text(body, encoding="utf-8")
        if self.bucket:
            self._upload(d, survey_id)
        return d

    def report(self, survey_id: str, fmt: str) -> Optional[str]:
        if "/" in survey_id or "\\" in survey_id or ".." in survey_id:
            return None
        p = self.persist_dir / survey_id / f"mission.{fmt}"
        return p.read_text(encoding="utf-8") if p.exists() else None

    def _upload(self, d: Path, survey_id: str) -> None:
        try:
            import boto3                               # optional dependency
            s3 = boto3.client("s3")
            for f in d.iterdir():
                s3.upload_file(str(f), self.bucket, f"results/{survey_id}/{f.name}")
        except Exception as e:                         # best-effort; local copy is authoritative
            log.warning("S3 upload skipped for %s: %s", survey_id, e)
