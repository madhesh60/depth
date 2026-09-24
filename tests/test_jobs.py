"""
Tests for the bounded background-job queue (src/dashboard/jobs.py) — no model needed.

Covers: submit → poll → done, progress updates, error capture, eviction of finished jobs only,
on-disk persistence of reports (restart-safe downloads) and path-traversal refusal.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dashboard.jobs import JobStore


def _wait(store: JobStore, jid: str, timeout: float = 5.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        rec = store.get(jid)
        if rec["status"] in ("done", "error"):
            return rec
        time.sleep(0.02)
    raise AssertionError(f"job {jid} did not finish")


def test_submit_progress_done(tmp_path):
    store = JobStore(persist_dir=tmp_path)

    def work(progress):
        for i in range(3):
            progress({"done": i + 1, "total": 3, "stage": "frame_done"})
        return {"answer": 42}

    jid = store.submit(work, total=3)
    rec = _wait(store, jid)
    assert rec["status"] == "done" and rec["result"] == {"answer": 42}
    assert rec["progress"]["done"] == 3 and rec["progress"]["total"] == 3
    assert store.counts()["done"] == 1


def test_error_is_captured_not_raised(tmp_path):
    store = JobStore(persist_dir=tmp_path)

    def boom(_progress):
        raise ValueError("bad frame")

    rec = _wait(store, store.submit(boom, total=1))
    assert rec["status"] == "error" and "ValueError" in rec["error"] and "bad frame" in rec["error"]


def test_eviction_keeps_unfinished_jobs(tmp_path):
    store = JobStore(max_jobs=2, persist_dir=tmp_path)
    ids = [store.submit(lambda p: {}, total=1) for _ in range(2)]
    for jid in ids:
        _wait(store, jid)
    newest = store.submit(lambda p: {}, total=1)
    _wait(store, newest)
    assert store.get(ids[0]) is None                     # oldest finished job evicted
    assert store.get(newest) is not None


def test_persist_and_report_from_disk(tmp_path):
    store = JobStore(persist_dir=tmp_path)
    store.persist("survey-x", {"survey_id": "survey-x"}, {"geojson": '{"type":"FeatureCollection"}', "csv": "a,b\n"})
    fresh = JobStore(persist_dir=tmp_path)               # simulates a server restart
    assert "FeatureCollection" in fresh.report("survey-x", "geojson")
    assert fresh.report("survey-x", "csv") == "a,b\n"
    assert fresh.report("survey-missing", "csv") is None
    assert fresh.report("../survey-x", "csv") is None    # path traversal refused
    assert fresh.report("a/b", "csv") is None


def _run_all():
    import tempfile
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} job tests passed")


if __name__ == "__main__":
    _run_all()
