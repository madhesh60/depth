"""
Smoke tests for the FastAPI service (src/dashboard/app.py) via Starlette's TestClient.

Health/samples/validation always run; the analyze+survey+report path runs only when the ONNX model
and dataset samples are present locally (skipped gracefully otherwise). Standalone: python tests/test_api.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from src.dashboard.app import app
from src.dashboard import samples as samples_mod
from src.detection.infer import DEFAULT_ONNX

client = TestClient(app)
_HAVE_MODEL = DEFAULT_ONNX.exists()
_HAVE_SAMPLES = len(samples_mod.list_samples()) > 0


def test_health():
    j = client.get("/api/health").json()
    assert j["status"] == "ok" and j["opencv"]


def test_samples_endpoint():
    j = client.get("/api/samples").json()
    assert "samples" in j and isinstance(j["samples"], list)


def test_analyze_requires_input():
    assert client.post("/api/analyze").status_code == 400


def test_health_reports_provenance_and_limits():
    j = client.get("/api/health").json()
    assert "cv2_file" in j and isinstance(j["is_cool_path"], bool)
    assert j["limits"]["max_frames"] >= 1 and j["limits"]["max_upload_mb"] > 0
    assert set(j["jobs"]) >= {"queued", "running", "done", "error"}


def test_upload_rejects_non_images():
    r = client.post("/api/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_upload_rejects_undecodable_image():
    r = client.post("/api/analyze", files={"file": ("broken.png", b"not really a png", "image/png")})
    assert r.status_code == 400


def test_survey_frame_limit(monkeypatch):
    import src.dashboard.app as app_mod
    monkeypatch.setattr(app_mod, "MAX_FRAMES", 2)
    files = [("files", (f"f{i}.png", b"x", "image/png")) for i in range(3)]
    assert client.post("/api/jobs/survey", files=files).status_code == 413


def test_sync_survey_redirects_big_runs_to_jobs(monkeypatch):
    if not _HAVE_SAMPLES:
        return
    import src.dashboard.app as app_mod
    monkeypatch.setattr(app_mod, "MAX_SYNC_FRAMES", 1)
    r = client.post("/api/survey?use_samples=1")
    assert r.status_code == 413 and "/api/jobs/survey" in r.json()["detail"]


def test_unknown_job_is_404():
    assert client.get("/api/jobs/nope").status_code == 404


def test_metrics_reports_host_and_provenance():
    j = client.get("/api/metrics").json()
    assert j["host"]["machine"] and j["host"]["vcpus"]
    assert isinstance(j["opencv"]["is_cool_path"], bool) and j["opencv"]["version"]
    assert "stages_ms" in j and j["frames_total"] >= 0


def test_analyze_and_report_path():
    if not (_HAVE_MODEL and _HAVE_SAMPLES):
        print("  skip  test_analyze_and_report_path (model/samples absent)")
        return
    sid = samples_mod.list_samples()[0]["id"]
    a = client.post(f"/api/analyze?sample={sid}")
    assert a.status_code == 200
    body = a.json()
    assert "counts" in body and "overlay_png" in body and body["overlay_png"].startswith("data:image")
    for c in body["candidates"]:
        assert c["verdict"] in ("confirmed", "review", "low_risk")
        assert c["trace"] and c["trace"][-1]["tool"] == "decide"

    s = client.post("/api/survey?use_samples=1&gps=synthetic")
    assert s.status_code == 200
    survey_id = s.json()["survey_id"]
    assert s.json()["mission"]["human_approval_required"] is True
    rep = client.get(f"/api/report/geojson?survey_id={survey_id}")
    assert rep.status_code == 200 and "FeatureCollection" in rep.text
    assert client.get("/api/report/gpx?survey_id=does-not-exist").status_code == 404
    md = client.get(f"/api/report/brief?survey_id={survey_id}")
    assert md.status_code == 200 and md.text.startswith("# Mission brief") and "human approval" in md.text
    assert md.headers["content-type"].startswith("text/markdown")
    b = client.get(f"/api/brief?survey_id={survey_id}&writer=template").json()
    assert b["writer"] == "template" and b["grounding"]["ok"] and b["facts"]["gps"] == "synthetic"
    assert client.get("/api/brief?survey_id=does-not-exist").status_code == 404

    # the same survey as a background job: queue → poll → result → report
    import time
    job = client.post("/api/jobs/survey?use_samples=1&gps=synthetic").json()
    rec = {}
    for _ in range(600):
        rec = client.get(f"/api/jobs/{job['job_id']}").json()
        if rec["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert rec["status"] == "done", rec.get("error")
    assert rec["result"]["survey_id"] == job["survey_id"]
    assert rec["progress"]["done"] == rec["progress"]["total"] == job["frames"]
    assert client.get(f"/api/report/csv?survey_id={job['survey_id']}").status_code == 200
    # evicted from memory → the brief is rebuilt from the persisted JSON report
    from src.dashboard import app as app_mod
    app_mod._SURVEYS.pop(job["survey_id"], None)
    bp = client.get(f"/api/brief?survey_id={job['survey_id']}&public=1").json()
    assert bp["grounding"]["ok"] and bp["facts"]["public"] is True and bp["facts"]["frames"] == job["frames"]

    # every analysed frame (analyze + both surveys) fed the live per-stage metrics
    m = client.get("/api/metrics").json()
    assert m["stages_ms"]["see"]["n"] >= 1 + 2 * job["frames"] and m["stages_ms"]["stage1"]["p50"] >= 0
    assert body["stage1"]["orientation"]["nadir"] in ("top", "unknown")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} api tests passed")


if __name__ == "__main__":
    _run_all()
