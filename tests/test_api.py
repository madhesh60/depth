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
        assert c["verdict"] in ("confirmed", "review", "rejected")
        assert c["trace"] and c["trace"][-1]["tool"] == "decide"

    s = client.post("/api/survey?use_samples=1&gps=synthetic")
    assert s.status_code == 200
    survey_id = s.json()["survey_id"]
    assert s.json()["mission"]["human_approval_required"] is True
    rep = client.get(f"/api/report/geojson?survey_id={survey_id}")
    assert rep.status_code == 200 and "FeatureCollection" in rep.text
    assert client.get("/api/report/gpx?survey_id=does-not-exist").status_code == 404


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} api tests passed")


if __name__ == "__main__":
    _run_all()
