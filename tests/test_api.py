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


def test_json_is_gzipped_when_the_client_accepts_it():
    r = client.get("/api/health", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200 and r.headers.get("content-encoding") == "gzip" and r.json()["status"] == "ok"
    assert client.get("/api/health", headers={"Accept-Encoding": "identity"}).headers.get("content-encoding") is None


def test_samples_endpoint():
    j = client.get("/api/samples").json()
    assert "samples" in j and isinstance(j["samples"], list)


def test_the_judges_guide_is_served_with_the_server_facts():
    """The For judges tab renders docs/JUDGES.md itself (one source) beside what this server runs."""
    j = client.get("/api/judges").json()
    on_disk = (Path(__file__).resolve().parents[1] / "docs" / "JUDGES.md").read_text(encoding="utf-8")
    assert j["markdown"] == on_disk and j["source"] == "docs/JUDGES.md"
    assert j["docs_base"].startswith("https://github.com/") and j["docs_base"].endswith("/docs/")
    assert j["server"]["model"] and j["server"]["opencv"] and isinstance(j["server"]["is_cool_path"], bool)


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
    assert r.status_code == 422                                  # a structured corrupt_frame incident
    assert r.json()["detail"]["incident"]["code"] == "corrupt_frame"


def test_analyze_withdraws_the_promise_on_a_picture_that_is_not_sonar():
    if not _HAVE_MODEL:
        print("  skip  test_analyze_withdraws_the_promise_on_a_picture_that_is_not_sonar (model absent)")
        return
    import cv2
    from test_applicability import not_sonar
    ok, buf = cv2.imencode(".jpg", not_sonar())
    r = client.post("/api/analyze?compute_s=60", files={"file": ("cartoon.jpg", buf.tobytes(), "image/jpeg")})
    assert ok and r.status_code == 200
    b = r.json()
    assert b["applicability"]["applies"] is False
    g = b["guarantees"]
    assert g["recall_promise"] is None and g["promise_withdrawn"]["was"] and "does NOT apply" in g["promise_withdrawn"]["why"]
    ac = b["agent_choice"]
    assert ac["promise_applies"] is False and ac["mode"]["name"] == "standard"      # no promise left to buy
    assert any(i["code"] == "frame_promise_not_applicable" for i in b["incidents"])
    if _HAVE_SAMPLES:                                                    # a demo frame keeps its promise
        s = client.post(f"/api/analyze?sample={samples_mod.list_samples()[0]['id']}&compute_s=60").json()
        assert s["applicability"]["applies"] and s["agent_choice"]["promise_applies"]
        assert s["guarantees"]["recall_promise"] and "promise_withdrawn" not in s["guarantees"]


def test_a_failed_closer_look_carries_the_standard_promise():
    """The tile pass failed -> first look only: the frame (and its cards) carry the standard promise, not 84%."""
    if not (_HAVE_MODEL and _HAVE_SAMPLES):
        print("  skip  test_a_failed_closer_look_carries_the_standard_promise (model/samples absent)")
        return
    sid = samples_mod.list_samples()[0]["id"]
    b = client.post(f"/api/analyze?sample={sid}&compute_s=60&simulate=tool_failed").json()
    assert b["agent_choice"]["mode"]["name"] == "look_harder" and b["perception"].get("fallback")
    std = client.get("/api/health").json()["calibration"]["guarantees"]["recall_promise"]
    assert b["guarantees"]["recall_promise"] == std < b["agent_choice"]["mode"]["promise"]


def test_a_withdrawn_promise_never_meets_a_recall_goal():
    if not _HAVE_MODEL:
        return
    import cv2
    from test_applicability import not_sonar
    ok, buf = cv2.imencode(".jpg", not_sonar())
    b = client.post("/api/analyze?compute_s=60&recall=0.78", files={"file": ("c.jpg", buf.tobytes(), "image/jpeg")}).json()
    assert b["agent_choice"]["promise_applies"] is False and b["agent_choice"]["goal_met"] is False


def test_survey_frame_limit(monkeypatch):
    import src.dashboard.service as service_mod
    monkeypatch.setattr(service_mod, "MAX_FRAMES", 2)
    files = [("files", (f"f{i}.png", b"x", "image/png")) for i in range(3)]
    assert client.post("/api/jobs/survey", files=files).status_code == 413


def test_sync_survey_redirects_big_runs_to_jobs(monkeypatch):
    if not _HAVE_SAMPLES:
        return
    import src.dashboard.service as service_mod
    monkeypatch.setattr(service_mod, "MAX_SYNC_FRAMES", 1)
    r = client.post("/api/survey?use_samples=1")
    assert r.status_code == 413 and "/api/jobs/survey" in r.json()["detail"]


def test_effort_curves_for_each_perception_mode():
    """The Study tab's effort panel: the standard look by default, the look-harder queue on request (each at its
    own promise); one card per object is reported; an unknown mode is refused."""
    r = client.get("/api/effort")
    if r.status_code == 404:                                  # no effort_curve.json for this model
        return
    std = r.json()
    assert std["mode"] == "standard" and std["curves"]["promise"] and "manual" in std["curves"]
    if "look_harder" in std["modes"]:
        lh = client.get("/api/effort?mode=look_harder").json()
        assert lh["mode"] == "look_harder" and lh["curves"]["promise"] > std["curves"]["promise"]
        assert lh["frames"] == std["frames"]                  # the same split, another queue
    assert client.get("/api/effort?mode=nope").status_code == 422


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
    a = client.post(f"/api/analyze?sample={sid}&compute_s=60")      # a pinned mode: the two calls compare
    assert a.status_code == 200
    body = a.json()
    assert "counts" in body and body["frame_png"].startswith("data:image") and "overlay_png" not in body   # lean default
    full = client.post(f"/api/analyze?sample={sid}&compute_s=60&images=full")
    assert full.json()["overlay_png"].startswith("data:image") and len(full.content) > len(a.content)
    for c in body["candidates"]:
        assert c["verdict"] in ("confirmed", "review", "low_risk")
        tools = [st["tool"] for st in c["trace"]]          # [look_harder] voi_check ... decide -> handoff
        if tools[0] == "look_harder":                    # the closer tile look proposed / raised this find
            tools = tools[1:]
        assert tools[0] == "voi_check" and "decide" in tools and tools[-1] == "handoff"

    s = client.post("/api/survey?use_samples=1&gps=synthetic")
    assert s.status_code == 200
    survey_id = s.json()["survey_id"]
    assert s.json()["mission"]["human_approval_required"] is True
    cf = s.json()["stage1_counterfactual"]                     # STUDY-12, live on every GPS survey
    assert cf["available"] and cf["n"] >= 1 and 0 <= cf["outside"] <= cf["n"]
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
    from src.dashboard import service as service_mod
    service_mod._SURVEYS.pop(job["survey_id"], None)
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
