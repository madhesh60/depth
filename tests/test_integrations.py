"""
Tests for the console integrations: OGC API – Features (src/dashboard/ogc.py) and signed webhooks
(src/dashboard/integrations.py). A real local HTTP receiver checks the HMAC signature end to end.
"""
from __future__ import annotations

import http.server
import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from src.agentic.approvals import ApprovalStore
from src.dashboard import app as app_mod
from src.dashboard import integrations as integ
from src.dashboard.ogc import build_features
from src.detection.infer import DEFAULT_ONNX

client = TestClient(app_mod.app)


def _survey_dict():
    tracked = [
        {"oid": "H001", "cls_name": "fishing_gear", "verdict": "review", "conf": 0.6, "evidence_score": 0.6, "frame_id": "F",
         "bbox": [0, 0, 9, 9], "lat": 37.8001, "lon": -76.1501, "geo_error_m": 4.0, "p_pot": 0.7, "shadow_quality": "clear",
         "height_rel": 0.02, "sightings": 1},
        {"oid": "H002", "cls_name": "wreck_debris", "verdict": "review", "conf": 0.5, "evidence_score": 0.5, "frame_id": "F",
         "bbox": [0, 0, 9, 9], "lat": 37.812345, "lon": -76.151234, "geo_error_m": 5.0, "p_pot": None, "shadow_quality": "none",
         "height_rel": 0.0, "sightings": 1}]
    mission = {"gps_synthetic": True, "review_queue": ["H001", "H002"], "budget": {"review_ids": ["H001"]},
               "recovery_route": [], "inspection_route": ["H001", "H002"], "inspection_length_m": 1400.0,
               "resurvey_plan": {"lines": [
                   {"id": "RS1", "start": [37.80, -76.15], "end": [37.801, -76.15], "targets": ["H001"], "status": "PLANNED"},
                   {"id": "RS2", "start": [37.81, -76.15], "end": [37.813, -76.15], "targets": ["H002"], "status": "PLANNED"}]}}
    return {"survey_id": "s-1", "tracked": tracked, "mission": mission}


def test_ogc_features_redaction_and_work_orders(tmp_path):
    ap = ApprovalStore(tmp_path)
    s = _survey_dict()
    hz = build_features("hazards", [s], ap, public=False)
    assert [f["id"] for f in hz] == ["s-1:H001", "s-1:H002"] and hz[0]["properties"]["review_rank"] == 1
    assert hz[0]["properties"]["gps_synthetic"] is True and hz[0]["geometry"]["coordinates"] == [-76.1501, 37.8001]
    pub = build_features("hazards", [s], ap, public=True)
    wreck = next(f for f in pub if f["properties"]["hazard_id"] == "H002")
    assert wreck["geometry"]["coordinates"] == [-76.15, 37.81] and wreck["properties"]["error_m"] == 1100.0
    assert [f["properties"]["pass_id"] for f in build_features("resurvey_passes", [s], ap, True)] == ["RS1"]   # RS2 would reveal it
    assert build_features("work_orders", [s], ap, False) == []
    r = ap.request("s-1", "inspect", ["H001"], "top card with a clear shadow")
    assert build_features("hazards", [s], ap, False)[0]["properties"]["approval"]["status"] == "pending"
    ap.decide(r["id"], "approved", "A. Person")
    wo = build_features("work_orders", [s], ap, False)
    assert len(wo) == 1 and wo[0]["geometry"] == {"type": "MultiPoint", "coordinates": [[-76.1501, 37.8001]]}
    assert wo[0]["properties"]["approved_by"] == "A. Person"


def test_ogc_endpoints_conform():
    land = client.get("/ogc").json()
    assert {"self", "service-desc", "conformance", "data"} <= {l["rel"] for l in land["links"]}
    assert "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/geojson" in client.get("/ogc/conformance").json()["conformsTo"]
    ids = [c["id"] for c in client.get("/ogc/collections").json()["collections"]]
    assert ids == ["hazards", "resurvey_passes", "routes", "work_orders"]
    r = client.get("/ogc/collections/hazards/items?limit=2")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/geo+json")
    d = r.json()
    assert d["type"] == "FeatureCollection" and d["numberReturned"] <= 2 and "numberMatched" in d
    assert client.get("/ogc/collections/hazards/items?bbox=1,2,3").status_code == 400
    assert client.get("/ogc/collections/nope/items").status_code == 404
    assert client.get("/ogc/collections/hazards/queryables").headers["content-type"].startswith("application/schema+json")


class _Receiver(http.server.BaseHTTPRequestHandler):
    got: list = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        _Receiver.got.append(({k.lower(): v for k, v in self.headers.items()}, body))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture()
def receiver(monkeypatch):
    monkeypatch.setenv("DEPTH_WEBHOOK_ALLOW_PRIVATE", "1")
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Receiver)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Receiver.got = []
    yield f"http://127.0.0.1:{srv.server_port}/hook"
    srv.shutdown()


def test_webhook_signed_delivery_and_guards(tmp_path, receiver, monkeypatch):
    wh = integ.Webhooks(tmp_path)
    hook = wh.add(receiver, ["approval.decided"])
    assert hook["secret"].startswith("whsec_") and wh.list()[0]["secret"].endswith("…")     # shown once
    assert wh.emit("survey.completed", {"x": 1}) == 0                                    # not subscribed
    assert wh.emit("approval.decided", {"id": "AR-1", "targets": ["H001"]}) == 1
    for _ in range(50):
        if _Receiver.got:
            break
        time.sleep(0.05)
    headers, body = _Receiver.got[0]
    assert headers["x-depth-event"] == "approval.decided"
    assert integ.verify(hook["secret"], headers["x-depth-signature"], body)              # the receiver's check
    assert not integ.verify("whsec_wrong", headers["x-depth-signature"], body)
    assert not integ.verify(hook["secret"], headers["x-depth-signature"], body + b" ")   # tampered body
    assert json.loads(body)["data"]["id"] == "AR-1"
    assert wh.send_now(hook["id"])["status"] == "delivered"
    monkeypatch.delenv("DEPTH_WEBHOOK_ALLOW_PRIVATE")
    for bad in ("http://127.0.0.1:9/x", "http://169.254.169.254/latest/meta-data", "ftp://example.org/x"):
        with pytest.raises(integ.WebhookError):
            wh.add(bad)                                                                  # SSRF / scheme guard


def test_webhook_api_admin_token_and_events(tmp_path, receiver, monkeypatch):
    app_mod._WEBHOOKS = integ.Webhooks(tmp_path / "wh")
    app_mod._APPROVALS = ApprovalStore(tmp_path / "ap")
    monkeypatch.setenv("DEPTH_ADMIN_TOKEN", "adm")
    assert client.post("/api/integrations/webhooks", json={"url": receiver}).status_code == 403
    h = client.post("/api/integrations/webhooks", json={"url": receiver, "events": ["approval.requested"]},
                    headers={"X-DEPTH-Admin": "adm"}).json()
    assert h["id"].startswith("wh_")
    r = client.post("/api/approvals", json={"survey_id": "s-x", "action": "resurvey", "targets": ["RS1"],
                                            "rationale": "two uncertain port-side targets"})
    assert r.status_code == 200
    for _ in range(50):
        if _Receiver.got:
            break
        time.sleep(0.05)
    assert _Receiver.got and _Receiver.got[0][0]["x-depth-event"] == "approval.requested"
    info = client.get("/api/integrations").json()
    assert info["mcp"]["url"].endswith("/mcp") and info["ogc"]["landing"].endswith("/ogc")
    assert client.delete(f"/api/integrations/webhooks/{h['id']}", headers={"X-DEPTH-Admin": "adm"}).status_code == 200


def test_survey_completed_event_is_emitted(tmp_path, monkeypatch):
    if not DEFAULT_ONNX.exists():
        pytest.skip("model absent")
    sent = []
    monkeypatch.setattr(app_mod._WEBHOOKS, "emit", lambda ev, data: sent.append((ev, data)) or 0)
    s = client.post("/api/survey?use_samples=1&gps=synthetic&budget_minutes=1").json()
    ev = [d for e, d in sent if e == "survey.completed"]
    assert ev and ev[0]["survey_id"] == s["survey_id"] and ev[0]["human_approval_required"] is True
    assert ev[0]["gps"] == "synthetic" and ev[0]["links"]["ogc_hazards"].endswith(s["survey_id"])
