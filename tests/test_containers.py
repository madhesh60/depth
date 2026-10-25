"""The container path: health probes, the build's provenance commit, and the Dockerfile / Kubernetes files agree
with the code (probe paths exist, one replica, torch-free, writes only to /data). The real build and the kind
deploy run in CI (.github/workflows/container.yml)."""
from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from src.bench import fingerprint
from src.dashboard import service as S
from src.dashboard.app import app

ROOT = Path(__file__).resolve().parents[1]
K8S = ROOT / "infra" / "k8s"
client = TestClient(app)


def test_liveness_and_readiness(monkeypatch):
    assert client.get("/healthz").json() == {"status": "ok"}
    monkeypatch.setitem(S._MODEL_STATE, "loaded", False)
    monkeypatch.setitem(S._MODEL_STATE, "error", None)
    r = client.get("/readyz")
    assert r.status_code == 503 and r.json()["detail"]["status"] == "loading"
    monkeypatch.setitem(S._MODEL_STATE, "error", "RuntimeError: no weights")
    assert client.get("/readyz").json()["detail"]["status"] == "error"
    monkeypatch.setitem(S._MODEL_STATE, "loaded", True)
    assert client.get("/readyz").json()["status"] == "ready"


def test_an_image_without_git_still_names_its_commit(monkeypatch):
    monkeypatch.setenv("DEPTH_GIT_COMMIT", "0123456789abcdef")
    assert fingerprint.git_commit() == "0123456789ab"


def _k8s() -> dict:
    docs = {}
    for name in yaml.safe_load((K8S / "kustomization.yaml").read_text())["resources"]:
        d = yaml.safe_load((K8S / name).read_text())
        docs[d["kind"]] = d
    return docs


def test_kubernetes_probes_hit_real_routes_and_one_replica():
    dep = _k8s()["Deployment"]
    assert dep["spec"]["replicas"] == 1 and dep["spec"]["strategy"]["type"] == "Recreate"   # surveys are in memory
    c = dep["spec"]["template"]["spec"]["containers"][0]
    routes = {r.path for r in app.routes}
    for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
        assert c[probe]["httpGet"]["path"] in routes, probe
    assert c["securityContext"]["readOnlyRootFilesystem"] is True
    assert {m["mountPath"] for m in c["volumeMounts"]} == {"/data", "/tmp"}


def test_dockerfile_is_torch_free_non_root_and_writes_only_to_data():
    df = (ROOT / "Dockerfile").read_text()
    assert "requirements.txt" in df and "requirements-train" not in df and "torch" not in df.replace("torch-free", "")
    assert "USER 10001" in df and "ln -s /data /app/runs" in df and "/readyz" in df
    wf = (ROOT / ".github" / "workflows" / "container.yml").read_text()
    image = yaml.safe_load((K8S / "kustomization.yaml").read_text())["images"][0]
    assert f"{image['newName']}:{image['newTag']}" in wf          # CI swaps exactly the image the base names
