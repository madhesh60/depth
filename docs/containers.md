# DEPTH in containers: Docker, Compose and Kubernetes

The same server as the EC2 demo ([https://13-216-140-101.sslip.io](https://13-216-140-101.sslip.io)) (studio, API, MCP at `/mcp`, OGC API at `/ogc`) in one CPU-only, torch-free image.
It runs **stock OpenCV 5** from PyPI. It is **not COOL**, and `/api/health` says so (`is_cool_path: false`).

## Run it

```bash
# Docker
docker build -t depth .                                   # fetches the detector, SHA-256 checked
docker run --rm -p 8000:8000 -v depth-data:/data depth    # http://localhost:8000

# Docker Compose (read-only root, no capabilities, a named volume)
docker compose up --build

# Kubernetes (any cluster; the image is published by CI from main)
kubectl apply -k infra/k8s
kubectl -n depth port-forward svc/depth 8000:80
```

Check any of them with `python tests/container_smoke.py --url http://localhost:8000`. It checks readiness, runs
one Analyze on a held-out sample and a Survey of the raw recording, and checks the provenance commit.

## What the image is

| | |
|---|---|
| Base | `python:3.10-slim-bookworm`, multi-stage (build tools never reach the final image) |
| Contents | `src/`, `webui/`, `models/<MODEL>/` (calibration + `best.onnx`), the 11 MB raw recording (`FETCH_RECORDING=0` to skip) |
| User | non-root, uid 10001; works with a read-only root filesystem |
| Writes | only `/data` (`runs/` points there): jobs, approvals, feedback, study results, audit tags, webhooks |
| Health | `/healthz` (process alive) · `/readyz` (detector loaded and warmed; 503 before) · Docker `HEALTHCHECK` |
| Provenance | `GIT_COMMIT` build arg → every result's stamp names the commit, as on EC2 |
| Size, memory | 731 MB image; about 340 MiB in use after a survey of the real recording (measured 2026-10-25) |

Behind a proxy that inspects TLS, pass its CA for the build only (it is not stored in the image):
`docker build --secret id=ca,src=ca.pem --build-arg HTTPS_PROXY=... -t depth .`

## Kubernetes (`infra/k8s/`)

Kustomize base: Namespace (Pod Security `restricted`), ConfigMap, PersistentVolumeClaim, Deployment, Service,
NetworkPolicy. The pod runs non-root with a read-only root, all capabilities dropped and the RuntimeDefault seccomp
profile. Startup and readiness probes use `/readyz`, so traffic arrives only after the detector is warm; liveness
uses `/healthz`. Secrets (`DEPTH_MCP_TOKEN`, `DEPTH_ADMIN_TOKEN`, `DEPTH_APPROVER_PIN`) come from an optional
Secret named `depth-secrets`.

**One replica, on purpose.** Surveys live in the server's memory, so a second replica would not see the first
one's surveys. There is no autoscaler for that reason.

## How it is tested

| Check | Where |
|---|---|
| Image built, run read-only with no capabilities, smoke test passed (Analyze, Survey of the raw recording, provenance commit) | locally, 2026-10-25, and CI `container.yml` on every change |
| The studio driven by headless Chrome against the container, guided demo to its last step, no failed step | locally, 2026-10-25 |
| Manifests valid against the Kubernetes 1.31 schema (`kubernetes-validate --strict`) | locally, 2026-10-25 |
| Deployed to a kind cluster; the same smoke test through the Service | CI `container.yml` |
| Image published to `ghcr.io/madhesh60/depth` | CI, `main` only |

The EC2 demo server does not use this image. It runs from systemd (`infra/depth.service`) so that it can use the
COOL AMI's OpenCV under `/opt/cool` once the subscription is active.
