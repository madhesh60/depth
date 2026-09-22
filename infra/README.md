# infra/ — AWS deployment & the COOL benchmark (deploy-ready, pending credits)

Everything here is written and locally verified but **not yet deployed** (AWS credits pending).
When credits land it's *deploy, not build*. The primary judged deliverable is **Best Use of
COOL**, so this is organised around that.

## Why this wins "Best Use of COOL"

Scorecard: verified COOL on Graviton **30%** + measured perf vs baseline **20%** +
architecture **25%** + innovation **15%** + reproducibility **10%**. Half the award is a
**reproducible Graviton-vs-x86 benchmark of the CPU workload** — not a fancier model. So the
CPU-side pipeline is deliberately built from COOL/KleidiCV-accelerated OpenCV ops (resize,
adaptive-gaussian threshold, contours) and benchmarked three ways.

Honest scoping (see `experiments.md` STUDY-01): classical CV is **not** the detector — it has
no discriminative power on this sonar. The COOL workload is the **sonar preprocessing /
tiling** pass; YOLO (`cv2.dnn`) does detection. OpenCV 5 is used substantively throughout:
preprocessing → tiling → DNN inference → overlay rendering, all CPU, all Graviton-capable.

## Components

| File | Role | Status |
|---|---|---|
| `lambda_handler.py` | Inference endpoint: image → `cv2.dnn` ONNX detect → geotagged JSON. No torch (small pkg, Arm-ready). Mirrors `src/detection/infer.py`. | ✅ local smoke-tested |
| `benchmark_graviton.sh` | Runs `src.cv_pipeline.benchmark` for one config, stamps an input-manifest hash, uploads JSON to S3. | ✅ ready |

## The 3-way COOL benchmark (the money shot)

Run the *same* frame set (≥1,000, pinned by sha256) in three configs:

```bash
# (A) x86 baseline — on a c7i.xlarge
LABEL=baseline_x86   OPENCV_FLAVOR=stock S3_BUCKET=$B ./infra/benchmark_graviton.sh
# (B) Graviton, stock OpenCV — on a c7g.xlarge / c8g.xlarge
LABEL=graviton_stock OPENCV_FLAVOR=stock S3_BUCKET=$B ./infra/benchmark_graviton.sh
# (C) Graviton, COOL/KleidiCV — SAME c7g/c8g instance
LABEL=graviton_cool  OPENCV_FLAVOR=cool  S3_BUCKET=$B ./infra/benchmark_graviton.sh
```
Each JSON records OpenCV build (KleidiCV detected y/n), `platform.machine` (aarch64 vs
x86_64), per-op p50/p95, throughput, and the input-manifest hash → self-documenting and
one-command reproducible. (B)→(C) = COOL's contribution on identical hardware; (A)→(C) =
Arm-vs-x86 headline.

## Planned AWS architecture

```
S3 (raw/ processed/ models/ reports/ benchmarks/)
  └─ API Gateway → Lambda (lambda_handler.py, ARM64/Graviton)
        ├─ Stage-1 preprocess + tiling (COOL-accelerated OpenCV)
        └─ Stage-2 cv2.dnn ONNX detect → geotagged JSON
  └─ DynamoDB (detections; geo + time GSIs)
  └─ Amplify dashboard (upload → overlays + downloadable report)
  └─ CloudWatch (latency/throughput) · SageMaker (training)
```

## Instance types & cost

| Purpose | Instance | Notes |
|---|---|---|
| COOL compute / benchmark (C,B) | `c7g.xlarge` / `c8g.xlarge` (Graviton3/4) | award-critical path |
| x86 baseline (A) | `c7i.xlarge` | apples-to-apples vCPU/GHz |
| Inference | Lambda **arm64** | serverless, scale-to-zero |
| Training | SageMaker `ml.g5.xlarge` (or Kaggle T4 for now) | |

**Cost discipline:** tear down benchmark EC2 immediately after each run; stay within the $150
grant + Free Tier; Lambda/DynamoDB scale to zero. Deploy order: **S3 → benchmark (COOL locked
first) → Lambda/API → DynamoDB → Amplify.**

## Deploy checklist (when credits land)
1. `aws configure` (AWS CLI still not installed locally — blocker B6).
2. Create S3 buckets/prefixes; upload `best.onnx` → `models/` and ≥1,000 frames → `frames/`.
3. Run the 3-way benchmark above; verify KleidiCV detected in `graviton_cool.json`.
4. Package `lambda_handler.py` + deps (arm64) → Lambda; set `MODEL_S3`; wire API Gateway.
5. Smoke-test the endpoint; record latency by `arch`.
