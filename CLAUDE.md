# Marine Debris Detection — Team Syndicate

Solo hackathon project (OpenCV AI Competition 2026 / AWS COOL Award — "Best Use of COOL",
stretch: "Agentic Vision"). Full proposal and rationale: [AGENT.md](AGENT.md).

## What this is

End-to-end pipeline that ingests side-scan sonar imagery and detects man-made marine debris
(ghost nets, pipes, structural fragments, rope) against natural seafloor clutter, then emits
geotagged JSON/CSV hazard reports via a web dashboard.

Two-stage detection:
1. **Classical OpenCV** (CPU, runs on AWS Graviton via COOL) — denoise, adaptive threshold,
   morphological filtering, contour geometry — produces candidate ROIs.
2. **YOLOv8/YOLO11-seg** (loaded via `cv2.dnn.readNetFromONNX`) — verifies/classifies ROIs,
   outputs instance masks + confidence scores.

Classes — the **live dataset is 4-class** (`data.yaml`, `nc=4`):
`0 fishing_gear, 1 pipe_cylinder, 2 structural_fragment, 3 natural_formation`
(`rope_line` from the original 5-class proposal is folded into `fishing_gear`; re-introducing
it is an open decision — see [docs/dataset_report.md](docs/dataset_report.md) §2).

## Planned AWS architecture

S3 (raw/processed/models/reports) → Lambda+API Gateway (Stage 1 on Graviton/COOL → Stage 2
YOLO inference) → DynamoDB (detections) → Amplify (dashboard) + CloudWatch (benchmarks).
SageMaker for training. Details/diagram in [AGENT.md](AGENT.md#4-planned-aws-architecture-and-services).

## Dataset state

`DATASET/` is git-ignored (too large — raw archives + 30k+ generated images). Pipeline already run:

- `01_raw_archives/` — 7 source datasets (AI4Shipwrecks, crab-pot, ICRA19, TrashCan, UATD, etc.)
- `02_intermediate_processing/unified/` — remapped/unified intermediate (`class_counts.json`).
- `03_yolo_ready_dataset/` — v0 split (superseded; kept for provenance).
- **`03_yolo_ready_dataset_v1/`** — the **clean, training-ready dataset** (`data.yaml`, `nc=4`):
  **train=26,485 (incl. 999 background) · val=1,253 · test=1,253**; leakage-free (split by
  source frame), full-frame boxes removed, corrupt-checked. Counts + cleaning stats in
  `manifest.json`. Per-class boxes: `fishing_gear`=21,823, `structural_fragment`=11,470,
  `natural_formation`=5,900, `pipe_cylinder`=1,867.
- `scripts/audit_dataset.py [root]` — ground-truth audit · `scripts/build_dataset_v1.py` —
  the v0→v1 cleaner · `scripts/visualize_labels.py` — box-on-image QA renders.
- `archive_scripts/run_remaining_steps.py` — frozen v0 build script (idempotent).

**Remaining (training-time, not dataset defects):** class imbalance (`fishing_gear` ≈ 11.7×
`pipe_cylinder`) → use class weights / focal loss / augmentation; mixed sensors (80% sonar /
20% optical, tagged in `manifest.json`) → domain-split experiment.

**Training should point at `DATASET/03_yolo_ready_dataset_v1/data.yaml`.**

## Environment

No venv — using the global Python 3.10 install, which already has `torch`, `torchvision`,
`opencv-python`, `numpy`, `PyYAML`, `Pillow`. Run `pip install -r requirements.txt` to add
`ultralytics`, `boto3`, `onnxruntime`, `fastapi`/`uvicorn` for training, AWS, and the dashboard API.

AWS CLI is **not installed** on this machine yet — needed before any SageMaker/S3/Lambda work.

## Repo layout (to be built out)

Nothing under `src/` yet — this is a fresh setup. Suggested structure as work starts:

```
src/
  cv_pipeline/    # Stage 1 classical OpenCV (denoise, threshold, morph, contours)
  detection/      # Stage 2 YOLO train/infer/export (ONNX)
  reporting/      # geotagging + JSON/CSV/GeoJSON report generation
  dashboard/      # FastAPI backend + upload/map UI
infra/            # AWS: Lambda handlers, SageMaker job configs, IaC
```

## Git & delivery workflow

Repository: **https://github.com/madhesh60/depth.git** (branch `main`).

- **Ship components, not dumps.** When a component of the system is completed and verified,
  **push it to GitHub** as its own small, self-contained commit (or a short series) — don't
  batch unrelated work into one giant commit.
- Use typed, imperative commit subjects: `feat:`, `fix:`, `data:`, `docs:`, `infra:`, `exp:`,
  `chore:`, `test:`. One logical change per commit.
- Follow **PLAN → IMPLEMENT → TEST → REVIEW → DOCUMENT → COMMIT** for each component; only
  push once it's tested and the docs (`progress.md`, `experiments.md`) are updated.
- Version the *code*, never the *data*: `DATASET/` data stays git-ignored; only the
  processing scripts under `DATASET/scripts/` are tracked (see `.gitignore`).
- End every commit message with the co-author trailer:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Key constraints (from the proposal — keep these in mind when implementing)

- Stage 1 must run CPU-only (Graviton, no GPU) — don't reach for GPU-only OpenCV ops there.
- Target: per-frame latency < 300ms, ≥5 FPS, mAP@0.5 ≥ 0.70, precision ≥ 0.80, recall ≥ 0.70.
- Every detection needs lat/lon + confidence + classification, traceable to its source frame.
- Benchmarks must be reported for both Graviton (COOL) and an x86 baseline — this is the
  primary judging criterion (Best Use of COOL).
