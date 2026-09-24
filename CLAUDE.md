# Marine Debris Detection — Team Syndicate

Solo hackathon project (OpenCV AI Competition 2026 / AWS COOL Award — "Best Use of COOL",
stretch: "Agentic Vision"). Full proposal and rationale: [AGENT.md](AGENT.md).

## What this is

End-to-end pipeline that ingests side-scan sonar imagery and detects man-made marine debris
(ghost nets, pipes, structural fragments, rope) against natural seafloor clutter, then emits
geotagged JSON/CSV hazard reports via a web dashboard.

Pipeline (roles updated 2026-09-22, see `experiments.md` STUDY-01):
1. **Classical OpenCV** (CPU, runs on AWS Graviton via COOL) — denoise, resize, adaptive
   threshold, morphology, contours — the **CPU sonar-preprocessing workload benchmarked for
   the COOL award** (Graviton vs x86). It is **not** a Stage-2 ROI gate: on this sonar data
   classical CV has no discriminative power (STUDY-01 — GT coverage caps ~72% at 141 ROIs/frame;
   fires more on empty seafloor than on debris), so ROI-gating the detector was retired.
2. **YOLO11 detector** (loaded via `cv2.dnn.readNetFromONNX`, `src/detection/infer.py`) — the
   actual detector, run **full-frame**; outputs boxes + class + confidence. EXP-001 baseline
   mAP@0.5 0.822. (`-seg` masks are an option, not yet trained.)

Classes — the **live dataset is 4-class** (`data.yaml`, `nc=4`):
`0 fishing_gear, 1 pipe_cylinder, 2 structural_fragment, 3 natural_formation`
(`rope_line` from the original 5-class proposal is folded into `fishing_gear`; re-introducing
it is an open decision — see [docs/dataset_report.md](docs/dataset_report.md) §2).

## Planned AWS architecture

S3 (raw/processed/models/reports) → Lambda+API Gateway (Stage 1 preprocessing on Graviton/COOL
→ Stage 2 YOLO full-frame inference) → DynamoDB (detections) → Amplify (dashboard) + CloudWatch
(benchmarks).
SageMaker for training. Details/diagram in [AGENT.md](AGENT.md#4-planned-aws-architecture-and-services).

## Dataset state

`DATASET/` is git-ignored (too large — raw archives + 30k+ generated images). Pipeline already run:

- `01_raw_archives/` — 7 source datasets (AI4Shipwrecks, crab-pot, ICRA19, TrashCan, UATD, etc.)
- `02_intermediate_processing/unified/` — remapped/unified intermediate (`class_counts.json`).
- `03_yolo_ready_dataset/` — v0 split (superseded; kept for provenance).
- **`03_yolo_ready_dataset_v1/`** — the **clean, training-ready dataset** (`data.yaml`, `nc=4`):
  **train=26,533 (incl. 1,120 background) · val=1,204 · test=1,276**; leakage-free (split by
  **recording/clip**, 5,618 groups — a whole video clip / sonar run stays in one split so
  near-identical consecutive frames can't leak), **jointly stratified by class box-mix + sensor
  domain** (greedy fill-equalisation on originals) so val **and** test each mirror train's
  per-class and sonar/optical distribution — all three splits are ~53% `fishing_gear` / 28%
  `structural_fragment` / 15% `natural_formation` / 4% `pipe_cylinder` and ~80/20 sonar/optical.
  All 4 classes + all 7 sources present in each split; full-frame + sliver boxes removed,
  corrupt-checked. Counts + cleaning stats in `manifest.json`. Per-class boxes (total):
  `fishing_gear`=21,487, `structural_fragment`=11,430, `natural_formation`=5,918,
  `pipe_cylinder`=1,821. Residual: `pipe_cylinder` is rare (only 65 test / 120 val boxes — lives
  in few clips) → per-class AP for it is high-variance; always report per-class metrics.
- `scripts/audit_dataset.py [root]` — ground-truth audit · `scripts/build_dataset_v1.py` —
  the v0→v1 cleaner · `scripts/visualize_labels.py` — box-on-image QA renders.
- `archive_scripts/run_remaining_steps.py` — frozen v0 build script (idempotent).

**Remaining (training-time, not dataset defects):** class imbalance (`fishing_gear` ≈ 11.7×
`pipe_cylinder`) → use class weights / focal loss / augmentation; mixed sensors (80% sonar /
20% optical, tagged in `manifest.json`) → domain-split experiment.

**EXP-001 (the deployed model) trained on v1. EXP-002 should train on
`DATASET/03_yolo_ready_dataset_v2b/data.yaml`** — v2b (`build_dataset_v2b.py`): 2-class sonar-only,
one copy per Roboflow frame (3,241 rotated copies dropped), val = held-out recordings Rec10/12/16,
test = 214 unique crab-pot frames; `test_official398/` (GhostVision only) + `test_xsonar/`
(Contact_sslo cross-sonar); `groups.json` for group bootstrap. See `docs/dataset_card.md`.

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
