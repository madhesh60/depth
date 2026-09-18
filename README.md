# Marine Debris Detection System — Team Syndicate

> Automated detection of man-made marine debris (ghost nets, pipes, structural
> fragments, rope) from side-scan sonar imagery, with geotagged hazard reporting
> and a live web dashboard.

**Repository:** https://github.com/madhesh60/depth.git
**Competition:** OpenCV AI Competition 2026 (Sponsor: AWS · Administrator: OpenCV Foundation)
**Team:** Syndicate (solo — Madhesh)
**Bonus award path:** Best Use of COOL · **Stretch:** Agentic Vision
**Grant:** AWS Cloud Compute Grant ($150) — awarded
**Final submission deadline:** 2026-10-26, 23:59 PT

---

## 1. What this is

Manual interpretation of side-scan sonar (SSS) is the operational bottleneck in
marine-debris surveys: a trained analyst needs 4–8 hours per mission, debris blends
into natural seafloor clutter, and classification is subjective. This project is an
**end-to-end, edge-deployable computer-vision pipeline** that ingests sonar imagery,
separates man-made anomalies from natural topology, and emits **geotagged JSON / CSV /
GeoJSON hazard reports** through a web dashboard.

### Two-stage hybrid detection

| Stage | Engine | Where it runs | Job |
|---|---|---|---|
| **Stage 1 — Classical CV** | OpenCV 5 (CPU only) | AWS Graviton via **COOL** | Denoise → adaptive threshold → morphology → contour geometry → candidate ROIs |
| **Stage 2 — Deep learning** | YOLOv8 / YOLO11-seg via `cv2.dnn.readNetFromONNX()` | x86 / GPU inference | Verify + classify + instance-segment each ROI, output confidence |

Stage 1 constrains the search space so Stage 2 runs on a fraction of each frame —
the design goal is a **60–80% reduction in deep-learning inference cost** versus
full-frame inference, and it is what makes edge/Graviton deployment viable.

### Detection classes (live dataset: `nc = 4`)

| ID | Class | Description |
|---|---|---|
| 0 | `fishing_gear` | Ghost nets, trawl fragments, monofilament tangles, rope/line (merged) |
| 1 | `pipe_cylinder` | Industrial piping, barrels, cylindrical debris |
| 2 | `structural_fragment` | Shipwreck components, hull plates, metal debris |
| 3 | `natural_formation` | Rock clusters, geological ridges (negative / control class) |

> The proposal ([`AGENT.md`](AGENT.md)) described 5 classes with `rope_line` separate; the
> current dataset folds `rope_line` into `fishing_gear` (4 classes). Whether to re-introduce
> `rope_line` is an open decision — see [`docs/dataset_report.md`](docs/dataset_report.md) §2.

---

## 2. Why this wins

The judging rubric rewards three things; the architecture targets each explicitly.

1. **Best Use of COOL (our target bonus prize).** Stage 1 is the *core workload* and runs
   on **AWS Graviton (Arm) via COOL**, with a fully automated, reproducible benchmark
   harness comparing Graviton vs. an x86 baseline on identical inputs — latency,
   throughput, CPU utilisation, and cost per 1,000 frames. See
   [`architecture.md` §7](architecture.md).
2. **Substantive OpenCV 5 use.** OpenCV 5 is the backbone of *both* stages — classical
   preprocessing (Stage 1) and DNN inference/compositing/visualisation (Stage 2) — not a
   thin wrapper.
3. **Agentic Vision (stretch).** A perception→decision→action loop: low-confidence
   detections trigger an automated re-scan / gain-adjustment request via an MCP-exposed
   toolset, so the *visual result changes what the system does next*. See
   [`architecture.md` §8](architecture.md).

---

## 3. Repository layout

```
MARINE_DEBRIS/
├── README.md                 # This file — front door + quickstart
├── architecture.md           # Peak system design, AWS, COOL, agentic loop, conventions
├── progress.md               # Living status log + milestone tracker
├── experiments.md            # ML experiment register (one entry per training run)
├── TODO.md                   # Phased, actionable backlog
├── CLAUDE.md                 # Working instructions for Claude Code sessions
├── AGENT.md                  # Original grant proposal (source of record)
├── requirements.txt          # Pinned Python dependencies
│
├── docs/                     # Reference material
│   ├── competition_rules.md  # Official rules digest
│   ├── dataset_report.md     # Verified dataset audit (regenerable)
│   └── claude_code_playbook.md

│
├── src/                      # (to be built — see architecture.md §5)
│   ├── cv_pipeline/          # Stage 1 classical OpenCV
│   ├── detection/            # Stage 2 YOLO train / infer / ONNX export
│   ├── reporting/            # Geotagging + JSON/CSV/GeoJSON report generation
│   ├── dashboard/            # FastAPI backend + upload/map UI
│   └── common/               # Config, schemas, logging, shared utilities
│
├── infra/                    # AWS: Lambda handlers, SageMaker configs, IaC, benchmarks
├── tests/                    # Unit + integration tests
│
└── DATASET/                  # git-ignored — see §5 (pushed to S3, not committed)
    ├── 01_raw_archives/
    ├── 02_intermediate_processing/
    ├── 03_yolo_ready_dataset/   # data.yaml lives here — TRAIN POINTS HERE
    ├── scripts/                 # active dataset tooling
    └── archive_scripts/         # frozen build scripts (idempotent)
```

> `src/`, `infra/`, `tests/`, `docs/` are the target structure and are built out as work
> proceeds — see [`TODO.md`](TODO.md) and [`progress.md`](progress.md) for what exists today.

---

## 4. Quickstart

```bash
# 1. Dependencies (global Python 3.10 — no venv, per project convention)
pip install -r requirements.txt

# 2. Train Stage 2 (once dataset fixes land — see TODO.md)
#    Dataset entry point:
#    DATASET/03_yolo_ready_dataset/data.yaml

# 3. Run the local dashboard (once implemented)
#    uvicorn src.dashboard.app:app --reload
```

**Environment notes**
- Global Python 3.10 already has `torch`, `torchvision`, `opencv-python`, `numpy`,
  `PyYAML`, `Pillow`. `requirements.txt` adds `ultralytics`, `boto3`, `onnxruntime`,
  `fastapi`/`uvicorn`.
- **AWS CLI is not yet installed** on this machine — required before any S3 / SageMaker /
  Lambda work.

---

## 5. Dataset

The full dataset (raw archives + generated images) is **git-ignored** and lives in
`DATASET/`; it is pushed to S3 rather than committed. **Training uses the cleaned v1 split**
(`DATASET/03_yolo_ready_dataset_v1/`, `data.yaml`):

| Split | Images | Notes |
|---|---:|---|
| train | 26,485 | incl. 999 background negatives |
| val | 1,253 | originals only (clean eval) |
| test | 1,253 | originals only (clean eval) |

Built from 7 source datasets (crab-pot, UATD, ICRA19, TrashCan, Marine PULSE,
SeabedObjects-KLSG, AI4Shipwrecks) remapped to the 4-class taxonomy. v1 was produced by
[`DATASET/scripts/build_dataset_v1.py`](DATASET/scripts/build_dataset_v1.py), which fixes the
audited v0 problems: **cross-split leakage → split by source frame (now 0 shared frames)**,
full-frame boxes removed, corrupt-checked, and empty-label images kept as background
negatives. Remaining *training-time* concerns (class imbalance, mixed sensors) are handled at
training, not by data surgery. Full before/after audit: [`docs/dataset_report.md`](docs/dataset_report.md)
(regenerate: `python DATASET/scripts/audit_dataset.py DATASET/03_yolo_ready_dataset_v1`).

---

## 6. Target metrics (acceptance criteria)

| Metric | Target |
|---|---|
| mAP@0.5 | ≥ 0.70 |
| mAP@0.5:0.95 | ≥ 0.45 |
| Precision | ≥ 0.80 |
| Recall | ≥ 0.70 |
| False-positive reduction (Stage 1 vs full-frame) | ≥ 60% |
| Per-frame latency | < 300 ms |
| Throughput | ≥ 5 FPS |

---

## 7. Project documents

| File | Purpose |
|---|---|
| [`architecture.md`](architecture.md) | System design, data flow, AWS/COOL, agentic loop, naming conventions |
| [`progress.md`](progress.md) | Current state, milestone timeline, weekly log |
| [`experiments.md`](experiments.md) | ML experiment register + error-analysis log |
| [`TODO.md`](TODO.md) | Phased backlog with acceptance criteria |
| [`AGENT.md`](AGENT.md) | Original grant proposal (immutable source of record) |
| [`CLAUDE.md`](CLAUDE.md) | Claude Code working instructions |
| [`docs/competition_rules.md`](docs/competition_rules.md) | Official rules & requirements digest |
| [`docs/dataset_report.md`](docs/dataset_report.md) | Verified dataset audit (regenerable) |
| [`docs/claude_code_playbook.md`](docs/claude_code_playbook.md) | Working method reference |

---

*Team Syndicate · OpenCV AI Competition 2026 · Best Use of COOL / Agentic Vision.*
