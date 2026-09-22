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

### Pipeline (roles updated 2026-09-22 — see [`experiments.md`](experiments.md) STUDY-01)

| Stage | Engine | Where it runs | Job |
|---|---|---|---|
| **Stage 1 — Classical CV** | OpenCV 5 (CPU only) | AWS Graviton via **COOL** | Sonar **preprocessing** (denoise, resize, adaptive threshold, contours) — the CPU workload **benchmarked for the COOL award** |
| **Stage 2 — YOLO11 detector** | `cv2.dnn.readNetFromONNX()` | x86 / GPU / Lambda | **Full-frame** detection → class + confidence (+ geotag) |

> **Honest architecture note.** We originally planned Stage 1 as an ROI *gate* that would cut
> Stage-2 false positives ≥60%. STUDY-01 disproved that on this sonar data — classical CV has
> no discriminative power here (GT coverage caps ~72% at 141 ROIs/frame; it fires *more* on
> empty seafloor than on debris). So **the ROI-gating and the ≥60% FP-reduction target are
> retired**: YOLO runs full-frame, and Stage 1 is repurposed as the COOL-benchmarked
> preprocessing workload. Documenting this negative result *with evidence* is itself a
> submission asset. OpenCV 5 is still used substantively end-to-end: preprocessing → tiling →
> `cv2.dnn` inference → overlay rendering.

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

1. **Best Use of COOL (our target bonus prize).** The Stage-1 sonar-preprocessing pass is the
   *core CPU workload* and runs on **AWS Graviton (Arm) via COOL**, with a reproducible 3-way
   benchmark harness (x86 / Graviton-stock / Graviton-COOL) on identical, hash-pinned inputs —
   latency (p50/p95), throughput, and per-op profile. Runner: [`infra/benchmark_graviton.sh`](infra/benchmark_graviton.sh).
   Half the award is exactly this reproducible benchmark.
2. **Substantive OpenCV 5 use.** OpenCV 5 runs the whole CPU path — preprocessing, tiling,
   `cv2.dnn` ONNX inference, and overlay rendering — not a thin wrapper. (Classical CV is *not*
   the detector; STUDY-01 showed why. That honesty is deliberate.)
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
│   ├── colab_baseline.md     # (historical) EXP-001 baseline run guide
│   ├── exp002_kaggle.md      # EXP-002 Kaggle "Save & Run All" guide
│   ├── references.md         # Published-work citations backing design choices
│   └── claude_code_playbook.md
│
├── src/                      # built out — see progress.md
│   ├── cv_pipeline/          # Stage 1 classical CV: pipeline, config, benchmark, tune_coverage
│   └── detection/            # train, infer (cv2.dnn), error_analysis, ablation_fp,
│                             #   fn_gallery, tiled_infer
│       # reporting/ + dashboard/ not built yet (next)
│
├── infra/                    # AWS (deploy-ready, not deployed): lambda_handler.py,
│   │                         #   benchmark_graviton.sh, README.md
├── runs/                     # git-ignored — training runs, weights, benchmark JSON
│
└── DATASET/                  # git-ignored — see §5 (pushed to S3, not committed)
    ├── 01_raw_archives/
    ├── 02_intermediate_processing/
    ├── 03_yolo_ready_dataset_v1/   # data.yaml lives here — TRAIN POINTS HERE
    ├── scripts/                    # active dataset tooling
    └── archive_scripts/            # frozen build scripts (idempotent)
```

> `reporting/`, `dashboard/`, and `tests/` are the remaining target structure — see
> [`TODO.md`](TODO.md) and [`progress.md`](progress.md) for exactly what exists today.

---

## 4. Quickstart

```bash
# 1. Dependencies (global Python 3.10 — no venv, per project convention)
pip install -r requirements.txt

# 2. Train Stage 2 (GPU; dataset entry point below). EXP-001 baseline is done;
#    next is EXP-002 (see docs/exp002_kaggle.md).
python src/detection/train.py --model yolo11s.pt --imgsz 1024 --batch 8 --epochs 40 --name EXP-002
#    Dataset entry point: DATASET/03_yolo_ready_dataset_v1/data.yaml

# 3. Detect with the exported ONNX (cv2.dnn — the AWS/Lambda path)
python -m src.detection.infer <image_or_dir>          # per-class conf thresholds by default

# 4. Deep error analysis (per-class, split sonar vs optical)
python -m src.detection.error_analysis

# 5. Local dashboard — not built yet (next)
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
| train | 26,533 | incl. 1,120 background negatives (99% sonar) |
| val | 1,204 | incl. 15 background |
| test | 1,276 | incl. 58 background |

Built from 7 source datasets (crab-pot, UATD, ICRA19, TrashCan, Marine PULSE,
SeabedObjects-KLSG, AI4Shipwrecks) remapped to the 4-class taxonomy. v1 was produced by
[`DATASET/scripts/build_dataset_v1.py`](DATASET/scripts/build_dataset_v1.py), which fixes the
audited v0 problems: **cross-split leakage → split by source frame (now 0 shared frames)**,
full-frame boxes removed, corrupt-checked, and empty-label images kept as background
negatives. Remaining *training-time* concerns (class imbalance, mixed sensors) are handled at
training, not by data surgery. Full before/after audit: [`docs/dataset_report.md`](docs/dataset_report.md)
(regenerate: `python DATASET/scripts/audit_dataset.py DATASET/03_yolo_ready_dataset_v1`).

---

## 6. Target metrics & current status

| Metric | Target | EXP-001 baseline (test) |
|---|---|---|
| mAP@0.5 | ≥ 0.70 | **0.822** ✅ |
| mAP@0.5:0.95 | ≥ 0.45 | **0.514** ✅ |
| Precision | ≥ 0.80 | **0.808** ✅ |
| Recall | ≥ 0.70 | **0.800** ✅ |
| Per-frame latency | < 300 ms | ~11.5 ms (T4) ✅ |
| Throughput | ≥ 5 FPS | ~87 FPS (T4) ✅ |
| ~~FP reduction (Stage 1)~~ | ~~≥ 60%~~ | **retired** — STUDY-01 |

> Aggregate targets are met, but the aggregate is **inflated by domain segregation**
> (`natural_formation` is optical-only, R 0.99). The real open gap is **`fishing_gear`
> (ghost nets) on sonar** — recall ~0.47 at conf 0.25, lifted to ~0.69 by per-class
> confidence thresholds; EXP-002 (higher resolution) targets the rest. Always read metrics
> **per class and split sonar vs optical** — see [`experiments.md`](experiments.md).

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
| [`docs/exp002_kaggle.md`](docs/exp002_kaggle.md) | EXP-002 Kaggle run guide (next training run) |
| [`docs/references.md`](docs/references.md) | Published-work citations backing design choices |
| [`infra/README.md`](infra/README.md) | AWS deploy plan + the 3-way COOL benchmark |
| [`docs/claude_code_playbook.md`](docs/claude_code_playbook.md) | Working method reference |

---

*Team Syndicate · OpenCV AI Competition 2026 · Best Use of COOL / Agentic Vision.*
