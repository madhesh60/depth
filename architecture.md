# Architecture — Marine Debris Detection System

**Status:** Design baseline · **Owner:** Madhesh (Team Syndicate) · **Last updated:** 2026-09-22

This document is the single source of truth for system design. It records *what* we are
building, *how* the pieces fit, *where* each runs, and *why* those choices win the
competition. Update it whenever a structural decision changes — not the code comments.

> **⚠️ 2026-09-22 architecture change (see `experiments.md` STUDY-01).** Empirically, classical
> Stage 1 has **no discriminative power** on this sonar data, so the original **two-stage
> ROI-gating** model below is **retired**: Stage 1 no longer feeds candidate ROIs to Stage 2,
> the **`roi_guided` mode and the ≥60% FP-reduction target are dropped**, and **YOLO runs
> full-frame** as the detector (EXP-001, mAP@0.5 0.822). Stage 1 is repurposed as the CPU
> **sonar-preprocessing workload** benchmarked for the COOL award. Sections §3–§4 below still
> describe the old gating design and are pending rewrite; read them with this banner in mind.

**Related:** [`README.md`](README.md) · [`AGENT.md`](AGENT.md) (proposal) ·
[`progress.md`](progress.md) · [`experiments.md`](experiments.md) · [`TODO.md`](TODO.md)

---

## 1. Design goals & guiding principles

| Principle | Consequence for the design |
|---|---|
| **Win the COOL award first** | Stage 1 is the *core workload*, runs on Graviton/Arm via COOL, and every design choice preserves a clean x86-vs-Arm benchmark comparison. |
| **OpenCV 5 is substantive, in both stages** | Classical CV (Stage 1) *and* DNN inference + compositing + visualisation (Stage 2) go through OpenCV 5. |
| **Edge-first, cloud-optional** | The full pipeline must run CPU-only on a laptop/AUV; AWS is where we scale and benchmark, not a hard dependency for a demo. |
| **Explainability sells** | The pipeline is inspectable stage-by-stage (raw → Stage 1 masks → Stage 2 detections) — a judge can see *why* each call was made. |
| **Reproducibility is a deliverable** | Every experiment, benchmark, and report is regenerable from pinned inputs + a script. |
| **Modular, config-driven** | No dataset-specific constants in code; behaviour is set by config files and versioned. |

**Non-negotiable acceptance metrics:** mAP@0.5 ≥ 0.70 · mAP@0.5:0.95 ≥ 0.45 ·
precision ≥ 0.80 · recall ≥ 0.70 · ~~FP reduction ≥ 60%~~ *(retired — STUDY-01)* ·
latency < 300 ms · throughput ≥ 5 FPS.

---

## 2. System overview

```
                          ┌──────────────────────────────┐
                          │   Side-scan sonar log +       │
                          │   ping metadata (lat/lon/     │
                          │   heading/depth/timestamp)    │
                          └───────────────┬───────────────┘
                                          │ upload
                                          ▼
     ┌───────────────────────────────────────────────────────────────────┐
     │                       INGESTION  (S3 /raw/)                         │
     └───────────────────────────────┬───────────────────────────────────┘
                                      │ S3 event → Lambda / local runner
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  STAGE 1 — CLASSICAL OpenCV 5      [ AWS Graviton / Arm · via COOL ]      │
  │  cvtColor(HSV/LAB) → denoise(NLMeans/median) → adaptiveThreshold/Otsu     │
  │  → morphologyEx(OPEN) → findContours → geometry filter → candidate ROIs   │
  └───────────────────────────────┬───────────────────────────────────────────┘
                                   │ N candidate ROIs (≪ full frame)
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  STAGE 2 — YOLOv8/YOLO11-seg      [ cv2.dnn.readNetFromONNX · x86/GPU ]   │
  │  ROI crop (boundingRect) → inference → NMS → mask → class + confidence    │
  │  → confidence filter (τ) → composite (addWeighted, polylines, putText)    │
  └───────────────────────────────┬───────────────────────────────────────────┘
             verified detections   │                 low-confidence (τ_low < c < τ)
                                   │                          │
                                   ▼                          ▼  [ STRETCH ]
  ┌───────────────────────────────────────┐   ┌─────────────────────────────────┐
  │  STAGE 3 — GEOTAG + REPORT             │   │  AGENTIC LOOP                   │
  │  fuse ping metadata → per-detection    │   │  perception → decision → action │
  │  record → JSON / CSV / GeoJSON         │   │  re-scan / gain-adjust request  │
  │  → DynamoDB + S3 /reports/             │   │  (MCP-exposed OpenCV/COOL tools)│
  └───────────────────┬────────────────────┘   └─────────────────────────────────┘
                      │
                      ▼
   Dashboard (Amplify/FastAPI) · CloudWatch benchmarks · downloadable reports
```

---

## 3. Stage 1 — Classical OpenCV (the COOL core workload)

Runs CPU-only so it is deployable at the edge and, critically, on **AWS Graviton via
COOL**. ~~Produces candidate ROIs that shrink Stage 2's search space.~~ *(RETIRED as an ROI
gate — STUDY-01; this is now the **sonar-preprocessing / benchmark workload**. The op table
and COOL relevance below still stand — those ops are exactly what we benchmark.)*

| # | Step | OpenCV 5 op | Rationale |
|---|---|---|---|
| 1 | Intensity-space transform | `cv2.cvtColor` (BGR→HSV / LAB) | Debris reflects acoustic signal strongly; isolate brightness (V / L) channel independent of illumination. |
| 2 | Speckle denoise | `cv2.fastNlMeansDenoising`, `cv2.medianBlur` | Suppress acoustic speckle without eroding true object edges. |
| 3 | Adaptive segmentation | `cv2.adaptiveThreshold`, `cv2.threshold(THRESH_OTSU)` | Per-region cutoff compensates for shadow gradients + motion-induced drift (heave/pitch/roll). |
| 4 | Morphological filtering | `cv2.morphologyEx(MORPH_OPEN)` | Drop transient sensor clusters; keep dense debris-consistent blobs. |
| 5 | Geometric contour classification | `cv2.findContours` + area/aspect/solidity/extent | Reject irregular natural formations; retain compact/elongated man-made profiles. |

**Contract:** input = single sonar frame (grayscale or RGB); output = list of ROI
bounding boxes + binary candidate masks + per-ROI geometric feature vector. Config-driven
thresholds (no hardcoding). Deterministic given the same frame + config.

**COOL relevance:** every op above is a "widely used vision operation" COOL accelerates on
Graviton — this is the workload we benchmark Arm-vs-x86.

---

## 4. Stage 2 — Deep-learning verification

| Component | Implementation | Notes |
|---|---|---|
| Architecture | **YOLO11s detect** (EXP-001; `-seg` is an option, not yet trained) | 4-class detection. mAP@0.5 0.822 baseline — see [`experiments.md`](experiments.md). |
| Inference path | `cv2.dnn.readNetFromONNX()` | Model exported to ONNX; loaded through OpenCV 5 DNN so the demo + Lambda path is pure OpenCV (no torch). `onnxruntime` for parity checks. |
| ~~ROI cropping~~ | ~~`cv2.boundingRect()`~~ | **Retired (STUDY-01)** — full-frame detection instead. |
| Post-process | NMS + confidence filter (τ, default 0.5) | Detections below τ discarded (noise-filtering requirement). |
| Compositing | `cv2.addWeighted`, `cv2.polylines`, `cv2.putText`, `cv2.rectangle` | Overlays for dashboard + transparency view. |

**Two inference modes** (in `src/detection/infer.py`):
- `full_frame` — YOLO on the whole frame. **This is production** (EXP-001, mAP@0.5 0.822).
- `roi_guided` — YOLO gated to Stage-1 ROIs. **RETIRED (STUDY-01):** it cut recall 0.71→0.15
  for a fake FP "reduction"; kept only as the evidence harness (`ablation_fp.py`).

The ROI-gating ablation is documented as a **negative result** (STUDY-01) — the honest outcome,
not a ≥60% win. Stage 1's real value is the COOL benchmark workload, not FP gating.

---

## 5. Software architecture (`src/`)

```
src/
├── common/            # config loader, pydantic schemas, logging, geo utils, constants
│   ├── config.py      # typed config; all thresholds/paths live here, never in logic
│   ├── schemas.py     # Detection, Report, PingMetadata dataclasses/pydantic models
│   └── geo.py         # pixel→lat/lon projection from ping metadata
├── cv_pipeline/       # STAGE 1
│   ├── preprocess.py  # cvtColor, denoise
│   ├── segment.py     # adaptive threshold + morphology
│   ├── contours.py    # contour extraction + geometric filtering → ROIs
│   └── pipeline.py    # orchestrates Stage 1, returns ROI list
├── detection/         # STAGE 2
│   ├── train.py       # ultralytics training entrypoint (reads data.yaml)
│   ├── export_onnx.py # .pt → .onnx + cv2.dnn load verification
│   ├── infer.py       # cv2.dnn inference, NMS, confidence filter (full_frame|roi_guided)
│   └── evaluate.py    # mAP/P/R/F1, confusion matrix, per-class, ablation report
├── reporting/         # STAGE 3
│   ├── geotag.py      # detection + ping metadata → geotagged record
│   └── report.py      # JSON / CSV / GeoJSON writers
├── dashboard/         # PRODUCT
│   ├── app.py         # FastAPI: upload, run pipeline, serve overlays + reports
│   ├── routes/        # /upload /detect /report /benchmark
│   └── ui/            # upload + interactive map + transparency view
└── agentic/           # STRETCH
    ├── tools.py       # OpenCV/COOL operations exposed as agent tools (MCP)
    └── loop.py        # perception→decision→action controller
```

**Rules:** config-driven (no magic numbers in logic), typed I/O contracts between stages,
each stage independently testable with fixtures under `tests/`.

---

## 6. AWS architecture

| Layer | Service | Role |
|---|---|---|
| Data ingestion | **Amazon S3** | `raw/ · processed/ · models/ · reports/ · benchmarks/` prefixes, organised per survey mission. |
| Model training | **Amazon SageMaker** | Fine-tune YOLO-seg; experiment tracking + Model Registry versioning. |
| Inference API | **Lambda + API Gateway** | Serverless endpoint: accept frame → Stage 1 → Stage 2 → JSON. |
| **COOL compute** | **AWS Graviton via COOL** | Executes Stage 1 classical workload on Arm — the award-critical path. |
| Detection storage | **DynamoDB** | Per-detection records; GSIs for geo + temporal queries. |
| Report generation | **S3 + Lambda** | On-demand JSON/CSV/GeoJSON. |
| Dashboard hosting | **AWS Amplify** | Upload / map / download UI. |
| Monitoring | **CloudWatch** | Latency, throughput, errors, utilisation → benchmark evidence. |

**Cost discipline (limited grant + Free Tier):** default to serverless + on-demand;
spin Graviton/x86 benchmark instances up only for measurement runs and tear down after;
keep the large dataset in S3, not in compute storage. Audit for idle resources weekly.

---

## 7. COOL benchmark harness (primary judging criterion)

This is a **first-class deliverable**, not an afterthought. Lives in `infra/benchmarks/`.

**Method:** run the identical Stage-1 pipeline over ≥1,000 fixed frames on:
- **Graviton (Arm) + COOL** — the claimed core workload.
- **x86 baseline** — matched instance class, same inputs, same config.

**Reported metrics** (per instance, CSV + chart, regenerable):

| Metric | Source |
|---|---|
| Per-frame latency (ms), p50/p95 | in-process timing + CloudWatch custom metric |
| Throughput (frames/s) sustained | continuous-load run |
| CPU utilisation (%) | CloudWatch EC2 metrics |
| Cost per 1,000 frames ($) | Cost Explorer, normalised by instance type |

**Evidence package for judges:** COOL version, Graviton instance type, exact deploy config,
input manifest hash, and a one-command reproduction script. See [`TODO.md`](TODO.md) Phase 4.

---

## 8. Agentic Vision loop (stretch)

Qualifies only if *visual evidence changes the next action*. Design:

```
perceive (Stage 1+2 on frame) → assess confidence
   ├─ c ≥ τ            → accept detection, write report
   ├─ τ_low ≤ c < τ    → DECIDE: request re-scan  → ACTION: emit re-scan/gain-adjust
   │                      task (higher gain, overlapping pass) → re-perceive
   └─ c < τ_low        → discard as clutter
```

- OpenCV/COOL operations are exposed as **MCP tools** (`src/agentic/tools.py`), so an agent
  can invoke `stage1_preprocess`, `run_inference`, `adjust_gain`, `request_rescan` and
  choose the next call based on the visual result.
- Deliverable: an agent workflow diagram + a **trace** showing a low-confidence detection
  triggering a re-scan that changes the outcome, plus task-success / failure-handling eval.

---

## 9. Data flow & contracts

1. **Ingest** → sonar frames + ping metadata to `S3://.../raw/<mission>/`.
2. **Trigger** → S3 event invokes Lambda (or local runner in dev).
3. **Stage 1 (COOL/Graviton)** → ROIs + candidate masks.
4. **Stage 2 (OpenCV DNN)** → verified, classified, segmented detections + confidence.
5. **Store** → detections → DynamoDB; overlays/artifacts → `processed/`.
6. **Report** → geotagged JSON/CSV/GeoJSON → `reports/`; served/downloadable.
7. **Monitor** → latency/throughput/errors → CloudWatch → benchmark dashboard.

**Detection record schema (canonical):**
```json
{
  "detection_id": "uuid",
  "mission_id": "string",
  "source_frame": "raw/<mission>/<frame>.png",
  "class_id": 0, "class_name": "fishing_gear",
  "confidence": 0.0,
  "bbox_px": [x, y, w, h],
  "mask_area_m2": 0.0,
  "lat": 0.0, "lon": 0.0,
  "depth_m": 0.0, "heading_deg": 0.0,
  "timestamp": "ISO-8601",
  "pipeline_version": "semver",
  "model_version": "yolo11s-seg_v1_20260920"
}
```
Every detection is traceable to its source frame — a hard requirement.

---

## 10. Naming conventions (repo + dataset)

Consistency here is graded indirectly (reproducibility, clarity) — enforce it.

**Documents** — top-level docs use the names fixed by this project: `README.md`,
`architecture.md`, `progress.md`, `experiments.md`, `TODO.md`, `CLAUDE.md`, `AGENT.md`.
Reference material lives under `docs/`: `competition_rules.md`, `claude_code_playbook.md`,
`dataset_report.md` (the earlier stray `comptetion.md` / `HOWTOWORK.md` were formalised and moved here).

**Python** — `snake_case` modules/functions, `PascalCase` classes, `UPPER_SNAKE` constants.
One responsibility per module (mirrors §5).

**Dataset** — keep the working numeric-stage prefix scheme (it reads top-to-bottom):
```
DATASET/
  01_raw_archives/                # immutable source datasets
  02_intermediate_processing/     # remapped/unified, class_counts.json
  03_yolo_ready_dataset/          # final split; data.yaml <- TRAIN POINTS HERE
      {train,val,test}/{images,labels}/
  scripts/                        # active tooling
  archive_scripts/                # frozen, idempotent build scripts
  exports/                        # generated reports/manifests
```
Image files carry a **source prefix** (`crabpot_`, `uatd_`, `icra_`, `vid_`, `mpulse_`,
`seabed_`, `shipwreck_`) so provenance is recoverable — keep this.

**Models** — `models/<arch>_<dataset-version>_<yyyymmdd>.{pt,onnx}`
(e.g. `yolo11s-seg_v1_20260920.pt`).

**Experiments** — IDs `EXP-NNN`; runs under `runs/EXP-NNN/` (git-ignored); logged in
[`experiments.md`](experiments.md).

**Dataset versions** — `vN` tag bumped whenever labels/splits change; recorded per
experiment so results are attributable to a snapshot.

**S3 prefixes** — `raw/ processed/ models/ reports/ benchmarks/`, each partitioned by
`<mission_id>/`.

**Git** — small, typed commits: `feat:`, `fix:`, `data:`, `infra:`, `docs:`, `exp:`.

---

## 11. Risks & mitigations

Risks below are re-derived from the verified audit ([`docs/dataset_report.md`](docs/dataset_report.md)),
which supersedes the earlier polygon/5-class assumptions.

| Risk | Impact | Mitigation |
|---|---|---|
| **Cross-split leakage** — same `crab_pot` source frame in multiple splits (361 train↔val, 359 train↔test, 118 val↔test) | Inflated val/test scores → model looks better than it is; invalidates evaluation evidence | Re-split by **grouped source frame** so all augmentations of a frame stay in one split; re-run audit to confirm 0 shared keys (TODO Phase 1). |
| 1,258 full-frame boxes (w&h>0.95, mostly Marine PULSE) | Degenerate boxes hurt localisation | Detect + drop/relabel (TODO Phase 1). |
| 1,713 tiny boxes (area < 0.0005) | Possible annotation noise | Min-box filter + sample-inspect. |
| Severe class imbalance (`fishing_gear` ≈ 7.8× `pipe_cylinder`) | Biased model | Class weights / focal loss / targeted augmentation; report per-class. |
| Mixed sensors (80% sonar, 20% optical) | Domain shift | Track sonar-only vs mixed metrics; consider sonar-only fine-tune. |
| Taxonomy drift: dataset is 4-class (`fishing_gear`) vs proposal's 5-class (separate `rope_line`) | Submission/eval mismatch | Resolve the §12 open decision before v1. |
| COOL/Graviton setup unfamiliar + AWS CLI not installed | Blocks the COOL bonus prize | Install AWS CLI early; validate COOL on a tiny workload before scaling (TODO Phase 4). |
| Solo builder, ~5.5 weeks to deadline | Scope risk | Ruthless phase gating; agentic loop is explicitly *stretch*. |

---

## 12. Open decisions

- **Taxonomy:** keep the live 4-class `fishing_gear` taxonomy, or re-introduce `rope_line`
  as a 5th class to match the proposal. (Affects `data.yaml`, all class tables, and the
  submission's evaluation story.)
- YOLOv8-seg vs YOLO11-seg vs detection-only YOLO — decide from EXP-001..003.
- Input resolution (640 vs 768/896 for small sonar targets).
- Whether to train sonar-only vs mixed-sensor as the primary model.
- Lambda inference vs a small always-warm container for the live demo.

_Log decisions as they are made; move resolved items into the relevant section above._
