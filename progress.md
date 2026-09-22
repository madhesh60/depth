# Progress Log — Marine Debris Detection System

**Team Syndicate · OpenCV AI Competition 2026**
This is the living status of the project. Update it at the end of every working session:
what moved, what's blocked, what's next. Newest entries at the top of §4.

**Related:** [`README.md`](README.md) · [`architecture.md`](architecture.md) ·
[`experiments.md`](experiments.md) · [`TODO.md`](TODO.md)

---

## 1. Timeline & hard dates

| Date | Milestone |
|---|---|
| 2026-08-13 | Proposal submission window opened |
| 2026-08-26 | Build phase begins |
| **2026-09-18** | **Today — foundation / planning** |
| 2026-09-21 → 10-02 | **Grant check-in** (30-min Zoom, required to unlock 2nd 50% of grant) |
| 2026-10-26 23:59 PT | **Final submission deadline** |
| 2026-10-27 → 11-09 | Judging |
| 2026-11-10 | Winners announced (OpenCV Live!) |

> **~5.5 weeks of build time remain.** The grant check-in is imminent — a concise,
> credible progress story is needed for it (see §5).

---

## 2. Milestone tracker

| Milestone | Status | Notes |
|---|---|---|
| Grant awarded ($150) | ✅ Done | Secured. 2nd half unlocked at check-in. |
| Proposal (AGENT.md) | ✅ Done | Source of record. |
| Dataset built (v0, 32,981 imgs) | ✅ Done | 7 sources remapped to 5 classes; split 30444/1263/1274. |
| Dataset audit | ✅ Done | Issues catalogued in `docs/dataset_report.md`. |
| Project docs (this set) | ✅ Done | README / architecture / progress / experiments / TODO. |
| Dataset fixes (v1) | ✅ Done | Clean split built: 0 leakage, full-frame boxes removed, 1,099 background negatives, 4-class taxonomy locked. `03_yolo_ready_dataset_v1/`. |
| Stage 1 classical CV | 🟡 In progress | **Reframed** (STUDY-01): not a Stage-2 ROI gate (no discriminative power on this sonar) — now the CPU **sonar-preprocessing workload** for the COOL benchmark. `src/cv_pipeline/` + 3-way harness. Local x86: 29 ms/frame, 30.5 FPS. Needs Graviton+COOL run. |
| Stage 1→2 wiring (cv2.dnn) | ✅ Done | `infer.py` (ONNX via `cv2.dnn`, full_frame + roi_guided) + `ablation_fp.py` + `tune_coverage.py`. ONNX verified loading/forward. |
| Stage 2 baseline train | ✅ Done | EXP-001 (yolo11s detect, v1, Kaggle T4): test mAP@0.5 **0.822**, P 0.808, R 0.800 — all targets met. |
| Evaluation + ablation | 🟡 In progress | Baseline (EXP-001) + STUDY-01 (ROI-gating negative result) logged. Open: `fishing_gear` recall 0.37 (EXP-002/004). FP-reduction target **retired**. |
| Reporting engine | ⬜ Not started | JSON/CSV/GeoJSON + geotagging. |
| Dashboard | ⬜ Not started | FastAPI + map + transparency view. |
| AWS deployment | ⬜ Not started | AWS CLI not yet installed. |
| **COOL benchmark (Arm vs x86)** | ⬜ Not started | **Bonus-prize deliverable (Best Use of COOL).** |
| Agentic loop (stretch) | ⬜ Not started | Only if ahead of schedule. |
| Submission package (report + video) | ⬜ Not started | Due 2026-10-26. |

Legend: ✅ done · 🟡 in progress · ⬜ not started · ⛔ blocked

---

## 3. Current state (2026-09-18)

**What exists**
- Grant secured; proposal finalised (`AGENT.md`).
- Dataset v0 built and audited: `DATASET/03_yolo_ready_dataset/data.yaml` is the training
  entry point. Build scripts frozen in `DATASET/archive_scripts/` (idempotent).
- Pinned dependencies (`requirements.txt`).
- Full project documentation set created (this commit).

**What does not exist yet**
- No `src/`, `infra/`, or `tests/` code — greenfield.
- Dataset v0 has **known blockers** (see §6) that must be fixed before the first real train.
- AWS CLI not installed; no cloud resources provisioned; COOL not yet exercised.

**Environment**
- Global Python 3.10 (no venv). `torch`, `torchvision`, `opencv-python`, `numpy`,
  `PyYAML`, `Pillow` present. `pip install -r requirements.txt` adds the rest.

---

## 4. Session log

### 2026-09-22 — Stage 1→2 wired via cv2.dnn; ROI-gating is a NEGATIVE result → Stage 1 reframed
- **Wired Stage 2 for the Lambda path:** extracted EXP-001 artifacts locally, verified
  `best.onnx` loads + runs through `cv2.dnn.readNetFromONNX` (output `(1,8,8400)`; 4 box + 4 cls).
  Built `src/detection/infer.py` (`YoloOnnxDetector`, full_frame + roi_guided modes, letterbox
  + NMS, CLI) and `src/detection/ablation_fp.py` (full vs ROI-gated TP/FP/recall + Stage-1 GT
  coverage), plus `src/cv_pipeline/tune_coverage.py` (geometry-filter sweep).
- **The two-stage FP premise fails on this data (STUDY-01).** On a 300-frame strided test
  sample: Stage-1 **GT coverage 24.7%**; ROI-gating drops recall **0.708→0.147** for a fake
  52% "FP reduction" (it just discards true detections). Sweeping geometry filters, coverage
  **caps at 72.4% with ALL filters off — at 141 ROIs/frame** (~28% of debris never segments).
  Killer probe: on **empty-seafloor** frames Stage 1 emits **60 ROIs/frame** vs **8** on frames
  with objects — it fires *more* on clutter than on debris. It thresholds brightness/geometry
  and ignores the real sonar cue (highlight + acoustic shadow).
- **Decision (locked):** retire the ROI-gating / ≥60% FP-reduction target; **YOLO is the
  detector**; **reframe Stage 1 as the CPU sonar-preprocessing workload for the COOL benchmark**
  (its resize/threshold/contour ops are still the COOL sweet-spot — it just no longer gates
  Stage 2). Negative result + evidence kept as a submission asset. Shadow-aware Stage 1 deferred.
- **Next:** attack `fishing_gear` recall (0.371) — pull confusion matrix + FN gallery, prep
  EXP-002 (aug ablation) / EXP-004 (oversample/focal) retrain configs for Kaggle.

### 2026-09-21 — EXP-001 baseline trained (Kaggle T4) — all aggregate targets met
- **First real training run is in.** YOLO11s (detect, not seg), dataset v1, 640, sonar-aware
  aug, 100 ep / batch 16 on a Kaggle T4. Logged as **EXP-001** in `experiments.md` with full
  per-class results.
- **Test-split result (1,276 imgs / 1,787 inst):** mAP@0.5 **0.822**, mAP@0.5:0.95 **0.514**,
  precision **0.808**, recall **0.800** — **every aggregate target cleared** (≥0.70 / ≥0.45 /
  ≥0.80 / ≥0.70). Inference ~11.5 ms/frame on T4.
- **The one problem: `fishing_gear`.** Recall **0.371** (AP@0.5 0.45) — we miss ~63% of the
  single most dangerous class (ghost nets / rope). Aggregate metrics pass only because the other
  three classes are strong (pipe_cylinder AP 0.94 on 65 boxes — high-variance; struct 0.923;
  natural 0.973). Counter to blocker B4, imbalance hurt the *majority* merged class, not the rare one.
- **Next:** pull `confusion_matrix.png` + a fishing_gear FN gallery to classify the misses;
  run **EXP-002** (originals-only vs baked-aug ablation), then attack fishing_gear recall
  (oversample / focal, EXP-004). Export `best.onnx` for the Stage-1→Stage-2 wiring.

### 2026-09-21 — Stage 1 CV pipeline + COOL benchmark harness; verified COOL scorecard
- **Verified what COOL actually is** (was an assumption): Cloud-Optimized OpenCV, a
  KleidiCV-accelerated Arm/Graviton build that speeds up **resize, adaptive-gaussian
  threshold, contour detection** (~1.5× avg). Proposal's expansion was correct.
- **Got the "Best Use of COOL" scorecard:** verified COOL integration on Graviton **30%** +
  measured performance vs baseline **20%** + architecture **25%** + innovation **15%** +
  reproducibility **10%**. ⇒ **half the award is a reproducible Graviton-vs-x86 benchmark of
  the core workload**, not a fancier model.
- **Built Stage 1** (`src/cv_pipeline/`): `config.py` (all tunables, COOL-friendly defaults),
  `pipeline.py` (`Stage1Pipeline.process()`, per-op timed, + `draw_candidates()`),
  `benchmark.py` (3-way harness), `README.md`. Deliberately composed of COOL-accelerated ops.
- **Tested on real sonar frames** (v1 test set): local x86 stock OpenCV 4.12, 400 frames →
  **29.2 ms/frame, 30.5 FPS**, avg 10 ROIs/frame. Per-op: **threshold+contours+resize = ~76%
  of compute** — i.e. the compute concentrates in exactly the ops COOL accelerates, so the
  Graviton+COOL speedup will be attributable. Well under 300 ms / above 5 FPS.
- **Key design calls:** `fastNlMeansDenoising` OFF by default (slow + NOT KleidiCV-accelerated,
  would dilute COOL's measured gain) — median denoise default, NLMeans kept in `QUALITY_PRESET`
  for the ablation. Benchmark is **three-way** (x86-stock / Graviton-stock / Graviton-COOL) so
  the Graviton-stock→Graviton-COOL delta isolates COOL on identical hardware.
- **Architecture guidance:** AWS footprint for the COOL award is small — EC2 Graviton (c7g/c8g)
  + COOL running Stage 1, EC2 x86 baseline, S3 for frames/results, CloudWatch/CSV for metrics.
  Lambda/DynamoDB/Amplify/SageMaker are overall-award/demo polish, built only after COOL locked.
- **Next:** run the benchmark on a Graviton instance with COOL vs stock (needs AWS CLI + EC2);
  wire Stage 1 → Stage 2 (ROI crop → `cv2.dnn.readNetFromONNX`) once EXP-001 weights land.

### 2026-09-18 — CRITICAL FIX: v1 split had dead classes; rebuilt & stratified
- **Bug found via audit:** the on-disk v1 dataset did **not** match its manifest — a stale/partial
  build had **zero `pipe_cylinder` and zero `structural_fragment` boxes in val AND test** (2 of 4
  classes unmeasurable), and `pipe_cylinder` was nearly gone overall (105 boxes). Any mAP would
  have silently excluded two classes. v0 source verified intact (all 4 classes healthy).
- **Fix:** rewrote the split to stratify by **(source, dominant-class)** at the recording/clip
  level — guarantees every source and every class is represented ~80/10/10 in val/test while
  staying leakage-free. Rebuilt from source. New counts: **train=27,380 / val=1,220 / test=957**,
  leakage **0/0/0**, **all 4 classes now present in every split** (val pipe=100/struct=479,
  test pipe=105/struct=551).
- **Residual (honest):** optical-sensor share still skews val 13% / test 30% (only a few optical
  video clips; whole clips can't be subdivided). Options for later: make val/test sonar-only, or
  accept + report the mix. Not a blocker for EXP-001 but note it when reading metrics.
- **Next:** eyeball QA renders → `pip install -r requirements.txt` → EXP-001 baseline.

### 2026-09-18 — Split refinement + Stage 2 trainer
- **Dataset v1 split hardened.** Replaced the base-frame split key with a source-aware
  `seq_group` (video-clip id for icra/vid, recording+channel for crabpot, wreck/object for
  shipwreck, survey site for mpulse, per-frame otherwise). Whole groups are placed greedily
  to keep per-class box share + image count near 80/10/10. Also drop degenerate **sliver**
  boxes (w or h < 0.01). Re-audit: **leakage 0/0/0**, 5,618 groups; counts refreshed
  (train=26,389 / val=1,172 / test=1,734; sensor split ~80/20 sonar/optical).
- **Stage 2 trainer added** (`src/detection/train.py`): reads v1 `data.yaml`, sonar-aware
  augmentation (hue/sat off, no rotation/vflip, along-track hflip only), prints inverse-freq
  class weights for the imbalance, and reports test-split mAP/P/R at the end. `--seg` switches
  to instance segmentation. Verified `--help` runs without the training deps installed.
- **Next:** eyeball QA renders → `pip install -r requirements.txt` → run **EXP-001** baseline.

### 2026-09-18 — Foundation, docs cleanup & dataset audit
- Read full context (AGENT.md proposal, competition rules, dataset report, CC playbook).
- Established peak architecture and winning plan (`architecture.md`).
- Created the formal documentation set: `README.md`, `architecture.md`, `progress.md`,
  `experiments.md`, `TODO.md`; defined naming conventions.
- Moved reference docs into `docs/` and rewrote them formally: `competition_rules.md`,
  `claude_code_playbook.md`, `dataset_report.md` (removed clumsy `comptetion.md`,
  `HOWTOWORK.md`, root `dataset_report.md`).
- Built `DATASET/scripts/audit_dataset.py` and ran a full ground-truth audit. **Key
  findings:** dataset is **4-class** (`fishing_gear`, not 5 w/ `rope_line`); **all boxes,
  zero polygons** (already converted); **cross-split leakage** from `crab_pot`; 1,258
  full-frame + 1,713 tiny boxes. Old report was stale → rewritten (`docs/dataset_report.md`).
- Built **dataset v1** (`03_yolo_ready_dataset_v1/`) via `build_dataset_v1.py`: source-frame
  grouped split (**leakage 0/0/0**, re-audited), full-frame boxes dropped (1,096), corrupt
  scan (0), 1,099 background negatives, 4-class taxonomy locked (`rope_line` stays merged).
  Added `visualize_labels.py` (QA renders in `DATASET/exports/qa_*`).
- **Next:** eyeball QA renders → Stage 1 CV (`src/cv_pipeline/`) → baseline train EXP-001 on v1.

---

## 5. Grant check-in talking points (Sep 21 – Oct 2)

Keep it concrete and honest:
1. **Problem & impact** — ghost-net detection from side-scan sonar; 4–8h manual review → minutes.
2. **Dataset** — 32,981 images / 47,881 boxes from 7 sources, unified to a 4-class taxonomy;
   audited with a documented, regenerable quality report.
3. **Architecture** — two-stage OpenCV 5 pipeline; Stage 1 classical CV is the **COOL core
   workload** on Graviton, benchmarked vs x86.
4. **Plan to deadline** — phased execution in `TODO.md`; primary path is Best Use of COOL.
5. **Evidence of active development** — this documentation set + dataset audit + committed repo.

---

## 6. Known blockers (must clear before first real training run)

Re-derived from the verified audit ([`docs/dataset_report.md`](docs/dataset_report.md),
`python DATASET/scripts/audit_dataset.py`). The earlier polygon/5-class blockers are
**resolved** — the dataset was reprocessed (polygons→boxes, `rope_line` merged into
`fishing_gear`, collapsed to 4 classes).

| # | Blocker | Status |
|---|---|---|
| B1 | Cross-split leakage | ✅ Resolved — v1 split by source frame; re-audit 0/0/0 |
| B2 | 1,258 full-frame boxes | ✅ Resolved — dropped in v1 (audit: 0 remain) |
| B3 | Tiny boxes | ✅ Handled — degenerate dropped; small-plausible kept + flagged for QA |
| B4 | Class imbalance (`fishing_gear` ≈ 11.7× `pipe_cylinder`) | 🟡 Open — training-time (weights/focal/aug), not a data defect |
| B5 | Taxonomy decision | ✅ Resolved — **4-class**, `rope_line` merged into `fishing_gear` |
| B6 | AWS CLI not installed | 🟡 Open — blocks cloud + COOL work |

Details and fixes tracked in [`TODO.md`](TODO.md) Phase 1 & Phase 4.

---

## 7. Metrics dashboard (fill as results arrive)

| Metric | Target | Current | Δ |
|---|---|---|---|
| mAP@0.5 | ≥ 0.70 | **0.822** (EXP-001) | +0.122 ✅ |
| mAP@0.5:0.95 | ≥ 0.45 | **0.514** (EXP-001) | +0.064 ✅ |
| Precision | ≥ 0.80 | **0.808** (EXP-001) | +0.008 ✅ |
| Recall | ≥ 0.70 | **0.800** (EXP-001) | +0.100 ✅ |
| `fishing_gear` recall (watch) | ≥ 0.70 | **0.371** (EXP-001) | −0.329 ❌ |
| ~~FP reduction (Stage 1)~~ | ~~≥ 60%~~ | **retired** (STUDY-01: gating costs recall) | — |
| Latency / frame (Stage 2, T4) | < 300 ms | ~11.5 ms (EXP-001) | ✅ |
| Throughput (Stage 2, T4) | ≥ 5 FPS | ~87 FPS (EXP-001) | ✅ |
| COOL: Graviton vs x86 latency | measured | — | — |

_Link each filled row to the `EXP-NNN` that produced it. Note: aggregate targets met, but
`fishing_gear` (mission-critical) recall is the open gap._
