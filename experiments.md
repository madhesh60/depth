# Experiment Register — Marine Debris Detection System

Every training / evaluation run gets one entry. This is the research loop, not a scoreboard:
each experiment states a **hypothesis**, records the exact config, reports metrics, and ends
with **error analysis** + the **next** experiment. No result is trusted without knowing the
dataset version, model, augmentation, and hyperparameters that produced it.

**Related:** [`architecture.md`](architecture.md) · [`progress.md`](progress.md) ·
[`TODO.md`](TODO.md)

---

## Conventions

- **ID:** `EXP-NNN` (zero-padded, monotonic). Run artifacts: `runs/EXP-NNN/` (git-ignored).
- **Dataset version:** `vN` — bump whenever labels/splits change. Never train on an unversioned dataset.
- **No test-set leakage:** the `test` split is touched **only** for final reporting, never for tuning.
- **One variable at a time** where possible, so an improvement is attributable.
- **Special attention to false negatives** on dangerous debris (`fishing_gear` — ghost nets/rope).
- Record per-class metrics + confusion matrix, not just aggregate mAP.
- Copy the template below for each new run; keep newest at the top of the log.

---

## Metric definitions

| Metric | Meaning | Target |
|---|---|---|
| Precision | TP / (TP+FP) — how trustworthy a detection is | ≥ 0.80 |
| Recall | TP / (TP+FN) — how much debris we catch | ≥ 0.70 |
| mAP@0.5 | mean AP at IoU 0.5 | ≥ 0.70 |
| mAP@0.5:0.95 | mean AP over IoU 0.5→0.95 | ≥ 0.45 |
| ~~FP reduction~~ | ~~FP(full-frame) → FP(ROI-guided)~~ — **retired**, see STUDY-01 (Stage-1 gating costs recall) | ~~≥ 60%~~ |

---

## Experiment template (copy for each run)

```
### EXP-NNN — <short title>
- Date:
- Status:            planned | running | done | discarded
- Hypothesis:        what we expect to change and why
- Baseline compared: EXP-MMM (or "none")

Config
- Model:            yolo11s-seg | yolov8s-seg | ...
- Task:             segment | detect
- Dataset version:  vN   (note fixes applied vs previous version)
- Input size:       640 | 768 | 896
- Epochs / Batch:   /
- Optimizer / LR:
- Augmentation:     flip, brightness, contrast, blur, mosaic, underwater color shift, ...
- Class weighting:  none | inverse-freq | focal
- Hardware:         local CPU/GPU | SageMaker <instance>
- Command / config: path or one-liner to reproduce

Results
- Precision:
- Recall:
- mAP@0.5:
- mAP@0.5:0.95:
- Per-class (P/R/AP):   # live 4-class taxonomy
    fishing_gear:
    pipe_cylinder:
    structural_fragment:
    natural_formation:
- Confusion matrix:  (link to runs/EXP-NNN/confusion_matrix.png)
- Inference latency:  full_frame __ ms | roi_guided __ ms
- FP reduction (Stage 1): __%

Error analysis
- False negatives grouped by cause (low-vis / occlusion / small object / blur / domain / annotation):
- False positives (natural formation misclassified, shadows, rock clusters):
- Worst class + likely reason:

Decision & next
- Keep / revert / iterate:
- Next experiment (EXP-___): the single change to try next, and why
```

---

## Planned experiment ladder

**Dataset v1 is ready** (`DATASET/03_yolo_ready_dataset_v1/data.yaml`, leakage-free, 4-class).
These are queued, not run:

Reprioritised 2026-09-22 after the `fishing_gear` FN analysis (small-object problem — see EXP-001
error analysis). Resolution/tiling first; aug-ablation demoted.

| ID | Hypothesis | Key change |
|---|---|---|
| ✅ EXP-001 | Establish a baseline (**done** — see log) | YOLO11s detect, v1, 640, sonar aug → mAP@0.5 0.822, fishing_gear R 0.371 |
| **EXP-002** (next) | **Higher resolution recovers small fishing_gear** (91% of misses <10% frame) | `--imgsz 1280` (else = EXP-001). Cheapest, highest-expected-value lever. |
| EXP-003 | **Tiled train+infer** (SAHI-style slicing) beats a single large frame for tiny targets — and doubles as Stage-1's reframed "tiling preprocessing" role | slice frames → detect per tile → merge |
| EXP-004 | Oversampling fishing_gear + small-object aug (copy-paste, scale-up mosaic) lifts recall | minority oversample + aug |
| EXP-005 | Sonar-only training beats mixed-sensor for the target domain | filter optical sources |
| EXP-006 | Aug ablation: are the ~62% baked-in v0 augs helping or just doubling online aug? | originals-only vs baked-aug (demoted) |

Promote each into the log below with full results as it runs.

---

## Experiment log

### STUDY-01 — Stage-1 classical-CV ROI-gating (NEGATIVE RESULT)
- Date:              2026-09-22
- Status:            done
- Hypothesis:        Stage-1 classical CV (denoise → adaptive threshold → contours → geometry
                     filter) proposes ROIs that (a) cover real debris and (b) suppress background,
                     so gating the YOLO detector to Stage-1 ROIs should cut false positives ≥60%
                     while keeping recall.
- Baseline compared: EXP-001 full-frame detector
- Method:           `src/detection/ablation_fp.py` (full_frame vs roi_guided, gate IoU 0.10,
                    match IoU 0.5) + `src/cv_pipeline/tune_coverage.py` (geometry-filter sweep),
                    300-frame strided test sample; background-ROI probe on 80 empty-label frames.

Results — **the premise fails on this data:**
- Stage-1 **GT coverage 24.7%** with default config (104/421 GT boxes fall in any ROI) →
  hard recall ceiling for roi_guided.
- roi_guided vs full_frame: recall **0.708 → 0.147** (only 20.8% of TPs kept) for FP 124 → 59
  (52.4% "reduction"). The FP drop is an artefact of discarding true detections.
- Geometry-filter sweep: even with **all filters off** ("ceiling" config) coverage tops out at
  **72.4%** — at **141 ROIs/frame**. ~28% of debris does not segment at all (low-contrast,
  blends into seafloor). `min_area_frac` was the dominant lever (0.30 → 0.63 coverage).
- **Discriminative-power probe:** on background (empty-seafloor) frames Stage 1 emits **60.2
  ROIs/frame** vs **8.0** on frames with objects; only **1%** of background frames emit zero.
  Stage 1 fires *more* on clutter than on debris.

Error analysis
- Stage 1 thresholds on **brightness + blob geometry**; the real side-scan-sonar debris cue is
  the **highlight + acoustic-shadow pair**, which this pipeline ignores. Rippled seafloor / sand
  waves / speckle generate abundant bright blobs → clutter dominates.
- Classical contour CV therefore works neither as a per-box recall gate (72% ceiling, huge ROI
  counts) nor as a frame-level triage (louder on empty background).

Decision & next
- **Retire the ROI-gating / ≥60% FP-reduction target.** YOLO (EXP-001) is the detector.
- **Reframe Stage 1** as the CPU **sonar-preprocessing workload** benchmarked for the COOL award
  (Graviton vs x86) — its ops (resize, adaptive threshold, contours) are still the COOL sweet-spot;
  it just no longer feeds ROIs to Stage 2. This negative result + evidence is a submission asset.
- Shadow-aware Stage 1 (highlight+shadow detection) is a possible future upgrade, deferred.

### EXP-001 — Baseline YOLO11s (detect) on dataset v1
- Date:              2026-09-21
- Status:            done
- Hypothesis:        Establish an honest first baseline on the clean v1 split — where do we
                     stand against the targets (mAP@0.5 ≥ 0.70, P ≥ 0.80, R ≥ 0.70) with a
                     stock YOLO11s and no per-class tricks, and which class is the weak link?
- Baseline compared: none (this is the reference point for EXP-002+)

Config
- Model:            yolo11s.pt (detect — **not** seg; 9.41 M params, 21.4 GFLOPs, fused)
- Task:             detect
- Dataset version:  v1  (`03_yolo_ready_dataset_v1`, leakage-free, 4-class, split by clip)
- Input size:       640
- Epochs / Batch:   100 (early-stopping patience 20) / 16
- Optimizer / LR:   Ultralytics defaults (auto optimizer + auto lr0)
- Augmentation:     sonar-aware — hsv_h=0, hsv_s=0, hsv_v=0.2; degrees=0, flipud=0,
                    fliplr=0.5; translate=0.1, scale=0.5; mosaic=1.0, close_mosaic=10
- Class weighting:  none (inverse-freq weights printed for reference only, not applied)
- Hardware:         Kaggle T4 (16 GB), GPU
- Command / config: `python src/detection/train.py --model yolo11s.pt --epochs 100 --batch 16 --name EXP-001`

Results  (test split — 1,276 images / 1,787 instances; all 4 targets met in aggregate ✅)
- Precision:        0.808   (≥ 0.80 ✅)
- Recall:           0.800   (≥ 0.70 ✅)
- mAP@0.5:          0.822   (≥ 0.70 ✅)
- mAP@0.5:0.95:     0.514   (≥ 0.45 ✅)
- Per-class (imgs / inst — P / R / AP@0.5 / AP@0.5:0.95):
    fishing_gear:         537 / 952 — 0.535 / 0.371 / 0.450 / 0.194   ⬅ worst; mission-critical
    pipe_cylinder:         65 /  65 — 0.909 / 0.954 / 0.940 / 0.523   (high-variance, 65 boxes)
    structural_fragment:  402 / 501 — 0.951 / 0.897 / 0.923 / 0.554
    natural_formation:    214 / 269 — 0.835 / 0.978 / 0.973 / 0.784
- Confusion matrix:  runs/EXP-001/confusion_matrix.png (in MyDrive/EXP-001_results)
- Inference latency:  T4 — 1.1 ms preprocess + 9.8 ms inference + 0.6 ms postprocess ≈ 11.5 ms/frame (~87 FPS)
- FP reduction (Stage 1): n/a — full-frame detector only; Stage-1→Stage-2 ROI study is EXP-later

Error analysis
- **`fishing_gear` is the failure mode, not the minority class.** Recall 0.371 means we miss
  ~63% of ghost-net/rope instances — the single most dangerous class (per experiments.md §Conventions).
  Its AP@0.5 (0.45) is the only class below the 0.70 target; aggregate metrics pass *only because*
  the other three classes are strong. This is the headline problem to fix.
- **Imbalance did NOT hurt the rare class.** Counter to blocker B4's worry, `pipe_cylinder`
  (rarest, 4% of boxes) scored AP@0.5 0.94 — but on just 65 test boxes, so treat as high-variance.
  The imbalance lever should target `fishing_gear` *recall*, not `pipe_cylinder`.
- **Why fishing_gear is hard — CONFIRMED (2026-09-22, `src/detection/fn_gallery.py`):** it's a
  **small-object detectability** problem, not confusion or the merged taxonomy.
    - Normalised confusion matrix: true fishing_gear → predicted **background 0.63** (missed
      outright), ~0.00 into any other class. And true **background → fishing_gear 0.79** — it is
      also the dominant false-positive sink (matches P 0.535). Hard both ways.
    - FN gallery (test split, conf 0.25): **517/952 (54%)** GT boxes missed; **91% of the misses
      are < 0.10 of frame width** (median long side **0.059**, 31% < 0.05). Eyeballed frames:
      small low-contrast bright returns embedded in heavy speckle/sand-wave clutter.
    - Note: some crabpot samples carry **baked-in rotation borders** from v0 augmentation
      (black triangular corners) — a possible extra confound; flag for a data spot-check.
- **Implication for next experiments:** the lever is **effective resolution** (higher `imgsz`
  and/or tiling so small targets are larger to the detector) + small-object augmentation, NOT
  the aug-on/off ablation. Reprioritised the ladder accordingly.

Decision & next
- **Keep** as the reference baseline. It clears every aggregate target, so v1 + the sonar-aware
  aug recipe is sound; the open problem is `fishing_gear` recall, not overall capacity.
- Next experiment (**EXP-002**): before touching the class problem, isolate whether the ~62%
  baked-in augmentation is helping or just doubling the online aug — originals-only train vs. this
  run (per `docs/colab_baseline.md` §7). Then attack `fishing_gear` recall via oversampling /
  focal (EXP-004) once the FN cause is confirmed from the confusion matrix.

<!-- template retained below for the next run:
### EXP-NNN — <short title>  (copy from the template section above) -->
