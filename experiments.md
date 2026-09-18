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
| FP reduction | FP(full-frame) → FP(ROI-guided), relative | ≥ 60% |

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

| ID | Hypothesis | Key change |
|---|---|---|
| EXP-001 | Establish a baseline | YOLO11s-seg, dataset v1, 640, default aug, 100 ep |
| EXP-002 | Bigger input helps small sonar targets | input 768/896 vs EXP-001 |
| EXP-003 | Arch comparison | YOLOv8s-seg vs YOLO11s-seg, else identical |
| EXP-004 | Class weighting lifts `pipe_cylinder` (minority) recall | focal / inverse-freq weighting |
| EXP-005 | Sonar-domain augmentation reduces FP on natural_formation | speckle/gain/shadow aug |
| EXP-006 | Sonar-only training beats mixed-sensor for the target domain | filter optical sources |

Promote each into the log below with full results as it runs.

---

## Experiment log

_None run yet. Dataset v1 is ready and the trainer is wired
(`python src/detection/train.py --model yolo11s.pt --name EXP-001`, add `--seg` for
instance masks). EXP-001 is next, after the visual-QA pass and once `ultralytics` is
installed (`pip install -r requirements.txt`)._

<!-- ### EXP-001 — Baseline YOLO11s-seg on dataset v1
     (copy the template above) -->
