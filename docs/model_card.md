# Model card — DEPTH detector (EXP-003, deployed since 2026-10-06)

## Overview

- **Architecture:** YOLO11s (detect), 2 classes `ghost_gear`, `wreck_debris`. Exported to ONNX
  (opset 12) and run through OpenCV 5 `cv2.dnn` at inference (torch-free deploy; `cv2.dnn` forward
  checked: output `(1, 6, 8400)` at 640 px).
- **Training:** dataset **v2b tiles** (`build_tiles.py`: full frames + 2×2 tiles, cut boxes inpainted),
  640 px, **explicit SGD lr 0.01** (nominal batch 64), seed 42, Kaggle T4. 104 of 150 epochs (early
  stop), 135 min. Sonar-aware augmentation: no hue / saturation shift, no rotation, no vertical flip,
  horizontal flip 0.5 (along-track), value ±0.2, translate 0.1, scale 0.5, mosaic.
- **Checkpoint selection:** best `ghost_gear` AP@0.5 on the held-out validation recordings (Rec10 / 12 /
  16), not ultralytics' default fitness: epoch 67.
- **Why EXP-003:** best of six clean runs (EXP-002–005). EXP-002 underfit (`optimizer=auto` built AdamW
  0.00167; tiles left cut objects unlabelled), EXP-004 / 005 changed the data and did not beat it
  ([`exp003_diagnosis.md`](exp003_diagnosis.md), [`exp005_diagnosis.md`](exp005_diagnosis.md)).
- **Earlier baseline:** EXP-001 (4 classes, dataset v1). Its numbers stay in
  [`eval_exp001.md`](eval_exp001.md) and [`calibration_exp001.md`](calibration_exp001.md).

## Intended use

Decision-support triage of side-scan sonar surveys for derelict fishing gear (crab pots) and wreck
debris, **with a human in the loop**. It ranks and proves candidates for a reviewer; it does not make
autonomous removal decisions.

## Out of scope

- Optical / camera imagery. v2b is sonar-only.
- Forward-looking sonar (UATD), a different sensor.
- Sonars it was not trained on without re-calibration. The val-tuned threshold does not transfer
  across sonars (recall 0.11 on `test_xsonar`).
- Any autonomous dispatch. A person approves every action ([`responsible_use.md`](responsible_use.md)).

## Metrics — read per class, not the aggregate

Deploy-faithful (`cv2.dnn`, the shipped ONNX), thresholds tuned on **val**, each test split scored once
([`eval_exp003_test.md`](eval_exp003_test.md), [`…_official398`](eval_exp003_test_official398.md),
[`…_xsonar`](eval_exp003_test_xsonar.md)):

| split | class | n_gt | AP@0.5 (95% CI) | AP@0.5:0.95 | P / R / F1 at val threshold |
|---|---|--:|--:|--:|--:|
| v2b test (214 unique crab-pot frames + 71 wreck) | **ghost_gear** | 285 | **0.482** (0.43–0.55) | 0.165 | 0.50 / 0.47 / 0.49 |
| v2b test | wreck_debris | 70 | 0.156 (0.10–0.25) | 0.047 | 0.35 / 0.11 / 0.17 |
| official 398 (GhostVision split) | ghost_gear | 567 | 0.391 | 0.135 | F1 **0.41** (GhostVision: 0.71–0.73) |
| cross-sonar `test_xsonar` | ghost_gear | 1265 | 0.341 (0.31–0.38) | 0.115 | 0.81 / 0.11 / 0.19 |

mAP@0.5 on v2b test: **0.319** (target 0.70, **not met**). The gap to GhostVision is real and reported as
is; the remaining problems are in the data (off-domain wreck screenshots, loose wreck boxes), not the
training recipe ([`exp003_diagnosis.md`](exp003_diagnosis.md) §3).

**Agent tiers (`docs/calibration_exp003.md`):** fit on the held-out validation recordings (163 unique
frames, 186 pots), verified once on unseen test (199 unique frames, 264 pots). Promise: **≥ 79% of pots
reach a human (95% confidence) — held on test at 81.1% (lower bound 76.7%)**. The requested 90% is not
achievable (recall ceiling 0.85), and no ≥ 85% precision promise is supportable (best 0.565), so nothing
is auto-confirmed: every find goes to a person, in P(pot) order. Re-look policies did not beat plain
detector confidence on the calibration recordings, so the agent does not spend them. The acoustic
shadow is evidence and relative height, never a gate.

Compared with EXP-001 on each model's own unseen data: recall promise ≥ 65% → **≥ 79%**, review cards
per frame 3.2 → 2.7 (test), ranking AUC 0.74 → 0.80 (val). There is no same-frames head-to-head: EXP-001
is not leakage-free on v2b test.

## Latency (the deployed CPU path)

`python -m src.bench.product_bench` on the full product path (decode → Stage 1 → `cv2.dnn` → evidence →
render), laptop x86 (stock OpenCV 5.0.0, 8 threads): **252 ms/frame p50, 3.3 fps**; the network is ~90%
of it (`cv2.dnn` forward alone 167 ms p50). The OpenCV-only stages run at 40 fps. Results:
[`benchmarks/`](../benchmarks), report [`cool_benchmark.md`](cool_benchmark.md). Graviton + COOL numbers
come from `infra/bench_cool.sh` on EC2 (**not yet run**).

## Limitations & ethics

- Small, low-relief pots in speckle are the weak point; wreck debris is weak (training wrecks are mostly
  off-domain fish-finder screenshots).
- Trained on one bay / one sonar family. Generalisation is unproven, and the cross-sonar test shows it.
- False negatives on ghost gear have real ecological cost, so `ghost_gear` runs hot (floor 0.05,
  recall-first) and uncertainty is routed to people rather than dropped.
- License: **AGPL-3.0** (Ultralytics YOLO). Weights: GitHub Release `exp003-v1` (`best.onnx`, 37,929,805
  bytes, SHA-256 `71339e99ad2b4b29f8d516967ce02e1fb5895434bf6f483cb517efe6750ba2f2`).
  `python -m src.detection.fetch_model` downloads it and refuses any file whose hash differs.
