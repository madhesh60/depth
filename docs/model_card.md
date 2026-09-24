# Model card — DEPTH detector (EXP-001)

## Overview

- **Architecture:** YOLO11s (detect), 9.41 M params, exported to ONNX and run through OpenCV 5
  `cv2.dnn` at inference (torch-free deploy).
- **Training:** dataset v1 (4-class), 640 px, 40 epochs, Kaggle T4. Sonar-aware augmentation
  (no hue/sat shift, no rotation/vflip, along-track hflip only). Best checkpoint ≈ epoch 25 (val
  peaked ep16, then overfit — a documented finding, EXP-001).
- **Successor:** EXP-002 trains on the honest sonar-only v2 split (`imgsz 1024`, ~25–30 ep).

## Intended use

Decision-support triage of side-scan sonar surveys for derelict fishing gear and wreck debris —
**with a human in the loop**. It ranks and proves candidates for a human reviewer; it does not make
autonomous removal decisions.

## Out of scope

- Optical/camera imagery (off-domain: optical debris recall is ~0 — the model maps optical →
  `natural_formation`).
- Forward-looking sonar (UATD) — different sensor; held out.
- Any autonomous dispatch — a human must approve every action (see [`responsible_use.md`](responsible_use.md)).

## Metrics (test split — read per class, not just the aggregate)

| | P | R | AP@0.5 |
|---|---:|---:|---:|
| **aggregate** | 0.808 | 0.800 | **0.822** *(inflated — see note)* |
| fishing_gear (sonar, the target) | 0.54 | 0.37 → **0.69** @conf 0.10 | 0.45 |
| structural_fragment | 0.95 | 0.90 | 0.92 |
| pipe_cylinder (65 boxes — high variance) | 0.91 | 0.95 | 0.94 |
| natural_formation (optical control) | 0.84 | 0.99 | 0.97 |

> The aggregate is **inflated by domain segregation** — `natural_formation` is optical-only and
> trivially separable. The honest headline is `fishing_gear` on sonar.

**Agent tiers (STUDY-07, `docs/calibration_exp001.md`):** fit on the unseen calibration recording
(v1 val Rec19, 66 unique frames), verified once on unseen test (v1 test, 92 unique frames). Promise:
**≥ 65% of pots reach a human (95% confidence) — held on test (86.2%, lower bound 80.5%)**. The
requested 90% is not achievable (recall ceiling 0.72 on the calibration recording) and no ≥ 85%
precision promise is supportable, so nothing is auto-confirmed. Re-look variants did not beat plain
detector confidence on unseen data; the earlier "CONFIRMED 0.737" was tuned on test and is 0.58 on
unseen frames — retired. The acoustic shadow is evidence + relative height, never a gate.

## Latency (the deployed CPU path)

Measured with `python -m src.bench.product_bench` on the full product path (decode → Stage 1 →
`cv2.dnn` → evidence → render): **238 ms/frame p50** on a laptop x86 CPU (stock OpenCV 5.0.0, all
threads; 522 ms single-thread); the network is ~86% of it. Graviton + COOL numbers come from
`infra/bench_cool.sh` on EC2. (The ~11.5 ms figure from training was a T4 GPU and is not the
deployed path.)

## Limitations & ethics

- Small-object recall is the weak point (91% of misses are <10% of frame width).
- Trained on one bay / one sonar brand — generalisation unproven.
- False negatives on ghost gear have real ecological cost → we run `fishing_gear` hot (recall-first)
  and route uncertainty to humans rather than dropping it.
- License: **AGPL-3.0** (Ultralytics YOLO). Weights hosting + hash: see release notes.
