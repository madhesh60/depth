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

**Agentic value (STUDY-04, `src/agentic/calibrate.py`):** raw hot-detector precision 0.60 → agent
**CONFIRMED** tier **0.737 @ 30% recall-share** (re-look persistence ⋃ high detector confidence);
REJECTED is recall-safe (91% of true pots retained, never deleted). The acoustic shadow is
non-discriminative here (CLEAR-rate 14.5% TP vs 14.6% FP) and is used as evidence + height, never a gate.

## Latency

~11.5 ms/frame on a T4 (GPU). On CPU (the live target) expect ~200 ms/frame — measure on the real
Graviton/x86 server (COOL benchmark), never quote the GPU number for the deployed CPU path.

## Limitations & ethics

- Small-object recall is the weak point (91% of misses are <10% of frame width).
- Trained on one bay / one sonar brand — generalisation unproven.
- False negatives on ghost gear have real ecological cost → we run `fishing_gear` hot (recall-first)
  and route uncertainty to humans rather than dropping it.
- License: **AGPL-3.0** (Ultralytics YOLO). Weights hosting + hash: see release notes.
