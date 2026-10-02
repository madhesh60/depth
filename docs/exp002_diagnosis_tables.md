# Diagnosis: EXP-002, EXP-002s

_`python -m src.detection.diagnose` · 2026-10-02 08:09 · deploy path (ONNX via cv2.dnn) · v2b val (234 frames) + 200 random training frames (seed 0) · test is not touched_

## Verdict

- **EXP-002**: **UNDERFIT**: ghost AP@0.3 0.57 on its own training frames (< 0.65; a working model scores 0.71). Train longer, or check the optimizer and the labels, before anything else.
- **EXP-002s**: **UNDERFIT**: ghost AP@0.3 0.54 on its own training frames (< 0.65; a working model scores 0.71). Train longer, or check the optimizer and the labels, before anything else.

## The run's own record

| model | optimizer built | epochs | minutes | final train cls loss | final val cls loss | exported |
|---|---|--:|--:|--:|--:|---|
| EXP-001 | trained on v1, which contains the v2b val recordings: optimistic upper bound |  |  |  |  |  |
| EXP-002 | not recorded (optimizer=auto) | 30 | 93.2 | 1.72175 | 6.98931 | epoch23.pt |
| EXP-002s | not recorded (optimizer=auto) | 25 | 38.8 | 1.56679 | 14.6333 | epoch21.pt |

### Validation — ghost gear (held-out recordings Rec10/12/16)

AP@0.5 / AP@0.3 (the product's loose match) · recall ceiling @0.05 · TP/FP @0.25

| source | images | ghost_gear boxes | EXP-001 | EXP-002 | EXP-002s |
|---|--:|--:|---|---|---|
| **all** | 234 | 186 | 0.659 / 0.72 · 0.91 · 132/105 | 0.264 / 0.38 · 0.83 · 17/39 | 0.294 / 0.40 · 0.81 · 9/20 |
| crabpot_Rec10 | 25 | 15 | 0.630 / 0.69 · 0.93 · 12/17 | 0.367 / 0.37 · 0.93 · 2/4 | 0.567 / 0.58 · 0.80 · 1/1 |
| crabpot_Rec12 | 91 | 98 | 0.795 / 0.84 · 0.94 · 79/42 | 0.338 / 0.44 · 0.80 · 10/11 | 0.361 / 0.44 · 0.76 · 6/4 |
| crabpot_Rec16 | 47 | 73 | 0.476 / 0.56 · 0.88 · 41/46 | 0.230 / 0.38 · 0.86 · 5/20 | 0.244 / 0.39 · 0.88 · 2/12 |
| seabed_natform | 51 | 0 | — · — · 0/0 | — · — · 0/4 | — · — · 0/3 |

### Validation — wreck debris

AP@0.5 / AP@0.3 (the product's loose match) · recall ceiling @0.05 · TP/FP @0.25

| source | images | wreck_debris boxes | EXP-001 | EXP-002 | EXP-002s |
|---|--:|--:|---|---|---|
| **all** | 234 | 20 | 0.886 / 0.91 · 0.95 · 15/6 | 0.009 / 0.04 · 0.05 · 0/17 | 0.013 / 0.04 · 0.10 · 1/39 |
| crabpot_Rec12 | 91 | 0 | — · — · 0/0 | — · — · 0/0 | — · — · 0/1 |
| seabed_natform | 51 | 0 | — · — · 0/1 | — · — · 0/9 | — · — · 0/28 |
| shipwreck | 20 | 20 | 0.901 / 0.93 · 0.95 · 15/5 | 0.022 / 0.09 · 0.05 · 0/8 | 0.028 / 0.14 · 0.10 · 1/10 |

### Fit check — ghost gear on the model's own training frames

AP@0.5 / AP@0.3 (the product's loose match) · recall ceiling @0.05 · TP/FP @0.25

| source | images | ghost_gear boxes | EXP-001 | EXP-002 | EXP-002s |
|---|--:|--:|---|---|---|
| **all** | 200 | 160 | 0.574 / 0.71 · 0.81 · 102/68 | 0.420 / 0.57 · 0.72 · 33/28 | 0.412 / 0.54 · 0.68 · 30/17 |

### Fit check — wreck debris on the model's own training frames

AP@0.5 / AP@0.3 (the product's loose match) · recall ceiling @0.05 · TP/FP @0.25

| source | images | wreck_debris boxes | EXP-001 | EXP-002 | EXP-002s |
|---|--:|--:|---|---|---|
| **all** | 200 | 60 | 0.958 / 0.98 · 0.95 · 56/4 | 0.682 / 0.78 · 0.78 · 28/3 | 0.900 / 0.91 · 0.95 · 48/8 |

_EXP-001 trained on v1, which contains the v2b val recordings: optimistic upper bound._
