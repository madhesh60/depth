# Diagnosis: EXP-003, EXP-004, EXP-004g

_`python -m src.detection.diagnose` · 2026-10-03 06:45 · deploy path (ONNX via cv2.dnn) · v2b val (234 frames) + 200 random training frames (seed 0) · test is not touched_

## Verdict

- **EXP-003**: fits (train AP@0.3 0.76) and transfers (val AP@0.3 0.51, AP@0.5 0.39): ready for onboarding.
- **EXP-004**: fits (train AP@0.3 0.74) and transfers (val AP@0.3 0.49, AP@0.5 0.36): ready for onboarding.
- **EXP-004g**: **GENERALISATION GAP**: ghost AP@0.3 train 0.86 vs val 0.45. It fits but does not transfer to new recordings: more varied data / augmentation, or less capacity.

## The run's own record

| model | optimizer built | epochs | minutes | final train cls loss | final val cls loss | exported |
|---|---|--:|--:|--:|--:|---|
| EXP-001 | trained on v1, which contains the v2b val recordings: optimistic upper bound |  |  |  |  |  |
| EXP-003 | {"type": "SGD", "lr": 0.01, "nominal_batch": 64, "accumulate": 4} | 104 | 134.9 | 0.95952 | 8.75373 | best_ghost.pt |
| EXP-004 | {"type": "SGD", "lr": 0.01, "nominal_batch": 64, "accumulate": 4} | 112 | 119.3 | 0.99978 | 11.4661 | best_ghost.pt |
| EXP-004g | {"type": "SGD", "lr": 0.01, "nominal_batch": 64, "accumulate": 4} | 103 | 113.9 | 1.06783 | 9.73205 | best_ghost.pt |

### Validation — ghost gear (held-out recordings Rec10/12/16)

AP@0.5 / AP@0.3 (the product's loose match) · recall ceiling @0.05 · TP/FP @0.25

| source | images | ghost_gear boxes | EXP-001 | EXP-003 | EXP-004 | EXP-004g |
|---|--:|--:|---|---|---|---|
| **all** | 234 | 186 | 0.659 / 0.72 · 0.91 · 132/105 | 0.394 / 0.51 · 0.77 · 73/80 | 0.365 / 0.49 · 0.74 · 86/106 | 0.326 / 0.45 · 0.64 · 88/169 |
| crabpot_Rec10 | 25 | 15 | 0.630 / 0.69 · 0.93 · 12/17 | 0.644 / 0.68 · 0.87 · 10/7 | 0.551 / 0.61 · 0.80 · 10/9 | 0.442 / 0.49 · 0.80 · 8/16 |
| crabpot_Rec12 | 91 | 98 | 0.795 / 0.84 · 0.94 · 79/42 | 0.434 / 0.54 · 0.77 · 39/35 | 0.439 / 0.58 · 0.70 · 47/37 | 0.371 / 0.55 · 0.56 · 43/53 |
| crabpot_Rec16 | 47 | 73 | 0.476 / 0.56 · 0.88 · 41/46 | 0.338 / 0.48 · 0.77 · 24/36 | 0.329 / 0.46 · 0.77 · 29/44 | 0.347 / 0.50 · 0.71 · 37/61 |
| seabed_natform | 51 | 0 | — · — · 0/0 | — · — · 0/2 | — · — · 0/16 | — · — · 0/39 |

### Validation — wreck debris

AP@0.5 / AP@0.3 (the product's loose match) · recall ceiling @0.05 · TP/FP @0.25

| source | images | wreck_debris boxes | EXP-001 | EXP-003 | EXP-004 | EXP-004g |
|---|--:|--:|---|---|---|---|
| **all** | 234 | 20 | 0.886 / 0.91 · 0.95 · 15/6 | 0.020 / 0.05 · 0.10 · 2/54 | 0.035 / 0.13 · 0.30 · 0/8 | 0.000 / nan · 0.00 · 0/0 |
| crabpot_Rec12 | 91 | 0 | — · — · 0/0 | — · — · 0/3 | — · — · 0/0 | — · — · 0/0 |
| seabed_natform | 51 | 0 | — · — · 0/1 | — · — · 0/36 | — · — · 0/4 | — · — · 0/0 |
| shipwreck | 20 | 20 | 0.901 / 0.93 · 0.95 · 15/5 | 0.057 / 0.14 · 0.10 · 2/15 | 0.067 / 0.20 · 0.30 · 0/4 | 0.000 / nan · 0.00 · 0/0 |

### Fit check — ghost gear on the model's own training frames

AP@0.5 / AP@0.3 (the product's loose match) · recall ceiling @0.05 · TP/FP @0.25

| source | images | ghost_gear boxes | EXP-001 | EXP-003 | EXP-004 | EXP-004g |
|---|--:|--:|---|---|---|---|
| **all** | 200 | 160 | 0.574 / 0.71 · 0.81 · 102/68 | 0.652 / 0.76 · 0.89 · 89/44 | 0.630 / 0.74 · 0.83 · 92/51 | 0.839 / 0.86 · 0.95 · 131/50 |

### Fit check — wreck debris on the model's own training frames

AP@0.5 / AP@0.3 (the product's loose match) · recall ceiling @0.05 · TP/FP @0.25

| source | images | wreck_debris boxes | EXP-001 | EXP-003 | EXP-004 | EXP-004g |
|---|--:|--:|---|---|---|---|
| **all** | 200 | 60 | 0.958 / 0.98 · 0.95 · 56/4 | 0.963 / 0.98 · 0.98 · 56/7 | 0.084 / 0.19 · 0.15 · 4/8 | 0.000 / nan · 0.00 · 0/0 |

_EXP-001 trained on v1, which contains the v2b val recordings: optimistic upper bound._
