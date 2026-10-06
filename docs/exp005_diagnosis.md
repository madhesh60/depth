# EXP-005 — stronger brightness and scale jitter made the detector worse

**Verdict.** Both EXP-005 runs are **worse than EXP-003** for ghost gear, so neither is onboarded and
the test split was not touched. A paired bootstrap over the same validation frames gives:

| run | seed | ΔAP@0.5 vs EXP-003 (95% CI) | chance it is better |
|---|---|---|---|
| EXP-005 | 42 | **−0.10** (−0.17 to −0.03) | 0% |
| EXP-005a | 7 | **−0.07** (−0.13 to −0.005) | 2% |

The two seeds agree with each other (Δ +0.035, CI −0.02 to +0.09), so this is the augmentation, not
seed luck.

The hypothesis behind the run was that stronger brightness and scale jitter would close the gap
between training frames and new recordings. **It is refuted.** The gap got wider, and confident false
alarms nearly doubled.

**EXP-003 stays the best clean model.** It is onboarded, with a recall promise of ≥ 79% held on test.

This is the sixth clean run since EXP-003 (EXP-003f, 004, 004g, 005, 005a), and none has beaten it.
Every training-recipe lever has been measured on validation:

- optimizer;
- tiles vs full frames;
- data cleanup;
- class set;
- augmentation strength;
- duplicate-box suppression rules;
- pseudo-labels.

What is left is **data**: the validation labels, more recordings, and the wreck set.

Tables: [`exp005_diagnosis_tables.md`](exp005_diagnosis_tables.md). Kit:
[`kaggle_training.md`](kaggle_training.md).

## 1. The runs (Kaggle T4 ×2, SGD 0.01, 640 px, fixed tiles, 150 epochs with early stopping)

| | EXP-003 (reference) | EXP-004 | **EXP-005** | **EXP-005a** |
|---|--:|--:|--:|--:|
| training data | v2b + tiles | cleaned¹ | cleaned¹ | cleaned¹ |
| brightness / scale jitter (`hsv_v` / `scale`) | 0.2 / 0.5 | 0.2 / 0.5 | **0.4 / 0.6** | **0.4 / 0.6** |
| seed | 42 | 42 | 42 | 7 |
| epochs (early stop) / minutes | 104 / 135 | 112 / 119 | 103 / 101 | 139 / 128 |
| fit: ghost AP@0.3 on its own training frames² | 0.76 | 0.74 | 0.79 | 0.68 |
| **val ghost AP@0.5** (deploy path, 95% CI) | **0.394** (0.33–0.48) | 0.365 (0.30–0.46) | 0.292 (0.24–0.37) | 0.327 (0.27–0.41) |
| val ghost AP@0.3 (the product's loose match) | **0.51** | 0.49 | 0.44 | 0.44 |
| the same, crab-pot recordings only (163 frames) | **0.406** / 0.52 | 0.395 / 0.53 | 0.321 / 0.48 | 0.363 / 0.49 |
| val ghost recall ceiling (conf 0.05, IoU 0.5) | **0.77** | 0.74 | 0.70 | 0.75 |
| val ghost TP / FP at conf 0.25 | **73 / 80** | 86 / 106 | 82 / **169** | 75 / **131** |
| val wreck recall ceiling (bar 0.5) | 0.10 | 0.30 | 0.30 | 0.00 |
| training-time val ghost AP@0.5, mean of the best 5 epochs³ | **0.344** | 0.324 | 0.270 | 0.279 |

¹ v2b without the 412 colour fish-finder screenshots (`--drop-sources seabed`) and the rotated
black-bordered copies (`--drop-rotated`).
² The fit sample is 200 random v2b training frames. Its wreck boxes are mostly the fish-finder
screenshots the cleaned runs never saw, so the wreck fit row in the tables is low for every cleaned
run (EXP-004: 0.08). That is expected, not a training failure.
³ Ultralytics' own validation during training, a different matcher from the deploy path; it is used
only to compare the curves.

Both runs trained properly:

- SGD at 0.01 was built (`model_meta.optimizer_built`);
- the new jitter values reached ultralytics;
- training loss fell smoothly;
- `best_ghost.pt` was selected on validation ghost AP (0.297 and 0.300; EXP-003's was 0.360);
- the ONNX export passed the `cv2.dnn` check.

## 2. The curves: lower from start to finish

Validation ghost AP@0.5 during training (ultralytics), as a 5-epoch running mean:

| epoch | 20 | 40 | 60 | 80 | 100 |
|---|--:|--:|--:|--:|--:|
| EXP-003 | 0.163 | 0.240 | 0.307 | 0.320 | 0.316 |
| EXP-004 | 0.148 | 0.169 | 0.213 | 0.288 | 0.287 |
| EXP-005 | 0.168 | 0.195 | 0.233 | 0.236 | 0.220 |
| EXP-005a | 0.112 | 0.207 | 0.248 | 0.267 | 0.250 |

The stronger jitter made the training task harder: the training box/cls loss at epoch 100 was
1.46 / 1.19, against EXP-003's 1.23 / 0.99. Validation never caught up, and both seeds plateau about
0.07 below EXP-003 from epoch 60 on. More epochs would not have closed it: EXP-005a ran 139 epochs
and finished at 0.246.

## 3. What went wrong

**1. The gap to new recordings widened, so it is not a brightness/scale problem.**
The gap between training-frame fit and validation (ghost AP@0.3) was:

| | gap |
|---|--:|
| EXP-003 | 0.25 |
| EXP-005 | 0.35 |
| EXP-005a | 0.24 |

EXP-005a kept the gap only by fitting less (0.68), not by transferring better. What separates the
training recordings from the held-out ones is not gain or range scale, the two things this jitter
varies.

**2. The loss is in ranking, as before.**
The recall ceiling barely moved (0.77 → 0.70 / 0.75). The models still *find* the pots, but confident
false alarms rose:

| false alarms at conf 0.25 | EXP-003 | EXP-004 | EXP-005 | EXP-005a |
|---|--:|--:|--:|--:|
| crab-pot recordings (Rec10 / 12 / 16) | 7 / 35 / 36 | 9 / 37 / 44 | 14 / 59 / 72 | 10 / 36 / 57 |
| natural-formation frames (no pots) | 2 | 16 | 24 | 28 |

More of them now rank above real pots: 83 and 75 false alarms sit above the median true-positive
confidence, against 54 for EXP-003. This is the failure mode EXP-004's analysis found, and the jitter
made it worse.

A *likely* mechanism, not measured:

- `scale 0.6` shrinks a 14–36 px pot to as little as 6–14 px in training, where a pot and a bright
  speckle blob look alike;
- `hsv_v 0.4` (±40% brightness) weakens the strong-return cue that separates a pot from the seabed
  around it.

**3. The cleanup removed the rock negatives.**
On the 51 validation natural-formation frames, false pots went 2 (EXP-003) → 16 (EXP-004) → 24–28
(EXP-005/005a). These frames come from the same source as the 412 dropped training screenshots, which
also showed the model what rock looks like.

On crab-pot recordings alone, EXP-004 matches EXP-003 (0.395 vs 0.406). So EXP-004's small overall loss
came from those frames. EXP-005's loss holds on the crab-pot recordings too (0.321 / 0.363). EXP-005 and EXP-004 share the
same data and the same seed (42), so the jitter is the cause there.

**4. Wreck debris still fails.**
The recall ceiling is 0.30 / 0.00 against the 0.5 bar, on only 20 validation boxes. Augmentation does
not fix the wreck class: it has 70 side-scan training frames with loose, edge-clipped boxes.

**5. Seed noise is about half the validation uncertainty.**
The two seeds differ by 0.035 AP. EXP-003 is a single seed (42), so part of its margin may be luck. Both
EXP-005 seeds are still below it, at P(gain) ≤ 0.02.

## 4. What to improve, in order

1. **Ship EXP-003 as the default model.** It is the best clean model, with a recall promise of
   ≥ 79% held on test at 81% (lower bound 76.7%) and fewer review cards than EXP-001. EXP-001's
   numbers are inflated because it trained on the validation recordings. Switch the default
   (`DEPTH_MODEL`) and publish the EXP-003 weights release.
2. **Audit the validation labels** (blinded, Audit tab, 1–2 people, about 15 minutes each).
   - The 19 confident unlabelled pots and the box-extent disagreements depress *every* model's
     score.
   - Correcting them (v2c, validation only, never test) gives an honest, sharper ruler, then all
     models are re-scored on the same labels.
   - This is the cheapest lift in the measured numbers.
3. **Stop the recipe search.** Six clean runs have measured every recipe lever above.
   - Another augmentation or cleanup run is unlikely to clear ±0.07 validation noise before the
     21 Oct freeze.
   - If GPU time is free, the only run worth making is **EXP-003's exact recipe with a second seed**.
     It reports EXP-003's own run-to-run spread in the technical report, an honesty number rather
     than a gain.
4. **Gains that need new data** (after the competition, unless data appears):
   - more held-out recordings (validation has 3, which is why the CI is ±0.07);
   - tighter side-scan shipwreck boxes and more of them for the wreck class;
   - the fish-finder screenshots kept as *rock negatives* only (no wreck boxes), which restores what
     the cleanup took away.

Do not use:

- stronger brightness/scale jitter (this record);
- a ghost-only model (EXP-004g);
- duplicate-box suppression or inner-box rules (EXP-004 analysis);
- pseudo-labels from training frames (only 8 found against 1,345 labels).
