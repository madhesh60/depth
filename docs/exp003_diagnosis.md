# EXP-003 — the fixed recipe: what it achieved, what is left, and why EXP-004 is about data

**Verdict.** The EXP-002 fixes worked.

- **Ghost gear:** EXP-003 (fixed tiles) fits its training frames and transfers to new recordings. It
  **raises the product's recall promise from ≥ 65% to ≥ 79%**, held on test at 81%, with fewer review
  cards per frame than EXP-001.
- **Wreck debris fails** (validation recall ceiling 0.10).
- **Head-to-head:** on the official GhostVision split, its F1 is 0.41 against their published
  0.71–0.73.

The remaining problems are in the **data**, not the training. EXP-004 targets them
([`kaggle_training.md`](kaggle_training.md)). Tables:
[`exp003_diagnosis_tables.md`](exp003_diagnosis_tables.md) · onboarding:
[`onboard_exp003.md`](onboard_exp003.md) · evaluations: [test](eval_exp003_test.md),
[official 398](eval_exp003_test_official398.md), [cross-sonar](eval_exp003_test_xsonar.md).

## 1. The two arms (Kaggle T4 ×2, explicit SGD 0.01, 640 px)

| | EXP-002 (failed) | **EXP-003** (fixed tiles) | EXP-003f (full frames) |
|---|--:|--:|--:|
| optimizer actually built | AdamW 0.00167 | **SGD 0.01** | **SGD 0.01** |
| epochs (early stop) / minutes | 30 / 93 | 104 of 150 / 135 | 273 of 300 / 141 |
| final train class loss | 1.72 | 0.96 | 0.70 |
| fit: ghost AP@0.3 on its own training frames | 0.57 | **0.76** | 0.92 |
| val ghost AP@0.5 / AP@0.3 (deploy path) | 0.26 / 0.38 | **0.39 / 0.51** | 0.36 / 0.49 |
| val ghost recall ceiling @0.05 | 0.83 | 0.77 | 0.69 |
| val wreck recall ceiling | 0.05 | 0.10 | 0.05 |
| `diagnose` verdict | underfit | **fits and transfers** | generalisation gap (overfits) |

**Tiles help.** At the same optimizer and similar step counts, the tiled arm generalises better and
the full-frame arm overfits. EXP-003 is the validation winner, so test was scored once, for EXP-003
only.

## 2. Test, scored once (EXP-003)

| | value | note |
|---|--:|---|
| recall promise (fit on val: 186 pots, 163 frames) | **≥ 79%** | EXP-001: ≥ 65% (134 pots, 66 frames) |
| recall on test at the review threshold | **81.1%** (95% lower bound 76.7%), **held** | 264 pots, 199 frames |
| recall ceiling at the detector floor (val / test) | 0.85 / 0.84 | EXP-001: 0.72 on its calibration recording |
| review cards per frame (val / test) | 3.5 / 2.7 | EXP-001: 4.2 / 3.2 |
| ranking AUC, real vs false (val) | 0.80 | EXP-001: 0.74 |
| ghost AP@0.5, v2b unique-frame test | 0.48 (95% CI 0.43–0.55) | F1 0.49 at the val-tuned threshold |
| wreck AP@0.5, v2b test | 0.16 | 70 boxes, 8 wrecks |
| **official 398-frame split (GhostVision)** | **F1 0.41**, AP 0.39 | GhostVision: F1 0.71–0.73. A real gap, reported as is |
| cross-sonar `test_xsonar` | AP 0.34; recall 0.11 at the val threshold (precision 0.81) | the threshold does not transfer across sonars |
| speed (`cv2.dnn`, laptop) | 167 ms/frame | EXP-001: 168 ms |

EXP-001's guarantee numbers come from its own unseen v1 frames, and EXP-003's from v2b. On the v2b
test itself EXP-001 is not leakage-free, because v1 contains those recordings. So each model's
promise is honest on its own unseen data, but there is no same-frames head-to-head between them.

## 3. What is left: the data

1. **The wreck class mostly learned from off-domain images.** Of the 459 training frames with wreck
   labels, **389 are "seabed" frames**: colour screenshots of consumer fish-finder displays, often
   down- or forward-imaging, with on-screen interface. Only 70 are side-scan shipwreck frames, from
   16 wrecks. Validation (4 other wrecks) and test (8) are side-scan. The model scores 0.96–1.0 on
   the wrecks it trained on and misses the validation ones.
2. **The side-scan shipwreck labels are loose.**
   - 45–57% of the boxes run off the frame edge (Roboflow crop copies);
   - many cover more than a quarter of the frame;
   - the texture inside a box is about the same as the rest of the frame (ratio ≈ 1.0).

   The wreck metric on 20 validation boxes is close to noise.
3. **Rotated copies.** 139 training frames are Roboflow rotations with black corners. They include
   all the TI/MC crab-pot singles; none of the grayscale target recordings (Rec\*) are rotated.
4. **Ghost gear still has a train/val gap** (fit 0.76 vs val 0.51 at IoU 0.3). The model fits better
   than it transfers to new recordings.

## 4. EXP-004 — clean the data, same recipe

Both arms keep EXP-003's recipe and leave out the "seabed" screenshots (412 frames, including their
natural-formation backgrounds) and the rotated copies (88 more). This is applied when the training
set is built on Kaggle (`build_tiles.py --drop-sources seabed --drop-rotated`).

- **EXP-004:** 2 classes.
- **EXP-004g:** ghost gear only (`--keep-classes 0`). Does the noisy wreck class hurt pot detection?

Pass bar, as before: fit AP@0.3 ≥ 0.65; validation ghost AP clearly above EXP-003's 0.39; and, for
the 2-class arm, a wreck recall ceiling > 0.5.

**Deployment:** EXP-001 stays the default for now. EXP-003 is registered (`models/EXP-003/`) and can
be switched on with `DEPTH_MODEL=EXP-003`. The default changes after EXP-004, to whichever model then
gives the best guarantee without losing what the product needs.
