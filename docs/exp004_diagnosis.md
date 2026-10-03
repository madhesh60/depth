# EXP-004 — cleaner data did not beat EXP-003; what limits the detector now

**Verdict.** EXP-004 (cleaned data, 2 classes) is **not better than EXP-003** for ghost gear. A paired
bootstrap over the same validation frames gives ΔAP@0.5 = −0.03 (95% CI −0.11 to +0.05). EXP-004g
(ghost gear only) is **worse**: ΔAP −0.07, 3% chance it is better, and it overfits. The cleanup did
help wreck debris (recall ceiling 0.10 → 0.30; false wrecks on rocky seabed 36 → 4), but not enough
(bar: 0.5).

**EXP-003 stays the best clean model.** It is onboarded, with a recall promise of ≥ 79% held on test.

The deeper finding: the detector already **finds 85% of validation pots**, and its misses are not
concentrated by size or contrast. What holds the score down is **ranking against false alarms**:

- about half of the top false alarms sit on a labelled pot with a box that disagrees;
- several look exactly like pots and are probably unlabelled;
- with 186 pots in 3 recordings, validation cannot resolve changes smaller than about ±0.07.

Tables: [`exp004_diagnosis_tables.md`](exp004_diagnosis_tables.md). Kit and procedure:
[`kaggle_training.md`](kaggle_training.md).

## 1. The runs (Kaggle T4 ×2, SGD 0.01, 640 px, fixed tiles; train without the 412 fish-finder screenshots and 88 rotated copies)

| | EXP-003 (reference) | EXP-004 (cleaned, 2 classes) | EXP-004g (cleaned, ghost only) |
|---|--:|--:|--:|
| epochs (early stop) / minutes | 104 / 135 | 112 / 119 | 103 / 114 |
| fit: ghost AP@0.3, own training frames¹ | 0.76 | 0.74 | 0.86 |
| **val ghost AP@0.5 / AP@0.3** (deploy path) | **0.394 / 0.51** | 0.365 / 0.49 | 0.326 / 0.45 |
| val ghost recall ceiling (conf 0.05, IoU 0.5) | **0.77** | 0.74 | 0.64 |
| val ghost TP / FP at conf 0.25 | 73 / 80 | 86 / 106 | 88 / 169 |
| val wreck recall ceiling | 0.10 | **0.30** | — |
| wreck false alarms on rocky seabed (val, conf 0.25) | 36 | **4** | — |
| `diagnose` verdict | fits and transfers | fits and transfers | generalisation gap |

¹ The fit sample is 200 random v2b training frames. For EXP-004 / 004g it includes frames they were not
trained on (the screenshots and rotated copies), so their wreck "fit" is not meaningful.

**Paired bootstrap** (2,000 resamples of the 234 validation frames; frame-level, so if anything optimistic):

| vs EXP-003 | ΔAP@0.5 [95% CI] | P(better) | ΔAP@0.3 [95% CI] |
|---|---|--:|---|
| EXP-004 | −0.029 [−0.105, +0.046] | 0.22 | −0.020 [−0.085, +0.042] |
| EXP-004g | −0.068 [−0.143, 0.000] | 0.03 | −0.054 [−0.135, +0.011] |
| EXP-003 itself | AP@0.5 0.394 [0.334, 0.481] | | AP@0.3 0.506 [0.442, 0.598] |

## 2. Did the model learn? Yes — what limits it is ranking, labels and validation size

**It sees the pots.** At the detector floor (conf ≥ 0.05, IoU ≥ 0.3), EXP-003 finds **158 of 186 (85%)**
validation pots, and EXP-004 finds 159. Misses are not concentrated anywhere:

| pot size (shortest side) | < 16 px | 16–24 | 24–32 | 32–48 | ≥ 48 |
|---|--:|--:|--:|--:|--:|
| EXP-003 found | 0.86 | 0.72 | 0.83 | 0.81 | 0.94 |
| EXP-004 found | 0.79 | 0.69 | 0.75 | 0.93 | 0.95 |

By local contrast, found rates are flat at 0.81–0.86. So **higher input resolution is not the lever**:
small pots are already found about as often as large ones.

**What holds the score down: false alarms rank above real pots.** EXP-003 makes 246 matches and
393 false alarms on validation:

- real pots get a median confidence of 0.21; 89 false alarms score above that;
- of the 40 most confident false alarms, **19 overlap a labelled pot** but disagree on the box (IoU
  0.05–0.3), often a box that also covers the shadow tail;
- **several look exactly like the matched pots** (a bright head with a shadow tail) and are probably
  unlabelled;
- only a handful are real clutter (bright rocks or blobs).

Box shapes are mostly head-only in every split (tall boxes are 4% in train, 10% in val, 5% in test).
So this is not a single convention bug: it is noisy localisation on 14–36 px objects, plus missing
labels.

**The wreck class matters for pots.** Without it (EXP-004g), the model fires on twice as many wrong
objects. The wreck class was absorbing structures that are otherwise mistaken for pots. Keep 2 classes.

## 3. What this means for the next training

- **Validation is noisy** (95% CI ±0.07 on AP), so small recipe changes cannot be told apart.
- **Larger capacity (YOLO11m) or 1024 px** would cost 2.5–3× the inference time, beyond the < 300 ms
  target, for a gain the data says is unlikely: misses are not small-object misses.
- **The measurable lever is the labels:**
  1. A blinded audit of EXP-003's confident false alarms (the Audit tab) separates unlabelled pots
     from clutter.
  2. Add the confirmed pots to train and val as **v2c**. Test is never edited.
  3. Retrain EXP-003's recipe on v2c (**EXP-005**), with two seeds to measure run-to-run noise.

  This is the only next run expected to move the number by more than the validation noise. More real
  side-scan crab-pot recordings (outreach to the dataset authors) would be the other lever.

**Deployment:** EXP-003 is the strongest clean model on the product's own metrics:

- recall promise ≥ 79% vs ≥ 65%, held on test;
- 2.7 vs 3.2 review cards per frame;
- ranking AUC 0.80 vs 0.74.

Its weak wreck class is the one reason to keep EXP-001 as the default.
