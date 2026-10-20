# EXP-002 post-mortem — why the v2b models failed, and what EXP-003 changes

**Verdict.** EXP-002 (1024 px) and EXP-002s (640 px) are **not deployable**. Their best ghost-gear
AP@0.5 on the held-out validation recordings is **0.25** for both. They were rejected on
validation alone; the test split was **not** scored. There are two causes:

1. **Underfit, caused by a bug in our training kit.** The kit passed `lr0 0.01` but left
   `optimizer=auto`. In ultralytics 8.4 that silently builds **AdamW at lr 0.002·5/(4+nc) = 0.00167**
   for any run of 10,000 optimizer steps or fewer, and ignores `lr0`. EXP-002 had 2,700 steps.
2. **The tiles taught "object = background".** When a tile cut a box below 60% visibility, the box
   lost its label but kept its pixels. This happened in 49% of wreck appearances in tiles and 8% of
   ghost-gear appearances.

Both are fixed in the kit (`src/detection/train.py`, `DATASET/scripts/build_tiles.py`), and the
EXP-003 notebook is ready (`docs/kaggle_training.md`). Every number below is reproducible:

```bash
python -m src.detection.diagnose --zip EXP-002_complete.zip --zip EXP-002s_complete.zip --train-sample 200
```

The full tables are in [`exp002_diagnosis_tables.md`](exp002_diagnosis_tables.md). They come from the
deploy path (ONNX through `cv2.dnn`), on v2b validation plus 200 random training frames (seed 0);
test is not touched.

## 1. What the runs recorded

| | EXP-002 | EXP-002s |
|---|--:|--:|
| input | 1024 px | 640 px |
| epochs run / minutes (Kaggle T4) | 30 / 93 | 25 (early stop) / 39 |
| exported checkpoint (best ghost AP on val) | epoch 23: **ghost AP50 0.250** | epoch 21: **ghost AP50 0.253** |
| ultralytics' own `best.pt` (fitness) | ghost AP50 0.231 | ghost AP50 0.218 |
| final train class loss | 1.72 | 1.57 |
| val class loss over training | 6–29 | 8–50 |

The training pipeline worked as designed:

- every epoch was validated;
- the ghost-AP selection beat ultralytics' fitness pick in both runs;
- the ONNX export passed the `cv2.dnn` check;
- the packages were complete.

The *model* is what failed.

## 2. Validation, per source (held-out recordings Rec10/12/16)

Ghost-gear AP@0.5 / recall ceiling at conf 0.05:

| source | boxes | EXP-001 (leaky, see below) | EXP-002 | EXP-002s |
|---|--:|--:|--:|--:|
| Rec10 | 15 | 0.63 / 0.93 | 0.37 / 0.93 | 0.57 / 0.80 |
| Rec12 | 98 | 0.80 / 0.94 | 0.34 / 0.80 | 0.36 / 0.76 |
| Rec16 | 73 | 0.48 / 0.88 | 0.23 / 0.86 | 0.24 / 0.88 |
| **all** | 186 | 0.66 / 0.91 | **0.26 / 0.83** | **0.29 / 0.81** |
| wreck_debris (shipwreck frames) | 20 | 0.89 / 0.95 | **0.01 / 0.05** | **0.01 / 0.10** |

- **The pots are proposed but not believed.** The recall ceiling of 0.83 means most pots get a box at
  some confidence. But only 56 ghost-gear detections in all of validation reach conf 0.25 (EXP-002:
  17 correct, 39 false), so the pots are
  ranked below clutter. That is the signature of an under-trained classifier.
- **Wreck debris collapsed:** it is almost never proposed.
- **EXP-001 is not a fair target here.** It trained on v1, which contains these recordings, so its
  validation numbers are an optimistic upper bound.

## 3. The fit check — the decisive measurement

Ghost-gear AP@0.5 on **200 random frames from each model's own training set**:

| | EXP-002 | EXP-002s |
|---|--:|--:|
| own training frames | **0.42** | **0.41** |
| held-out validation | 0.26 | 0.29 |

A model that scores only 0.42 on frames it trained on is **underfit**: no amount of regularisation or
new data would fix it. The loss curves agree:

- the training losses were still falling at the last epoch;
- EXP-001 reached EXP-002's *final* class loss (1.72) within about 2 epochs.

## 4. Cause 1 — the optimizer was not the one we asked for

| | EXP-001 (worked) | EXP-002 |
|---|---|---|
| requested | `optimizer=auto`, `lr0 0.01` | `optimizer=auto`, `lr0 0.01` |
| optimizer steps (nominal batch 64) | 26,533/64 × 40 ≈ **16,600** | 5,746/64 × 30 ≈ **2,700** |
| what ultralytics built | SGD, lr 0.01 | **AdamW, lr 0.00167** (`lr0` ignored) |
| final train class loss | 0.51 | 1.72 |

The rule is in `ultralytics/engine/trainer.py`, `build_optimizer`. With `name == "auto"`:

- `lr_fit = round(0.002 * 5 / (4 + nc), 6)`;
- AdamW at `lr_fit` if `iterations ≤ 10000`;
- otherwise SGD-family at 0.01.

The dedupe to one copy per frame made v2b about 15× smaller than v1. That flipped the run below the
threshold.

**Fix:**

- `train.py` now passes an explicit `--optimizer` (default **SGD**, so `--lr0` is honoured).
- A callback records the optimizer ultralytics really built (`optimizer_built` in
  `model_meta.json`) and warns on a mismatch.
- Epochs rise so both EXP-003 arms get about 8–11 k SGD steps.
- The CPU smoke test confirms `{"type": "SGD", "lr": 0.01}`.

## 5. Cause 2 — tiles left objects unlabelled

`build_tiles.py` cut each training frame into 2×2 overlapping tiles. A box cut below `min_vis` 0.6
was dropped from the tile's labels, **but its pixels stayed**. Counted over v2b train:

| | appearances in a tile | kept (labelled) | dropped but ≥ 20% visible |
|---|--:|--:|--:|
| wreck_debris (median box 12.5% of the frame) | 1,670 | 702 | **819 (49%)** |
| ghost_gear (median box 0.3%) | 2,426 | 2,165 | 191 (8%) |

About 80% of the training images were tiles, so the model was repeatedly told that half a wreck is
seabed. That fits the wreck collapse, and it penalised partial pots too.

**Fix:**

- A frame is tiled only if every box is small (longest side ≤ 25% of the frame). The 388 wreck and
  seabed frames stay full-frame only; tiles cannot help a large object anyway.
- Any box a tile still cuts below 60% visibility has its visible part **inpainted**
  (`cv2.inpaint`, Telea) from the surrounding seabed: 307 boxes over the training set.
- New tiled set: 1,773 full frames + 3,066 tiles.
- A visual check is in `runs/_diag/inpaint_check.jpg`, which is not tracked. The pot head is
  removed; its shadow streak remains, and a shadow with no pot is not a pot.

## 6. What did not cause it (checked)

- **Export / deploy path:** ultralytics' own validation (0.25) and our `cv2.dnn` evaluation (0.26)
  agree.
- **Loose boxes:** at the product's loose match (IoU 0.3), validation ghost AP rises only to 0.38
  (EXP-002) and 0.40 (EXP-002s); EXP-001 reaches 0.72. Box tightness is not why they failed.
- **Validation labels:** EXP-001 scores 0.66 on them, so they are usable. The val set is the same
  for every model.
- **Resolution:** 1024 px did not beat 640 px (0.25 vs 0.25). Because both runs were underfit,
  STUDY-13's question is still open. EXP-003 runs at 640, which is 2.6× cheaper at inference, and
  asks the cleaner question first: do tiles help at all?

## 7. EXP-003 — the corrected run (`docs/kaggle_training.md`)

| | GPU 0: **EXP-003** | GPU 1: **EXP-003f** |
|---|---|---|
| training set | full frames + fixed tiles (4,839 images) | full frames only (1,773) |
| input / batch | 640 / 16 | 640 / 16 |
| optimizer | **SGD lr 0.01 (explicit)** | **SGD lr 0.01 (explicit)** |
| epochs (patience = ¼) | 150 (~11.4 k steps) | 300 (~8.4 k steps) |
| selection | best ghost-gear AP@0.5 on val (tracked every epoch) | same |

**Pass bar before onboarding** (`python -m src.detection.diagnose`):

- ghost **AP@0.3 ≥ 0.65 on its own training frames** (not underfit). The bar was set before any
  EXP-003 result, from the reference: EXP-001, a working model, scores 0.71 there (0.57 at IoU 0.5,
  because the 14–36 px label boxes disagree by a few pixels); underfit EXP-002 scores 0.57;
- validation ghost AP clearly above 0.29;
- wreck recall ceiling above 0.5.

Only then is test scored, once, by `onboard_model`.
