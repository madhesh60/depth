# Training on Kaggle — the procedure (last run: EXP-005, negative)

This page is the procedure for every training run: train on Kaggle, check with `diagnose`, then plug
in with ONE command.

History:

- EXP-002 was underfit ([`exp002_diagnosis.md`](exp002_diagnosis.md)).
- EXP-003 fixed that and raised the recall promise from 65% to 79%
  ([`exp003_diagnosis.md`](exp003_diagnosis.md)).
- EXP-004's data cleanup helped wrecks but not pots ([`exp004_diagnosis.md`](exp004_diagnosis.md)).
- EXP-005's stronger brightness/scale jitter made pots **worse** on both seeds (Δ −0.10 / −0.07). The
  recipe search is done; EXP-003 stays the best ([`exp005_diagnosis.md`](exp005_diagnosis.md)).

**What EXP-005 builds on** (all measured on the held-out validation recordings):

| finding | run | EXP-005 |
|---|---|---|
| explicit SGD 0.01 trains properly | EXP-002 → 003 | kept |
| fixed tiles beat full frames | EXP-003 vs 003f | kept |
| cleaned data: wrecks better, pots unchanged | EXP-004 | kept |
| two classes beat ghost-only | EXP-004g | kept |
| misses are not small-object misses (85% found at the floor) | analysis | no 1024 px / bigger model |
| duplicate boxes on one pot: suppression rules make AP worse (−0.05 to −0.21) | analysis | none |
| pseudo-labels: only 8 confident unlabelled pots in train, vs 1,345 labels | analysis | not used |
| **fit on training frames ≫ fit on new recordings** (AP@0.3 0.74 vs 0.49; val peaks mid-run) | EXP-003/004 | stronger brightness + scale jitter → **worse** (EXP-005) |

| | EXP-005 | EXP-005a |
|---|---|---|
| recipe | EXP-003's (SGD 0.01, fixed tiles, 640 px, 150 ep, 2 classes) on EXP-004's cleaned data | same |
| augmentation | `--hsv-v 0.4 --scale 0.6` (was 0.2 / 0.5) | same |
| seed | 42 | 7 |

Validation AP has a 95% CI of ±0.07. Two seeds show whether the change is real, and the better run on
validation is kept.

**Tested before the run:**

- the exact EXP-005a command was dry-run on CPU (cleaned set, the new flags reached ultralytics);
- every notebook cell parses;
- kit tests (14) pass.

---

## 0. Before you start (once)

1. The **depth-v2b** Kaggle dataset from EXP-002 is reused as is; nothing to upload. If it is gone,
   rebuild `runs/kaggle_upload/depth-v2b.zip` (command at the end) and create it again as a
   **Private** dataset.
2. Your Kaggle account needs phone verification, otherwise GPU and Internet stay disabled.

## 1. The notebook (import it, no pasting)

1. Kaggle → **Code → New Notebook** → **File → Import Notebook** → upload
   **`notebooks/exp005_kaggle.ipynb`** from this repo.
2. Right panel:
   - **Add Input** → **depth-v2b**;
   - **Accelerator → GPU T4 ×2**;
   - **Internet → On**.
3. Either **Save Version → Save & Run All (Commit)**, which keeps running if you close the browser,
   or run cells 1–5 in order in the open session and keep the tab open.

What the five code cells do:

1. Clone the repo and install **ultralytics 8.4.157** (pinned). Assert a GPU.
2. Find the dataset automatically and check the split sizes (train 1,773 · val 234 · test 285 ·
   official 398 · cross-sonar 555).
3. Build the cleaned training set (about 2 minutes): `v2b_clean_p`, 1,273 frames + about 2,670 tiles.
4. Launch both trainings, each in its **own process group**, so Stop/Interrupt on a cell cannot kill
   them. It refuses to start a second copy. Logs go to `/kaggle/working/logs/<name>.log`.
5. **Status / wait / collect.** It is safe to run at any time and never starts or stops a training.
   Per run it shows:
   - the phase;
   - epoch / total, val mAP50 / recall, and the **train class loss**. It should fall well below 1.0;
     EXP-002 stalled at 1.7;
   - min/epoch and time left, GPU use, and the latest log line.
   
   It also warns about duplicates and stalls. At the end it copies both `*_complete.zip` files to
   **Output**.

**Expected time:** about 2–2.5 h per arm, running in parallel (EXP-003 took 2.2 h on a similar set);
Kaggle's limit is 12 h. Disk use is about 2 GB.

**While it runs, check:**

- the first epoch finishes within ~5 minutes;
- the train class loss falls steadily, below ~1.2 by about epoch 30;
- val mAP50 passes ~0.15 within the first ~40 epochs (EXP-003 reached 0.15 by epoch 41).

**If something goes wrong:**

| symptom | fix |
|---|---|
| `CUDA out of memory` | in cell 4 change `"--batch", "16"` to `"8"`, then run cells 4–5 again |
| `dataset not found` | the dataset is not attached: Add Input → depth-v2b |
| `wrong ultralytics version` / pip error | Internet is off, or the phone is not verified |
| cell 5 says `STOPPED BEFORE FINISHING` | the printed log lines say why; fix, then run cells 4–5 again |
| cell 5 says `N copies … are running` | run the printed `!kill <pid>` in a new cell |
| a cell errors while training | the trainings are separate processes and keep going: just run cell 5 |
| train class loss stays above ~1.5 after 30 epochs | still underfit: send the cell-5 output to Claude before spending more GPU time |

## 2. Back on your machine — check first, then plug in

Download both zips from **Output** into the repo root. Then run the check: per-source validation
(AP at IoU 0.5 and at the product's loose match, IoU 0.3), plus the fit on each model's own training
frames. It never touches test.

```bash
python -m src.detection.diagnose --zip EXP-003_complete.zip --zip EXP-005_complete.zip --zip EXP-005a_complete.zip
```

**Pass bar:**

- ghost AP@0.3 ≥ 0.65 on its own training frames (not underfit). The bar was set before any
  EXP-003 result, from the reference: EXP-001, a working model, scores 0.71 there (0.57 at IoU 0.5,
  because the 14–36 px label boxes disagree by a few pixels); underfit EXP-002 scores 0.57;
- validation ghost AP above EXP-003's 0.39 (deploy path, IoU 0.5), confirmed by the paired bootstrap;
- a wreck recall ceiling above 0.5.

Onboard only the models that pass:

```bash
python -m src.detection.onboard_model --zip EXP-005_complete.zip --max-ms 700   # or EXP-005a
```

Onboarding:

1. Checks the ONNX sha256 and runs a `cv2.dnn` forward pass.
2. Times the model against EXP-001.
3. Registers `models/<name>/`.
4. **Fits the guaranteed tiers on the validation recordings and verifies them once on test.**
5. Evaluates on the v2b test, the **official 398-frame split** (GhostVision head-to-head) and the
   cross-sonar test.
6. Writes `docs/onboard_<name>.md`, with loud warnings if anything fails.

**Pick the winner on validation, never on test:** the higher ghost AP and recall ceiling within the
speed budget. Then switch:

```bash
DEPTH_MODEL=EXP-005 python -m uvicorn src.dashboard.app:app --port 8000
```

## Success criteria

- recall ceiling at the detector floor on the calibration recordings: **0.72 → ≥ 0.85**
  (EXP-003: 0.85 on val; promise ≥ 79%, held on test at 81%)
- crab-pot AP@0.5 on the v2b unique-frame test: **≥ 0.60**
- official 398-frame split F1 within 0.05 of GhostVision (0.71–0.73)
- recall promise ≥ 90% (95% confidence), held on test

Log EXP-005 / EXP-005a in `experiments.md` with the diagnosis and the onboarding reports linked,
**including if they miss these targets**.

---

Rebuild the upload package (only if the Kaggle dataset is gone or the data changed):

```bash
python -c "import zipfile,pathlib; s=pathlib.Path('DATASET/03_yolo_ready_dataset_v2b'); z=zipfile.ZipFile('runs/kaggle_upload/depth-v2b.zip','w'); [z.write(p, p.relative_to(s.parent).as_posix()) for p in sorted(s.rglob('*')) if p.is_file()]; z.write('webui/samples/ATTRIBUTION.md','03_yolo_ready_dataset_v2b/DATA_ATTRIBUTION.md'); z.close()"
```
