# Training on Kaggle — the current run is EXP-004

This page is the procedure for every training run: train on Kaggle, check with `diagnose`, then plug
in with ONE command. History: EXP-002 was underfit ([`exp002_diagnosis.md`](exp002_diagnosis.md)).
EXP-003 fixed that and raised the recall promise from 65% to 79%
([`exp003_diagnosis.md`](exp003_diagnosis.md)).

**Why EXP-004.** EXP-003 fits its training frames and transfers to new recordings for ghost gear.
What is left is **data**:

- the wreck class learned from 389 off-domain colour fish-finder screenshots (validation wreck recall
  ceiling 0.10);
- 139 training frames are rotated copies with black corners.

| | EXP-003 (onboarded) | **EXP-004** | **EXP-004g** |
|---|---|---|---|
| recipe | SGD 0.01 (explicit), fixed tiles, 640 px, 150 ep | same | same |
| training data | v2b train | **without** the "seabed" screenshots (412 frames) and rotated copies (88 more) | same |
| classes | ghost_gear, wreck_debris | same | **ghost_gear only** (wreck boxes removed; their frames stay as negatives) |
| question | — | does clean data help? | does the noisy wreck class hurt pot detection? |

The cleanup happens when the training set is built on Kaggle (`build_tiles.py --drop-sources seabed
--drop-rotated [--keep-classes 0]`), so nothing needs re-uploading. Val and test are untouched; for the
ghost-only arm, val and test are copied with the same single class.

**Tested before the run.**

- **Locally:** a dry build of the ghost-only set (1,273 frames + 2,672 tiles; 412 + 88 frames left
  out; all 186 val / 285 test ghost boxes kept).
- **Kit tests (13 passed):**
  - rotated copies are detected, but not a sonogram's black nadir stripe;
  - class filtering;
  - the notebook flags;
  - the status cell.
- The training path itself is unchanged from EXP-003, which ran end to end on Kaggle.

---

## 0. Before you start (once)

1. The **depth-v2b** Kaggle dataset from EXP-002 is reused as is; nothing to upload. If it is gone,
   rebuild `runs/kaggle_upload/depth-v2b.zip` (command at the end) and create it again as a
   **Private** dataset.
2. Your Kaggle account needs phone verification, otherwise GPU and Internet stay disabled.

## 1. The notebook (import it, no pasting)

1. Kaggle → **Code → New Notebook** → **File → Import Notebook** → upload
   **`notebooks/exp004_kaggle.ipynb`** from this repo.
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
3. Build both cleaned training sets (about 3 minutes): `v2b_clean` (2 classes) and `v2b_clean_g`
   (ghost gear only), each about 1,273 frames + 2,672 tiles.
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
python -m src.detection.diagnose --zip EXP-004_complete.zip --zip EXP-004g_complete.zip
```

**Pass bar:**

- ghost AP@0.3 ≥ 0.65 on its own training frames (not underfit). The bar was set before any
  EXP-003 result, from the reference: EXP-001, a working model, scores 0.71 there (0.57 at IoU 0.5,
  because the 14–36 px label boxes disagree by a few pixels); underfit EXP-002 scores 0.57;
- validation ghost AP clearly above EXP-003's 0.39 (deploy path, IoU 0.5);
- for the 2-class arm, a wreck recall ceiling above 0.5. The ghost-only arm has no wreck class by design.

Onboard only the models that pass:

```bash
python -m src.detection.onboard_model --zip EXP-004_complete.zip --max-ms 700   # or EXP-004g
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
DEPTH_MODEL=EXP-004 python -m uvicorn src.dashboard.app:app --port 8000
```

## Success criteria

- recall ceiling at the detector floor on the calibration recordings: **0.72 → ≥ 0.85**
  (EXP-003: 0.85 on val; promise ≥ 79%, held on test at 81%)
- crab-pot AP@0.5 on the v2b unique-frame test: **≥ 0.60**
- official 398-frame split F1 within 0.05 of GhostVision (0.71–0.73)
- recall promise ≥ 90% (95% confidence), held on test

Log EXP-004 / EXP-004g in `experiments.md` with the diagnosis and the onboarding reports linked,
**including if they miss these targets**.

---

Rebuild the upload package (only if the Kaggle dataset is gone or the data changed):

```bash
python -c "import zipfile,pathlib; s=pathlib.Path('DATASET/03_yolo_ready_dataset_v2b'); z=zipfile.ZipFile('runs/kaggle_upload/depth-v2b.zip','w'); [z.write(p, p.relative_to(s.parent).as_posix()) for p in sorted(s.rglob('*')) if p.is_file()]; z.write('webui/samples/ATTRIBUTION.md','03_yolo_ready_dataset_v2b/DATA_ATTRIBUTION.md'); z.close()"
```
