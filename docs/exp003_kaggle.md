# EXP-003 on Kaggle — train, check, then ONE command to plug it in

**Why this run matters most.** EXP-001's recall ceiling caps its recall promise at 65%
(`docs/calibration_exp001.md`: it never proposes about 20–28% of crab pots, even at conf 0.05). No
triage can recover a pot the detector never proposes; only a better detector can.

EXP-002 tried and **failed**: ghost-gear AP 0.25 on the held-out recordings. The cause was found and
fixed — see [`exp002_diagnosis.md`](exp002_diagnosis.md):

1. **Underfit.** `optimizer=auto` silently trained with AdamW at lr 0.00167 for 2,700 steps and
   ignored `lr0`. The models scored only AP 0.42 on their own training frames.
2. **Tiles taught "object = background".** Cut objects lost their labels but kept their pixels.

| | EXP-002 (failed) | **EXP-003 / EXP-003f** |
|---|---|---|
| optimizer | `auto` → AdamW lr 0.00167 (`lr0` ignored) | **explicit SGD lr 0.01**; the optimizer actually built is recorded and checked |
| optimizer steps | ~2,700 | **~11.4 k (EXP-003) / ~8.4 k (EXP-003f)**, near EXP-001's 16.6 k |
| training set | full frames + 2×2 tiles (5,746); cut boxes left unlabelled | **EXP-003:** full frames + fixed tiles (4,839): large-object frames are not tiled, cut boxes are inpainted. **EXP-003f:** full frames only (1,773) |
| input | 1024 and 640 (no difference while underfit) | **640** for both (2.6× cheaper at inference) |
| epochs / patience | 30 / 8 | **150 / 37 (EXP-003)** and **300 / 75 (EXP-003f)**; mosaic off for the last 10 |
| checkpoint selection | every epoch saved (~40 MB each) and validated afterwards | best **ghost-gear AP@0.5** tracked during training (no per-epoch files); shortlist re-validated |
| before onboarding | — | **`diagnose`:** per-source val + fit check on the model's own training frames |

The comparison is fair because both arms have the same optimizer and a similar step count. So
EXP-003 vs EXP-003f answers whether tiles help.

**Tested before the run (CPU smoke test).**

- **Training:** the run built `SGD lr 0.01`, the per-epoch ghost AP was tracked, and the shortlist
  selection picked a checkpoint.
- **Packaging:** the ONNX export passed the `cv2.dnn` check, and the zip was complete.
- **Monitor cell:** tested against a simulated process list.

---

## 0. Before you start (once)

1. The **depth-v2b** Kaggle dataset from EXP-002 is reused as is; nothing to upload. If it is gone,
   rebuild `runs/kaggle_upload/depth-v2b.zip` (command at the end) and create it again as a
   **Private** dataset.
2. Your Kaggle account needs phone verification, otherwise GPU and Internet stay disabled.

## 1. The notebook (import it, no pasting)

1. Kaggle → **Code → New Notebook** → **File → Import Notebook** → upload
   **`notebooks/exp003_kaggle.ipynb`** from this repo.
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
3. Build both training sets (about 2 minutes): `v2b_tiles` (4,839 images) and `v2b_full` (1,773).
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

**Expected time:** about 3–3.5 h per arm, running in parallel; Kaggle's limit is 12 h. Disk use is
about 2 GB.

**While it runs, check:**

- the first epoch finishes within ~5 minutes;
- the train class loss falls steadily, below ~1.2 by about epoch 30;
- val mAP50 passes EXP-002's 0.12 within the first ~20 epochs.

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

Download both zips from **Output** into the repo root. Then run the check: per-source validation,
plus the fit on each model's own training frames. It never touches test.

```bash
python -m src.detection.diagnose --zip EXP-003_complete.zip --zip EXP-003f_complete.zip
```

**Pass bar:**

- ghost AP ≥ 0.70 on its own training frames;
- validation ghost AP clearly above EXP-002's 0.29;
- wreck recall ceiling above 0.5.

Onboard only the models that pass:

```bash
python -m src.detection.onboard_model --zip EXP-003_complete.zip --max-ms 700
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
DEPTH_MODEL=EXP-003 python -m uvicorn src.dashboard.app:app --port 8000
```

## Success criteria

- recall ceiling at the detector floor on the calibration recordings: **0.72 → ≥ 0.85**
- crab-pot AP@0.5 on the v2b unique-frame test: **≥ 0.60**
- official 398-frame split F1 within 0.05 of GhostVision (0.71–0.73)
- recall promise ≥ 90% (95% confidence), held on test

Log EXP-003 / EXP-003f in `experiments.md` with the diagnosis and the onboarding reports linked,
**including if they miss these targets**.

---

Rebuild the upload package (only if the Kaggle dataset is gone or the data changed):

```bash
python -c "import zipfile,pathlib; s=pathlib.Path('DATASET/03_yolo_ready_dataset_v2b'); z=zipfile.ZipFile('runs/kaggle_upload/depth-v2b.zip','w'); [z.write(p, p.relative_to(s.parent).as_posix()) for p in sorted(s.rglob('*')) if p.is_file()]; z.write('webui/samples/ATTRIBUTION.md','03_yolo_ready_dataset_v2b/DATA_ATTRIBUTION.md'); z.close()"
```
