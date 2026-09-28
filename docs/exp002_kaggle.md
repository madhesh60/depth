# EXP-002 on Kaggle — train, then ONE command to plug it in

**Why this run matters most.** EXP-001 never proposes about 20–28% of crab pots even at conf 0.05
(`docs/calibration_exp001.md`: recall ceiling 0.72 on the calibration recording), so its recall
*promise* is capped at 65%. No triage can recover a pot the detector never proposes; only a better
detector can. EXP-002 attacks exactly that with:

- clean v2b data;
- tiles for small objects;
- model selection on the same sonar as test;
- two input sizes, chosen on held-out recordings within a speed budget.

| | EXP-001 (deployed) | **EXP-002 / EXP-002s** |
|---|---|---|
| data | v1 (4 classes, optical + sonar, Roboflow copies, leaks the official test) | **v2b** (2 classes, sonar only, one copy per frame, official split locked) |
| training set | v1 train | v2b train as **full frames + 2×2 overlapping tiles** (5,746 images, `build_tiles.py`) |
| input | 640 px | **1024 px (EXP-002)** and **640 px (EXP-002s)** in parallel; STUDY-13 showed resolution must be *chosen* |
| exported checkpoint | ultralytics `best.pt` (0.9·mAP50-95 fitness, both classes) | **best ghost-gear AP@0.5 on the held-out recordings** Rec10/12/16 (every epoch saved and validated; the table is recorded) |
| schedule | 100 ep, overfit after ~16–20 | **30 ep, patience 8, cosine LR, mosaic off for the last 5** |
| augmentation | sonar-aware | the same: no hue, rotation or vertical flip; along-track flip only. Ultralytics `copy_paste` is **not** used (it is a no-op for box-only labels). |
| speed | — | onboarding times each model against EXP-001 on the same machine; `--max-ms` gate |

**Tested end to end before the run.** A CPU smoke test (2 epochs, 2% of the data) went through
training, per-epoch checkpoints, ghost-AP selection, ONNX export, the `cv2.dnn` check, the zip, and
onboarding with calibration, all three evaluations and the speed table. The weak smoke model was
correctly refused by the gate.

---

## 0. Before you start (5 minutes, once)

1. The upload package is already built: **`runs/kaggle_upload/depth-v2b.zip`** (620 MB). Rebuild it if
   the dataset changes (see the command at the end).
2. Kaggle → **Datasets → New Dataset** → drag in `depth-v2b.zip` → title **depth-v2b** → **Private** →
   Create. Kaggle unpacks the zip; wait until processing finishes (a few minutes).
3. Kaggle account → **Settings → phone verification** must be done, otherwise GPU and Internet stay
   disabled.

## 1. The notebook (import it, no pasting)

1. Kaggle → **Code → New Notebook** → **File → Import Notebook** → upload
   **`notebooks/exp002_kaggle.ipynb`** from this repo.
2. Right panel:
   - **Add Input** → your **depth-v2b** dataset;
   - **Accelerator → GPU T4 ×2**;
   - **Internet → On**.
3. **Save Version → Save & Run All (Commit)**. It keeps running if you close the browser. With 2 GPUs,
   EXP-002 (1024 px) and EXP-002s (640 px) train at the same time; with 1 GPU they run one after the
   other.

What the six cells do:

1. Clone the repo and install **ultralytics 8.4.157** (pinned: the version the kit was tested with).
   Asserts a GPU.
2. Find the dataset automatically and check the split sizes (train 1,773 · val 234 · test 285 ·
   official 398 · cross-sonar 555).
3. Build the tiled set (about 30 s; 5,746 training images).
4. Launch both trainings. Logs go to `/kaggle/working/logs/<name>.log`.
5. Wait, printing epoch / val mAP50 / recall every 10 minutes. After training, each run validates
   every epoch's checkpoint and exports the best one for ghost gear.
6. Copy `EXP-002_complete.zip` and `EXP-002s_complete.zip` to the **Output** tab, and print which
   checkpoint each run exported.

**Expected time:** about 2–3.5 h for the 1024-px run and about 1 h for the 640-px run (parallel), plus
~10–15 min of checkpoint selection. Kaggle's limit is 12 h. Disk use is about 3 GB (no image cache —
a disk cache at 1024 px would need ~27 GB and would crash the run).

**While it runs** (open the running version → Logs), check:

- the first epoch starts within ~5 minutes;
- `val mAP50` rises over the first 5–10 epochs;
- `GPU mem` stays under ~14 GB.

**If something goes wrong:**

| symptom | fix |
|---|---|
| `CUDA out of memory` | in cell 4 set the batch for the 1024-px run to `4`, then run again |
| `dataset not found` | the dataset is not attached: Add Input → depth-v2b |
| `wrong ultralytics version` / pip error | Internet is off, or the phone is not verified |
| one run fails, the other finishes | download the one zip that exists; re-run later with only the failed run in `RUNS` |
| stopped at 12 h | lower `--epochs` to 20 in cell 4 (patience 8 usually stops earlier anyway) |

## 2. Back on your machine — one command per model

Download both zips from the notebook's **Output** tab into the repo root, then:

```bash
python -m src.detection.onboard_model --zip EXP-002_complete.zip  --max-ms 700
python -m src.detection.onboard_model --zip EXP-002s_complete.zip --max-ms 700
```

Each command does the following:

1. Unpacks the zip.
2. Checks the ONNX sha256 and runs a `cv2.dnn` forward pass (OpenCV 5).
3. **Times it against EXP-001** on this machine.
4. Registers `models/<name>/`.
5. **Fits the guaranteed tiers on the v2b validation recordings and verifies them once on test.**
6. Evaluates deploy-faithfully on:
   - the v2b unique-frame test;
   - the **official 398-frame split** (the GhostVision head-to-head, leakage-free for v2b);
   - the **cross-sonar** `test_xsonar`.
7. Writes `docs/onboard_<name>.md`. It warns loudly (do NOT switch) if calibration fails, the recall
   promise is weak, or the model is too slow.

**Pick the winner by the validation recordings, never by test:**

- the higher recall ceiling / recall promise in the onboarding report;
- within the speed budget.

If 1024 px wins on validation but is too slow on the laptop, decide on the Graviton + COOL numbers
(`python -m src.bench.product_bench`). COOL may be what makes the more accurate model affordable, and
that is the COOL story, measured.

Switch the product:

```bash
DEPTH_MODEL=EXP-002 python -m uvicorn src.dashboard.app:app --port 8000     # or EXP-002s
```

On the COOL server, upload `models/<name>/best.onnx` to S3 and set `DEPTH_MODEL=<name>`
(`infra/setup_cool_instance.sh` / `infra/depth.service`). Then re-run
`python -m src.agentic.effort` and the audit build for the new model, and publish its weights as a
GitHub Release (`fetch_model`).

## Success criteria (from the review, §10)

- recall ceiling at the detector floor on the calibration recordings: **0.72 → ≥ 0.85**
- crab-pot AP@0.5 on the v2b unique-frame test: **≥ 0.60**
- official 398-frame split F1 within 0.05 of GhostVision (0.71–0.73)
- recall promise ≥ 90% (95% confidence), held on test

Log the runs as **EXP-002 / EXP-002s** in `experiments.md` with the onboarding reports linked,
including if they miss these targets.

---

Rebuild the upload package (only if the dataset changed):

```bash
python -c "import zipfile,pathlib; s=pathlib.Path('DATASET/03_yolo_ready_dataset_v2b'); z=zipfile.ZipFile('runs/kaggle_upload/depth-v2b.zip','w'); [z.write(p, p.relative_to(s.parent).as_posix()) for p in sorted(s.rglob('*')) if p.is_file()]; z.write('webui/samples/ATTRIBUTION.md','03_yolo_ready_dataset_v2b/DATA_ATTRIBUTION.md'); z.close()"
```
