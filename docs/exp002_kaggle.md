# EXP-002 on Kaggle — train, then ONE command to plug it in

**Why this run matters most.** EXP-001 never proposes ~20–28% of crab pots even at conf 0.05
(`docs/calibration_exp001.md`: recall ceiling 0.72 on the calibration recording), so its recall
*promise* is capped at 65%. No triage can recover a pot the detector never proposes — only a better
detector can. EXP-002 attacks exactly that: higher input resolution, tiles for small objects, clean
v2b data, and model selection on the same sonar as test.

| | EXP-001 (deployed) | **EXP-002** |
|---|---|---|
| data | v1 (4 classes, optical + sonar, Roboflow copies, leaks the official test) | **v2b** (2 classes, sonar only, one copy per frame, official split locked) |
| model selection (`best.pt`) | v1 val (optical-heavy) | **held-out recordings Rec10/12/16** — the same Humminbird sonar as test |
| input | 640 px | **1024 px** |
| small objects | — | **full frames + 2×2 overlapping tiles** (`build_tiles.py`) |
| extra positives | — | optional **sonar-aware copy-paste** (pots + shadow tails, same range, Poisson-blended) |
| schedule | 100 ep, overfit after ~16–20 | **30 ep, patience 8, cosine LR, mosaic off for the last 5** |
| augmentation | sonar-aware | same (no hue/rotation/vertical flip); ultralytics `copy_paste` is **not** used — it is a no-op for box-only labels |

## 0. Upload the dataset once (local machine)

```bash
cd DATASET && tar -czf v2b.tar.gz 03_yolo_ready_dataset_v2b   # ~ 2,300 images + labels + manifest
```
Kaggle → **Datasets → New dataset** → upload `v2b.tar.gz` (Kaggle unpacks it) → name it `depth-v2b`.
Private is fine. (The images are CC-BY-SA-4.0 — keep the attribution file in the upload.)

## 1. Notebook (GPU T4 ×1, Internet ON) — paste these cells

```python
!git clone -q https://github.com/madhesh60/depth.git && cd depth && git log --oneline -1
%cd depth
!pip install -q ultralytics
```

```python
# find the uploaded dataset root (the folder that contains data.yaml + train/ val/ test/)
import glob; print(glob.glob("/kaggle/input/**/data.yaml", recursive=True))
SRC = "/kaggle/input/depth-v2b/03_yolo_ready_dataset_v2b"      # ← adjust to the printed path
```

```python
# tiled training set in /kaggle/working (val/test stay the source full frames)
!python DATASET/scripts/build_tiles.py --src {SRC} --out /kaggle/working/v2b_tiles
# variant with the sonar-aware copy-paste (+600 synthetic frames) — for EXP-002p
!python DATASET/scripts/build_tiles.py --src {SRC} --out /kaggle/working/v2b_tiles_paste --paste 600
```

```python
# EXP-002: 1024 px, full frames + tiles  (~1.5-3 h on a T4)
!python src/detection/train.py --data /kaggle/working/v2b_tiles/data.yaml --name EXP-002 --device 0
```

Optional second run (a new notebook version, or the same one if time allows — Kaggle stops at 12 h):
```python
!python src/detection/train.py --data /kaggle/working/v2b_tiles_paste/data.yaml --name EXP-002p --device 0
```

```python
# the zips train.py wrote (weights .pt + verified .onnx + model_meta.json + curves)
!cp runs/EXP-002_complete.zip /kaggle/working/ 2>/dev/null; cp runs/EXP-002p_complete.zip /kaggle/working/ 2>/dev/null; ls -la /kaggle/working/*.zip
```

Run it as **Save Version → Save & Run All (Commit)** so it keeps going if your browser disconnects.
If it prints `CUDA out of memory`, add `--batch 4`.

What `train.py` does after training: exports `best.pt` → ONNX (opset 12, static 1024), **loads it
through `cv2.dnn` and runs a forward pass** (the deploy path — a broken export fails here, not in the
demo), writes `model_meta.json` (classes, imgsz, data, args, best epoch + val metrics, ONNX sha256),
and zips everything.

## 2. Back on your machine — one command

Download `EXP-002_complete.zip` from the notebook's **Output** tab into the repo root, then:

```bash
python -m src.detection.onboard_model --zip EXP-002_complete.zip
```

It unpacks → checks the ONNX sha256 and a `cv2.dnn` forward pass → registers `models/EXP-002/`
→ **fits the guaranteed tiers on the v2b validation recordings and verifies them once on test** →
evaluates deploy-faithfully on the v2b unique-frame test, the **official 398-frame split
(GhostVision head-to-head, leakage-free for v2b)** and the **cross-sonar** `test_xsonar` → writes
`docs/onboard_exp002.md` with the numbers next to EXP-001's. It warns loudly (and recommends NOT
switching) if calibration fails or the recall promise is weak. Run the same for `EXP-002p` and keep
the better one **by validation**, not by test.

Switch the product:
```bash
DEPTH_MODEL=EXP-002 python -m uvicorn src.dashboard.app:app --port 8000
```
On the COOL server: upload `models/EXP-002/best.onnx` to S3 and set `DEPTH_MODEL=EXP-002`
(`infra/setup_cool_instance.sh` / `infra/depth.service`).

## Success criteria (from the review, §10)

- recall ceiling at the detector floor on the calibration recordings: **0.72 → ≥ 0.85**
- crab-pot AP@0.5 on the v2b unique-frame test: **≥ 0.60**
- official 398-frame split F1 within 0.05 of GhostVision (0.71–0.73)
- recall promise ≥ 90% (95% confidence), held on test

Log the run as **EXP-002** in `experiments.md` with the onboarding report linked — including if it
misses these targets.
