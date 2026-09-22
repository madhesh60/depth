# EXP-001 Baseline Training on Google Colab (free T4 GPU)

> **⚠️ Historical / superseded (2026-09-22).** EXP-001 was ultimately run on **Kaggle T4 for
> 40 epochs** (not Colab/100 — the args below are the original template). For the **next run
> (EXP-002) use [`exp002_kaggle.md`](exp002_kaggle.md)** — it has the correct settings
> (`imgsz 1024 / batch 8 / 40 ep`) and the unattended "Save & Run All" flow. This file is kept
> for provenance of how the baseline was set up.

The local machine (MX330 / 2 GB, CPU-only torch) can't train this dataset. Run on a cloud
**T4 (16 GB)** instead. This reuses the tracked trainer `src/detection/train.py` — only the
*data* is uploaded separately (the `DATASET/` tree is git-ignored).

**Target (from CLAUDE.md):** mAP@0.5 ≥ 0.70, precision ≥ 0.80, recall ≥ 0.70 — reported
**per-class** (pipe_cylinder AP is high-variance; see dataset report).

---

## 0. One-time: make the upload bundle (run locally)

```bash
cd DATASET
tar cf 03_yolo_ready_dataset_v1.tar 03_yolo_ready_dataset_v1     # ~1-1.5 GB
```

Upload `DATASET/03_yolo_ready_dataset_v1.tar` to the **root of your Google Drive** (`MyDrive/`).
The `.tar` stays git-ignored — do not commit it.

---

## 1. Colab: enable GPU

`Runtime → Change runtime type → T4 GPU`, then:

```python
!nvidia-smi        # confirm a T4 (16 GB) is attached
```

## 2. Code + deps

> **The repo must be public** for this clone to work. Colab can't answer an
> interactive credential prompt, so cloning a *private* repo over HTTPS fails with
> `fatal: could not read Username for 'https://github.com'` — and then every later
> cell dies with `No such file or directory: .../train.py`. Either make the repo
> public (GitHub → Settings → Danger Zone → Change visibility), or clone with a
> read-only token: `!git clone https://<TOKEN>@github.com/madhesh60/depth.git`.

```python
!git clone https://github.com/madhesh60/depth.git /content/depth
!ls /content/depth/src/detection/train.py    # sanity-check: must print the file
!pip -q install ultralytics
```

## 3. Data (from Drive)

```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/depth/DATASET
!tar xf /content/drive/MyDrive/03_yolo_ready_dataset_v1.tar -C /content/depth/DATASET
!ls /content/depth/DATASET/03_yolo_ready_dataset_v1
```

## 4. Pin absolute paths in data.yaml

`data.yaml` uses split-relative paths; give Ultralytics an absolute root so it can't miss them:

```python
import yaml, pathlib
root = pathlib.Path('/content/depth/DATASET/03_yolo_ready_dataset_v1')
d = yaml.safe_load((root / 'data.yaml').read_text())
d['path'] = str(root)
(root / 'data.yaml').write_text(yaml.safe_dump(d, sort_keys=False))
print(d)
```

## 5. Train (yolo11s, detect) — EXP-001 used 40 epochs on Kaggle

`train.py` auto-detects the GPU and applies the sonar-aware augmentation
(no hue/sat, no rotation/vflip, along-track flip on) and prints per-class metrics.

```python
!cd /content/depth && python src/detection/train.py \
    --model yolo11s.pt --epochs 100 --batch 16 --name EXP-001
```

T4 fits `yolo11s @ 640, batch 16`. EXP-001 actually ran **40 epochs (~4 h)** and overfit after
~epoch 20 (see `experiments.md`). If you OOM, drop `--batch` to 8. The final block prints test
mAP/precision/recall.

## 6. Export ONNX + save results back to Drive

The AWS pipeline loads the detector via `cv2.dnn.readNetFromONNX`, so export ONNX now:

```python
!cd /content/depth && yolo export model=runs/EXP-001/weights/best.pt format=onnx opset=12
!cp -r /content/depth/runs/EXP-001 /content/drive/MyDrive/EXP-001_results
```

Download `best.pt`, `best.onnx`, `results.png`, and the per-class table from
`MyDrive/EXP-001_results`.

---

## 7. After the run — record it

EXP-001 is logged in `experiments.md` (done). The ladder was **reprioritised** after the deep
error analysis (fishing_gear is a small-object sonar problem):

- **EXP-002** — higher resolution (`imgsz 1024`) for small fishing_gear. **← next**, see
  [`exp002_kaggle.md`](exp002_kaggle.md).
- **EXP-003** — sonar-only training (optical debris is a total miss and off-product).
- Aug-ablation (originals-only) demoted to EXP-006.
