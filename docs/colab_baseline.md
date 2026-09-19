# EXP-001 Baseline Training on Google Colab (free T4 GPU)

The local machine (MX330 / 2 GB, CPU-only torch) can't train this dataset. Run the baseline on
a Colab **T4 (16 GB)** instead. This reuses the tracked trainer `src/detection/train.py` — only
the *data* is uploaded separately (the `DATASET/` tree is git-ignored).

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

```python
!git clone https://github.com/madhesh60/depth.git
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

## 5. Train (yolo11s, detect, 100 epochs)

`train.py` auto-detects the GPU and applies the sonar-aware augmentation
(no hue/sat, no rotation/vflip, along-track flip on) and prints per-class metrics.

```python
!cd /content/depth && python src/detection/train.py \
    --model yolo11s.pt --epochs 100 --batch 16 --name EXP-001
```

T4 fits `yolo11s @ 640, batch 16`. Expect **~2-4 h** for 100 epochs (early-stopping patience 20).
If you OOM, drop `--batch` to 8. The final block prints test mAP/precision/recall.

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

Log EXP-001 in `experiments.md` with the **per-class** mAP/precision/recall (not just aggregate),
note pipe_cylinder AP as high-variance (65 test boxes), then commit + push. Next experiments:

- **EXP-002** — originals-only train (drop baked augs) vs this run: is the 62% baked augmentation
  helping or just doubling online augmentation? (See dataset report / CLAUDE.md.)
- **EXP-006** — sonar-only fine-tune for the mission domain.
- Minority handling (oversample / focal) only if pipe/rope recall is poor here.
