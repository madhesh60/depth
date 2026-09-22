# EXP-002 — higher-resolution retrain (Kaggle, unattended)

**Goal:** recover small-object `fishing_gear` recall by training at a larger input size.
EXP-001 error analysis: 91% of missed `fishing_gear` boxes are <10% of frame width — a
small-object problem; higher resolution is the cheapest, highest-value lever.

**Change vs EXP-001:** only `--imgsz 640 → 1024` and `--batch 16 → 8`, `--epochs 40`
(EXP-001 overfit after ~epoch 20, so 40 is plenty). Everything else identical.

---

## Why "Save & Run All" (read this — it saves your run)

- Higher resolution is ~2.5× slower → ~10–12 min/epoch → **40 epochs ≈ 8–10 h**.
- A normal interactive run **dies if your internet drops**. A **committed** run executes on
  Kaggle's own servers — close the laptop, lose wifi, it keeps going.
- Kaggle hard-stops any run at **12 h**, so 1024/40ep fits; 1280/60ep would NOT.

**How:** top-right → **Save Version → "Save & Run All (Commit)"** → wait for the green tick.
Do **not** press the ▶ play buttons for the real run.

---

## Notebook cells (paste in order)

**1. GPU on:** Settings (right panel) → Accelerator → **GPU T4 x1**. Attach the same dataset
you used for EXP-001.

**2. Get the code**
```python
!git clone https://github.com/madhesh60/depth.git
%cd depth
!git log --oneline -1     # confirm it's the latest commit
```

**3. Install trainer**
```python
!pip install -q ultralytics
```

**4. Find your dataset yaml** (copy the path it prints)
```python
!find /kaggle/input -name "data*.yaml"
```

**5. Train EXP-002** (paste your data path into `--data`)
```python
!python src/detection/train.py \
  --model yolo11s.pt \
  --data /kaggle/input/PASTE_YOUR_PATH/data.yaml \
  --imgsz 1024 \
  --batch 8 \
  --epochs 40 \
  --name EXP-002
```
*If it prints `CUDA out of memory`: change `--batch 8` to `--batch 4` and re-commit.*

**6. Zip results** (last cell — runs after training)
```python
!cd runs && zip -r /kaggle/working/EXP-002_complete.zip EXP-002
```

---

## After it finishes
- Open the committed version → **Output** → download `EXP-002_complete.zip`.
- **Back it up** (weights are not in git — they're too large).
- Send me the zip. I'll run the same deep analysis (`error_analysis.py`) — per-class, split
  sonar vs optical — and compare to EXP-001.

**Success = `fishing_gear` sonar recall climbs from ~0.47 toward ~0.70** while the other
classes hold. Watch the val curve in `results.png` for the overfitting point too.
