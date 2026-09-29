"""Writes notebooks/exp002_kaggle.ipynb — the EXP-002 training notebook for Kaggle (import it with
File → Import Notebook, or paste the cells). Regenerate: python notebooks/make_exp002_notebook.py"""
import json
from pathlib import Path

ULTRALYTICS = "8.4.157"          # the version the kit is smoke-tested with (requirements-train.txt)

cells = [
    ("markdown", """# DEPTH — EXP-002 training (Kaggle, GPU T4 ×2)

Trains the next DEPTH detector on dataset **v2b** and packages each model for one-command onboarding.

* **EXP-002** — YOLO11s at **1024 px** on full frames + 2×2 tiles (GPU 0)
* **EXP-002s** — the same at **640 px** (GPU 1) — STUDY-13 showed resolution must be *chosen* on
  held-out recordings, not assumed; onboarding compares the two within a speed budget.

Each run saves every epoch and exports the checkpoint with the best **ghost-gear AP@0.5 on the
held-out validation recordings** (not ultralytics' default box-tightness fitness).

**Settings:** Accelerator **GPU T4 ×2** · Internet **On** · then **Save Version → Save & Run All
(Commit)**. With one GPU the two runs go one after the other. Outputs: `EXP-002_complete.zip`,
`EXP-002s_complete.zip` in the Output tab."""),
    ("code", f"""# 1. environment — the repo + the EXACT ultralytics version the kit was tested with
!nvidia-smi --query-gpu=index,name,memory.total --format=csv
!rm -rf depth && git clone -q https://github.com/madhesh60/depth.git && cd depth && git log --oneline -1
%cd /kaggle/working/depth
!pip install -q ultralytics=={ULTRALYTICS}
import torch, ultralytics, cv2
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__, "| CUDA GPUs", torch.cuda.device_count(), "| cv2", cv2.__version__)
assert ultralytics.__version__ == "{ULTRALYTICS}", "wrong ultralytics version"
assert torch.cuda.is_available(), "no GPU - set Accelerator to GPU T4 x2"
"""),
    ("code", """# 2. find the uploaded dataset (the folder holding data.yaml + train/ val/ test/) — no manual path
import glob, os
hits = [p for p in glob.glob("/kaggle/input/**/data.yaml", recursive=True) if os.path.isdir(os.path.join(os.path.dirname(p), "train"))]
assert hits, "dataset not found - add the depth-v2b dataset to this notebook (Add Input)"
SRC = os.path.dirname(sorted(hits, key=len)[0])
for s in ("train", "val", "test", "test_official398", "test_xsonar"):
    d = os.path.join(SRC, s, "images")
    print(f"{s:>17}: {len(os.listdir(d)) if os.path.isdir(d) else 'missing'} images")
print("SRC =", SRC)
assert len(os.listdir(os.path.join(SRC, "train", "images"))) > 1500, "train split looks incomplete"
"""),
    ("code", """# 3. tiled training set in /kaggle/working (val/test stay the untouched full frames)
!python DATASET/scripts/build_tiles.py --src "{SRC}" --out /kaggle/working/v2b_tiles
!cat /kaggle/working/v2b_tiles/data.yaml
!echo "train images: $(ls /kaggle/working/v2b_tiles/train/images | wc -l)"
!df -h /kaggle/working | tail -1
"""),
    ("code", """# 4. launch both runs (parallel on 2 GPUs, else one after the other) — logs in /kaggle/working/logs
import subprocess, time, torch
os.makedirs("/kaggle/working/logs", exist_ok=True)
DATA = "/kaggle/working/v2b_tiles/data.yaml"
RUNS = [  # name, imgsz, batch
    ("EXP-002", 1024, 8),
    ("EXP-002s", 640, 16),
]
def cmd(name, imgsz, batch, dev):
    return ["python", "src/detection/train.py", "--data", DATA, "--name", name, "--imgsz", str(imgsz),
            "--batch", str(batch), "--device", str(dev), "--workers", "2", "--cache", "none",
            "--epochs", "30", "--patience", "8", "--notes", f"Kaggle T4, {imgsz}px, tiles, ghost-AP50 selection"]
ngpu = torch.cuda.device_count()
procs = []
if ngpu >= 2:
    for dev, (name, imgsz, batch) in enumerate(RUNS):
        log = open(f"/kaggle/working/logs/{name}.log", "w")
        procs.append((name, subprocess.Popen(cmd(name, imgsz, batch, dev), stdout=log, stderr=subprocess.STDOUT)))
        print("started", name, "on GPU", dev)
else:
    print("one GPU: the runs will go one after the other (cell 5 starts the second)")
    name, imgsz, batch = RUNS[0]
    log = open(f"/kaggle/working/logs/{name}.log", "w")
    procs.append((name, subprocess.Popen(cmd(name, imgsz, batch, 0), stdout=log, stderr=subprocess.STDOUT)))
T0 = time.time()
"""),
    ("code", """# 5. wait for the runs (this cell blocks — needed for Save & Run All); progress every 10 min
import csv, glob as _g
def progress():
    for name, *_ in RUNS:
        rc = f"/kaggle/working/depth/runs/{name}/results.csv"
        if os.path.exists(rc):
            rows = list(csv.DictReader(open(rc)))
            if rows:
                r = {k.strip(): v for k, v in rows[-1].items()}
                print(f"  {name}: epoch {int(float(r['epoch']))}  val mAP50 {float(r['metrics/mAP50(B)']):.3f}  "
                      f"recall {float(r['metrics/recall(B)']):.3f}")
pending = list(procs)
while pending:
    time.sleep(600)
    print(f"--- {(time.time() - T0) / 3600:.1f} h"); progress()
    still = []
    for name, p in pending:
        if p.poll() is None:
            still.append((name, p))
        else:
            print(name, "finished with exit code", p.returncode)
            if ngpu < 2 and name == RUNS[0][0]:                    # one GPU: start the 640 run now
                n2, s2, b2 = RUNS[1]
                log = open(f"/kaggle/working/logs/{n2}.log", "w")
                still.append((n2, subprocess.Popen(cmd(n2, s2, b2, 0), stdout=log, stderr=subprocess.STDOUT)))
    pending = still
print("all runs done in", round((time.time() - T0) / 3600, 2), "h")
"""),
    ("code", """# 6. collect the packages into the Output tab + print what was selected
import json, shutil
for name, *_ in RUNS:
    z = f"/kaggle/working/depth/runs/{name}_complete.zip"
    meta_p = f"/kaggle/working/depth/runs/{name}/model_meta.json"
    if not os.path.exists(z):
        print(f"!! {name}: no package - see /kaggle/working/logs/{name}.log"); continue
    shutil.copy(z, "/kaggle/working/")
    m = json.load(open(meta_p))
    sel = (m.get("selection") or {}).get("picked") or {}
    ub = (m.get("selection") or {}).get("ultralytics_best") or {}
    print(f"{name}: {m['train_minutes']} min | exported {sel.get('checkpoint')} ghost_gear AP50 {sel.get('ghost_ap50')} "
          f"(ultralytics' pick {ub.get('ghost_ap50')}) | {m.get('cv2_dnn_verified')}")
!rm -rf /kaggle/working/v2b_tiles          # 629 MB of tiles: not needed any more
!ls -la /kaggle/working/*.zip
!tail -3 /kaggle/working/logs/*.log
"""),
    ("markdown", """## Next — on your own machine

Download `EXP-002_complete.zip` and `EXP-002s_complete.zip` from the **Output** tab into the repo root, then:

```bash
python -m src.detection.onboard_model --zip EXP-002_complete.zip  --max-ms 700
python -m src.detection.onboard_model --zip EXP-002s_complete.zip --max-ms 700
```

Keep the model with the better **recall ceiling / recall promise on the validation recordings within
the speed budget** (never pick by test), then `DEPTH_MODEL=<name>`."""),
]


def as_source(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


nb = {"cells": [{"cell_type": t, "metadata": {}, "source": as_source(s), **({"outputs": [], "execution_count": None} if t == "code" else {})}
                for t, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("exp002_kaggle.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
