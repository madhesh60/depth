"""Writes notebooks/exp003_kaggle.ipynb — the EXP-003 training notebook for Kaggle (File → Import
Notebook). Regenerate: python notebooks/make_kaggle_notebook.py"""
import json
from pathlib import Path

ULTRALYTICS = "8.4.157"          # the version the kit is smoke-tested with (requirements-train.txt)

cells = [
    ("markdown", """# DEPTH — EXP-003 training (Kaggle, GPU T4 ×2)

EXP-002 failed (ghost-gear AP 0.25 on the held-out recordings). The post-mortem is in
`docs/exp002_diagnosis.md`. It found two causes, and both are fixed in this kit:

1. **Underfit.** `optimizer=auto` silently trained with AdamW at lr 0.00167 for 2,700 steps, and the
   requested `lr0 0.01` was ignored. The models reached only AP 0.42 on their *own training frames*.
   This kit uses an **explicit SGD optimizer at lr 0.01** with enough steps, and records the optimizer
   ultralytics actually built.
2. **Tiles taught "object = background".** Objects cut by a tile kept their pixels but lost their
   label: 49% of wreck appearances and 8% of ghost-gear appearances. Frames with large objects are no
   longer tiled, and cut objects are inpainted.

Two arms, both at 640 px with the same optimizer and a similar number of steps:

* **EXP-003**: full frames + the fixed 2×2 tiles, 150 epochs (GPU 0)
* **EXP-003f**: full frames only, 300 epochs (GPU 1)

This shows whether tiles help. Each run keeps the epoch with the best **ghost-gear AP@0.5 on the
held-out validation recordings**.

**Settings:** Accelerator **GPU T4 ×2** · Internet **On** · add the **depth-v2b** dataset (Add Input).
Then either **Save Version → Save & Run All (Commit)** (runs unattended), or run cells 1–5 in order and
keep the tab open. Cell 5 shows what is running, waits, and collects; it is safe to re-run at any
time."""),
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
    ("code", """# 3. the two training sets in /kaggle/working (val/test stay the untouched full frames)
!python DATASET/scripts/build_tiles.py --src "{SRC}" --out /kaggle/working/v2b_tiles
!python DATASET/scripts/build_tiles.py --src "{SRC}" --out /kaggle/working/v2b_full --no-tiles
!echo "tiles set: $(ls /kaggle/working/v2b_tiles/train/images | wc -l) images | full-frame set: $(ls /kaggle/working/v2b_full/train/images | wc -l) images"
!df -h /kaggle/working | tail -1
"""),
    ("code", """# 4. launch both runs (parallel on 2 GPUs, else one after the other) — logs in /kaggle/working/logs
#    Run it ONCE: it refuses to start a second copy while a training is alive. Cell 5 shows the status.
import shlex, subprocess, torch
os.makedirs("/kaggle/working/logs", exist_ok=True)
RUNS = [  # name, training set, epochs  (both: 640 px, batch 16, SGD lr0 0.01 - ~9-11 k optimizer steps)
    ("EXP-003", "v2b_tiles", 150),
    ("EXP-003f", "v2b_full", 300),
]
def cmd(name, data, epochs, dev):
    return ["python", "-u", "src/detection/train.py", "--data", f"/kaggle/working/{data}/data.yaml", "--name", name,
            "--imgsz", "640", "--batch", "16", "--device", str(dev), "--workers", "2", "--cache", "none",
            "--optimizer", "SGD", "--lr0", "0.01", "--epochs", str(epochs), "--close-mosaic", "10",
            "--notes", f"Kaggle T4, 640px, {data}, SGD 0.01, ghost-AP50 selection"]
live = subprocess.run(["bash", "-lc", "ps -eo args="], capture_output=True, text=True).stdout
assert "src/detection/train.py" not in live, "a training is already running - do not start it twice; run cell 5"
# start_new_session: the trainings get their own process group, so Stop/Interrupt on a cell cannot kill them
if torch.cuda.device_count() >= 2:
    for dev, (name, data, epochs) in enumerate(RUNS):
        log = open(f"/kaggle/working/logs/{name}.log", "w")
        subprocess.Popen(cmd(name, data, epochs, dev), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print("started", name, "on GPU", dev)
else:
    chain = " ; ".join(f"{shlex.join(cmd(n, d, e, 0))} > /kaggle/working/logs/{n}.log 2>&1" for n, d, e in RUNS)
    subprocess.Popen(["bash", "-c", chain], start_new_session=True)
    print("one GPU: the runs go one after the other:", ", ".join(n for n, *_ in RUNS))
print("now run cell 5: it shows what is running and waits until the models are packaged")
"""),
    ("code", Path(__file__).with_name("monitor_cell.py").read_text(encoding="utf-8")),
    ("markdown", """## Next — on your own machine

Download `EXP-003_complete.zip` and `EXP-003f_complete.zip` from the **Output** panel into the repo
root and tell Claude "EXP-003 done". The diagnosis (per-source val, fit on training frames) comes
first, then onboarding:

```bash
python -m src.detection.diagnose --zip EXP-003_complete.zip --zip EXP-003f_complete.zip
python -m src.detection.onboard_model --zip EXP-003_complete.zip --max-ms 700
```

Keep the model with the better **ghost-gear AP and recall ceiling on the validation recordings within
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
out = Path(__file__).with_name("exp003_kaggle.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
