"""Writes notebooks/exp005_kaggle.ipynb — the EXP-005 training notebook for Kaggle (File → Import
Notebook). Regenerate: python notebooks/make_kaggle_notebook.py"""
import json
from pathlib import Path

ULTRALYTICS = "8.4.157"          # the version the kit is smoke-tested with (requirements-train.txt)

cells = [
    ("markdown", """# DEPTH — EXP-005 training (Kaggle, GPU T4 ×2)

What the previous runs established:

- explicit SGD fixes the underfit (EXP-002 → EXP-003);
- fixed tiles help (EXP-003 beat full frames);
- cleaned data helps wrecks and is neutral for pots (EXP-004);
- two classes beat ghost-only (EXP-004g).

What still limits the model is **ranking against false alarms**, and the gap between training frames
and new recordings (`docs/exp004_diagnosis.md`). A post-processing fix for duplicate boxes was tested
on validation and rejected.

EXP-005 = EXP-004's cleaned data and EXP-003's recipe (SGD 0.01, fixed tiles, 640 px, 2 classes), with

* **stronger brightness and scale jitter** (`--hsv-v 0.4 --scale 0.6`, up from 0.2 / 0.5). Sonar gain
  differs between recordings, and the models fit training frames far better than new ones (AP@0.3 0.74
  vs 0.49; validation peaks mid-run and then falls).

The same recipe runs twice with different seeds: **EXP-005** (seed 42, GPU 0) and **EXP-005a** (seed 7,
GPU 1). Validation noise is ±0.07 AP, so two runs show whether the change is real. The better one on
validation is kept. (Reviewed pseudo-labels were also tested: only 8 confident unlabelled pots exist in
the training frames, against 1,345 labels, so they are not used.)

Validation and test are untouched. Each run keeps the epoch with the best **ghost-gear AP@0.5 on the
held-out validation recordings**.

**Settings:** Accelerator **GPU T4 ×2** · Internet **On** · add the **depth-v2b** dataset (Add Input).
Then either **Save Version → Save & Run All (Commit)** (runs unattended), or run cells 1–5 in order and
keep the tab open."""),
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
    ("code", """# 3. the cleaned training set in /kaggle/working (val/test untouched)
!python DATASET/scripts/build_tiles.py --src "{SRC}" --out /kaggle/working/v2b_clean_p --drop-sources seabed --drop-rotated
!echo "training set: $(ls /kaggle/working/v2b_clean_p/train/images | wc -l) images"
!df -h /kaggle/working | tail -1
"""),
    ("code", """# 4. launch both runs (parallel on 2 GPUs, else one after the other) — logs in /kaggle/working/logs
#    Run it ONCE: it refuses to start a second copy while a training is alive. Cell 5 shows the status.
import shlex, subprocess, torch
os.makedirs("/kaggle/working/logs", exist_ok=True)
RUNS = [  # name, training set, epochs, extra flags  (both: 640 px, batch 16, SGD lr0 0.01, fixed tiles, 2 classes)
    ("EXP-005", "v2b_clean_p", 150, ["--hsv-v", "0.4", "--scale", "0.6", "--seed", "42"]),
    ("EXP-005a", "v2b_clean_p", 150, ["--hsv-v", "0.4", "--scale", "0.6", "--seed", "7"]),
]
def cmd(name, data, epochs, dev, extra):
    return ["python", "-u", "src/detection/train.py", "--data", f"/kaggle/working/{data}/data.yaml", "--name", name,
            "--imgsz", "640", "--batch", "16", "--device", str(dev), "--workers", "2", "--cache", "none",
            "--optimizer", "SGD", "--lr0", "0.01", "--epochs", str(epochs), "--close-mosaic", "10",
            "--notes", f"Kaggle T4, 640px, {data}, SGD 0.01 {' '.join(extra)}"] + extra
live = subprocess.run(["bash", "-lc", "ps -eo args="], capture_output=True, text=True).stdout
assert "src/detection/train.py" not in live, "a training is already running - do not start it twice; run cell 5"
# start_new_session: the trainings get their own process group, so Stop/Interrupt on a cell cannot kill them
if torch.cuda.device_count() >= 2:
    for dev, (name, data, epochs, extra) in enumerate(RUNS):
        log = open(f"/kaggle/working/logs/{name}.log", "w")
        subprocess.Popen(cmd(name, data, epochs, dev, extra), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print("started", name, "on GPU", dev)
else:
    chain = " ; ".join(f"{shlex.join(cmd(n, d, e, 0, x))} > /kaggle/working/logs/{n}.log 2>&1" for n, d, e, x in RUNS)
    subprocess.Popen(["bash", "-c", chain], start_new_session=True)
    print("one GPU: the runs go one after the other:", ", ".join(n for n, *_ in RUNS))
print("now run cell 5: it shows what is running and waits until the models are packaged")
"""),
    ("code", Path(__file__).with_name("monitor_cell.py").read_text(encoding="utf-8")),
    ("markdown", """## Next — on your own machine

Download `EXP-005_complete.zip` and `EXP-005a_complete.zip` from the **Output** panel into the repo
root and tell Claude "EXP-005 done". The diagnosis comes first (validation per source, fit check,
paired comparison with EXP-003); test is scored once, only for the validation winner:

```bash
python -m src.detection.diagnose --zip EXP-003_complete.zip --zip EXP-005_complete.zip --zip EXP-005a_complete.zip
python -m src.detection.onboard_model --zip EXP-005_complete.zip --max-ms 700
```"""),
]


def as_source(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


nb = {"cells": [{"cell_type": t, "metadata": {}, "source": as_source(s), **({"outputs": [], "execution_count": None} if t == "code" else {})}
                for t, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("exp005_kaggle.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
