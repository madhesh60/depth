# ==== DEPTH TRAINING - STATUS / WAIT / COLLECT ====================================================
# Run it in a NEW cell, any time, as often as you like. It only LOOKS: it never starts, restarts or
# stops a training.
#   1. shows what each training is doing right now - read from the live process list and the run
#      files, so it works after a cell error or a kernel restart
#   2. while anything is still running: waits and prints a fresh status every CHECK_EVERY_MIN minutes
#   3. when nothing runs any more: copies the model packages to /kaggle/working (the Output panel)
# While it waits: keep this tab open, and do NOT press Stop/Interrupt or Restart. A Jupyter interrupt
# goes to the kernel's whole process group, which includes trainings launched without their own session.
import csv, json, os, re, shutil, subprocess, sys, time

WORK = "/kaggle/working"
RUNS_DIR, LOGS = f"{WORK}/depth/runs", f"{WORK}/logs"
NAMES = ["EXP-005", "EXP-005a"]          # the same recipe, seeds 42 and 7
EPOCHS = {"EXP-005": 150, "EXP-005a": 150}   # cell 4 (early stopping may end a run sooner)
CHECK_EVERY_MIN = 5
QUIET_WARN_MIN = 30                       # alive but nothing written for this long -> "may be stuck"


def sh(cmd):
    try:
        return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def processes():
    """({run name: [(pid, seconds alive)]} for live trainings, whether any train.py process is alive).
    Data-loader workers (children of a training) and launcher shells are not counted as trainings."""
    procs = []
    for line in sh("ps -eo pid=,ppid=,stat=,etimes=,args=").splitlines():
        f = line.split(None, 4)
        if len(f) == 5 and "src/detection/train.py" in f[4] and not f[2].startswith("Z"):
            procs.append(f)
    py = {f[0] for f in procs if os.path.basename(f[4].split()[0]).startswith("python")}
    runs = {}
    for pid, ppid, _, et, args in procs:
        tok = args.split()
        if pid not in py or ppid in py or "--name" not in tok[:-1]:
            continue
        runs.setdefault(tok[tok.index("--name") + 1], []).append((int(pid), int(et) if et.isdigit() else 0))
    return runs, bool(procs)


def epochs_of(n):
    """--epochs of the run, from the args.yaml ultralytics writes when it starts."""
    p = f"{RUNS_DIR}/{n}/args.yaml"
    if os.path.exists(p):
        for line in open(p):
            if line.startswith("epochs:"):
                return int(line.split(":")[1])
    return EPOCHS.get(n, 0) if isinstance(EPOCHS, dict) else EPOCHS


def rows(n):
    p = f"{RUNS_DIR}/{n}/results.csv"
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]


def log_tail(n, k=4):
    p = f"{LOGS}/{n}.log"
    if not os.path.exists(p):
        return ["(no log file)"]
    with open(p, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 16000))
        txt = f.read().decode("utf-8", "replace")
    txt = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", txt).replace("\r", "\n")
    return [l.strip() for l in txt.split("\n") if l.strip()][-k:] or ["(log is empty)"]


def quiet_seconds(n):
    """Seconds since the run last wrote anything (log, metrics, checkpoints, selection)."""
    paths = [f"{LOGS}/{n}.log", f"{RUNS_DIR}/{n}/results.csv"]
    for d in (f"{RUNS_DIR}/{n}/weights", f"{RUNS_DIR}/{n}/select"):
        if os.path.isdir(d):
            paths += [os.path.join(d, x) for x in os.listdir(d)]
    t = [os.path.getmtime(p) for p in paths if os.path.exists(p)]
    return time.time() - max(t) if t else None


def phase(n):
    """What a live run is doing, read from its files."""
    run = f"{RUNS_DIR}/{n}"
    if os.path.isdir(f"{run}/select"):
        w = f"{run}/weights"
        total = len([x for x in os.listdir(w) if x.endswith(".pt") and x != "best_depth.pt"]) if os.path.isdir(w) else 0
        return f"CHOOSING THE BEST CHECKPOINT (validating {len(os.listdir(f'{run}/select'))} of {total})"
    if os.path.exists(f"{run}/selection.csv") or os.path.exists(f"{run}/weights/best_depth.pt"):
        return "EXPORTING ONNX + PACKAGING (a few minutes)"
    return "TRAINING"


def gpus():
    out = sh("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")
    parts = []
    for line in out.splitlines():
        try:
            i, u, m, t = [x.strip() for x in line.split(",")]
            parts.append(f"GPU{i} {u}% busy, {float(m) / 1024:.1f}/{float(t) / 1024:.0f} GB")
        except ValueError:
            pass
    return " | ".join(parts) or "GPU info n/a"


def status():
    runs, any_alive = processes()
    print(time.strftime("\n=== %H:%M:%S (Kaggle clock) ==="), "|", gpus())
    for n in NAMES:
        r, alive = rows(n), runs.get(n, [])
        if alive:
            state = phase(n)
        elif os.path.exists(f"{RUNS_DIR}/{n}_complete.zip"):
            state = "FINISHED - package ready"
        elif not r and not os.path.exists(f"{LOGS}/{n}.log"):
            state = "NOT STARTED YET (queued)" if any_alive else "NOT STARTED"
        else:
            state = "STOPPED BEFORE FINISHING"
        print(f"{n:>9}: {state}")
        if r:
            last = r[-1]
            best = max(float(x.get("metrics/mAP50(B)") or 0) for x in r)
            info = (f"           epoch {len(r)}/{epochs_of(n)} | val mAP50 {float(last.get('metrics/mAP50(B)') or 0):.3f}"
                    f" (best so far {best:.3f}) | val recall {float(last.get('metrics/recall(B)') or 0):.3f}"
                    f" | train cls loss {float(last.get('train/cls_loss') or 0):.2f}")
            if state == "TRAINING":
                per = (float(last.get("time") or 0) or max(et for _, et in alive)) / len(r)
                left = max(0, epochs_of(n) - len(r)) * per / 60
                info += f" | {per / 60:.1f} min/epoch -> at most ~{left:.0f} min + ~5 min to choose/export"
            print(info)
        elif alive:
            print(f"           starting - no finished epoch yet ({max(et for _, et in alive) / 60:.0f} min since launch)")
        if len(alive) > 1:
            newest = [str(p) for p, _ in sorted(alive, key=lambda x: x[1])[:-1]]
            print(f"           !! {len(alive)} copies of {n} are running (cell 4 was run twice); they share one folder."
                  f" Stop the newest with:  !kill {' '.join(newest)}")
        if alive:
            q = quiet_seconds(n)
            if q is not None and q > QUIET_WARN_MIN * 60:
                print(f"           !! alive but nothing written for {q / 60:.0f} min - may be stuck (see GPU % above)")
            print("           log:", log_tail(n, 1)[0][:150])
        elif state == "STOPPED BEFORE FINISHING":
            print("           last log lines:")
            for l in log_tail(n, 8):
                print("             ", l[:170])
    sys.stdout.flush()
    return any_alive


alive = status()
while alive:
    print(f"   ... still running - next check in {CHECK_EVERY_MIN} min. Keep this tab open; do NOT press Stop.")
    sys.stdout.flush()
    time.sleep(CHECK_EVERY_MIN * 60)
    alive = status()

print("\n=== nothing is running any more - collecting the results ===")
ok = 0
for n in NAMES:
    z = f"{RUNS_DIR}/{n}_complete.zip"
    if not os.path.exists(z):
        print(f"!! {n}: NO package - it stopped before finishing. Last log lines:")
        for l in log_tail(n, 15):
            print("    ", l[:200])
        continue
    shutil.copy(z, f"{WORK}/{n}_complete.zip")
    mp = f"{RUNS_DIR}/{n}/model_meta.json"
    m = json.load(open(mp)) if os.path.exists(mp) else {}
    sel = m.get("selection") or {}
    pick, ub = sel.get("picked") or {}, sel.get("ultralytics_best") or {}
    print(f"OK {n}: {(m.get('best') or {}).get('epochs_run')} epochs in {m.get('train_minutes')} min | exported "
          f"{pick.get('checkpoint')} -> ghost_gear AP50 {pick.get('ghost_ap50')} on val (ultralytics' best.pt: "
          f"{ub.get('ghost_ap50')}) | {m.get('cv2_dnn_verified')}")
    ok += 1
if ok == len(NAMES):
    shutil.rmtree(f"{WORK}/v2b_tiles", ignore_errors=True)       # the tiled copy is not needed any more
print(sh(f"ls -la {WORK}/*_complete.zip 2>/dev/null") or "no packages in /kaggle/working")
if ok:
    print("\nDONE. Download each *_complete.zip: right-hand panel -> Output (/kaggle/working) -> ... -> Download.")
