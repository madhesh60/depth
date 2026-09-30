# ==== DEPTH EXP-002 - STATUS / WAIT / COLLECT ====================================================
# Safe to run at any time, any number of times. It NEVER starts or restarts a training.
# 1) shows whether each training is RUNNING / FINISHED / STOPPED (reads the real process list + files,
#    so it works even after a cell error or a kernel restart)
# 2) if any training is still running: waits, printing progress every few minutes (+ time left)
# 3) when both are done: copies the model packages to /kaggle/working and says what was exported
import os, csv, json, shutil, subprocess, time

WORK = "/kaggle/working"
RUNS_DIR, LOGS = f"{WORK}/depth/runs", f"{WORK}/logs"
NAMES = ["EXP-002", "EXP-002s"]          # 1024 px and 640 px
EPOCHS = 30
CHECK_EVERY_MIN = 5


def sh(cmd):
    return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True).stdout.strip()


def secs(etime):                          # ps etime: [[dd-]hh:]mm:ss
    d = 0
    if "-" in etime:
        d, etime = etime.split("-")
    p = [int(x) for x in etime.split(":")]
    while len(p) < 3:
        p.insert(0, 0)
    return int(d) * 86400 + p[0] * 3600 + p[1] * 60 + p[2]


def running():
    """{run name: seconds running} for training processes that are alive right now."""
    out, alive = sh("ps -eo etime,args | grep 'src/detection/train.py' | grep -v grep"), {}
    for line in out.splitlines():
        et, args = (line.strip().split(None, 1) + [""])[:2]
        for n in NAMES:
            if f"--name {n} " in args + " ":
                alive[n] = max(alive.get(n, 0), secs(et))
    return alive


def epochs(n):
    p = f"{RUNS_DIR}/{n}/results.csv"
    if not os.path.exists(p):
        return []
    return [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(p))]


def log_tail(n, k=4):
    p = f"{LOGS}/{n}.log"
    if not os.path.exists(p):
        return ["(no log file)"]
    with open(p, "rb") as f:
        f.seek(0, 2); f.seek(max(0, f.tell() - 8000))
        txt = f.read().decode("utf-8", "replace")
    return [l.strip() for l in txt.replace("\r", "\n").split("\n") if l.strip()][-k:]


def status():
    alive = running()
    print(time.strftime("\n=== %H:%M:%S ==="), "| GPUs:", sh("nvidia-smi --query-gpu=index,utilization.gpu,memory.used "
                                                          "--format=csv,noheader | tr '\\n' ';'") or "n/a")
    for n in NAMES:
        rows = epochs(n)
        done = os.path.exists(f"{RUNS_DIR}/{n}_complete.zip")
        state = "FINISHED (package ready)" if done else ("RUNNING" if n in alive else "STOPPED")
        line = f"{n:>9}: {state}"
        if rows:
            last = rows[-1]
            best = max(float(r["metrics/mAP50(B)"]) for r in rows)
            line += (f" | epoch {len(rows)}/{EPOCHS} | val mAP50 {float(last['metrics/mAP50(B)']):.3f}"
                     f" (best {best:.3f}) | recall {float(last['metrics/recall(B)']):.3f}")
            if n in alive and not done:
                per = alive[n] / max(1, len(rows))
                left = max(0, EPOCHS - len(rows)) * per / 60
                line += f" | running {alive[n] / 3600:.1f} h, ~{left:.0f} min of training left + ~15 min selection"
        elif n in alive:
            line += f" | starting (running {alive[n] / 60:.0f} min, no finished epoch yet)"
        print(line)
        if state == "STOPPED":
            print("           last log lines:")
            for l in log_tail(n, 6):
                print("             ", l[:160])
    return alive


alive = status()
while alive:
    time.sleep(CHECK_EVERY_MIN * 60)
    alive = status()

print("\n=== no training process is running any more - collecting ===")
ok = 0
for n in NAMES:
    z = f"{RUNS_DIR}/{n}_complete.zip"
    if not os.path.exists(z):
        print(f"!! {n}: no package. It stopped before finishing - last log lines:")
        for l in log_tail(n, 15):
            print("   ", l[:200])
        continue
    shutil.copy(z, f"{WORK}/{n}_complete.zip")
    m = json.load(open(f"{RUNS_DIR}/{n}/model_meta.json"))
    sel = (m.get("selection") or {}).get("picked") or {}
    ub = (m.get("selection") or {}).get("ultralytics_best") or {}
    print(f"OK {n}: trained {m.get('train_minutes')} min | exported {sel.get('checkpoint')} "
          f"-> ghost_gear AP50 {sel.get('ghost_ap50')} (ultralytics' own pick: {ub.get('ghost_ap50')}) | "
          f"{m.get('cv2_dnn_verified')}")
    ok += 1
if ok == len(NAMES):
    shutil.rmtree(f"{WORK}/v2b_tiles", ignore_errors=True)
print(sh(f"ls -la {WORK}/*_complete.zip 2>/dev/null") or "no packages in /kaggle/working")
print("\nDownload the *_complete.zip files: right-hand panel -> Output (/kaggle/working) -> ... -> Download.")
