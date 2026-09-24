"""
onboard_model.py — ONE command to plug a freshly trained detector into DEPTH (review I-3 / I-5).

    python -m src.detection.onboard_model --zip EXP-002_complete.zip
    python -m src.detection.onboard_model --run-dir runs/EXP-002            # already unpacked

Steps (each is the existing, tested tool — this only wires them in the right order):

1. **unpack** ``<name>_complete.zip`` (written by ``train.py``) → ``runs/<name>/``;
2. **verify** the ONNX: sha256 matches ``model_meta.json``; ``cv2.dnn`` loads it and a forward pass
   at the trained ``imgsz`` returns ``(1, 4 + nc, N)`` — the exact deploy path;
3. **register** it: ``models/<name>/best.onnx`` (git-ignored) + ``models/<name>/model_meta.json`` and a
   first ``calibration.json`` (names, imgsz, per-class floors, guaranteed class = class 0);
4. **calibrate** the guaranteed tiers on the dataset's held-out validation recordings and verify
   them ONCE on test (``src.agentic.calibrate``; v2b: val = Rec10/12/16, test = unique crab-pot
   frames) → recall/precision promises, P(pot) bins, the agent-vs-confidence comparison;
5. **evaluate** deploy-faithfully (``src.detection.evaluate``): v2b unique-frame test, the official
   398-frame split (GhostVision head-to-head — leakage-free for v2b-trained models) and the
   cross-sonar ``test_xsonar`` (orange Contact crops), thresholds tuned on val only;
6. **report** ``docs/onboard_<name>.md`` — the numbers next to EXP-001's, and how to switch the
   app (``DEPTH_MODEL=<name>``) locally and on the COOL server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V2B = REPO / "DATASET" / "03_yolo_ready_dataset_v2b"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_onnx(onnx: Path, imgsz: int, nc: int) -> str:
    import cv2
    import numpy as np
    net = cv2.dnn.readNetFromONNX(str(onnx))
    net.setInput(cv2.dnn.blobFromImage(np.zeros((imgsz, imgsz, 3), np.uint8), 1 / 255.0, (imgsz, imgsz), swapRB=True))
    out = net.forward()
    if out.ndim != 3 or out.shape[0] != 1 or out.shape[1] != 4 + nc:
        raise SystemExit(f"cv2.dnn output {out.shape} != (1, {4 + nc}, N) - wrong model/imgsz, not onboarded")
    return f"cv2.dnn {cv2.__version__}: forward OK at {imgsz}px, output {tuple(out.shape)}"


def _run(cmd: list[str], env: dict, log: Path) -> int:
    print("  $", " ".join(cmd))
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"    -> exit {r.returncode} (log {log.relative_to(REPO)})")
    return r.returncode


def _metric(path: Path, *keys):
    try:
        d = json.loads(path.read_text())
        for k in keys:
            d = d[k]
        return d
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip", help="<name>_complete.zip from train.py")
    g.add_argument("--run-dir", help="an unpacked runs/<name> directory")
    ap.add_argument("--data-root", default=str(V2B), help="dataset with val/ test/ test_official398/ test_xsonar/")
    ap.add_argument("--pattern", default="*wcp_ss_*.jpg", help="the product's sonogram frames (calibration)")
    ap.add_argument("--skip-calibrate", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap frames per split (smoke test only)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()

    # 1. unpack --------------------------------------------------------------------------------
    if a.zip:
        z = Path(a.zip)
        with zipfile.ZipFile(z) as zf:
            top = sorted({n.split("/")[0] for n in zf.namelist() if "/" in n})
            if len(top) != 1:
                raise SystemExit(f"unexpected zip layout: {top}")
            zf.extractall(REPO / "runs")
        run = REPO / "runs" / top[0]
    else:
        run = Path(a.run_dir).resolve()
    meta_p = run / "model_meta.json"
    if not meta_p.exists():
        raise SystemExit(f"{meta_p} missing - was this produced by src/detection/train.py?")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    name, names, imgsz = meta["name"], meta["names"], int(meta["imgsz"])
    onnx = run / "weights" / "best.onnx"
    print(f"== onboarding {name}: classes {names}, imgsz {imgsz}")

    # 2. verify --------------------------------------------------------------------------------
    if not onnx.exists():
        raise SystemExit(f"{onnx} missing (train.py exports it unless --no-export)")
    sha = _sha256(onnx)
    if meta.get("onnx_sha256") and meta["onnx_sha256"] != sha:
        raise SystemExit("ONNX sha256 does not match model_meta.json - corrupted download?")
    msg = verify_onnx(onnx, imgsz, len(names))
    print("  ", msg)

    # 3. register ------------------------------------------------------------------------------
    mdir = REPO / "models" / name
    mdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx, mdir / "best.onnx")
    meta.update({"onnx_sha256": sha, "onboarded": time.strftime("%Y-%m-%dT%H:%M:%S"), "cv2_dnn_check": msg})
    (mdir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    g0 = names[0]
    from src.detection.calibration import save_calibration
    save_calibration({"model": name, "names": names,
                      "detector": {"imgsz": imgsz, "iou_nms": 0.45, "class_aware_nms": True,
                                   "conf": {n: (0.05 if n == g0 else 0.25) for n in names},
                                   "relook_conf": 0.05, "input": "raw"},
                      "tiers": {"guaranteed_class": g0, "non_hazard_classes": []}}, model=name)
    print(f"   registered models/{name}/ (best.onnx + model_meta.json + calibration.json)")

    env = dict(os.environ, DEPTH_MODEL=name, DEPTH_ONNX=str(mdir / "best.onnx"), PYTHONIOENCODING="utf-8")
    log = REPO / "runs" / name / "onboard.log"
    root = Path(a.data_root)
    tag = name.lower().replace("-", "")
    lim = ["--limit", str(a.limit)] if a.limit else []

    # 4. calibrate the guaranteed tiers (val recordings → verified once on test) ---------------
    cal_rc = None
    if not a.skip_calibrate:
        cal_rc = _run([sys.executable, "-m", "src.agentic.calibrate", "--model", name, "--onnx", str(mdir / "best.onnx"),
              "--cal-root", str(root / "val"), "--ver-root", str(root / "test"), "--pattern", a.pattern,
              "--cls-name", g0, "--cls-id", "0"] + lim, env, log)

    # 5. deploy-faithful evaluation -------------------------------------------------------------
    evals = {}
    if not a.skip_eval:
        common = [sys.executable, "-m", "src.detection.evaluate", "--weights", str(mdir / "best.onnx"),
                  "--root", str(root), "--names", ",".join(names), "--model", name,
                  "--bootstrap", str(a.bootstrap)] + lim
        for split, extra in (("test", []),
                             ("test_official398", ["--official-split", str(root / "official_crabpot_test.txt"),
                                                   "--leakage-free"]),
                             ("test_xsonar", [])):
            if not (root / split / "images").exists():
                continue
            out = REPO / "docs" / f"eval_{tag}_{split}.md"
            if _run(common + ["--test-split", split, "--out", str(out)] + extra, env, log) == 0:
                evals[split] = out

    # 6. report ---------------------------------------------------------------------------------
    cal = json.loads((mdir / "calibration.json").read_text(encoding="utf-8"))
    gtee = (cal.get("tiers") or {}).get("guarantees") or {}
    old = json.loads((REPO / "models" / "EXP-001" / "calibration.json").read_text(encoding="utf-8"))
    ogt = old["tiers"]["guarantees"]
    # gate: say loudly when the new model must NOT replace the demo model
    warn = []
    rp_new, rp_old = gtee.get("recall_promise"), ogt.get("recall_promise")
    if cal_rc not in (None, 0):
        warn.append(f"**CALIBRATION FAILED (exit {cal_rc})** - see `runs/{name}/onboard.log`; the agent would run "
                    f"this model on the legacy rules (no calibrated promise). Do not switch the demo.")
    elif not a.skip_calibrate and (rp_new is None or rp_new < 0.3):
        warn.append(f"**Recall promise too weak ({rp_new})** - the detector does not reach enough pots on the "
                    f"calibration recordings. Do not switch the demo.")
    elif rp_new is not None and rp_old is not None and rp_new < rp_old:
        warn.append(f"Recall promise {rp_new} is below EXP-001's {rp_old} (different splits - compare the v2b "
                    f"evaluation reports before switching).")
    L = [f"# Onboarding {name}", "", *[f"> {w}" for w in warn], *([""] if warn else []),
         f"_`python -m src.detection.onboard_model` · {time.strftime('%Y-%m-%d %H:%M')}_", "",
         f"- classes `{names}` · input {imgsz}px · ONNX sha256 `{sha[:16]}…` · {msg}",
         f"- trained: best epoch {meta.get('best', {}).get('epoch')} of {meta.get('best', {}).get('epochs_run')} "
         f"(val mAP50 {meta.get('best', {}).get('val_mAP50')}), {meta.get('train_minutes')} min; data `{Path(meta.get('data', '')).parent.name}`",
         "", "## Guarantees (calibrated on held-out val recordings, verified once on test)", "",
         "| | EXP-001 (v1 val/test) | " + name + " (v2b val/test) |", "|---|--:|--:|",
         f"| recall ceiling at the detector floor (calibration) | {ogt.get('recall_ceiling')} | {gtee.get('recall_ceiling')} |",
         f"| recall promise (95%) | ≥ {ogt.get('recall_promise')} | ≥ {gtee.get('recall_promise')} |",
         f"| requested 90% achievable | {ogt.get('requested_recall_achievable')} | {gtee.get('requested_recall_achievable')} |",
         f"| precision promise (auto-confirm) | {ogt.get('precision_promise')} | {gtee.get('precision_promise')} |",
         f"| held on test | {(ogt.get('verified_on_test') or {}).get('recall_promise_held')} | "
         f"{(gtee.get('verified_on_test') or {}).get('recall_promise_held')} |",
         "", "_Different splits: EXP-001 is scored on its own unseen frames (v1); the fair model-vs-model "
         "comparison is the v2b test below, where EXP-001 is NOT leakage-free (v1 contains those recordings)._", "",
         "## Evaluation reports", ""]
    L += [f"- `{s}` → [{p.relative_to(REPO).as_posix()}]({p.relative_to(REPO).as_posix()})" for s, p in evals.items()] or ["- (skipped)"]
    L += ["", "## Switch the product to this model", "",
          "```bash", f"DEPTH_MODEL={name} python -m uvicorn src.dashboard.app:app --port 8000   # local", "```",
          f"COOL server: `aws s3 cp models/{name}/best.onnx s3://<bucket>/models/{name}/best.onnx`, then set "
          f"`DEPTH_MODEL={name}` in `infra/depth.service` (or re-run `setup_cool_instance.sh` with `DEPTH_MODEL={name}`).",
          "", "Log every number above in `experiments.md` (EXP-002 entry) before switching the demo.", ""]
    rp = REPO / "docs" / f"onboard_{tag}.md"
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"\n== done -> {rp.relative_to(REPO)}")
    for w in warn:
        print("!! " + w.replace("**", ""))
    print(f"   switch: DEPTH_MODEL={name}" + ("   (NOT recommended - see warnings)" if warn else ""))


if __name__ == "__main__":
    main()
