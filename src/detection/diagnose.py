"""
diagnose.py — why a trained model is (not) good, before it is onboarded. VAL + TRAIN only, never test.

The deployed artifact (ONNX through ``cv2.dnn``) runs over:

* the held-out **validation** recordings, split **per source**: ghost-gear and wreck AP@0.5, the
  recall ceiling at conf 0.05, and TP/FP at conf 0.25;
* a fixed random sample of the model's **own training frames**. This fit check separates
  *underfit* (low AP even on frames it trained on) from a *generalisation gap* (high train AP, low
  val AP). EXP-002 was the first case: AP 0.42 on its training frames.

It also reads the run's own record from the package: the optimizer ultralytics actually built, the
final train / val class loss, and the epochs run. Optionally it includes a baseline model mapped to
the same two classes (EXP-001: ``fishing_gear`` → ghost_gear, ``pipe_cylinder`` /
``structural_fragment`` → wreck_debris). EXP-001 trained on v1, which contains the v2b validation
recordings, so its val numbers are an optimistic upper bound and not a fair target.

    python -m src.detection.diagnose --zip EXP-003_complete.zip --zip EXP-003f_complete.zip
    python -m src.detection.diagnose --zip EXP-002_complete.zip --train-sample 200 --out docs/exp002_diagnosis_tables.md
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.detection.evaluate import _load_gt, _match_image, voc_ap   # noqa: E402
from src.detection.infer import YoloOnnxDetector                     # noqa: E402

V2B = REPO / "DATASET" / "03_yolo_ready_dataset_v2b"
TO_V2B = {"ghost_gear": 0, "fishing_gear": 0, "wreck_debris": 1, "pipe_cylinder": 1, "structural_fragment": 1}
CLASSES = ("ghost_gear", "wreck_debris")
# Verdict thresholds use the product's loose match (IoU 0.3): with 14-36 px boxes the label boxes
# themselves disagree by a few pixels, so even a working model cannot score high at IoU 0.5 on them.
# Calibrated on the reference BEFORE any EXP-003 result existed (2026-10-02): on 200 v2b training
# frames, EXP-001 (a working model) scores ghost AP@0.3 0.71 (AP@0.5 0.57); underfit EXP-002 0.57.
UNDERFIT_AP = 0.65           # ghost AP@0.3 on the model's own training frames below this = underfit
GAP_AP = 0.30                # train AP@0.3 - val AP@0.3 above this = generalisation gap


def group(name: str) -> str:
    """Source of a v2b frame: crab-pot recording, other crab-pot survey, or the other datasets."""
    low = name.lower()
    if low.startswith("rec"):
        return "crabpot_" + name.split("_")[0].replace("Rec0", "Rec")
    for p in ("bc_post", "baycove"):
        if low.startswith(p):
            return "crabpot_" + p
    for p in ("shipwreck", "seabed", "mpulse"):
        if low.startswith(p):
            return p + ("_natform" if "natform" in low else "")
    return "crabpot_other" if low.startswith(("ti0", "mc0", "bb_")) else name.split("_")[0]


class Model:
    def __init__(self, name: str, onnx: Path, imgsz: int, names: list[str], record: dict | None = None):
        self.name, self.onnx, self.imgsz, self.names, self.record = name, onnx, imgsz, names, record or {}
        self.cmap = {i: TO_V2B[n] for i, n in enumerate(names) if n in TO_V2B}

    @classmethod
    def from_zip(cls, zp: Path, work: Path) -> "Model":
        z = zipfile.ZipFile(zp)
        meta_name = next(n for n in z.namelist() if n.endswith("model_meta.json"))
        run = meta_name.split("/")[0]
        meta = json.loads(z.read(meta_name))
        out = work / run
        if not (out / "weights" / "best.onnx").exists():
            for n in z.namelist():
                if n.endswith(("best.onnx", "model_meta.json", "results.csv")):
                    z.extract(n, work)
        rows = list(csv.DictReader(io.StringIO(z.read(f"{run}/results.csv").decode()))) if f"{run}/results.csv" in z.namelist() else []
        rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in rows]
        last = rows[-1] if rows else {}
        record = {"optimizer_built": meta.get("optimizer_built") or "not recorded (optimizer=auto)",
                  "epochs_run": len(rows), "train_minutes": meta.get("train_minutes"),
                  "final_train_cls_loss": last.get("train/cls_loss"), "final_val_cls_loss": last.get("val/cls_loss"),
                  "exported": ((meta.get("selection") or {}).get("picked") or {}).get("checkpoint")}
        return cls(run, out / "weights" / "best.onnx", int(meta["imgsz"]), list(meta["names"]), record)

    @classmethod
    def baseline(cls, model: str = "EXP-001") -> "Model | None":
        cal = REPO / "models" / model / "calibration.json"
        onnx = REPO / "runs" / model / "weights" / "best.onnx"
        if not (cal.exists() and onnx.exists()):
            return None
        c = json.loads(cal.read_text(encoding="utf-8"))
        return cls(model, onnx, int(c.get("imgsz", 640)), list(c["names"]),
                   {"note": "trained on v1, which contains the v2b val recordings: optimistic upper bound"})


def frames(split: str, sample: int = 0, seed: int = 0) -> list[Path]:
    ims = sorted((V2B / split / "images").iterdir())
    if sample:
        random.Random(seed).shuffle(ims)
        ims = ims[:sample]
    return ims


def run(model: Model, split: str, ims: list[Path], cache_dir: Path) -> list[dict]:
    """Per image: per-class labelled PR points at IoU 0.5 (``pc``) and at the product's loose match,
    IoU 0.3 (``pc30``: 14-36 px boxes fail 0.5 on a few pixels), + n_gt; cached per model/split/sample."""
    key = f"v2|{model.onnx}|{model.imgsz}|{len(ims)}|{ims[0].name if ims else ''}"
    cache = cache_dir / f"{model.name}_{split}_{len(ims)}.json"
    if cache.exists():
        blob = json.loads(cache.read_text())
        if blob.get("key") == key:
            return blob["records"]
    det = YoloOnnxDetector(model.onnx, names=model.names, imgsz=model.imgsz, conf_thres=0.001)
    recs, t0 = [], time.time()
    for ip in ims:
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        dets = [(model.cmap[d.cls_id], d.conf, d.bbox) for d in det.detect(img) if d.cls_id in model.cmap]
        gts = _load_gt(V2B / split / "labels" / f"{ip.stem}.txt", w, h)
        pc, pc30 = {}, {}
        for c in (0, 1):
            dc, gc = [(cf, b) for k, cf, b in dets if k == c], [b for k, b in gts if k == c]
            pc[str(c)] = {"pts": _match_image(dc, gc, 0.5), "n_gt": len(gc)}
            pc30[str(c)] = {"pts": _match_image(dc, gc, 0.3), "n_gt": len(gc)}
        recs.append({"name": ip.name, "group": group(ip.name), "pc": pc, "pc30": pc30})
    print(f"  {model.name} {split}: {len(recs)} images in {time.time() - t0:.0f}s", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "records": recs}))
    return recs


def stats(recs: list[dict], c: int) -> dict:
    pts = [p for r in recs for p in r["pc"][str(c)]["pts"]]
    n = sum(r["pc"][str(c)]["n_gt"] for r in recs)
    pts30 = [p for r in recs if "pc30" in r for p in r["pc30"][str(c)]["pts"]]
    return {"n_gt": n, "ap50": voc_ap(pts, n) if n else float("nan"),
            "ap30": voc_ap(pts30, n) if n and pts30 else float("nan"),
            "ceiling": sum(t for cf, t in pts if cf >= 0.05) / n if n else float("nan"),
            "tp25": sum(t for cf, t in pts if cf >= 0.25), "fp25": sum(1 - t for cf, t in pts if cf >= 0.25)}


def _cell(s: dict) -> str:
    return (f"{s['ap50']:.3f} / {s['ap30']:.2f} · {s['ceiling']:.2f} · {s['tp25']}/{s['fp25']}" if s["n_gt"]
            else f"— · — · 0/{s['fp25']}")


def table(title: str, models: list[Model], results: dict, split: str, c: int, per_source: bool = True) -> list[str]:
    groups = sorted({r["group"] for r in results[(models[0].name, split)]})
    L = [f"### {title}", "", "AP@0.5 / AP@0.3 (the product's loose match) · recall ceiling @0.05 · TP/FP @0.25", "",
         "| source | images | " + CLASSES[c] + " boxes | " + " | ".join(m.name for m in models) + " |",
         "|---|--:|--:|" + "---|" * len(models)]
    for g in ["**all**"] + (groups if per_source else []):
        sel = lambda m: [r for r in results[(m.name, split)] if g == "**all**" or r["group"] == g]
        ss = [stats(sel(m), c) for m in models]
        if not ss[0]["n_gt"] and not any(s["fp25"] for s in ss) and g != "**all**":
            continue
        L.append(f"| {g} | {len(sel(models[0]))} | {ss[0]['n_gt']} | " + " | ".join(_cell(s) for s in ss) + " |")
    return L + [""]


def verdict(m: Model, results: dict) -> str:
    tr, va = stats(results[(m.name, "train")], 0), stats(results[(m.name, "val")], 0)
    if tr["ap30"] < UNDERFIT_AP:
        v = (f"**UNDERFIT**: ghost AP@0.3 {tr['ap30']:.2f} on its own training frames (< {UNDERFIT_AP}; a working "
             f"model scores 0.71). Train longer, or check the optimizer and the labels, before anything else.")
    elif tr["ap30"] - va["ap30"] > GAP_AP:
        v = (f"**GENERALISATION GAP**: ghost AP@0.3 train {tr['ap30']:.2f} vs val {va['ap30']:.2f}. It fits but does "
             f"not transfer to new recordings: more varied data / augmentation, or less capacity.")
    else:
        v = (f"fits (train AP@0.3 {tr['ap30']:.2f}) and transfers (val AP@0.3 {va['ap30']:.2f}, AP@0.5 "
             f"{va['ap50']:.2f}): ready for onboarding.")
    return v


def main():
    ap = argparse.ArgumentParser(description="Diagnose trained models on val + their own training frames (never test).")
    ap.add_argument("--zip", action="append", default=[], help="a <name>_complete.zip package (repeatable)")
    ap.add_argument("--baseline", default="EXP-001", help="baseline model mapped to v2b classes ('' = none)")
    ap.add_argument("--train-sample", type=int, default=200, help="random training frames for the fit check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="markdown report (default docs/diagnose_<names>.md)")
    a = ap.parse_args()
    work = REPO / "runs" / "_diagnose"
    models = [Model.from_zip(Path(z), work) for z in a.zip]
    base = Model.baseline(a.baseline) if a.baseline else None
    if base:
        models.insert(0, base)
    if not models:
        raise SystemExit("nothing to diagnose: pass --zip <name>_complete.zip")
    splits = {"val": frames("val"), "train": frames("train", a.train_sample, a.seed)}
    results = {(m.name, s): run(m, s, ims, work / "cache") for m in models for s, ims in splits.items()}

    names = [m.name for m in models if m is not base]
    L = [f"# Diagnosis: {', '.join(names)}", "",
         f"_`python -m src.detection.diagnose` · {time.strftime('%Y-%m-%d %H:%M')} · deploy path (ONNX via cv2.dnn) · "
         f"v2b val ({len(splits['val'])} frames) + {len(splits['train'])} random training frames (seed {a.seed}) · "
         f"test is not touched_", "", "## Verdict", ""]
    L += [f"- **{m.name}**: {verdict(m, results)}" for m in models if m is not base]
    L += ["", "## The run's own record", "", "| model | optimizer built | epochs | minutes | final train cls loss | "
          "final val cls loss | exported |", "|---|---|--:|--:|--:|--:|---|"]
    for m in models:
        r = m.record
        L.append(f"| {m.name} | {json.dumps(r.get('optimizer_built')) if isinstance(r.get('optimizer_built'), dict) else r.get('optimizer_built', r.get('note', ''))} | "
                 f"{r.get('epochs_run', '')} | {r.get('train_minutes', '')} | {r.get('final_train_cls_loss', '')} | "
                 f"{r.get('final_val_cls_loss', '')} | {r.get('exported', '')} |")
    L.append("")
    L += table("Validation — ghost gear (held-out recordings Rec10/12/16)", models, results, "val", 0)
    L += table("Validation — wreck debris", models, results, "val", 1)
    L += table("Fit check — ghost gear on the model's own training frames", models, results, "train", 0, per_source=False)
    L += table("Fit check — wreck debris on the model's own training frames", models, results, "train", 1, per_source=False)
    if base:
        L += [f"_{base.name} {base.record['note']}._", ""]
    out = Path(a.out) if a.out else REPO / "docs" / f"diagnose_{'_'.join(names)}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
