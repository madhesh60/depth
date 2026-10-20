"""evaluate.py — honest, deploy-faithful evaluation of the Stage-2 ONNX detector.

Runs the **exact artifact that ships** (``best.onnx`` via ``cv2.dnn`` / ``YoloOnnxDetector``
— no torch) over a YOLO split and reports, in one pass, the baseline metrics plus four evaluation
gaps an early project review found:

* **per-class AP@0.5** (VOC all-points), plus P/R/F1 (the baseline)
* **VAL-tuned per-class confidence thresholds** — the test split is scored ONCE
* **per-source and per-sensor tables** — the aggregate mAP hides the truth
* **bootstrap 95% CIs** on the headline metrics
* **GhostVision head-to-head** on the official crab-pot split
  — *gated on a leakage-free model*. EXP-001 trained on v1, which mixed the official
  crab-pot test frames into training, so the head-to-head is **SKIPPED** for it (printing a
  leaked number would be dishonest). v2b models (EXP-002+) respect the split.
* **COCO-style mAP@0.5:0.95**, a **confusion matrix** at the val-tuned thresholds and a
  **PR curve** image (``docs/img/pr_<model>_<split>.png``).

Design notes (why this is trustworthy):
  * Same ``cv2.dnn`` decode/NMS path as the studio and the MCP server, so the
    numbers describe the deployed system — not a torch re-run that never ships.
  * Detections are collected once at a low floor (conf 0.001) and cached; every threshold,
    table and bootstrap replicate is computed from that cache (fast, deterministic).
  * AP is Pascal-VOC all-points at IoU 0.5. This is an *independent* implementation, so values
    may differ by a hair from ultralytics' 101-point interpolation — the purpose here is
    val-tuned thresholds, per-domain tables and CIs, not to reproduce a single headline number.

Usage (defaults target the DEPLOYED model — ``$DEPTH_MODEL``, else ``calibration.SHIPPED_MODEL``
— its weights under ``models/<MODEL>/`` (``python -m src.detection.fetch_model``), its class names,
input size and thresholds from ``models/<MODEL>/calibration.json``, and the dataset it was scored on):
    python -m src.detection.evaluate                                # EXP-003 on v2b test
    python -m src.detection.evaluate --test-split test_official398 --official-split         DATASET/03_yolo_ready_dataset_v2b/official_crabpot_test.txt --leakage-free
    python -m src.detection.evaluate --model EXP-001                # the earlier 4-class baseline
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.detection.calibration import DEFAULT_MODEL, load_calibration
from src.detection.infer import YoloOnnxDetector, _iou

REPO = Path(__file__).resolve().parents[2]
DATASET_V1 = REPO / "DATASET" / "03_yolo_ready_dataset_v1"
DATASET_V2B = REPO / "DATASET" / "03_yolo_ready_dataset_v2b"
LEAKED_ON_OFFICIAL = {"EXP-001"}        # trained on v1, which mixed the official crab-pot test in
IOU_COCO = np.round(np.arange(0.50, 0.95 + 1e-9, 0.05), 2)
CACHE_VERSION = 2                       # v2 caches keep raw dets + GT (mAP@0.5:0.95, confusion)


def default_weights(model: str) -> Path:
    """``models/<MODEL>/best.onnx`` (where ``fetch_model`` installs it), else the training run."""
    for p in (REPO / "models" / model / "best.onnx", REPO / "runs" / model / "weights" / "best.onnx"):
        if p.exists():
            return p
    return REPO / "models" / model / "best.onnx"


def default_root(model: str) -> Path:
    """The dataset a model is scored on: EXP-001 → v1 (its 4 classes), every later model → v2b."""
    return DATASET_V1 if model == "EXP-001" else DATASET_V2B

# Source is the filename prefix (crabpot_..., uatd_..., icra_...). Sensor follows from source.
SONAR_SOURCES = {"crabpot", "mpulse", "seabed", "shipwreck", "uatd"}
OPTICAL_SOURCES = {"icra", "vid", "trashcan"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


_CRABPOT = ("rec", "bc_post", "baycove", "ti0")


def source_of(name: str) -> str:
    """Filename prefix (v1: ``crabpot_…``, ``uatd_…``). v2b drops the ``crabpot_`` prefix, so its
    PINGMapper recordings / mosaics are mapped back to ``crabpot`` and the orange Contact crops to
    ``crabpot_xsonar`` (the cross-sonar test)."""
    low = name.lower()
    if low.startswith("contact_"):
        return "crabpot_xsonar"
    if low.startswith(_CRABPOT):
        return "crabpot"
    return name.split("_", 1)[0]


def _strip_ext(name: str) -> str:
    """Drop a trailing image extension only — ``Path.stem`` would also eat the Roboflow
    ``.rf.<hash>`` part of names listed without an extension (0/398 official frames matched)."""
    for e in IMG_EXTS:
        if name.lower().endswith(e):
            return name[: -len(e)]
    return name


def sensor_of(src: str) -> str:
    if src in OPTICAL_SOURCES:
        return "optical"
    return "sonar"  # default sonar (all remaining prefixes are sonar sources)


# ----------------------------------------------------------------------------- data model
def _load_gt(label_path: Path, w: int, h: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Read a YOLO label file → [(cls_id, (x1,y1,x2,y2) in pixels)]. Missing file = background."""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        cx, cy, bw, bh = (float(x) for x in p[1:5])
        x1 = int(round((cx - bw / 2) * w))
        y1 = int(round((cy - bh / 2) * h))
        x2 = int(round((cx + bw / 2) * w))
        y2 = int(round((cy + bh / 2) * h))
        out.append((c, (x1, y1, x2, y2)))
    return out


def _match_image(dets_cls, gts_cls, iou_thr: float) -> list[tuple[float, int]]:
    """Greedy descending-confidence matching for ONE class in ONE image.

    Returns [(conf, tp)] per detection (tp=1 if it claims an unused GT at IoU>=thr, else 0).
    Within-image labels are stable under later confidence thresholding, so they are computed
    once and reused for AP, every P/R/F1 threshold, and the bootstrap.
    """
    dets_cls = sorted(dets_cls, key=lambda d: -d[0])  # (conf, box) desc by conf
    used = [False] * len(gts_cls)
    labeled = []
    for conf, box in dets_cls:
        best_iou, best_j = 0.0, -1
        for j, gbox in enumerate(gts_cls):
            if used[j]:
                continue
            i = _iou(box, gbox)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_j >= 0 and best_iou >= iou_thr:
            used[best_j] = True
            labeled.append((conf, 1))
        else:
            labeled.append((conf, 0))
    return labeled


def collect_split(detector: YoloOnnxDetector, split_dir: Path, names, cache: Path,
                  iou_thr: float = 0.5, limit: int = 0) -> list[dict]:
    """Run the detector over a split → per-image, per-class labeled PR points + n_gt.

    Cached to JSON keyed on the ONNX path so re-runs (and the bootstrap) are instant.
    """
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if limit:
        imgs = imgs[:limit]
    key = (f"v{CACHE_VERSION}|{detector_net_id(detector)}|{len(imgs)}|iou{iou_thr}"
           f"|floor{detector.conf_floor}|imgsz{detector.imgsz}")
    if cache.exists():
        blob = json.loads(cache.read_text())
        if blob.get("key") == key:
            print(f"  [cache] {split_dir.name}: {len(blob['records'])} images")
            return blob["records"]

    records = []
    t0 = time.time()
    for k, ip in enumerate(imgs):
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        dets = detector.detect(img)                       # conf >= floor, NMS applied
        gts = _load_gt(lbl_dir / f"{ip.stem}.txt", w, h)
        src = source_of(ip.name)
        per_class = {}
        for cid in range(len(names)):
            d_c = [(d.conf, d.bbox) for d in dets if d.cls_id == cid]
            g_c = [b for c, b in gts if c == cid]
            per_class[str(cid)] = {
                "pts": _match_image(d_c, g_c, iou_thr),
                "n_gt": len(g_c),
            }
        records.append({"name": ip.name, "source": src, "sensor": sensor_of(src),
                        "per_class": per_class,
                        "dets": [[d.cls_id, round(float(d.conf), 5), *map(int, d.bbox)] for d in dets],
                        "gts": [[c, *b] for c, b in gts]})
        if (k + 1) % 200 == 0:
            print(f"  {split_dir.name}: {k + 1}/{len(imgs)}  ({time.time() - t0:.0f}s)")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "records": records}))
    print(f"  {split_dir.name}: {len(records)} images in {time.time() - t0:.0f}s -> {cache.name}")
    return records


def detector_net_id(detector: YoloOnnxDetector) -> str:
    return str(getattr(detector, "_onnx_path", "onnx"))


# ----------------------------------------------------------------------------- metrics (pure)
def _gather(records, cid: int):
    """Concatenate labeled PR points and total n_gt for class `cid` over a set of images."""
    pts, n_gt = [], 0
    for r in records:
        pc = r["per_class"][str(cid)]
        pts.extend(pc["pts"])
        n_gt += pc["n_gt"]
    return pts, n_gt


def voc_ap(pts, n_gt: int) -> float:
    """Pascal-VOC all-points AP at a fixed IoU from labeled (conf, tp) points."""
    if n_gt == 0:
        return float("nan")            # class absent in this scope — undefined, not zero
    if not pts:
        return 0.0
    pts = sorted(pts, key=lambda x: -x[0])
    tp = np.array([p[1] for p in pts], dtype=float)
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def prf_at(pts, n_gt: int, t: float) -> tuple[float, float, float, int, int, int]:
    """Precision/Recall/F1 (+ TP/FP/FN counts) at confidence threshold `t`."""
    tp = sum(1 for c, hit in pts if c >= t and hit == 1)
    fp = sum(1 for c, hit in pts if c >= t and hit == 0)
    fn = n_gt - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / n_gt if n_gt else float("nan")
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1, tp, fp, fn


def tune_thresholds(val_records, names, grid=None) -> dict[str, float]:
    """Per-class threshold that maximises F1 on VALIDATION (test is never consulted)."""
    if grid is None:
        grid = np.round(np.arange(0.05, 0.90 + 1e-9, 0.05), 2)
    tuned = {}
    for cid, nm in enumerate(names):
        pts, n_gt = _gather(val_records, cid)
        if n_gt == 0:
            tuned[nm] = 0.25
            continue
        best_t, best_f1 = 0.25, -1.0
        for t in grid:
            _, _, f1, *_ = prf_at(pts, n_gt, float(t))
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        tuned[nm] = best_t
    return tuned


def bootstrap_ci(records, cid: int, b: int, seed: int = 0):
    """Resample images with replacement → distribution of AP@0.5 for class `cid`."""
    rng = np.random.default_rng(seed)
    per_img = [(r["per_class"][str(cid)]["pts"], r["per_class"][str(cid)]["n_gt"])
               for r in records]
    n = len(per_img)
    if n == 0:
        return (float("nan"),) * 3
    aps = []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        pts, n_gt = [], 0
        for j in idx:
            p, g = per_img[j]
            pts.extend(p)
            n_gt += g
        a = voc_ap(pts, n_gt)
        if not np.isnan(a):
            aps.append(a)
    if not aps:
        return (float("nan"),) * 3
    lo, med, hi = np.percentile(aps, [2.5, 50, 97.5])
    return float(lo), float(med), float(hi)


def ap_at_iou(records, cid: int, iou_thr: float) -> float:
    """AP for class `cid` re-matched at `iou_thr` from the cached raw detections + GT."""
    pts, n_gt = [], 0
    for r in records:
        d_c = [(d[1], tuple(d[2:6])) for d in r["dets"] if d[0] == cid]
        g_c = [tuple(g[1:5]) for g in r["gts"] if g[0] == cid]
        pts.extend(_match_image(d_c, g_c, iou_thr))
        n_gt += len(g_c)
    return voc_ap(pts, n_gt)


def ap_coco(records, cid: int) -> float:
    """COCO-style AP@0.5:0.95 — the mean of AP over IoU 0.50, 0.55, …, 0.95 (VOC all-points each)."""
    aps = [ap_at_iou(records, cid, float(t)) for t in IOU_COCO]
    return float("nan") if any(np.isnan(a) for a in aps) else float(np.mean(aps))


def confusion_matrix(records, names, tuned, iou_thr: float = 0.5) -> np.ndarray:
    """(C+1)×(C+1) counts, rows = predicted, cols = true; index C = background.

    Detections kept at each class's VAL-tuned threshold; class-agnostic greedy matching (highest
    confidence first, best-IoU unused GT ≥ `iou_thr`) so a box on the right object with the wrong
    label shows as confusion, not as one FP plus one FN.
    """
    C = len(names)
    cm = np.zeros((C + 1, C + 1), int)
    for r in records:
        dets = sorted((d for d in r["dets"] if d[0] < C and d[1] >= tuned[names[d[0]]]),
                      key=lambda d: -d[1])
        gts = [g for g in r["gts"] if g[0] < C]
        used = [False] * len(gts)
        for d in dets:
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if not used[j]:
                    i = _iou(tuple(d[2:6]), tuple(g[1:5]))
                    if i > best:
                        best, bj = i, j
            if bj >= 0 and best >= iou_thr:
                used[bj] = True
                cm[d[0], gts[bj][0]] += 1
            else:
                cm[d[0], C] += 1                 # false alarm on background
        for j, g in enumerate(gts):
            if not used[j]:
                cm[C, g[0]] += 1                 # missed object
    return cm


def pr_curve(pts, n_gt: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(recall, precision, confidence) along the ranked detections."""
    if not pts or not n_gt:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    pts = sorted(pts, key=lambda x: -x[0])
    tp = np.array([q[1] for q in pts], float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1.0 - tp)
    return ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-12), np.array([q[0] for q in pts])


def plot_pr(test_records, names, tuned, title: str, out_png: Path) -> Optional[Path]:
    """PR curve per class with the val-tuned operating point marked. Matplotlib is a dev-only
    dependency; without it the image is skipped and the report says so."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
    for cid, nm in enumerate(names):
        pts, n_gt = _gather(test_records, cid)
        if n_gt == 0:
            continue
        rec, prec, _ = pr_curve(pts, n_gt)
        ln, = ax.plot(rec, prec, lw=1.6, label=f"{nm}  AP@0.5 {voc_ap(pts, n_gt):.3f}")
        p, r, *_ = prf_at(pts, n_gt, tuned[nm])
        ax.plot([r], [p], "o", color=ln.get_color(), ms=6)
        ax.annotate(f"val-t {tuned[nm]:.2f}", (r, p), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color=ln.get_color())
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


# ----------------------------------------------------------------------------- report
def _fmt(x, nd=3):
    return "—" if (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def build_report(names, tuned, shipped, test_records, val_records, args) -> tuple[str, dict]:
    lines, js = [], {"names": names, "tuned_thresholds": tuned, "shipped_thresholds": shipped}
    A = lines.append
    try:
        wp = Path(args.weights).resolve().relative_to(REPO).as_posix()
    except ValueError:
        wp = Path(args.weights).name

    split = args.test_split
    A(f"# {args.model} evaluation — `{split}` split — deploy-faithful (cv2.dnn ONNX), honest per-domain\n")
    A(f"_Model_: `{wp}` · _Data_: `{Path(args.root).name}/{split}` ({len(test_records)} images) · "
      f"_AP_: VOC all-points @IoU 0.5 and COCO-style @0.5:0.95 · _NMS IoU_: {args.iou_nms} · "
      f"_input_: {args.imgsz} px · _bootstrap_: {args.bootstrap} resamples\n")
    A("> Evaluated on the **exact ONNX artifact the studio and the MCP server run** (torch-free "
      f"`cv2.dnn`). Thresholds are tuned on **{args.val_split}**; the **{split}** split below is "
      "scored once.\n")

    # 1. Per-class on TEST, at val-tuned thresholds (with bootstrap CI on AP)
    A(f"\n## 1. Per-class ({split})\n")
    A("| class | n_gt | AP@0.5 | AP 95% CI | AP@0.5:0.95 | P@val-t | R@val-t | F1@val-t | val-t | shipped-t |")
    A("|---|--:|--:|:--:|--:|--:|--:|--:|--:|--:|")
    ap_list, coco_list = [], []
    js["per_class"] = {}
    for cid, nm in enumerate(names):
        pts, n_gt = _gather(test_records, cid)
        ap = voc_ap(pts, n_gt)
        lo, med, hi = bootstrap_ci(test_records, cid, args.bootstrap, args.seed)
        p, r, f1, tp, fp, fn = prf_at(pts, n_gt, tuned[nm])
        apc = ap_coco(test_records, cid)
        if not np.isnan(ap):
            ap_list.append(ap)
        if not np.isnan(apc):
            coco_list.append(apc)
        A(f"| {nm} | {n_gt} | {_fmt(ap)} | {_fmt(lo,2)}–{_fmt(hi,2)} | {_fmt(apc)} | {_fmt(p)} | "
          f"{_fmt(r)} | {_fmt(f1)} | {tuned[nm]:.2f} | {shipped.get(nm,0.25):.2f} |")
        js["per_class"][nm] = {"n_gt": n_gt, "ap50": ap, "ap_ci": [lo, hi], "ap50_95": apc,
                               "P": p, "R": r, "F1": f1, "TP": tp, "FP": fp, "FN": fn,
                               "val_t": tuned[nm], "shipped_t": shipped.get(nm, 0.25)}
    m = float(np.mean(ap_list)) if ap_list else float("nan")
    mc = float(np.mean(coco_list)) if coco_list else float("nan")
    A(f"\n**mAP@0.5 (mean over classes present): {_fmt(m)}** · **mAP@0.5:0.95: {_fmt(mc)}**  "
      f"— *read the per-class rows: the mean hides the mission class.*")
    js["mAP50"], js["mAP50_95"] = m, mc

    # 1b. Confusion matrix + PR curve
    A(f"\n### Confusion matrix ({split}, val-tuned thresholds, IoU ≥ 0.5)\n")
    A("Rows = predicted, columns = true. The `background` row counts missed objects; the "
      "`background` column counts false alarms. Matching is class-agnostic, so a right box with the "
      "wrong label shows as confusion.\n")
    cm = confusion_matrix(test_records, names, tuned)
    labels = list(names) + ["background"]
    A("| predicted \\ true | " + " | ".join(labels) + " |")
    A("|---|" + "--:|" * len(labels))
    for i, nm in enumerate(labels):
        A(f"| **{nm}** | " + " | ".join(str(int(v)) for v in cm[i]) + " |")
    js["confusion"] = {"labels": labels, "rows_pred_cols_true": cm.tolist()}
    tag = args.model.lower().replace("-", "")
    png = plot_pr(test_records, names, tuned, f"{args.model} · {split} · PR (dot = val-tuned threshold)",
                  REPO / "docs" / "img" / f"pr_{tag}_{split}.png")
    if png is not None:
        rel = Path(os.path.relpath(png, Path(args.out).resolve().parent)).as_posix()
        A(f"\n![PR curve]({rel})\n")
        js["pr_curve_png"] = png.relative_to(REPO).as_posix()
    else:
        A("\n_PR curve skipped: matplotlib is not installed (dev-only dependency)._\n")

    # 2. Per-sensor
    A("\n## 2. Per-sensor (the product is SONAR)\n")
    A("| sensor | class | n_gt | AP@0.5 | P | R | F1 |")
    A("|---|---|--:|--:|--:|--:|--:|")
    js["per_sensor"] = {}
    for sensor in ("sonar", "optical"):
        rs = [r for r in test_records if r["sensor"] == sensor]
        js["per_sensor"][sensor] = {}
        for cid, nm in enumerate(names):
            pts, n_gt = _gather(rs, cid)
            if n_gt == 0:
                continue
            ap = voc_ap(pts, n_gt)
            p, r, f1, *_ = prf_at(pts, n_gt, tuned[nm])
            A(f"| {sensor} | {nm} | {n_gt} | {_fmt(ap)} | {_fmt(p)} | {_fmt(r)} | {_fmt(f1)} |")
            js["per_sensor"][sensor][nm] = {"n_gt": n_gt, "ap50": ap, "P": p, "R": r, "F1": f1}

    # 3. Per-source
    A("\n## 3. Per-source (where the boxes really come from)\n")
    A("| source | sensor | class | n_gt | AP@0.5 | R@val-t |")
    A("|---|---|---|--:|--:|--:|")
    js["per_source"] = {}
    sources = sorted({r["source"] for r in test_records})
    for src in sources:
        rs = [r for r in test_records if r["source"] == src]
        sens = rs[0]["sensor"] if rs else "?"
        js["per_source"][src] = {}
        for cid, nm in enumerate(names):
            pts, n_gt = _gather(rs, cid)
            if n_gt == 0:
                continue
            ap = voc_ap(pts, n_gt)
            _, r, _, *_ = prf_at(pts, n_gt, tuned[nm])
            A(f"| {src} | {sens} | {nm} | {n_gt} | {_fmt(ap)} | {_fmt(r)} |")
            js["per_source"][src][nm] = {"n_gt": n_gt, "ap50": ap, "R": r}

    # 4. GhostVision head-to-head (gated)
    A("\n## 4. GhostVision head-to-head (official crab-pot split)\n")
    if args.official_split and args.leakage_free and args.model not in LEAKED_ON_OFFICIAL:
        res = eval_official_split(names, test_records, val_records, tuned, args)  # noqa
        A(res)
        js["ghostvision"] = {"status": "computed"}
    elif args.model not in LEAKED_ON_OFFICIAL:
        A(f"Not run on `{split}`. {args.model} is leakage-free for the official split; its head-to-head "
          f"is the `test_official398` report ([`eval_{tag}_test_official398.md`]"
          f"(eval_{tag}_test_official398.md)). GhostVision (JMSE 2026) reports F1 ≈ 0.71–0.73 there.")
        js["ghostvision"] = {"status": "see test_official398"}
    else:
        A("**SKIPPED — EXP-001 is not leakage-free for this split.** EXP-001 trained on dataset "
          "**v1**, which mixed the official crab-pot test frames into training; scoring them here "
          "would report a leaked (optimistic) number. Dataset **v2** locks the official 398-frame "
          "split (`official_crabpot_test.txt`); EXP-003 is scored on it:\n\n"
          "```\npython -m src.detection.evaluate --test-split test_official398 \\\n"
          "    --official-split DATASET/03_yolo_ready_dataset_v2b/official_crabpot_test.txt "
          "--leakage-free\n```\n\n"
          "GhostVision (JMSE 2026) reports F1 ≈ 0.71–0.73 on this split — the target to match.")
        js["ghostvision"] = {"status": "skipped", "reason": "EXP-001 leaked on v1"}

    A("\n---\n_Generated by `src/detection/evaluate.py`. AP is an independent VOC all-points "
      "implementation over the deployed `cv2.dnn` path; expect small differences from "
      "ultralytics 101-pt interpolation._")
    return "\n".join(lines), js


def eval_official_split(names, test_records, val_records, tuned, args) -> str:
    """Ghost_gear (== fishing_gear) F1/P/R on the locked official split. Only call when
    the model is leakage-free for this split (EXP-002 on v2)."""
    paths = {ln.strip() for ln in Path(args.official_split).read_text().splitlines() if ln.strip()}
    stems = {_strip_ext(Path(p).name) for p in paths}
    rs = [r for r in test_records if _strip_ext(r["name"]) in stems]
    cid = next((names.index(n) for n in ("ghost_gear", "fishing_gear") if n in names), 0)
    pts, n_gt = _gather(rs, cid)
    ap = voc_ap(pts, n_gt)
    p, r, f1, tp, fp, fn = prf_at(pts, n_gt, tuned[names[cid]])
    return (f"Frames matched: **{len(rs)}** · n_gt {n_gt}\n\n"
            f"| metric | value | GhostVision |\n|---|--:|--:|\n"
            f"| F1 | {_fmt(f1)} | 0.71–0.73 |\n| Precision | {_fmt(p)} | — |\n"
            f"| Recall | {_fmt(r)} | — |\n| AP@0.5 | {_fmt(ap)} | — |\n")


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model name: weights, names, input size, thresholds from models/<MODEL>/; "
                         "caches go to runs/<MODEL>/eval")
    ap.add_argument("--weights", default=None, help="default models/<MODEL>/best.onnx")
    ap.add_argument("--root", default=None, help="dataset root with val/ and test/ (default: v1 for "
                                                  "EXP-001, else v2b)")
    ap.add_argument("--names", default=None, help="default: the model's calibration.json")
    ap.add_argument("--imgsz", type=int, default=None, help="default: the model's calibration.json")
    ap.add_argument("--iou-nms", type=float, default=None, help="NMS IoU (default: the model's calibration)")
    ap.add_argument("--iou-match", type=float, default=0.50, help="TP IoU threshold")
    ap.add_argument("--conf-floor", type=float, default=0.001, help="collection floor for the PR curve")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap images per split (smoke test)")
    ap.add_argument("--official-split", default=None)
    ap.add_argument("--leakage-free", action="store_true",
                    help="assert the model did NOT train on the official split (enables §4)")
    ap.add_argument("--out", default=None, help="default docs/reports/eval_<model>_<test-split>.md")
    ap.add_argument("--val-split", default="val", help="split dir used to tune thresholds")
    ap.add_argument("--test-split", default="test",
                    help="split dir scored once (e.g. test | test_official398 | test_xsonar)")
    args = ap.parse_args()

    cal = load_calibration(args.model)
    args.weights = args.weights or str(default_weights(args.model))
    args.root = args.root or str(default_root(args.model))
    args.imgsz = args.imgsz or cal.imgsz
    args.iou_nms = args.iou_nms if args.iou_nms is not None else cal.iou_nms
    args.out = args.out or str(REPO / "docs" / "reports" / f"eval_{args.model.lower().replace('-', '')}_{args.test_split}.md")
    names = args.names.split(",") if args.names else cal.names
    root = Path(args.root)
    if not (root / args.test_split / "images").exists():
        raise SystemExit(f"no images under {root / args.test_split} - DATASET/ is git-ignored; build it "
                         f"with DATASET/scripts/ (see docs/dataset_card.md) or pass --root")
    # Collect at a low floor for ALL classes so we recover the full PR curve, then threshold ourselves.
    det = YoloOnnxDetector(args.weights, names=names, imgsz=args.imgsz,
                           conf_thres=args.conf_floor, iou_thres=args.iou_nms,
                           class_aware_nms=cal.class_aware_nms)
    det._onnx_path = args.weights
    eval_dir = REPO / "runs" / args.model / "eval"

    print(f"Collecting {args.val_split.upper()} detections ...")
    val_records = collect_split(det, root / args.val_split, names, eval_dir / f"cache_{args.val_split}.json",
                                args.iou_match, args.limit)
    print(f"Collecting {args.test_split.upper()} detections ...")
    test_records = collect_split(det, root / args.test_split, names, eval_dir / f"cache_{args.test_split}.json",
                                 args.iou_match, args.limit)

    tuned = tune_thresholds(val_records, names)
    print(f"\nVal-tuned thresholds: {tuned}")

    report, js = build_report(names, tuned, cal.conf, test_records, val_records, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    (eval_dir / f"metrics_{args.test_split}.json").write_text(json.dumps(js, indent=2))
    print(f"\nWrote {out}  and  {eval_dir / f'metrics_{args.test_split}.json'}")
    print(f"mAP@0.5 (classes present): {_fmt(js['mAP50'])}")


if __name__ == "__main__":
    main()
