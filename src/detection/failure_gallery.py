"""
failure_gallery.py — failure cases with numbers (the rules require "evaluation evidence including
failure cases / limitations").

On the VERIFICATION split (unseen, unique frames — the one the recall promise was checked on):

* **misses** — labelled pots that NO candidate reached (IoU ≥ 0.3 at the detector floor 0.05): the
  pots the agent can never route to a human. 12 rendered, chosen across sizes;
* **false alarms** — the 12 most confident candidates that match no label;
* **why pots are missed** — per labelled pot, three measurements from the image itself: box size
  (px), slant range (row), and local contrast (mean of the box / mean of a ring around it); missed vs
  found pots compared with medians. This tells EXP-002 what to fix, instead of guessing.

Writes ``docs/img/failure_misses.jpg``, ``docs/img/failure_false_alarms.jpg``, ``docs/failure_gallery.md``.

    python -m src.detection.failure_gallery [--model EXP-001 --split test]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
V1 = REPO / "DATASET" / "03_yolo_ready_dataset_v1"


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _contrast(g: np.ndarray, b) -> float:
    x1, y1, x2, y2 = [int(v) for v in b]
    H, W = g.shape
    w, h = max(2, x2 - x1), max(2, y2 - y1)
    inner = g[max(0, y1):min(H, y2), max(0, x1):min(W, x2)].astype(np.float32)
    ring = g[max(0, y1 - h):min(H, y2 + h), max(0, x1 - w):min(W, x2 + w)].astype(np.float32)
    rs = ring.sum() - inner.sum(); rn = ring.size - inner.size
    return float(inner.mean() / max(1.0, rs / max(1, rn))) if inner.size else 0.0


def _tile(img, b, color, others=(), side=180, label=""):
    x1, y1, x2, y2 = [int(v) for v in b]
    H, W = img.shape[:2]
    s = int(max(60, 5 * max(x2 - x1, y2 - y1)))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    a0, b0 = max(0, cx - s // 2), max(0, cy - s // 2)
    crop = img[b0:min(H, b0 + s), a0:min(W, a0 + s)].copy()
    for ob, oc in others:
        cv2.rectangle(crop, (int(ob[0]) - a0, int(ob[1]) - b0), (int(ob[2]) - a0, int(ob[3]) - b0), oc, 1)
    cv2.rectangle(crop, (x1 - a0, y1 - b0), (x2 - a0, y2 - b0), color, 1)
    crop = cv2.resize(crop, (side, side), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(crop, (0, side - 18), (side, side), (10, 14, 18), -1)
    cv2.putText(crop, label, (4, side - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 238, 244), 1, cv2.LINE_AA)
    return crop


def _reading(st: dict) -> str:
    out = []
    if st["edge"]["missed"] is not None and st["edge"]["missed"] > 2 * (st["edge"]["found"] or 0.01):
        out.append(f"**{st['edge']['missed']:.0%} of the missed pots touch the frame edge** vs "
                   f"{st['edge']['found']:.0%} of the found ones — objects cut by the frame edge are the largest "
                   f"single failure mode. The edge is a chunk boundary OR a Roboflow crop (most of these frames are "
                   f"augmented copies); seam inference across chunk boundaries did NOT recover them (STUDY-11b), which "
                   f"points at crop edges — a dataset artefact v2b removes by keeping one un-cropped copy per frame")
    if (st["contrast"]["missed"] or 0) < (st["contrast"]["found"] or 0):
        out.append(f"missed pots are **lower-contrast** ({st['contrast']['missed']} vs {st['contrast']['found']}) and "
                   f"**farther in range** (row {st['row']['missed']} vs {st['row']['found']}) — range fall-off; "
                   f"training on Stage-1 range-gain-normalised frames targets these")
    if (st["size"]["missed"] or 0) >= (st["size"]["found"] or 0):
        out.append(f"they are **not smaller** ({st['size']['missed']} vs {st['size']['found']} px) — the "
                   f"'small-object' explanation does not hold on this split")
    else:
        out.append(f"they are smaller ({st['size']['missed']} vs {st['size']['found']} px) — higher input resolution helps")
    if (st["thin"]["missed"] or 0) > 0:
        out.append(f"{st['thin']['missed']:.0%} are degenerate (≤ 4 px) labels — not detector failures")
    return "; ".join(out) + "."


def run(model: str = "EXP-001", split: str = "test", n: int = 12) -> dict:
    blob = json.loads((REPO / "runs" / "calib" / f"{model}_{split}.json").read_text())
    root = V1 / split
    GREEN, AMBER, GREY = (80, 200, 100), (40, 170, 235), (140, 140, 140)
    pots, fas = [], []
    for fr in blob["frames"]:
        img = cv2.imread(str(root / "images" / fr["name"]))
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cands = sorted(fr["cands"], key=lambda c: -c["conf"])
        used = set()
        for c in cands:                                   # greedy one-to-one (same as calibration)
            best, bj = 0.0, -1
            for j, gb in enumerate(fr["gt"]):
                if j not in used and _iou(c["bbox"], gb) > best:
                    best, bj = _iou(c["bbox"], gb), j
            if bj >= 0 and best >= 0.3:
                used.add(bj)
            else:
                fas.append({"frame": fr["name"], "bbox": c["bbox"], "conf": c["conf"], "img": img, "gt": fr["gt"]})
        for j, gb in enumerate(fr["gt"]):
            reach = max((_iou(c["bbox"], gb) for c in cands), default=0.0)
            H, W = g.shape
            edge = gb[0] <= 3 or gb[1] <= 3 or gb[2] >= W - 3 or gb[3] >= H - 3      # cut by the chunk boundary
            pots.append({"frame": fr["name"], "bbox": gb, "found": reach >= 0.3, "img": img, "cands": cands,
                         "size": float(max(gb[2] - gb[0], gb[3] - gb[1])), "row": float((gb[1] + gb[3]) / 2),
                         "contrast": _contrast(g, gb), "edge": float(edge),
                         "thin": float(min(gb[2] - gb[0], gb[3] - gb[1]) <= 4)})
    missed = [p for p in pots if not p["found"]]
    found = [p for p in pots if p["found"]]
    med = lambda xs, k: round(float(np.median([x[k] for x in xs])), 2) if xs else None
    stats = {k: {"missed": med(missed, k), "found": med(found, k)} for k in ("size", "row", "contrast")}
    share = lambda xs, k: round(float(np.mean([x[k] for x in xs])), 3) if xs else None
    stats["edge"] = {"missed": share(missed, "edge"), "found": share(found, "edge")}
    stats["thin"] = {"missed": share(missed, "thin"), "found": share(found, "thin")}
    # 12 misses spread across sizes; 12 most confident false alarms
    ms = sorted(missed, key=lambda p: p["size"])
    pick_m = [ms[int(i)] for i in np.linspace(0, len(ms) - 1, min(n, len(ms)))] if ms else []
    pick_f = sorted(fas, key=lambda f: -f["conf"])[:n]
    tiles_m = [_tile(p["img"], p["bbox"], GREEN, [(c["bbox"], GREY) for c in p["cands"] if _iou(c["bbox"], p["bbox"]) > 0],
                     label=f"{int(p['size'])}px c{p['contrast']:.2f}") for p in pick_m]
    tiles_f = [_tile(f["img"], f["bbox"], AMBER, [(gb, GREEN) for gb in f["gt"]], label=f"conf {f['conf']:.2f}") for f in pick_f]
    out = REPO / "docs" / "img"; out.mkdir(parents=True, exist_ok=True)

    def sheet(tiles, path):
        if not tiles:
            return
        rows = [np.hstack(tiles[i:i + 6] + [np.zeros_like(tiles[0])] * (6 - len(tiles[i:i + 6]))) for i in range(0, len(tiles), 6)]
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    sheet(tiles_m, out / "failure_misses.jpg")
    sheet(tiles_f, out / "failure_false_alarms.jpg")
    res = {"model": model, "split": split, "frames": len(blob["frames"]), "pots": len(pots), "missed": len(missed),
           "false_alarms": len(fas), "stats": stats}
    L = ["# Failure gallery — what EXP-001 gets wrong on unseen frames", "",
         f"_`python -m src.detection.failure_gallery` · {model} · verification split `{split}` "
         f"({res['frames']} unique frames, {res['pots']} labelled pots) · detector floor 0.05 · match IoU ≥ 0.3_", "",
         f"## Missed pots — {len(missed)} of {len(pots)} ({len(missed) / max(1, len(pots)):.0%}) never reach a human", "",
         "Green = the labelled pot; grey = any candidate that touched it (below IoU 0.3). Caption: size (px), "
         "local contrast (box mean ÷ ring mean).", "", "![missed pots](img/failure_misses.jpg)", "",
         "| measured on every labelled pot | missed (median) | found (median) |", "|---|--:|--:|",
         f"| size, longest side (px) | {stats['size']['missed']} | {stats['size']['found']} |",
         f"| local contrast (box ÷ ring) | {stats['contrast']['missed']} | {stats['contrast']['found']} |",
         f"| slant range (row, px) | {stats['row']['missed']} | {stats['row']['found']} |",
         f"| **touches the frame edge** (chunk boundary or augmentation crop) | {stats['edge']['missed']:.0%} | {stats['edge']['found']:.0%} |",
         f"| degenerate label (≤ 4 px thin) | {stats['thin']['missed']:.0%} | {stats['thin']['found']:.0%} |", "",
         "**Reading (data-driven):** " + _reading(stats), "",
         f"## False alarms — the {len(pick_f)} most confident of {len(fas)}", "",
         "Amber = the false alarm; green = labelled pots nearby. Several look like real, unlabelled returns — the "
         "blinded audit (Audit tab, STUDY-10) measures how many.", "", "![false alarms](img/failure_false_alarms.jpg)", "",
         "_Crops: PINGEcosystem crab-pot dataset, CC-BY-SA-4.0 (see `webui/samples/ATTRIBUTION.md`)._", "",
         "---", "_Regenerated by the script; do not edit by hand._", ""]
    (REPO / "docs" / "failure_gallery.md").write_text("\n".join(L), encoding="utf-8")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default="EXP-001")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    print(json.dumps(run(a.model, a.split), indent=1))


if __name__ == "__main__":
    main()
