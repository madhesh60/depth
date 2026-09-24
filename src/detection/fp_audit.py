"""
fp_audit.py — are the detector's "false alarms" really false? A blinded label-noise audit (review M-4).

Some "false positives" are probably real objects nobody labelled (bright, compact, pot-like returns
with no box — review §3.1). If so, precision is understated, and the human-review design is
necessary rather than a crutch. This measures it, rigorously:

* **Exact band, not a sample:** every false alarm at or above a confidence cut-off on the
  CALIBRATION split is audited (``--n`` most confident; EXP-001: the top 60, conf ≥ 0.176). Inside
  that band the audited precision is exact: (TP + FP judged real) / (TP + FP).
* **Blind:** the auditor sees only the crop and its wider context — never the confidence, never the
  label. False alarms and **catch trials** (known labelled pots, ``--catch``) are drawn identically
  and shuffled together.
* **Catch trials measure the auditor:** if someone does not recognise the known pots, their "real
  object" calls on false alarms are not credible — the summary reports catch accuracy next to the
  result, and inter-annotator agreement (Cohen's κ) when two or more people audit.
* Tags: ``real`` (a real object — unlabelled pot / gear) · ``clutter`` (natural seabed, rock, weed)
  · ``noise`` (speckle / artefact / water-column) · ``unsure``.

Build once (writes shippable crops to ``webui/audit/<model>_<split>/`` — CC-BY-SA like the samples):
    python -m src.detection.fp_audit build [--n 60 --catch 15]
Tag in the app (**Audit** tab); summary: ``python -m src.detection.fp_audit summary``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
TAGS = ("real", "clutter", "noise", "unsure")
V1 = REPO / "DATASET" / "03_yolo_ready_dataset_v1"


def audit_dir(model: str = "EXP-001", split: str = "val") -> Path:
    return REPO / "webui" / "audit" / f"{model}_{split}"


def manifest_path(model: str = "EXP-001", split: str = "val") -> Path:
    """The answer key (which items are catch trials) lives OUTSIDE webui/ — never served statically,
    so the audit stays blind."""
    return REPO / "models" / model / f"fp_audit_{split}.json"


def tags_path() -> Path:
    return Path(os.environ.get("DEPTH_AUDIT_TAGS", REPO / "runs" / "audit" / "tags.jsonl"))


def _crop(img: np.ndarray, bbox, frac: float, out: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = img.shape[:2]
    side = int(max(48, frac * max(x2 - x1, y2 - y1)))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    a, b = max(0, cx - side // 2), max(0, cy - side // 2)
    c, d = min(W, a + side), min(H, b + side)
    crop = img[b:d, a:c].copy()
    s = out / max(1, max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    # identical marking for false alarms and catch trials (blind): a thin corner-bracket box
    bx1, by1, bx2, by2 = int((x1 - a) * s), int((y1 - b) * s), int((x2 - a) * s), int((y2 - b) * s)
    L = max(6, (bx2 - bx1) // 3)
    for (px, py, dx, dy) in ((bx1, by1, 1, 1), (bx2, by1, -1, 1), (bx1, by2, 1, -1), (bx2, by2, -1, -1)):
        cv2.line(crop, (px, py), (px + dx * L, py), (235, 190, 30), 1, cv2.LINE_AA)
        cv2.line(crop, (px, py), (px, py + dy * L), (235, 190, 30), 1, cv2.LINE_AA)
    return crop


def build(model: str = "EXP-001", split: str = "val", n: int = 60, catch: int = 15, seed: int = 0,
          root: Optional[Path] = None) -> dict:
    from src.agentic.calibrate import label
    blob = json.loads((REPO / "runs" / "calib" / f"{model}_{split}.json").read_text())
    root = Path(root or (V1 / split))
    frames = blob["frames"]
    items = []
    for fi, fr in enumerate(frames):
        s, tp, _, _ = label([fr], lambda c: c["conf"])
        cands = sorted(fr["cands"], key=lambda c: -c["conf"])      # label() order = conf desc
        for c, ok in zip(cands, tp):
            items.append({"frame": fr["name"], "bbox": c["bbox"], "conf": c["conf"], "tp": bool(ok)})
    fps = sorted([it for it in items if not it["tp"]], key=lambda it: -it["conf"])[:n]
    cut = fps[-1]["conf"] if fps else 1.0
    band_tp = sum(1 for it in items if it["tp"] and it["conf"] >= cut)
    rng = random.Random(seed)
    tps = [it for it in items if it["tp"]]
    catches = rng.sample(tps, min(catch, len(tps)))
    chosen = [dict(it, kind="fp") for it in fps] + [dict(it, kind="catch") for it in catches]
    rng.shuffle(chosen)
    out = audit_dir(model, split)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, it in enumerate(chosen):
        img = cv2.imread(str(root / "images" / it["frame"]))
        if img is None:
            continue
        iid = f"a{i:03d}"
        cv2.imwrite(str(out / f"{iid}.jpg"), _crop(img, it["bbox"], 4.0, 256), [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(out / f"{iid}_ctx.jpg"), _crop(img, it["bbox"], 12.0, 200), [cv2.IMWRITE_JPEG_QUALITY, 80])
        manifest.append({"id": iid, "kind": it["kind"], "frame": it["frame"], "bbox": it["bbox"], "conf": it["conf"]})
    meta = {"model": model, "split": split, "created": time.strftime("%Y-%m-%d"), "n_fp": len(fps),
            "n_catch": len(catches), "conf_cut": round(cut, 4), "band_tp": band_tp, "band_fp": len(fps),
            "band_precision_raw": round(band_tp / max(1, band_tp + len(fps)), 4), "items": manifest,
            "license": "CC-BY-SA-4.0 (crops of the PINGEcosystem crab-pot dataset; see webui/samples/ATTRIBUTION.md)"}
    manifest_path(model, split).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------------- tags
_LOCK = threading.Lock()


def public_items(model: str = "EXP-001", split: str = "val") -> list[dict]:
    """What the auditor may see: ids + image URLs only (blind — no kind, no confidence)."""
    mf = manifest_path(model, split)
    if not mf.exists():
        return []
    m = json.loads(mf.read_text(encoding="utf-8"))
    base = f"audit/{model}_{split}"
    return [{"id": it["id"], "crop": f"{base}/{it['id']}.jpg", "context": f"{base}/{it['id']}_ctx.jpg"} for it in m["items"]]


def add_tag(item_id: str, tag: str, annotator: str) -> dict:
    if tag not in TAGS:
        raise ValueError(f"tag must be one of {TAGS}")
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "item": str(item_id)[:10], "tag": tag,
           "annotator": (annotator or "anon").strip()[:40] or "anon"}
    p = tags_path()
    with _LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def _latest_tags() -> dict[str, dict[str, str]]:
    """annotator → item → latest tag."""
    p = tags_path()
    out: dict[str, dict[str, str]] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.setdefault(r["annotator"], {})[r["item"]] = r["tag"]
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def _kappa(a: dict[str, str], b: dict[str, str]) -> Optional[float]:
    common = [i for i in a if i in b]
    if len(common) < 5:
        return None
    po = sum(a[i] == b[i] for i in common) / len(common)
    ca, cb = Counter(a[i] for i in common), Counter(b[i] for i in common)
    pe = sum(ca[t] * cb[t] for t in TAGS) / (len(common) ** 2)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def summary(model: str = "EXP-001", split: str = "val") -> dict:
    mf = manifest_path(model, split)
    if not mf.exists():
        return {"built": False}
    m = json.loads(mf.read_text(encoding="utf-8"))
    kind = {it["id"]: it["kind"] for it in m["items"]}
    tags = _latest_tags()
    per = {}
    for ann, t in tags.items():
        fp_t = Counter(tag for i, tag in t.items() if kind.get(i) == "fp")
        c_t = [tag for i, tag in t.items() if kind.get(i) == "catch"]
        per[ann] = {"tagged": sum(1 for i in t if i in kind), "fp_tags": dict(fp_t),
                    "catch_tagged": len(c_t), "catch_real": sum(1 for x in c_t if x == "real"),
                    "catch_accuracy": round(sum(1 for x in c_t if x == "real") / len(c_t), 3) if c_t else None}
    # pooled: majority tag per false-alarm item across annotators (ties → unsure)
    votes: dict[str, Counter] = {}
    for t in tags.values():
        for i, tag in t.items():
            if kind.get(i) == "fp":
                votes.setdefault(i, Counter())[tag] += 1
    maj = {}
    for i, c in votes.items():
        top = c.most_common()
        maj[i] = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else "unsure"
    n_fp_tagged = len(maj)
    n_real = sum(1 for v in maj.values() if v == "real")
    tp, fp = m["band_tp"], m["band_fp"]
    complete = n_fp_tagged == fp
    real_lo, real_hi = _wilson(n_real, n_fp_tagged)
    out = {"built": True, "model": model, "split": split, "conf_cut": m["conf_cut"], "band_tp": tp, "band_fp": fp,
           "band_precision_raw": m["band_precision_raw"], "annotators": list(tags), "per_annotator": per,
           "fp_tagged": n_fp_tagged, "fp_taxonomy": dict(Counter(maj.values())),
           "fp_real_share": round(n_real / n_fp_tagged, 3) if n_fp_tagged else None,
           "fp_real_share_ci95": [real_lo, real_hi], "complete": complete,
           "band_precision_audited": round((tp + n_real) / (tp + fp), 4) if complete else None}
    anns = list(tags)
    if len(anns) >= 2:
        out["kappa_first_two"] = _kappa(tags[anns[0]], tags[anns[1]])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", default="EXP-001"); b.add_argument("--split", default="val")
    b.add_argument("--n", type=int, default=60); b.add_argument("--catch", type=int, default=15)
    s = sub.add_parser("summary")
    s.add_argument("--model", default="EXP-001"); s.add_argument("--split", default="val")
    a = ap.parse_args()
    if a.cmd == "build":
        m = build(a.model, a.split, a.n, a.catch)
        print(json.dumps({k: v for k, v in m.items() if k != "items"}, indent=1))
    else:
        print(json.dumps(summary(a.model, a.split), indent=1))


if __name__ == "__main__":
    main()
