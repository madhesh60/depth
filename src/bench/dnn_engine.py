"""
dnn_engine.py — STUDY-20: what does OpenCV 5's new DNN engine buy DEPTH's detector? Speed, same answers.

OpenCV 5 has two ONNX engines behind ``cv2.dnn.readNetFromONNX(path, engine)``: the classic one (OpenCV 4's
importer and layers) and the new graph engine (``ENGINE_NEW``). ``ENGINE_AUTO`` — DEPTH's default — tries
the new one first. A ``cv2.dnn.Net`` cannot say which engine built it, so this study (a) identifies the
engine ENGINE_AUTO picks for the deployed model, (b) times the engines on the product's own input and
(c) checks that they give the same answers.

**Protocol (fixed in this file before the first run)**

1. Model: the deployed ONNX (``models/$DEPTH_MODEL/best.onnx``; its SHA-256 is recorded). Input: the
   product's own pre-processing (``infer._letterbox`` + ``cv2.dnn.blobFromImage``) of the frames in
   ``--frames`` (default: the 8 shipped samples — only their pixels are used, no label is read).
2. Identification: ENGINE_AUTO's raw output on every timing frame is compared with ENGINE_NEW's and
   ENGINE_CLASSIC's; "auto = X" only if it is bit-identical to X on every frame.
3. Timing: ``--rounds`` (5) interleaved rounds; each round runs every engine in a rotated order, ``--n``
   (20) timed forwards each (``setInput`` + ``forward``), after 3 untimed warm-up forwards per engine.
   Thread settings ``--threads`` (default ``0,1``: OpenCV's default pool, then 1 thread). Reported per
   engine: pooled median and p90, and the per-round medians. A speed-up is claimed only when the per-round
   median ranges do not overlap (the STUDY-17 rule, ``compare.separation``); otherwise "within noise".
4. Same answers: on ``--parity-frames`` (default: the calibration split, v2b ``val`` ``*wcp_ss_*`` unique
   frames, when the dataset is present; else the timing frames) both engines' outputs go through the
   product's decoder at the calibrated thresholds. A detection matches when the class is equal, IoU ≥ 0.9,
   |Δscore| ≤ 0.01 and the tier at τ_review is equal (the STUDY-17 parity rule); every mismatch is
   listed. The test split is never read.
5. Decision rule: the default stays ``auto`` if it loads the faster engine and the answers match; a tier
   flip is reported against the calibrated promise, which was fit with the default engine.

    python -m src.bench.dnn_engine --label laptop_x86 [--rounds 5] [--n 20] [--threads 0,1]
    python -m src.bench.dnn_engine --report-only      # rebuild the report from benchmarks/*__dnn_engine.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "docs" / "reports" / "dnn_engine.md"
BENCH = REPO / "benchmarks"
VAL = REPO / "DATASET" / "03_yolo_ready_dataset_v2b" / "val"
ENGINES = ("classic", "new", "auto")
CONST = {"classic": "ENGINE_CLASSIC", "new": "ENGINE_NEW", "auto": "ENGINE_AUTO"}
PARITY = {"iou": 0.9, "dscore": 0.01}                      # = compare.PARITY (STUDY-17)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _rel(p) -> str:
    p = Path(p)
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def load_net(onnx: Path, engine: str):
    """A network on one named engine, and the seconds the load took."""
    t0 = time.perf_counter()
    net = cv2.dnn.readNetFromONNX(str(onnx), getattr(cv2.dnn, CONST[engine]))
    return net, time.perf_counter() - t0


def prepare(paths: list[Path], imgsz: int) -> list[dict]:
    """The product's pre-processing (``YoloOnnxDetector.forward_raw``) per frame."""
    from src.detection.infer import _letterbox
    out = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        canvas, scale, padx, pady = _letterbox(im, imgsz)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (imgsz, imgsz), swapRB=True)
        out.append({"id": p.stem, "blob": blob, "scale": scale, "padx": padx, "pady": pady, "hw": im.shape[:2]})
    return out


def forward(net, blob: np.ndarray) -> np.ndarray:
    net.setInput(blob)
    return np.array(net.forward(), copy=True)


def identify(nets: dict, frames: list[dict]) -> dict:
    """Which engine does ENGINE_AUTO load? Bit-identical raw output on every frame, or 'undetermined'."""
    same = {e: True for e in ("new", "classic")}
    max_d = 0.0
    for fr in frames:
        a = forward(nets["auto"], fr["blob"])
        for e in same:
            same[e] = same[e] and bool(np.array_equal(a, forward(nets[e], fr["blob"])))
        max_d = max(max_d, float(np.abs(forward(nets["new"], fr["blob"]) - forward(nets["classic"], fr["blob"])).max()))
    hit = [e for e, ok in same.items() if ok]
    return {"auto_equals": hit[0] if len(hit) == 1 else "undetermined", "frames": len(frames),
            "auto_bit_identical": same, "new_vs_classic_max_abs_output": round(max_d, 6),
            "layers": {e: len(n.getLayerNames()) for e, n in nets.items()}}


def time_engines(nets: dict, frames: list[dict], rounds: int, n: int, threads: int) -> dict:
    """Interleaved rounds (rotated engine order) of ``n`` timed forwards per engine."""
    if threads > 0:
        cv2.setNumThreads(threads)
    else:
        cv2.setNumThreads(-1)                                  # OpenCV's default pool
    names = list(nets)
    for e in names:                                            # untimed warm-up
        for i in range(3):
            forward(nets[e], frames[i % len(frames)]["blob"])
    per = {e: [] for e in names}
    for r in range(rounds):
        k = r % len(names)
        for e in names[k:] + names[:k]:
            ms = []
            for i in range(n):
                blob = frames[i % len(frames)]["blob"]
                t0 = time.perf_counter()
                nets[e].setInput(blob)
                nets[e].forward()
                ms.append((time.perf_counter() - t0) * 1000)
            per[e].append(ms)
    out = {"threads": cv2.getNumThreads(), "rounds": rounds, "n_per_round": n, "engines": {}}
    for e, rs in per.items():
        pooled = np.concatenate([np.asarray(x) for x in rs])
        out["engines"][e] = {"median_ms": round(float(np.median(pooled)), 1),
                             "p90_ms": round(float(np.percentile(pooled, 90)), 1),
                             "mean_ms": round(float(pooled.mean()), 1),
                             "round_medians_ms": [round(float(np.median(x)), 1) for x in rs]}
    c, w = out["engines"]["classic"], out["engines"]["new"]
    ratios = [a / b for a, b in zip(c["round_medians_ms"], w["round_medians_ms"]) if b]
    out["classic_over_new"] = {"pooled": round(c["median_ms"] / w["median_ms"], 3) if w["median_ms"] else None,
                               "per_round": [round(min(ratios), 3), round(max(ratios), 3)] if ratios else None,
                               "saved_ms": round(c["median_ms"] - w["median_ms"], 1),
                               "separation": separation(c["round_medians_ms"], w["round_medians_ms"])}
    return out


def separation(slow: list[float], fast: list[float]) -> str:
    """STUDY-17's claim rule on per-round values (``compare.separation``)."""
    if len(slow) < 2 or len(fast) < 2:
        return "single round - no claim"
    if max(fast) < min(slow):
        return "round ranges separate"
    if max(slow) < min(fast):
        return "round ranges separate (slower)"
    return "round ranges overlap - within noise, no claim"


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _tier(d, tiers) -> str:
    if tiers is None or d.cls_name != tiers.guaranteed_class:
        return "review"                                        # outside the guarantee: a person decides
    return tiers.tier(d.conf, d.conf).value


def compare_answers(ref: list, other: list, tiers) -> tuple[dict, list[dict]]:
    """Greedy same-class matching (by score) of one frame's detections: ``ref`` (the default engine)
    against ``other``. Returns the counts and the mismatches."""
    used = [False] * len(other)
    c = {"ref": len(ref), "other": len(other), "match": 0, "differ": 0, "missing": 0, "tier_flips": 0}
    bad, max_d = [], 0.0
    for d in sorted(ref, key=lambda x: -x.conf):
        best, bj = 0.0, -1
        for j, o in enumerate(other):
            if not used[j] and o.cls_name == d.cls_name:
                v = _iou(d.bbox, o.bbox)
                if v > best:
                    best, bj = v, j
        if bj < 0 or best < PARITY["iou"]:
            c["missing"] += 1
            bad.append({"kind": "missing", "cls": d.cls_name, "bbox": list(d.bbox), "score": round(d.conf, 4)})
            continue
        used[bj] = True
        o = other[bj]
        dsc = abs(o.conf - d.conf)
        max_d = max(max_d, dsc)
        flip = _tier(d, tiers) != _tier(o, tiers)
        c["tier_flips"] += int(flip)
        if dsc <= PARITY["dscore"] and not flip:
            c["match"] += 1
        else:
            c["differ"] += 1
            bad.append({"kind": "differ", "cls": d.cls_name, "bbox": list(d.bbox), "score": round(d.conf, 4),
                        "other_score": round(o.conf, 4), "iou": round(best, 3), "tier_flip": flip})
    extra = [o for j, o in enumerate(other) if not used[j]]
    c["extra"] = len(extra)
    bad += [{"kind": "extra", "cls": o.cls_name, "bbox": list(o.bbox), "score": round(o.conf, 4)} for o in extra]
    c["max_dscore"] = round(max_d, 5)
    return c, bad


def parity(nets: dict, frames: list[dict], det, tiers) -> dict:
    """Both engines' raw outputs through the product's decoder (``YoloOnnxDetector.decode``)."""
    tot = {"frames": len(frames), "ref": 0, "other": 0, "match": 0, "differ": 0, "missing": 0, "extra": 0,
           "tier_flips": 0, "max_dscore": 0.0, "max_abs_output": 0.0}
    mism = []
    for fr in frames:
        outs = {e: forward(nets[e], fr["blob"]) for e in ("new", "classic")}
        tot["max_abs_output"] = max(tot["max_abs_output"], float(np.abs(outs["new"] - outs["classic"]).max()))
        dets = {e: det.decode(o[0].T, fr["scale"], fr["padx"], fr["pady"], fr["hw"]) for e, o in outs.items()}
        c, bad = compare_answers(dets["new"], dets["classic"], tiers)
        for k in ("ref", "other", "match", "differ", "missing", "extra", "tier_flips"):
            tot[k] += c[k]
        tot["max_dscore"] = max(tot["max_dscore"], c["max_dscore"])
        mism += [{"frame": fr["id"], **b} for b in bad]
    tot["max_abs_output"] = round(tot["max_abs_output"], 6)
    tot["max_dscore"] = round(tot["max_dscore"], 5)
    return {**tot, "ref_engine": "new", "other_engine": "classic", "rule": PARITY, "mismatches": mism[:50]}


def _frame_paths(src: Path) -> list[Path]:
    return sorted(p for p in (src.rglob("*") if src.is_dir() else [src]) if p.suffix.lower() in IMG_EXTS)


def run(a) -> dict:
    from src.bench.fingerprint import fingerprint, sha256_file
    from src.detection.calibration import calibration_path, load_calibration
    from src.detection.infer import DEFAULT_ONNX, PER_CLASS_CONF, YoloOnnxDetector
    from src.agentic.policy import GuaranteedTiers
    onnx = Path(a.onnx) if a.onnx else DEFAULT_ONNX
    cal = load_calibration()
    if not hasattr(cv2.dnn, "ENGINE_NEW"):
        raise SystemExit(f"OpenCV {cv2.__version__} has no ENGINE_NEW - this study needs OpenCV 5")
    timing = prepare(_frame_paths(Path(a.frames)), cal.imgsz)
    if not timing:
        raise SystemExit(f"no frames in {a.frames}")
    if a.parity_frames:
        ppaths, psplit = _frame_paths(Path(a.parity_frames)), _rel(a.parity_frames)
    elif (VAL / "images").is_dir():
        from src.agentic.calibrate import list_frames
        ppaths, psplit = list_frames(VAL, "*wcp_ss_*.jpg"), f"{_rel(VAL)} *wcp_ss_* (calibration split, unique frames)"
    else:
        ppaths, psplit = _frame_paths(Path(a.frames)), f"{_rel(a.frames)} (the timing frames)"

    nets, load_s = {}, {}
    for e in ENGINES:
        loads = []
        for _ in range(3):
            net, s = load_net(onnx, e)
            loads.append(s)
        nets[e], load_s[e] = net, round(float(np.median(loads)), 3)
    print(f"[engines] loaded {', '.join(f'{e} {s:.2f} s' for e, s in load_s.items())}")
    ident = identify(nets, timing)
    print(f"[identify] ENGINE_AUTO = {ident['auto_equals']} (layers {ident['layers']})")

    det = YoloOnnxDetector(onnx, conf_thres=PER_CLASS_CONF)  # the product's decoder + thresholds
    det.net = None                                            # its own network is not needed here
    tiers = GuaranteedTiers.from_calibration(cal)
    par = parity(nets, prepare(ppaths, cal.imgsz), det, tiers)
    par["split"] = psplit
    print(f"[parity] {par['match']}/{par['ref']} detections identical on {par['frames']} frames "
          f"(differ {par['differ']}, missing {par['missing']}, extra {par['extra']}, tier flips {par['tier_flips']})")

    timings = []
    for t in [int(x) for x in str(a.threads).split(",") if x.strip()]:
        r = time_engines(nets, timing, a.rounds, a.n, t)
        timings.append(r)
        e = r["engines"]
        print(f"[timing] threads {r['threads']}: classic {e['classic']['median_ms']} ms · new {e['new']['median_ms']} ms "
              f"· auto {e['auto']['median_ms']} ms -> classic/new {r['classic_over_new']['pooled']}x "
              f"({r['classic_over_new']['separation']})")
    return {"label": a.label, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "workload": "dnn_engine",
            "fingerprint": fingerprint(onnx, calibration_path()),
            "config": {"model": cal.data.get("model"), "onnx": _rel(onnx), "onnx_sha256": sha256_file(onnx),
                       "imgsz": cal.imgsz, "timing_frames": _rel(a.frames), "n_timing_frames": len(timing),
                       "rounds": a.rounds, "n_per_round": a.n, "threads": a.threads},
            "load_s_median": load_s, "identification": ident, "parity": par, "timing": timings}


# ---------------------------------------------------------------------------------------- report
def _machine(r: dict) -> str:
    fp = r["fingerprint"]
    ec2 = fp.get("ec2") or {}
    return f"{fp['host']['machine']} · {ec2.get('instance-type') or fp['host'].get('cpu_model', '')[:30]}"


def _ocv(r: dict) -> str:
    o = r["fingerprint"]["opencv"]
    return f"{o['version']} {'**COOL**' if o['is_cool_path'] else 'stock'}"


def _product_share(label: str) -> Optional[str]:
    """The ``cv2.dnn`` forward's share of the product frame on the same machine (product_bench), if run."""
    p = BENCH / f"{label}__product.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    ops, tot = r.get("ops_ms") or {}, (r.get("stages_ms") or {}).get("total", {}).get("mean")
    if "dnn_forward" not in ops or not tot:
        return None
    return f"{ops['dnn_forward']['mean'] / tot:.0%} of the {tot:.0f}-ms product frame (`{label}__product.json`)"


def report(runs: list[dict]) -> str:
    L = ["# STUDY-20 — OpenCV 5's new DNN engine on DEPTH's detector: speed and same answers", "",
         "_`python -m src.bench.dnn_engine` · protocol in the module docstring (fixed before the first run). "
         "Every number names its machine; no number here is from a test split._", "",
         "OpenCV 5 loads an ONNX model with one of two engines: the **classic** one (OpenCV 4's importer and "
         "layers) or the **new** graph engine. DEPTH asks for `ENGINE_AUTO`, which tries the new one first. "
         "A `cv2.dnn.Net` cannot say which engine built it, so the product now loads the engines explicitly "
         "and records the one that loaded (`infer.ENGINE` → `/api/health` `runtime.engine`, the provenance "
         "stamp on every result, and every benchmark JSON).", ""]
    L += ["## Which engine does ENGINE_AUTO load?", "",
          "| run | machine | OpenCV | ENGINE_AUTO = | bit-identical on | layers (classic · new) | new vs classic, max raw output gap |",
          "|---|---|---|---|---|---|---|"]
    for r in runs:
        i = r["identification"]
        L.append(f"| `{r['label']}` | {_machine(r)} | {_ocv(r)} | **{i['auto_equals']}** | {i['frames']} of "
                 f"{i['frames']} frames | {i['layers']['classic']} · {i['layers']['new']} | "
                 f"{i['new_vs_classic_max_abs_output']:.4f} |")
    L += ["", "## Speed — one `cv2.dnn` forward of the 640² detector input", "",
          "Interleaved rounds, rotated engine order, the same frames for every engine. Median (p90) per "
          "forward; the speed-up is claimed only if the per-round medians do not overlap.", "",
          "| run | threads | classic | new | auto (default) | classic ÷ new | saved per forward | per-round ratio | claim |",
          "|---|--:|--:|--:|--:|--:|--:|---|---|"]
    for r in runs:
        for t in r["timing"]:
            e, s = t["engines"], t["classic_over_new"]
            pr = s.get("per_round") or [None, None]
            L.append(f"| `{r['label']}` | {t['threads']} | {e['classic']['median_ms']} ({e['classic']['p90_ms']}) ms "
                     f"| **{e['new']['median_ms']}** ({e['new']['p90_ms']}) ms | {e['auto']['median_ms']} ms "
                     f"| **{s['pooled']}×** | {s['saved_ms']} ms | {pr[0]}–{pr[1]}× | {s['separation']} |")
    L += ["", "## Same answers?", "",
          "Both engines' outputs through the product's decoder at the calibrated thresholds. A detection "
          "matches when the class is equal, IoU ≥ 0.9, |Δscore| ≤ 0.01 and its tier at τ_review is equal "
          "(STUDY-17's parity rule).", "",
          "| run | frames | detections (new) | identical | differ | missing | extra | tier flips | max \\|Δscore\\| |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for r in runs:
        p = r["parity"]
        L.append(f"| `{r['label']}` | {p['frames']} · {p['split']} | {p['ref']} | **{p['match']}** | {p['differ']} | "
                 f"{p['missing']} | {p['extra']} | **{p['tier_flips']}** | {p['max_dscore']:.4f} |")
    L += ["", "**Noise floor.** `auto` and `new` load the same engine (bit-identical outputs), so the gap between "
          "their medians is this machine's timing noise, not an engine effect:", "",
          "| run | threads | auto vs new (same engine) | rounds where classic was slower |", "|---|--:|--:|--:|"]
    for r in runs:
        for t in r["timing"]:
            e = t["engines"]
            gap = abs(e["auto"]["median_ms"] - e["new"]["median_ms"]) / e["new"]["median_ms"]
            slower = sum(c > w for c, w in zip(e["classic"]["round_medians_ms"], e["new"]["round_medians_ms"]))
            L.append(f"| `{r['label']}` | {t['threads']} | {gap:.0%} | {slower} of {t['rounds']} |")
    L += ["", "## What it means for DEPTH", ""]
    for r in runs:
        best = max(r["timing"], key=lambda t: t["threads"])
        s, i, p = best["classic_over_new"], r["identification"], r["parity"]
        share = _product_share(r["label"])
        e = best["engines"]
        if s["separation"] == "round ranges separate":           # the new engine is faster
            speed = (f"it runs the detector **{s['pooled']}×** as fast as the classic engine ({s['saved_ms']} ms saved "
                     "per forward; round ranges separate)")
        elif s["separation"] == "round ranges separate (slower)":  # the classic engine is faster
            speed = (f"the **classic** engine is faster here: {e['classic']['median_ms']} vs {e['new']['median_ms']} ms "
                     f"per forward (**{e['new']['median_ms'] / e['classic']['median_ms']:.2f}×**, "
                     f"{-s['saved_ms']} ms saved per forward; round ranges separate)")
        else:
            speed = f"classic ÷ new is {s['pooled']}×, but the rounds overlap: **no speed claim on this machine**"
        L.append(f"- `{r['label']}` ({_machine(r)}, {best['threads']} threads): ENGINE_AUTO loads the "
                 f"**{i['auto_equals']}** engine; {speed}" + (f"; the forward is {share}" if share else "")
                 + f". Answers: {p['match']} of {p['ref']} detections identical, {p['tier_flips']} tier flip(s).")
    L += ["- The look-harder mode runs 5 forwards per frame, so the per-forward saving counts five times there.",
          "- Where the parity table shows 0 tier flips, the calibrated promise does not depend on the engine: "
          "forcing the classic one (`DEPTH_DNN_ENGINE=classic`) changes no decision on those frames.",
          "- Which engine is faster depends on the machine, so the EC2 server does not assume it: "
          "`DEPTH_DNN_ENGINE=fastest` (`infra/depth.service`) loads both when the server starts, checks they give the "
          "same output on a fixed input, times them in interleaved rounds and keeps the classic engine only if it was "
          "faster in every round (`infer.faster_engine`); `/api/health` → `runtime.engine_choice` shows the evidence. "
          "The same rule runs on a COOL machine, whatever its build favours.",
          "- More rows (x86 and Graviton at 4 vCPU, COOL) are added by the same command on each machine "
          "(`infra/bench_cool.sh`, run on every machine by `infra/bench_fleet.sh`).", "",
          "---", "_Regenerated by the script from `benchmarks/*__dnn_engine.json`; do not edit by hand._"]
    return "\n".join(L) + "\n"


def load_runs(d: Path = BENCH) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(d.glob("*__dnn_engine.json"))]


def main():
    ap = argparse.ArgumentParser(description="STUDY-20: OpenCV 5 DNN engines (classic vs new) - speed + same answers")
    ap.add_argument("--label", help="machine label, e.g. laptop_x86 | baseline_x86 | graviton_stock | graviton_cool")
    ap.add_argument("--onnx", default="", help="ONNX model (default: the deployed model)")
    ap.add_argument("--frames", default=str(REPO / "webui" / "samples"), help="timing frames (pixels only)")
    ap.add_argument("--parity-frames", default="", help="parity frames (default: the calibration split if present)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--n", type=int, default=20, help="timed forwards per engine per round")
    ap.add_argument("--threads", default="0,1", help="comma list; 0 = OpenCV's default pool")
    ap.add_argument("--out", default=str(BENCH))
    ap.add_argument("--report-only", action="store_true", help="rebuild the report from the saved JSONs")
    a = ap.parse_args()
    if not a.report_only:
        if not a.label:
            ap.error("--label is required (or --report-only)")
        res = run(a)
        od = Path(a.out)
        od.mkdir(parents=True, exist_ok=True)
        (od / f"{a.label}__dnn_engine.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(f"-> {_rel(od / (a.label + '__dnn_engine.json'))}")
    runs = load_runs(Path(a.out))
    if runs:
        REPORT.write_text(report(runs), encoding="utf-8")
        print(f"-> {_rel(REPORT)} ({len(runs)} run(s))")


if __name__ == "__main__":
    main()
