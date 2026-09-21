"""
benchmark.py — Stage 1 latency/throughput benchmark for the COOL award.

Runs the identical Stage-1 pipeline over a folder of sonar frames and reports per-op and
total latency, throughput (FPS), and the OpenCV build fingerprint (whether COOL / KleidiCV /
NEON is active). Run it in each of the three configurations and compare the JSON outputs:

    (A) x86        + stock OpenCV      -> baseline_x86.json
    (B) Graviton   + stock OpenCV      -> graviton_stock.json
    (C) Graviton   + COOL              -> graviton_cool.json

The (B)->(C) delta on identical hardware isolates COOL's contribution (the 30% "verified
COOL integration" criterion); the (A)->(C) delta is the headline Arm-vs-x86 story. Because
the pipeline is pure CPU OpenCV, the same code and config run unchanged in all three.

Usage
-----
    python -m src.cv_pipeline.benchmark --images DATASET/03_yolo_ready_dataset_v1/test/images \
        --limit 500 --repeats 3 --label graviton_cool --out runs/bench
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from .config import Stage1Config, QUALITY_PRESET
from .pipeline import Stage1Pipeline

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def opencv_fingerprint() -> dict:
    """Capture the OpenCV build so a benchmark run is self-documenting re: COOL/KleidiCV."""
    info = cv2.getBuildInformation()
    flags = {
        key: (key.lower() in info.lower())
        for key in ("KleidiCV", "Carotene", "NEON", "IPP", "OpenCL", "TBB")
    }
    return {
        "opencv_version": cv2.__version__,
        "cool_or_kleidicv_detected": flags["KleidiCV"],
        "accel_flags": flags,
        "cpu_threads": cv2.getNumThreads(),
    }


def platform_fingerprint() -> dict:
    return {
        "machine": platform.machine(),          # x86_64 vs aarch64 (Graviton)
        "processor": platform.processor(),
        "system": platform.system(),
        "python": platform.python_version(),
    }


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[k]


def run(images_dir: Path, limit: int, repeats: int, warmup: int,
        cfg: Stage1Config) -> dict:
    paths = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise SystemExit(f"no images under {images_dir}")
    paths = paths[:limit]
    pipe = Stage1Pipeline(cfg)

    # warmup (fills caches / first-call init so timings reflect steady state)
    for p in paths[:warmup]:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            pipe.process(img)

    per_op: dict[str, list[float]] = {}
    totals: list[float] = []
    n_candidates: list[int] = []
    n_frames = 0
    wall0 = time.perf_counter()
    for _ in range(repeats):
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            res = pipe.process(img)
            for op, ms in res.timings_ms.items():
                per_op.setdefault(op, []).append(ms)
            totals.append(res.total_ms)
            n_candidates.append(len(res.candidates))
            n_frames += 1
    wall = time.perf_counter() - wall0

    op_stats = {
        op: {"mean_ms": round(statistics.mean(v), 3),
             "median_ms": round(statistics.median(v), 3),
             "p95_ms": round(_percentile(v, 95), 3)}
        for op, v in per_op.items()
    }
    return {
        "opencv": opencv_fingerprint(),
        "platform": platform_fingerprint(),
        "config": cfg.to_dict(),
        "n_frames_timed": n_frames,
        "repeats": repeats,
        "per_op_ms": op_stats,
        "total_ms": {
            "mean": round(statistics.mean(totals), 3),
            "median": round(statistics.median(totals), 3),
            "p95": round(_percentile(totals, 95), 3),
        },
        "throughput_fps": round(n_frames / wall, 2),
        "avg_candidates_per_frame": round(statistics.mean(n_candidates), 2),
        "meets_latency_budget_300ms": statistics.mean(totals) < 300.0,
        "meets_throughput_5fps": (n_frames / wall) >= 5.0,
    }


def print_summary(r: dict, label: str) -> None:
    print(f"\n===== Stage-1 benchmark [{label}] =====")
    print(f"platform : {r['platform']['machine']} | OpenCV {r['opencv']['opencv_version']} | "
          f"COOL/KleidiCV: {r['opencv']['cool_or_kleidicv_detected']} | "
          f"threads {r['opencv']['cpu_threads']}")
    print(f"frames   : {r['n_frames_timed']}  (x{r['repeats']} repeats)")
    print(f"{'op':<18}{'mean(ms)':>10}{'median':>10}{'p95':>10}")
    for op, s in r["per_op_ms"].items():
        print(f"{op:<18}{s['mean_ms']:>10}{s['median_ms']:>10}{s['p95_ms']:>10}")
    tot = r["total_ms"]
    print(f"{'TOTAL':<18}{tot['mean']:>10}{tot['median']:>10}{tot['p95']:>10}")
    print(f"throughput: {r['throughput_fps']} FPS | "
          f"<300ms: {r['meets_latency_budget_300ms']} | "
          f">=5FPS: {r['meets_throughput_5fps']} | "
          f"avg ROIs/frame: {r['avg_candidates_per_frame']}")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Stage-1 for the COOL award.")
    p.add_argument("--images", required=True, help="folder of sonar frames (recursed)")
    p.add_argument("--limit", type=int, default=500, help="max distinct frames to use")
    p.add_argument("--repeats", type=int, default=3, help="passes over the frame set")
    p.add_argument("--warmup", type=int, default=10, help="untimed warmup frames")
    p.add_argument("--label", default="run", help="config label, e.g. graviton_cool")
    p.add_argument("--preset", choices=["default", "quality"], default="default")
    p.add_argument("--out", default="runs/bench", help="output dir for the JSON")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = QUALITY_PRESET if a.preset == "quality" else Stage1Config()
    result = run(Path(a.images), a.limit, a.repeats, a.warmup, cfg)
    print_summary(result, a.label)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{a.label}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
