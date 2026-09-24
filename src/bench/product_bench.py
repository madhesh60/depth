"""
product_bench.py — benchmark the workload DEPTH actually runs (review M-5), per stage, with provenance.

The old COOL benchmark timed the retired Stage-1 contour pipeline. This one times the product path
exactly as ``/api/analyze`` runs it, per frame:

    decode (cv2.imdecode) → stage1 (canonicalise: luminance · orientation · bottom track · detector
    input) → see (YOLO11 via cv2.dnn: letterbox + forward + decode + class-aware NMS) → prove+decide
    (evidence: water column · thin-line shadow · tiers) → render (overlay + cv2.imencode JPEG)

Workloads: ``product`` (all of the above) · ``cv`` (everything except the network: the OpenCV
image-processing share where COOL/KleidiCV applies, plus the slant→ground ``cv2.remap`` and range
gain) · ``detect`` (decode + network only).

It reports p50/p95/mean per stage, single-stream throughput, and **cost**: $ per 1,000 frames and
$ + compute seconds per **survey-hour** of sonar (frames per survey-hour = ping rate × 3600 /
pings per frame × channels — the ping rate is a survey setting, passed in and recorded). The
fingerprint (``fingerprint.py``) proves where it ran: ``cv2.__file__`` under ``/opt/cool`` ⇒ COOL.

    python -m src.bench.product_bench --frames webui/samples --n 200 --label laptop_x86 \
        [--workload product|cv|detect] [--threads 0] [--price-hour 0.0] [--ping-rate-hz 15]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_frames(src: Path) -> tuple[list[tuple[str, bytes]], str]:
    """Encoded frame bytes (decode is part of the timed work) + a content manifest sha256."""
    paths = sorted(p for p in (src.rglob("*") if src.is_dir() else [src]) if p.suffix.lower() in IMG_EXTS)
    frames, h = [], hashlib.sha256()
    for p in paths:
        b = p.read_bytes()
        frames.append((p.stem, b))
        h.update(p.name.encode()); h.update(hashlib.sha256(b).digest())
    return frames, h.hexdigest()


def _stats(v: list[float]) -> dict:
    a = np.asarray(v, dtype=float)
    return {"p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "mean": round(float(a.mean()), 2)}


class Workload:
    def __init__(self, kind: str):
        self.kind = kind
        from src.cv_pipeline.canonical import Canonicaliser
        self.stage1 = Canonicaliser()
        self.agent = None
        if kind in ("product", "detect"):
            from src.agentic.agent import ReLookAgent
            self.agent = ReLookAgent()
            self.agent.perceptor.warmup()

    def run(self, frame_id: str, data: bytes) -> dict[str, float]:
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        t["decode"] = (time.perf_counter() - t0) * 1000

        if self.kind == "product":
            from src.agentic.agent import render
            res = self.agent.run_frame(img, frame_id=frame_id)
            t["stage1"] = res.stage_ms["stage1"]
            t["see"] = res.stage_ms["see"]
            t["prove_decide"] = res.stage_ms["prove_decide"]
            t0 = time.perf_counter()
            ok, _ = cv2.imencode(".jpg", render(img, res), [cv2.IMWRITE_JPEG_QUALITY, 85])
            t["render"] = (time.perf_counter() - t0) * 1000
        elif self.kind == "detect":
            t0 = time.perf_counter()
            self.agent.perceptor.perceive(img)
            t["see"] = (time.perf_counter() - t0) * 1000
        else:                                        # cv: every OpenCV image op, no network
            t0 = time.perf_counter()
            cf = self.stage1.process(img, frame_id)
            x = self.stage1.detector_input(img, cf, "gain")
            if cf.measured:
                self.stage1.ground_view(cf)
            lb = cv2.resize(x, (640, 640), interpolation=cv2.INTER_LINEAR)          # letterbox resize
            blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (640, 640), swapRB=True)
            t["stage1"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            clahe = cv2.createCLAHE(2.0, (8, 8)).apply(cf.gray)                    # evidence view
            vis = img.copy()
            for i in range(4):
                cv2.rectangle(vis, (40 + 60 * i, 40), (80 + 60 * i, 80), (0, 170, 235), 2)
            ok, _ = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            t["render"] = (time.perf_counter() - t0) * 1000
            del blob, clahe
        t["total"] = sum(v for k, v in t.items())
        return t


def survey_hour(mean_ms: float, price_hour: float, ping_rate_hz: float, pings_per_frame: int,
                channels: int) -> dict:
    frames = ping_rate_hz * 3600.0 / pings_per_frame * channels
    compute_s = frames * mean_ms / 1000.0
    return {"frames": round(frames, 1), "compute_s": round(compute_s, 2),
            "real_time_factor": round(compute_s / 3600.0, 5),
            "usd": round(compute_s * price_hour / 3600.0, 6) if price_hour else None}


def main():
    ap = argparse.ArgumentParser(description="Benchmark the DEPTH product workload per stage (COOL case).")
    ap.add_argument("--frames", default=str(REPO / "webui" / "samples"), help="dir of sonar frames")
    ap.add_argument("--n", type=int, default=200, help="timed frames (the set is cycled)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--workload", choices=["product", "cv", "detect"], default="product")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("DEPTH_THREADS", "0") or 0),
                    help="cv2.setNumThreads (0 = OpenCV default = all cores)")
    ap.add_argument("--label", required=True, help="e.g. baseline_x86 | graviton_stock | graviton_cool")
    ap.add_argument("--price-hour", type=float, default=0.0, help="instance on-demand $/h (+ software fee)")
    ap.add_argument("--ping-rate-hz", type=float, default=15.0, help="sonar ping rate (survey setting)")
    ap.add_argument("--pings-per-frame", type=int, default=640)
    ap.add_argument("--channels", type=int, default=2, help="port + starboard")
    ap.add_argument("--out", default=str(REPO / "runs" / "bench"))
    a = ap.parse_args()

    from src.detection.infer import configure_runtime, DEFAULT_ONNX
    from src.detection.calibration import calibration_path
    from .fingerprint import fingerprint
    runtime = configure_runtime(a.threads if a.threads > 0 else None)

    frames, manifest = load_frames(Path(a.frames))
    if not frames:
        raise SystemExit(f"no frames in {a.frames}")
    wl = Workload(a.workload)
    for i in range(a.warmup):
        wl.run(*frames[i % len(frames)])
    per: dict[str, list[float]] = {}
    t0 = time.perf_counter()
    for i in range(a.n):
        for k, v in wl.run(*frames[i % len(frames)]).items():
            per.setdefault(k, []).append(v)
    wall = time.perf_counter() - t0

    stages = {k: _stats(v) for k, v in per.items()}
    mean_ms = stages["total"]["mean"]
    out = {
        "label": a.label, "workload": a.workload, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fingerprint": fingerprint(DEFAULT_ONNX, calibration_path()),
        "runtime": runtime,
        "config": {"n": a.n, "warmup": a.warmup, "threads_requested": a.threads, "frames_dir": a.frames,
                   "unique_frames": len(frames), "input_manifest_sha256": manifest},
        "stages_ms": stages,
        "throughput_fps": round(a.n / wall, 2),
        "cost": {"price_hour_usd": a.price_hour,
                 "usd_per_1k_frames": round(mean_ms / 1000.0 * a.price_hour / 3600.0 * 1000, 6) if a.price_hour else None,
                 "survey_hour": survey_hour(mean_ms, a.price_hour, a.ping_rate_hz, a.pings_per_frame, a.channels),
                 "assumptions": {"ping_rate_hz": a.ping_rate_hz, "pings_per_frame": a.pings_per_frame,
                                 "channels": a.channels}},
    }
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    fp = od / f"{a.label}__{a.workload}.json"
    fp.write_text(json.dumps(out, indent=2))
    fpr = out["fingerprint"]
    print(f"[{a.label}/{a.workload}] {fpr['host']['machine']} | OpenCV {fpr['opencv']['version']} "
          f"({'COOL' if fpr['opencv']['is_cool_path'] else 'stock'}) | threads {runtime['threads']}")
    for k, v in stages.items():
        print(f"  {k:<13} p50 {v['p50']:>8.2f} ms   p95 {v['p95']:>8.2f} ms   mean {v['mean']:>8.2f} ms")
    sh = out["cost"]["survey_hour"]
    print(f"  throughput {out['throughput_fps']} fps | survey-hour: {sh['frames']} frames -> {sh['compute_s']} s "
          f"compute (real-time factor {sh['real_time_factor']})" + (f" | ${sh['usd']}" if sh["usd"] is not None else ""))
    print(f"  -> {fp}")


if __name__ == "__main__":
    main()
