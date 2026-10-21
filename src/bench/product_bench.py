"""
product_bench.py — benchmark the workload DEPTH actually runs: per stage AND per OpenCV op, with
provenance, CPU use, peak memory, interleaved repeats and an output-parity record (review M-5).

Workloads
    product  exactly what ``/api/analyze`` runs per frame: decode (``cv2.imdecode``) → stage1
             (canonicalise: luminance · orientation · bottom track · detector input) → see (letterbox +
             blob → ``cv2.dnn`` forward → decode + class-aware NMS) → prove+decide (evidence, re-looks,
             tiers) → render (overlay + ``cv2.imencode``). Reported as the product stages AND as an op
             split — decode · stage1 · dnn_pre · dnn_forward · dnn_post · evidence · render — that shows
             where the time goes, i.e. where COOL can act (OpenCV image ops vs the network forward).
    cv       the product path's OpenCV image ops alone, no network: decode → stage1 → letterbox + blob →
             render. (The slant→ground ``remap`` is NOT here: it runs only in the 3D twin, not in analyze.)
    survey   a REAL raw sonar recording end to end — Humminbird read → sonogram chunks → seabed
             cross-check → the agent on every chunk → mission plan. Gives the MEASURED real-time factor
             (compute seconds per second of sonar) and $ per survey-hour = real-time factor × $/h.
             Default recording: PINGMapper's 1-h Solix file when fetched (``humminbird fetch --large``),
             else the 150-s sample.
    detect   decode + network only.

``--repeats R`` interleaves the workloads in ONE process (product, cv, survey, product, cv, survey, …) so
drift — thermal, noisy neighbours — lands evenly on all of them; each JSON keeps every repeat plus the
pooled per-frame stats. CPU utilisation = process CPU seconds / wall seconds / vCPUs over a timed loop;
peak RSS = the process high-water mark when that repeat ended (repeat 1 is clean for the first workload).
``--parity`` writes ``<label>__parity.json``: every detection (class, box, score, tier) per frame, so
``compare.py`` can check that COOL changes the speed, not the answers.

    python -m src.bench.product_bench --label laptop_x86 --workloads product,cv,survey --repeats 3 --parity \
        [--n 120] [--threads 0] [--price-hour 0.0] [--recording PATH] [--frames webui/samples]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RECORDINGS = (REPO / "DATASET" / "external" / "pingmapper_large" / "Test-Large-DS.DAT",
              REPO / "DATASET" / "external" / "pingmapper_sample" / "Test-Small-DS.DAT")
OPS = ("decode", "stage1", "dnn_pre", "dnn_forward", "dnn_post", "evidence", "render")


def load_frames(src: Path) -> tuple[list[tuple[str, bytes]], str]:
    """Encoded frame bytes (decode is part of the timed work) + a content manifest sha256."""
    paths = sorted(p for p in (src.rglob("*") if src.is_dir() else [src]) if p.suffix.lower() in IMG_EXTS)
    frames, h = [], hashlib.sha256()
    for p in paths:
        b = p.read_bytes()
        frames.append((p.stem, b))
        h.update(p.name.encode())
        h.update(hashlib.sha256(b).digest())
    return frames, h.hexdigest()


def _stats(v: list[float]) -> dict:
    a = np.asarray(v, dtype=float)
    return {"p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "mean": round(float(a.mean()), 2)}


def peak_rss_mb() -> float | None:
    """Process high-water resident memory (MB)."""
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / (1e6 if sys.platform == "darwin" else 1024.0), 1)
    except Exception:
        try:
            import psutil
            mi = psutil.Process().memory_info()
            return round(getattr(mi, "peak_wset", mi.rss) / 1e6, 1)
        except Exception:
            return None


def candidate_record(c) -> dict:
    """What parity compares: class, box, detector score, tier (+ the evidence score, reported only)."""
    ev = getattr(c, "evidence", None)
    return {"cls": c.cls_name, "bbox": [int(v) for v in c.bbox], "conf": round(float(c.conf), 5),
            "tier": c.verdict.value if c.verdict is not None else None,
            "evidence_score": round(float(ev.evidence_score), 5) if ev is not None else None}


class Workload:
    """``run(frame_id, bytes)`` → per-frame ms by stage (+ the op split for ``product``)."""

    def __init__(self, kind: str, pipe=None):
        self.kind = kind
        from src.cv_pipeline.canonical import Canonicaliser
        self.stage1 = Canonicaliser()
        self.pipe = pipe
        self.last = None
        if kind in ("product", "detect") and self.pipe is None:
            from src.agentic.pipeline import AgenticPipeline
            self.pipe = AgenticPipeline()
            self.pipe.agent.perceptor.warmup()

    def run(self, frame_id: str, data: bytes) -> dict[str, float]:
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        t["decode"] = (time.perf_counter() - t0) * 1000

        if self.kind == "product":
            from src.agentic.agent import render
            det = self.pipe.agent.perceptor.detector
            for k in det.prof:
                det.prof[k] = 0
            res = self.pipe.run_frame(img, frame_id=frame_id)       # what /api/analyze calls
            self.last = res
            t["stage1"] = res.stage_ms["stage1"]
            t["see"] = res.stage_ms["see"]
            t["prove_decide"] = res.stage_ms["prove_decide"]
            t0 = time.perf_counter()
            cv2.imencode(".jpg", render(img, res), [cv2.IMWRITE_JPEG_QUALITY, 85])
            t["render"] = (time.perf_counter() - t0) * 1000
            p = dict(det.prof)
            t["dnn_pre"], t["dnn_forward"], t["dnn_post"] = p["pre_ms"], p["forward_ms"], p["post_ms"]
            t["evidence"] = max(0.0, t["see"] + t["prove_decide"] - p["pre_ms"] - p["forward_ms"] - p["post_ms"])
            t["forwards"] = float(p["forwards"])
        elif self.kind == "detect":
            t0 = time.perf_counter()
            self.pipe.agent.perceptor.perceive(img)
            t["see"] = (time.perf_counter() - t0) * 1000
        else:                                        # cv: the product path's OpenCV image ops, no network
            t0 = time.perf_counter()
            cf = self.stage1.process(img, frame_id)
            x = self.stage1.detector_input(img, cf, "gain")
            t["stage1"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            from src.detection.infer import _letterbox, _CAL
            canvas, *_ = _letterbox(x if x.ndim == 3 else cv2.cvtColor(x, cv2.COLOR_GRAY2BGR), _CAL.imgsz)
            cv2.dnn.blobFromImage(canvas, 1 / 255.0, (_CAL.imgsz, _CAL.imgsz), swapRB=True)
            t["dnn_pre"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            vis = img.copy()
            for i in range(4):
                cv2.rectangle(vis, (40 + 60 * i, 40), (80 + 60 * i, 80), (0, 170, 235), 2)
            cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            t["render"] = (time.perf_counter() - t0) * 1000
        t["total"] = sum(v for k, v in t.items()
                         if k in ("decode", "stage1", "see", "prove_decide", "render", "dnn_pre"))
        if self.kind == "product":                       # dnn_pre is inside see: do not count it twice
            t["total"] -= t["dnn_pre"]
        return t


class SurveyWorkload:
    """A whole raw recording, as the Survey tab's 'Raw recording' source runs it."""

    def __init__(self, dat: Path, pipe):
        from src.cv_pipeline.humminbird import read_recording, validate
        self.dat, self.pipe = dat, pipe
        val = validate(read_recording(dat))
        ch = val["channels"].get("port") or next(iter(val["channels"].values()))
        self.duration_s = float(ch["duration_s"])
        self.sha256 = hashlib.sha256(dat.read_bytes()).hexdigest()

    def run(self) -> dict[str, float]:
        from src.cv_pipeline.humminbird import read_recording, frames_and_track
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        rec = read_recording(self.dat)
        t["read"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        frames, track, _meta = frames_and_track(rec)                # sonogram chunks + seabed cross-check
        t["sonogram_crosscheck"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        res = self.pipe.run_survey(frames, track=track, survey_id="bench")
        t["agent_plan"] = (time.perf_counter() - t0) * 1000
        t["total"] = t["read"] + t["sonogram_crosscheck"] + t["agent_plan"]
        t["frames"] = float(len(frames))
        t["hazards"] = float(sum(len(fr.candidates) for fr in res.frames))
        return t


def _rel(path: str) -> str:
    """Repo-relative when inside the repo (results are committed; no local user paths)."""
    try:
        return Path(path).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return Path(path).name


def survey_hour(mean_ms: float, price_hour: float, ping_rate_hz: float, pings_per_frame: int,
                channels: int) -> dict:
    """The MODELLED survey-hour for a per-frame workload (the survey workload measures it instead)."""
    frames = ping_rate_hz * 3600.0 / pings_per_frame * channels
    compute_s = frames * mean_ms / 1000.0
    return {"frames": round(frames, 1), "compute_s": round(compute_s, 2),
            "real_time_factor": round(compute_s / 3600.0, 5),
            "usd": round(compute_s * price_hour / 3600.0, 6) if price_hour else None}


def _timed(fn, *args) -> tuple[object, float, float]:
    w0, c0 = time.perf_counter(), time.process_time()
    out = fn(*args)
    return out, time.perf_counter() - w0, time.process_time() - c0


def main():
    ap = argparse.ArgumentParser(description="Benchmark the DEPTH product workload per stage (COOL case).")
    ap.add_argument("--frames", default=str(REPO / "webui" / "samples"), help="dir of sonar frames")
    ap.add_argument("--n", type=int, default=120, help="timed frames per repeat (the set is cycled)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--workloads", default="product",
                    help="comma list of product,cv,survey,detect (interleaved per repeat)")
    ap.add_argument("--workload", choices=["product", "cv", "detect", "survey"], help="(single; legacy)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("DEPTH_THREADS", "0") or 0),
                    help="cv2.setNumThreads (0 = OpenCV default = all cores)")
    ap.add_argument("--label", required=True, help="e.g. baseline_x86 | graviton_stock | graviton_cool")
    ap.add_argument("--price-hour", type=float, default=0.0, help="instance on-demand $/h (+ software fee)")
    ap.add_argument("--price-source", default="", help="where the price came from (recorded)")
    ap.add_argument("--recording", default="", help="raw .DAT for the survey workload (default: 1-h, else 150-s)")
    ap.add_argument("--parity", action="store_true", help="write <label>__parity.json (detections per frame)")
    ap.add_argument("--ping-rate-hz", type=float, default=15.0, help="sonar ping rate (modelled survey-hour)")
    ap.add_argument("--pings-per-frame", type=int, default=640)
    ap.add_argument("--channels", type=int, default=2, help="port + starboard")
    ap.add_argument("--out", default=str(REPO / "benchmarks"))
    a = ap.parse_args()
    kinds = [a.workload] if a.workload else [k.strip() for k in a.workloads.split(",") if k.strip()]

    from src.detection.infer import configure_runtime, runtime_info, DEFAULT_ONNX
    from src.detection.calibration import calibration_path
    from .fingerprint import fingerprint
    configure_runtime(a.threads if a.threads > 0 else None)
    fpr = fingerprint(DEFAULT_ONNX, calibration_path())
    vcpus = os.cpu_count() or 1

    frames, manifest = load_frames(Path(a.frames))
    if not frames and set(kinds) - {"survey"}:
        raise SystemExit(f"no frames in {a.frames}")
    pipe = None
    if set(kinds) & {"product", "detect", "survey"}:
        from src.agentic.pipeline import AgenticPipeline
        pipe = AgenticPipeline()
        pipe.agent.perceptor.warmup()
    wls = {k: Workload(k, pipe) for k in kinds if k != "survey"}
    survey = None
    if "survey" in kinds:
        dat = Path(a.recording) if a.recording else next((p for p in RECORDINGS if p.exists()), None)
        if dat is None or not dat.exists():
            raise SystemExit("survey workload: no recording (python -m src.cv_pipeline.humminbird fetch [--large])")
        survey = SurveyWorkload(dat, pipe)

    parity = {}
    for k, wl in wls.items():                                   # warm-up (+ the parity pass for product)
        for i in range(max(a.warmup, len(frames) if (a.parity and k == "product") else 0)):
            fid, data = frames[i % len(frames)]
            wl.run(fid, data)
            if a.parity and k == "product" and i < len(frames):
                parity[fid] = [candidate_record(c) for c in wl.last.candidates]

    reps: dict[str, list[dict]] = {k: [] for k in kinds}
    pooled: dict[str, dict[str, list[float]]] = {k: {} for k in kinds}
    for r in range(a.repeats):
        for k in kinds:                                         # interleaved: product, cv, survey, product, …
            if k == "survey":
                t, wall, cpu = _timed(survey.run)
                rt = t["total"] / 1000.0 / survey.duration_s
                reps[k].append({"repeat": r + 1, "wall_s": round(wall, 2), "cpu_util": round(cpu / wall / vcpus, 3),
                                "peak_rss_mb": peak_rss_mb(), "real_time_factor": round(rt, 5),
                                **{f"{m}_ms": round(v, 1) for m, v in t.items() if m not in ("frames", "hazards")},
                                "frames": int(t["frames"]), "hazards": int(t["hazards"])})
                for m, v in t.items():
                    pooled[k].setdefault(m, []).append(v)
                continue
            per: dict[str, list[float]] = {}

            def loop():
                for i in range(a.n):
                    for m, v in wls[k].run(*frames[i % len(frames)]).items():
                        per.setdefault(m, []).append(v)
            _, wall, cpu = _timed(loop)
            reps[k].append({"repeat": r + 1, "wall_s": round(wall, 2), "fps": round(a.n / wall, 2),
                            "cpu_util": round(cpu / wall / vcpus, 3), "peak_rss_mb": peak_rss_mb(),
                            "total_ms": _stats(per["total"])})
            for m, v in per.items():
                pooled[k].setdefault(m, []).extend(v)

    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    common = {"label": a.label, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "fingerprint": fpr,
              "runtime": runtime_info()}                          # + the cv2.dnn engine that loaded
    for k in kinds:
        R = reps[k]
        if k == "survey":
            rtf = [x["real_time_factor"] for x in R]
            med = float(np.median(rtf))
            out = {**common, "workload": "survey",
                   "config": {"repeats": a.repeats, "recording": _rel(str(survey.dat)), "recording_sha256": survey.sha256,
                              "sonar_duration_s": survey.duration_s, "threads_requested": a.threads},
                   "stages_ms": {m: _stats(v) for m, v in pooled[k].items() if m not in ("frames", "hazards")},
                   "repeats": R,
                   "survey": {"sonar_duration_s": survey.duration_s, "frames": R[0]["frames"],
                              "real_time_factor_median": round(med, 5),
                              "real_time_factor_range": [round(min(rtf), 5), round(max(rtf), 5)],
                              "compute_s_per_survey_hour": round(med * 3600, 1),
                              "usd_per_survey_hour": round(med * a.price_hour, 6) if a.price_hour else None},
                   "cost": {"price_hour_usd": a.price_hour, "price_source": a.price_source}}
        else:
            stages = {m: _stats(v) for m, v in pooled[k].items() if m != "forwards"}
            mean_ms = stages["total"]["mean"]
            fps = [x["fps"] for x in R]
            out = {**common, "workload": k,
                   "config": {"n": a.n, "repeats": a.repeats, "warmup": a.warmup, "threads_requested": a.threads,
                              "frames_dir": _rel(a.frames), "unique_frames": len(frames),
                              "input_manifest_sha256": manifest},
                   "stages_ms": stages,
                   "ops_ms": {m: stages[m] for m in OPS if m in stages} if k == "product" else None,
                   "forwards_per_frame": round(float(np.mean(pooled[k]["forwards"])), 2) if "forwards" in pooled[k] else None,
                   "repeats": R,
                   "throughput_fps": round(float(np.median(fps)), 2),
                   "throughput_fps_range": [min(fps), max(fps)],
                   "cpu_util": round(float(np.median([x["cpu_util"] for x in R])), 3),
                   "peak_rss_mb": R[0]["peak_rss_mb"],
                   "cost": {"price_hour_usd": a.price_hour, "price_source": a.price_source,
                            "usd_per_1k_frames": round(mean_ms / 1000.0 * a.price_hour / 3600.0 * 1000, 6) if a.price_hour else None,
                            "survey_hour": survey_hour(mean_ms, a.price_hour, a.ping_rate_hz, a.pings_per_frame, a.channels),
                            "assumptions": {"ping_rate_hz": a.ping_rate_hz, "pings_per_frame": a.pings_per_frame,
                                            "channels": a.channels}}}
        fp = od / f"{a.label}__{k}.json"
        fp.write_text(json.dumps(out, indent=2))
        _print(out, fp)
    from . import cool_attest
    rc = cool_attest.receipt()
    if rc.get("aws_signed_identity"):                            # the machine's AWS-signed identity, checkable
        fp = od / f"{a.label}__receipt.json"                     # with: python -m src.bench.verify_cool <file>
        fp.write_text(json.dumps(rc, indent=2))
        print(f"  receipt (AWS-signed identity of this machine) -> {fp}")
    if a.parity and parity:
        fp = od / f"{a.label}__parity.json"
        fp.write_text(json.dumps({**common, "workload": "parity", "input_manifest_sha256": manifest,
                                  "frames": parity}, indent=2))
        print(f"  parity: {sum(len(v) for v in parity.values())} detections on {len(parity)} frames -> {fp}")


def _print(out: dict, fp: Path) -> None:
    f = out["fingerprint"]
    print(f"[{out['label']}/{out['workload']}] {f['host']['machine']} | OpenCV {f['opencv']['version']} "
          f"({'COOL' if f['opencv']['is_cool_path'] else 'stock'}) | threads {out['runtime']['threads']}")
    for k, v in out["stages_ms"].items():
        print(f"  {k:<20} p50 {v['p50']:>9.2f} ms   p95 {v['p95']:>9.2f} ms   mean {v['mean']:>9.2f} ms")
    if out["workload"] == "survey":
        s = out["survey"]
        print(f"  {s['sonar_duration_s']:.0f} s of sonar, {s['frames']} frames: real-time factor {s['real_time_factor_median']} "
              f"(range {s['real_time_factor_range']}) -> {s['compute_s_per_survey_hour']} s compute per survey-hour"
              + (f" = ${s['usd_per_survey_hour']}" if s["usd_per_survey_hour"] is not None else ""))
    else:
        print(f"  throughput {out['throughput_fps']} fps (range {out['throughput_fps_range']}) | cpu {out['cpu_util']:.0%} "
              f"of {f['host']['vcpus']} vCPU | peak RSS {out['peak_rss_mb']} MB")
    print(f"  -> {fp}")


if __name__ == "__main__":
    main()
