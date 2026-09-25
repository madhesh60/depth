"""
provenance.py — every result says exactly what produced it (observability, review §3.9 / Agentic-Vision
"observability" criterion; ties the COOL provenance into the product outputs).

``stamp()`` returns: model name + ONNX sha256, calibration sha256 (the promises in force), OpenCV
version + where ``cv2`` was loaded from (``/opt/cool`` ⇒ COOL), code commit, host architecture / EC2
type, UTC time. It is attached to ``/api/analyze`` responses and to every exported report, and heads
the agent decision log (``trace`` export) — so any hazard in a GeoJSON can be traced back to the model,
thresholds, OpenCV build and code that produced it.

Hashes are computed once per process (the ONNX is ~38 MB).
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=4)
def _static(model_path: str, calibration_path: str) -> dict:
    from src.bench.fingerprint import sha256_file, opencv_provenance, host, git_commit, ec2_identity
    ocv = opencv_provenance()
    h = host()
    ec2 = ec2_identity() if os.environ.get("DEPTH_PROVENANCE_EC2", "1") == "1" else {}
    return {
        "model_onnx_sha256": sha256_file(Path(model_path)),
        "calibration_sha256": sha256_file(Path(calibration_path)),
        "opencv": {"version": ocv["version"], "cv2_file": ocv["cv2_file"], "is_cool_path": ocv["is_cool_path"]},
        "host": {"machine": h["machine"], "vcpus": h["vcpus"], "instance_type": ec2.get("instance-type")},
        "code_commit": git_commit(),
    }


def stamp() -> dict:
    from src.detection.infer import DEFAULT_ONNX
    from src.detection.calibration import calibration_path, load_calibration
    cal = load_calibration()
    s = dict(_static(str(DEFAULT_ONNX), str(calibration_path())))
    s["model"] = cal.data.get("model")
    s["guarantees"] = {k: (cal.guarantees or {}).get(k) for k in ("recall_promise", "precision_promise")}
    s["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return s


def short(s: dict) -> str:
    """One-line human form for GPX/KML descriptions and the UI footer."""
    ocv = s["opencv"]
    return (f"DEPTH {s.get('code_commit') or '?'} · model {s.get('model')} "
            f"{(s.get('model_onnx_sha256') or '')[:12]} · calib {(s.get('calibration_sha256') or '')[:12]} · "
            f"OpenCV {ocv['version']} {'COOL' if ocv['is_cool_path'] else 'stock'} · {s['host']['machine']}"
            + (f" {s['host']['instance_type']}" if s['host'].get('instance_type') else "") + f" · {s['created_utc']}")
