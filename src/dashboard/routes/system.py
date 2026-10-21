"""
routes/system.py — health, the COOL receipt, metrics, the sample frames and the failure catalogue.
"""
from __future__ import annotations

import os

import cv2
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from .. import samples as samples_mod
from src.agentic.approvals import pin_required
from src.agentic.core import calibrated_modes
from src.agentic.incidents import DRILLS, catalogue as incident_catalogue
from src.bench import cool_attest
from src.detection.calibration import load_calibration
from src.detection.infer import DEFAULT_ONNX, runtime_info
from .. import service as S

router = APIRouter()


# ---- API ---------------------------------------------------------------------------------------
@router.get("/api/health")
def health():
    cal = load_calibration()
    summary = cal.summary()
    summary["detector_floor"] = cal.conf.get(cal.guaranteed_class)
    cv2_file = getattr(cv2, "__file__", "") or ""
    return {
        "status": "ok", "opencv": cv2.__version__,
        "cv2_file": cv2_file, "is_cool_path": "/opt/cool" in cv2_file,
        "runtime": runtime_info(),                  # threads + the cv2.dnn engine that loaded
        # models/<M>/best.onnx (fetch_model) or runs/<M>/weights/best.onnx (a training run)
        "model": DEFAULT_ONNX.parent.name if DEFAULT_ONNX.parent.parent.name == "models" else DEFAULT_ONNX.parent.parent.name,
        "model_loaded": S._MODEL_STATE["loaded"], "model_loading": S._MODEL_STATE["loading"],
        "model_error": S._MODEL_STATE["error"], "warmup_ms": S._MODEL_STATE["warmup_ms"],
        "samples": len(samples_mod.list_samples()), "calibration": summary, "jobs": S._JOBS.counts(),
        "limits": {"max_upload_mb": S.MAX_UPLOAD_MB, "max_frames": S.MAX_FRAMES, "max_sync_frames": S.MAX_SYNC_FRAMES},
        "recording": {"available": S.RECORDING_DAT.exists(), "name": S.RECORDING_DAT.stem},
        "approver_pin_required": pin_required(),
        # where a person finds the PIN, e.g. "judges: the PIN is in the submission form" (never the PIN itself)
        "approver_pin_hint": os.environ.get("DEPTH_APPROVER_HINT", "")[:120] or None,
        "modes": [m.to_dict() for m in calibrated_modes(cal)],
        "ms_per_pass": S._PIPELINE.ms_per_pass() if S._PIPELINE is not None else None,
        "interactive_s": S.INTERACTIVE_S,
        "cool": cool_attest.summary(),
    }


@router.get("/api/cool")
def cool_receipt():
    """The COOL execution receipt (src/bench/cool_attest.py): the AWS-signed instance identity document
    (AMI, Marketplace product code, arm64, instance type) exactly as IMDS serves it, the OpenCV objects this
    process mapped, and the COOL build read from the machine. Verify: python -m src.bench.verify_cool <url>"""
    return cool_attest.receipt()


@router.get("/api/metrics")
def metrics():
    """Rolling per-stage latency on this server + host / OpenCV (COOL) provenance."""
    out = S._METRICS.summary()
    out["runtime"] = runtime_info()
    return out


@router.get("/api/samples")
def samples():
    return {"samples": samples_mod.list_samples()}


@router.get("/api/sample_thumb/{sample_id}")
def sample_thumb(sample_id: str, size: int = Query(220, ge=32, le=640)):
    """Downscaled JPEG of a sample frame — for the gallery cards (real sonar, not a placeholder)."""
    p = samples_mod.sample_path(sample_id)
    if not p or not p.exists():
        raise HTTPException(404, f"sample not found: {sample_id}")
    img = cv2.imread(str(p))
    if img is None:
        raise HTTPException(400, "could not read sample")
    scale = size / max(1, max(img.shape[:2]))
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise HTTPException(500, "encode failed")
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"})


@router.get("/api/incidents")
def incidents_catalogue():
    """The failure modes DEPTH handles (fallback + human action) and the drills the studio can run."""
    return {"catalogue": incident_catalogue(), "drills": list(DRILLS)}
