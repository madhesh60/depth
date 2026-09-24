"""
fingerprint.py — where did this run happen, on which OpenCV build? (COOL provenance)

A benchmark number is only evidence if it says exactly what produced it. This records:

* **OpenCV provenance** — version, ``cv2.__file__`` and ``is_cool_path`` (COOL is an AMI whose
  OpenCV lives under ``/opt/cool``; that path — not "KleidiCV in the build", which the stock Arm
  wheel also has — is the proof a run used COOL), plus build flags for information;
* **host** — CPU architecture (``aarch64`` = Graviton), vCPUs, CPU model;
* **EC2 identity** (best effort, IMDSv2, 0.3 s timeout) — instance type + AMI id;
* **inputs** — git commit, model sha256, calibration sha256.
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional

import cv2

REPO = Path(__file__).resolve().parents[2]


def sha256_file(p: Path, chunk: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def opencv_provenance() -> dict:
    info = cv2.getBuildInformation()
    f = getattr(cv2, "__file__", "") or ""
    return {
        "version": cv2.__version__,
        "cv2_file": f,
        "is_cool_path": "/opt/cool" in f,
        "threads": cv2.getNumThreads(),
        "build_flags": {k: (k.lower() in info.lower()) for k in ("KleidiCV", "NEON", "IPP", "TBB", "OpenMP")},
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "cpu part")):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def ec2_identity(timeout: float = 0.3) -> dict:
    """Instance type + AMI via IMDSv2; ``{}`` off EC2 (fast fail, never raises)."""
    try:
        req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        tok = urllib.request.urlopen(req, timeout=timeout).read().decode()
        out = {}
        for key in ("instance-type", "ami-id", "placement/region"):
            r = urllib.request.Request(f"http://169.254.169.254/latest/meta-data/{key}",
                                       headers={"X-aws-ec2-metadata-token": tok})
            out[key.split("/")[-1]] = urllib.request.urlopen(r, timeout=timeout).read().decode()
        return out
    except Exception:
        return {}


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def host() -> dict:
    return {"machine": platform.machine(), "system": platform.system(), "python": platform.python_version(),
            "vcpus": os.cpu_count(), "cpu_model": _cpu_model()}


def fingerprint(model: Optional[Path] = None, calibration: Optional[Path] = None, ec2: bool = True) -> dict:
    return {"opencv": opencv_provenance(), "host": host(), "ec2": ec2_identity() if ec2 else {},
            "git_commit": git_commit(),
            "model_sha256": sha256_file(model) if model else None,
            "calibration_sha256": sha256_file(calibration) if calibration else None}
