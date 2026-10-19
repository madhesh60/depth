"""
cool_attest.py — the COOL execution receipt: evidence that THIS server process runs OpenCV from COOL on
an AWS Graviton instance launched from the COOL Marketplace AMI ("Verified COOL on Graviton").

Three layers, strongest first:

1. **AWS-signed identity** (IMDSv2 ``/latest/dynamic/instance-identity/{document,signature,rsa2048}``).
   AWS signs a JSON document that names the instance's AMI (``imageId``), its Marketplace product codes
   (``marketplaceProductCodes`` — the COOL listing's code), ``architecture`` (``arm64`` = Graviton),
   ``instanceType`` and ``region``. Anyone can check the signature offline against AWS's published
   certificate (``python -m src.bench.verify_cool <url>``); the server cannot forge it.
2. **What the process actually loaded** (Linux ``/proc/self/maps``): every OpenCV shared object mapped
   into this process, and whether ALL of them live under ``/opt/cool`` (no second OpenCV beside it).
   Reported by this open-source code: strong, but self-reported.
3. **The COOL build, read from the machine** (never typed in): version-like files under ``/opt/cool``,
   dpkg / pip package versions, and the lines of ``cv2.getBuildInformation()`` that name the build.

``receipt()`` is served at ``GET /api/cool``; ``summary()`` rides in ``/api/health``; the SHA-256 of the
signed document is stamped into every result's provenance, tying each output to the attested machine.
Off EC2 everything degrades to ``{"on_ec2": false}`` without touching the network.

The signed document contains the AWS account id and the instance's private IP. AWS does not treat
either as secret; serving the document unmodified is what makes the signature checkable.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import cv2

IMDS = "http://169.254.169.254"
COOL_ROOT = os.environ.get("DEPTH_COOL_ROOT", "/opt/cool")
_LOCK = threading.Lock()
_IDENTITY: dict = {}                      # cached signed identity (or {"_failed": t} when IMDS failed)
_BUILD_KEYS = re.compile(r"general configuration for opencv|version control|host:|target:|baseline:|dispatched|"
                         r"kleidi|neon|sve|arm compute|acl|openblas|lapack|parallel framework|dnn|"
                         r"cpu/hw features|compiler|cool", re.I)


# ---- layer 1: the AWS-signed instance identity ------------------------------------------------
def on_ec2() -> Optional[bool]:
    """True/False from the DMI vendor (no network); None when it cannot be told (then IMDS is tried)."""
    if os.environ.get("DEPTH_FORCE_IMDS") == "1":
        return True
    if not sys.platform.startswith("linux"):
        return False
    seen = False
    for p in ("/sys/devices/virtual/dmi/id/sys_vendor", "/sys/devices/virtual/dmi/id/board_vendor"):
        try:
            v = Path(p).read_text().strip().lower()
        except OSError:
            continue
        seen = True
        if "amazon" in v:
            return True
    return False if seen else None


def _imds(path: str, token: str, timeout: float) -> str:
    req = urllib.request.Request(f"{IMDS}{path}", headers={"X-aws-ec2-metadata-token": token})
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


def signed_identity(timeout: float = 1.0) -> dict:
    """``{"document", "signature", "rsa2048"}`` exactly as IMDSv2 serves them; ``{}`` off EC2.
    Cached after the first success; a failure on EC2 is retried after 60 s."""
    with _LOCK:
        if "document" in _IDENTITY:
            return dict(_IDENTITY)
        if time.time() - _IDENTITY.get("_failed", -1e9) < 60:
            return {}
        if on_ec2() is False:
            _IDENTITY["_failed"] = float("inf")             # not EC2: never ask again
            return {}
        try:
            req = urllib.request.Request(f"{IMDS}/latest/api/token", method="PUT",
                                         headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"})
            tok = urllib.request.urlopen(req, timeout=timeout).read().decode()
            got = {k: _imds(f"/latest/dynamic/instance-identity/{k}", tok, timeout)
                   for k in ("document", "signature", "rsa2048")}
        except Exception:
            _IDENTITY["_failed"] = time.time()
            return {}
        _IDENTITY.clear()
        _IDENTITY.update({"document": got["document"], "signature": got["signature"].strip(),
                          "rsa2048": got["rsa2048"].strip()})
        return dict(_IDENTITY)


def claims(document: str) -> dict:
    """The fields the receipt relies on, parsed from the signed document (trust them only after the
    signature checks out — ``verify_cool`` does that)."""
    d = json.loads(document)
    return {"architecture": d.get("architecture"), "instance_type": d.get("instanceType"),
            "image_id": d.get("imageId"), "marketplace_product_codes": d.get("marketplaceProductCodes") or [],
            "billing_products": d.get("billingProducts"), "region": d.get("region"),
            "availability_zone": d.get("availabilityZone"), "instance_id": d.get("instanceId"),
            "pending_time": d.get("pendingTime")}


def document_sha256(document: Optional[str]) -> Optional[str]:
    return hashlib.sha256(document.encode()).hexdigest() if document else None


# ---- layer 2: what this process mapped ---------------------------------------------------------
def _is_opencv(path: str) -> bool:
    base = path.rsplit("/", 1)[-1].lower()
    return base.startswith(("libopencv", "cv2")) or "/opencv" in path.lower()


def mapped_libraries(maps_text: Optional[str] = None, cool_root: str = COOL_ROOT) -> dict:
    """OpenCV objects mapped into this process and whether all of them come from ``cool_root``.
    ``maps_text`` (a ``/proc/<pid>/maps`` dump) is for tests; ``{"available": False}`` off Linux."""
    if maps_text is None:
        try:
            maps_text = Path("/proc/self/maps").read_text()
        except OSError:
            return {"available": False, "reason": "no /proc/self/maps on this OS"}
    roots = {cool_root.rstrip("/") + "/", os.path.realpath(cool_root).rstrip("/") + "/"}
    files = sorted({ln.split(None, 5)[5].strip() for ln in maps_text.splitlines()
                    if len(ln.split(None, 5)) == 6 and ln.split(None, 5)[5].startswith("/")})
    files = [f for f in files if ".so" in f or f.endswith((".dylib", ".pyd"))]
    under = lambda f: any(f.startswith(r) for r in roots)
    ocv = [f for f in files if _is_opencv(f)]
    outside = [f for f in ocv if not under(f)]
    return {"available": True, "cool_root": cool_root,
            "opencv_libs": ocv, "opencv_outside_cool": outside,
            "all_opencv_from_cool": bool(ocv) and not outside,
            "cool_tree_libs": [f for f in files if under(f)],
            "shared_objects_mapped": len(files)}


# ---- layer 3: the COOL build, read from the machine --------------------------------------------
def _small_text(p: Path, limit: int = 400) -> Optional[str]:
    try:
        if p.is_file() and p.stat().st_size <= 64_000:
            return p.read_text(errors="replace").strip()[:limit]
    except OSError:
        pass
    return None


def cool_build(cool_root: str = COOL_ROOT) -> dict:
    root = Path(cool_root)
    out: dict = {"cool_root_exists": root.is_dir()}
    if root.is_dir():
        try:
            out["cool_root_entries"] = sorted(p.name for p in root.iterdir())[:40]
            venvs = root / "venvs"
            if venvs.is_dir():
                out["cool_venvs"] = sorted(p.name for p in venvs.iterdir())[:20]
        except OSError:
            pass
        found = {}
        for p in list(root.glob("*")) + list(root.glob("*/*")):
            if re.search(r"version|release|manifest|build[-_]?info", p.name, re.I):
                t = _small_text(p)
                if t:
                    found[str(p)] = t
            if len(found) >= 12:
                break
        out["version_files"] = found
    try:
        r = subprocess.run(["dpkg-query", "-W", "-f", "${Package} ${Version}\n"], capture_output=True,
                           text=True, timeout=5)
        out["dpkg"] = [ln for ln in r.stdout.splitlines() if re.search(r"cool|opencv|kleidi", ln, re.I)][:20]
    except Exception:
        pass
    try:
        from importlib import metadata
        out["python_dists"] = sorted({f"{d.metadata['Name']} {d.version}" for d in metadata.distributions()
                                      if re.search(r"opencv|cool|kleidi|numpy", d.metadata.get("Name") or "", re.I)})
    except Exception:
        pass
    info = cv2.getBuildInformation()
    out["cv2_version"] = cv2.__version__
    out["cv2_build_lines"] = [ln.strip() for ln in info.splitlines() if _BUILD_KEYS.search(ln)][:40]
    return out


# ---- the receipt -------------------------------------------------------------------------------
def rehearsal() -> bool:
    """A stock-Ubuntu dress rehearsal of the stack (``infra/cfn/depth.yaml`` Rehearsal=true): NOT COOL."""
    return os.environ.get("DEPTH_REHEARSAL") == "1"


def receipt() -> dict:
    ident = signed_identity()
    doc = ident.get("document")
    cv2_file = getattr(cv2, "__file__", "") or ""
    return {
        "what": "COOL execution receipt - check it: python -m src.bench.verify_cool <this server's URL>",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rehearsal": rehearsal(),
        "on_ec2": bool(doc),
        "aws_signed_identity": ({"document": doc, "signature": ident["signature"], "rsa2048": ident["rsa2048"],
                                 "document_sha256": document_sha256(doc),
                                 "how_to_verify": "RSA-SHA256 'signature' or PKCS7 'rsa2048' against AWS's published "
                                                  "certificate for the region (EC2 User Guide: verify-iid)"}
                                if doc else None),
        "claims": claims(doc) if doc else None,
        "process": {"pid": os.getpid(), "python": sys.executable, "cv2_file": cv2_file,
                    "cv2_from_cool": cv2_file.startswith(COOL_ROOT.rstrip("/") + "/"),
                    **mapped_libraries()},
        "cool_build": cool_build(),
    }


def summary() -> dict:
    """The short form for ``/api/health`` (no network off EC2; cached on EC2)."""
    ident = signed_identity()
    doc = ident.get("document")
    c = claims(doc) if doc else {}
    maps = mapped_libraries()
    return {"rehearsal": rehearsal(), "on_ec2": bool(doc), "aws_signed": bool(doc),
            "architecture": c.get("architecture"), "instance_type": c.get("instance_type"),
            "image_id": c.get("image_id"), "marketplace_product_codes": c.get("marketplace_product_codes"),
            "opencv_all_from_cool": maps.get("all_opencv_from_cool") if maps.get("available") else None,
            "document_sha256": document_sha256(doc), "receipt": "/api/cool"}
