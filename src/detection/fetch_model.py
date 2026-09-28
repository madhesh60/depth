"""
fetch_model.py — get the trained detector a fresh clone needs (weights are not in git: 38 MB).

    python -m src.detection.fetch_model                  # EXP-001 (the deployed model)
    python -m src.detection.fetch_model --model EXP-002  # once it is released
    python -m src.detection.fetch_model --url <url> --sha256 <hex>   # a mirror (S3, Zenodo, …)

Downloads to ``models/<MODEL>/best.onnx`` (the first place the runtime looks, see
``infer._default_onnx``), streaming to a temporary file, and keeps it **only if its SHA-256 matches**
the published hash — the same hash every provenance stamp carries, so a result can be traced to the
exact weights. An existing file with the right hash is left alone.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RELEASES = {
    "EXP-001": {"url": "https://github.com/madhesh60/depth/releases/download/exp001-v1/best.onnx",
                "sha256": "55f827db9bd5cbf89a87d50c767ecbf17594b9c8c654a428a80118ab3537c19e",
                "bytes": 37932951, "license": "AGPL-3.0 (Ultralytics YOLO11s, trained on dataset v1)"},
}


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(model: str = "EXP-001", url: str | None = None, sha256: str | None = None,
          dest: Path | None = None, quiet: bool = False) -> Path:
    rel = RELEASES.get(model, {})
    url, want = url or rel.get("url"), (sha256 or rel.get("sha256") or "").lower()
    if not url or not want:
        raise SystemExit(f"no published release for {model}: pass --url and --sha256")
    dest = dest or REPO / "models" / model / "best.onnx"
    if dest.exists() and sha256_of(dest) == want:
        if not quiet:
            print(f"{dest} already present (sha256 ok)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    h, n = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        for b in iter(lambda: r.read(1 << 20), b""):
            f.write(b); h.update(b); n += len(b)
            if not quiet and total:
                print(f"\r  {n / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    if not quiet:
        print()
    if h.hexdigest() != want:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"sha256 mismatch for {url}: got {h.hexdigest()[:16]}…, expected {want[:16]}… - not installed")
    os.replace(tmp, dest)
    if not quiet:
        print(f"installed {dest} ({n / 1e6:.1f} MB, sha256 {want[:16]}…)")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default=os.environ.get("DEPTH_MODEL", "EXP-001"))
    ap.add_argument("--url")
    ap.add_argument("--sha256")
    a = ap.parse_args()
    fetch(a.model, a.url, a.sha256)


if __name__ == "__main__":
    sys.exit(main())
