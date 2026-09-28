"""Tests for src/detection/fetch_model.py: keep a download only when its SHA-256 matches."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.fetch_model import RELEASES, fetch, sha256_of


def test_downloads_and_verifies(tmp_path):
    src = tmp_path / "remote.onnx"
    src.write_bytes(b"onnx-bytes" * 1000)
    want = hashlib.sha256(src.read_bytes()).hexdigest()
    dest = tmp_path / "models" / "X" / "best.onnx"
    assert fetch("X", src.as_uri(), want, dest, quiet=True) == dest and sha256_of(dest) == want
    assert fetch("X", "file:///does/not/matter", want, dest, quiet=True) == dest     # present + ok: no download


def test_mismatch_is_not_installed(tmp_path):
    src = tmp_path / "remote.onnx"
    src.write_bytes(b"tampered")
    dest = tmp_path / "best.onnx"
    with pytest.raises(SystemExit):
        fetch("X", src.as_uri(), "0" * 64, dest, quiet=True)
    assert not dest.exists() and not dest.with_suffix(".part").exists()


def test_release_hash_matches_the_deployed_model():
    local = Path(__file__).resolve().parents[1] / "runs" / "EXP-001" / "weights" / "best.onnx"
    if not local.exists():
        pytest.skip("model absent")
    assert sha256_of(local) == RELEASES["EXP-001"]["sha256"] and local.stat().st_size == RELEASES["EXP-001"]["bytes"]
