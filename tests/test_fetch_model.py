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


def test_missing_release_is_a_clear_message_not_a_traceback(tmp_path):
    """An unpublished release (HTTP 404) or no network: a SystemExit that says what to do, no partial file."""
    dest = tmp_path / "models" / "X" / "best.onnx"
    with pytest.raises(SystemExit) as e:
        fetch("X", (tmp_path / "not-published.onnx").as_uri(), "0" * 64, dest, quiet=True)
    msg = str(e.value)
    assert "could not download the X detector weights" in msg and "--url" in msg and "--sha256" in msg
    assert not dest.exists() and not dest.with_suffix(".part").exists()


@pytest.mark.parametrize("model,local", [("EXP-001", ("runs", "EXP-001", "weights", "best.onnx")),
                                         ("EXP-003", ("models", "EXP-003", "best.onnx"))])
def test_release_hash_matches_the_local_model(model, local):
    p = Path(__file__).resolve().parents[1].joinpath(*local)
    if not p.exists():
        pytest.skip("model absent")
    assert sha256_of(p) == RELEASES[model]["sha256"] and p.stat().st_size == RELEASES[model]["bytes"]
