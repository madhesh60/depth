"""
The server's upload caches stay bounded on a long-running public demo (frames and their names).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.dashboard.service as S


def _upload(i: int) -> SimpleNamespace:
    img = np.full((32, 32, 3), i % 255, np.uint8)
    ok, buf = cv2.imencode(".png", img)
    return SimpleNamespace(content_type="image/png", filename=f"frame_{i:04d}.png", file=io.BytesIO(buf.tobytes()))


def test_frames_and_names_stay_bounded(monkeypatch):
    monkeypatch.setattr(S, "MAX_UPLOAD_CACHE", 3)
    monkeypatch.setattr(S, "_UPLOADS", type(S._UPLOADS)())
    monkeypatch.setattr(S, "_UPLOAD_SHA", type(S._UPLOAD_SHA)())
    for i in range(40):
        S._decode_upload(_upload(i))
    assert len(S._UPLOADS) == 3 and len(S._UPLOAD_SHA) == 12
    assert "frame_0039" in S._UPLOAD_SHA and "frame_0000" not in S._UPLOAD_SHA      # the newest names are kept
