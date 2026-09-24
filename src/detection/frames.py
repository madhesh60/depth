"""
frames.py — unique-frame helpers shared by evaluation and calibration.

The crab-pot archive was exported from Roboflow with augmentation baked in, so one sonar frame can
appear 2-12 times (crops, brightness, **rotations**). Every metric must count a frame once
(``docs/dataset_card.md``, v2b). These helpers mirror ``DATASET/scripts/build_dataset_v2b.py``:

* :func:`frame_key`    — the frame a file came from (everything before ``.rf.<hash>``, split
  prefixes and ``_png``/``_jpg``/``_aug`` suffixes removed);
* :func:`unique_frames` — keep ONE least-changed copy per frame (smallest black border = un-rotated,
  then brightness nearest the copies' median).
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

BLACK_LEVEL = 4


def frame_key(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"^crabpot_(train|valid|test)_", "", stem)
    stem = stem.split(".rf.")[0].replace("_aug0", "")
    return re.sub(r"(_png)?(_jpg)?$", "", stem)


def recording_of(name: str) -> str | None:
    m = re.search(r"rec0*(\d+)", frame_key(name).lower())
    return f"Rec{int(m.group(1))}" if m else None


def _copy_stats(p: Path) -> tuple[float, float]:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        return 1.0, 0.0
    return float((im.max(axis=2) <= BLACK_LEVEL).mean()), float(im.mean())


def unique_frames(paths: Iterable[Path]) -> list[Path]:
    """One least-changed copy per frame key (deterministic)."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        groups[frame_key(Path(p).name)].append(Path(p))
    out = []
    for _, ps in sorted(groups.items()):
        if len(ps) == 1:
            out.append(ps[0])
            continue
        st = {p: _copy_stats(p) for p in ps}
        med = float(np.median([v[1] for v in st.values()]))
        out.append(min(ps, key=lambda p: (round(st[p][0], 3), abs(st[p][1] - med), p.name)))
    return out
