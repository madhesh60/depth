"""
Tests for the single threshold source (src/detection/calibration.py) and class-aware NMS.
Standalone: python tests/test_calibration.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.calibration import DEFAULTS, load_calibration, save_calibration
from src.detection.infer import nms, PER_CLASS_CONF


def test_shipped_calibration_matches_runtime():
    cal = load_calibration()
    assert cal.conf == PER_CLASS_CONF                        # detector reads the same file
    assert 0 < cal.iou_nms < 1 and cal.class_aware_nms is True


def test_missing_file_falls_back_to_defaults():
    cal = load_calibration(path=Path(tempfile.gettempdir()) / "does_not_exist_depth.json")
    assert cal.conf == DEFAULTS["detector"]["conf"] and cal.path is None
    assert cal.is_fitted is False


def test_partial_file_is_merged_and_save_roundtrips():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "calibration.json"
        p.write_text(json.dumps({"tiers": {"tau_review": 0.2, "tau_confirm": 0.6}}))
        cal = load_calibration(path=p)
        assert cal.tau_review == 0.2 and cal.tau_confirm == 0.6 and cal.is_fitted
        assert cal.conf["fishing_gear"] == DEFAULTS["detector"]["conf"]["fishing_gear"]
        save_calibration({"tiers": {"tau_confirm": 0.7}}, path=p)
        cal2 = load_calibration(path=p)
        assert cal2.tau_confirm == 0.7 and cal2.tau_review == 0.2      # partial update kept the rest


def test_class_aware_nms_keeps_overlapping_boxes_of_different_classes():
    boxes = np.array([[10, 10, 20, 20], [11, 11, 20, 20]], float)    # IoU ≈ 0.8
    confs = np.array([0.9, 0.8])
    cls = np.array([0, 2])                                           # pot next to a wreck box
    assert len(nms(boxes, confs, cls, 0.1, 0.45, class_aware=True)) == 2
    assert len(nms(boxes, confs, cls, 0.1, 0.45, class_aware=False)) == 1
    same = np.array([0, 0])
    assert len(nms(boxes, confs, same, 0.1, 0.45, class_aware=True)) == 1


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} calibration tests passed")


if __name__ == "__main__":
    _run_all()
