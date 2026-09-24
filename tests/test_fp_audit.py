"""
Tests for the blinded false-alarm audit (src/detection/fp_audit.py): the public item list is blind,
tags validate, the majority tag pools annotators, audited precision is exact only once the whole
band is tagged, catch accuracy and Cohen's kappa are reported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection import fp_audit


def _setup(tmp_path, monkeypatch, n_fp=4, n_catch=2):
    items = [{"id": f"a{i:03d}", "kind": "fp", "frame": "f", "bbox": [0, 0, 5, 5], "conf": 0.5} for i in range(n_fp)]
    items += [{"id": f"c{i:03d}", "kind": "catch", "frame": "f", "bbox": [0, 0, 5, 5], "conf": 0.9} for i in range(n_catch)]
    mf = tmp_path / "fp_audit_val.json"
    mf.write_text(json.dumps({"band_tp": 6, "band_fp": n_fp, "band_precision_raw": 0.6, "conf_cut": 0.3, "items": items}))
    monkeypatch.setattr(fp_audit, "manifest_path", lambda model="EXP-001", split="val": mf)
    monkeypatch.setenv("DEPTH_AUDIT_TAGS", str(tmp_path / "tags.jsonl"))
    return items


def test_public_items_are_blind(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    pub = fp_audit.public_items()
    assert len(pub) == 6 and all(set(p) == {"id", "crop", "context"} for p in pub)   # no kind, no conf


def test_tag_validation(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    try:
        fp_audit.add_tag("a000", "maybe", "x")
        raise AssertionError("should raise")
    except ValueError:
        pass


def test_summary_audited_precision_catch_and_kappa(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    for i, t in enumerate(["real", "real", "clutter", "noise"]):
        fp_audit.add_tag(f"a{i:03d}", t, "A")
    fp_audit.add_tag("c000", "real", "A"); fp_audit.add_tag("c001", "clutter", "A")
    s = fp_audit.summary()
    assert s["complete"] and s["fp_real_share"] == 0.5
    assert s["band_precision_audited"] == round((6 + 2) / 10, 4)           # exact once the band is tagged
    assert s["per_annotator"]["A"]["catch_accuracy"] == 0.5
    for i, t in enumerate(["real", "clutter", "clutter", "noise"]):           # a second auditor disagrees once
        fp_audit.add_tag(f"a{i:03d}", t, "B")
    s2 = fp_audit.summary()
    assert s2["fp_taxonomy"].get("unsure", 0) == 1                           # tie on a001 → unsure
    assert s2["kappa_first_two"] is None or -1 <= s2["kappa_first_two"] <= 1


def test_incomplete_band_gives_no_audited_precision(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    fp_audit.add_tag("a000", "real", "A")
    s = fp_audit.summary()
    assert not s["complete"] and s["band_precision_audited"] is None and s["fp_real_share_ci95"][0] is not None
