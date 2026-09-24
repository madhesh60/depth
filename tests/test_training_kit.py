"""
Tests for the EXP-002 kit: tile label clipping (build_tiles.py), official-split name matching and
source mapping (evaluate.py), and the calibrate AUC-gain guard for degenerate models.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_tiles", REPO / "DATASET" / "scripts" / "build_tiles.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

from src.detection.evaluate import _strip_ext, source_of


def test_tiles_cover_frame_with_overlap():
    t = bt.tiles_for(640, 640, 0.6, 0)
    assert len(t) == 4 and t[0] == (0, 0, 384, 384) and t[3] == (256, 256, 384, 384)


def test_tile_box_clipping_and_min_visibility():
    W = H = 640
    # a 40×40 box centred at (100,100): fully inside tile 0
    lab = [(0, 100 / W, 100 / H, 40 / W, 40 / H)]
    out = bt.tile_boxes(lab, W, H, 0, 0, 384, 384, 0.6)
    assert len(out) == 1 and abs(out[0][1] - 100 / 384) < 1e-6 and abs(out[0][3] - 40 / 384) < 1e-6
    # a box straddling the tile edge with only 25% visible is dropped; 75% visible is kept (clipped)
    half = [(0, 394 / W, 100 / H, 40 / W, 40 / H)]          # x 374..414 → 10 of 40 px inside a 0..384 tile
    assert bt.tile_boxes(half, W, H, 0, 0, 384, 384, 0.6) == []
    most = [(0, 374 / W, 100 / H, 40 / W, 40 / H)]          # x 354..394 → 30 of 40 px inside
    kept = bt.tile_boxes(most, W, H, 0, 0, 384, 384, 0.6)
    assert len(kept) == 1 and abs(kept[0][3] - 30 / 384) < 1e-6


def test_official_split_names_match_without_eating_the_hash():
    listed = "Rec9_wcp_ss_port_00001_jpg.rf.cf8037272b75606be7164ab0922c8a26"
    on_disk = listed + ".jpg"
    assert _strip_ext(Path(listed).name) == _strip_ext(on_disk)
    assert Path(listed).stem != _strip_ext(on_disk)            # the old bug: .stem ate ".rf.<hash>"


def test_source_mapping_for_v2b_names():
    assert source_of("Rec9_wcp_ss_port_00001.jpg") == "crabpot"
    assert source_of("BC_POST_T2_01.jpg") == "crabpot"
    assert source_of("Contact_9_sslo_png.jpg") == "crabpot_xsonar"
    assert source_of("crabpot_train_Rec6.jpg") == "crabpot" and source_of("uatd_1.jpg") == "uatd"


def test_auc_gain_guard_on_degenerate_frames():
    from src.agentic.calibrate import auc_gain_ci
    # every candidate is a false positive → AUC undefined on every resample → "no gain", not a crash
    frames = [{"gt": [], "cands": [{"bbox": [0, 0, 5, 5], "conf": 0.3, "rl_single": 0.0, "rl_single_clahe": 0.0,
                                    "rl_mosaic": 0.0, "rl_mosaic_clahe": 0.0, "shadow": "none",
                                    "shadow_contrast": 0.0, "shadow_run": 0}]} for _ in range(5)]
    assert auc_gain_ci(frames, "agent_single", reps=20) == (0.0, 0.0, 0.0)
