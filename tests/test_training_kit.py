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


def test_kaggle_safe_defaults(monkeypatch):
    """No image cache (would overflow Kaggle's disk); ghost-gear-first selection; and an EXPLICIT
    optimizer - 'auto' silently trained EXP-002 with AdamW at lr 0.00167 and ignored lr0."""
    import src.detection.train as tr
    monkeypatch.setattr(sys, "argv", ["train.py"])
    a = tr.parse_args()
    assert a.cache == "none" and a.select == "ghost_ap50" and a.imgsz == 640
    assert a.optimizer == "SGD" and a.lr0 == 0.01 and a.epochs >= 100 and a.patience == 0


def test_notebook_pins_the_tested_ultralytics():
    import json
    repo = Path(__file__).resolve().parents[1]
    pin = next(l.split("==")[1].strip() for l in (repo / "requirements-train.txt").read_text().splitlines()
               if l.startswith("ultralytics=="))
    nb = json.loads((repo / "notebooks" / "exp003_kaggle.ipynb").read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert f"ultralytics=={pin}" in src and "--cache\", \"none\"" in src and "EXP-003f" in src
    assert "\"--optimizer\", \"SGD\"" in src and "--no-tiles" in src


def test_notebook_monitor_cell_reports_and_collects(tmp_path, monkeypatch):
    """Execute the notebook's status/wait/collect cell against a simulated Kaggle folder: one run
    finished (package copied), one stopped with a traceback (shown); no process alive. The kind of
    bug that broke the first Kaggle run (stale variables, wrong unpacking) would fail here."""
    import json, subprocess, zipfile
    repo = Path(__file__).resolve().parents[1]
    nb = json.loads((repo / "notebooks" / "exp003_kaggle.ipynb").read_text(encoding="utf-8"))
    cell = next("".join(c["source"]) for c in nb["cells"] if "STATUS / WAIT / COLLECT" in "".join(c["source"]))
    w = tmp_path
    (w / "logs").mkdir(); (w / "depth/runs/EXP-003").mkdir(parents=True); (w / "depth/runs/EXP-003f").mkdir(parents=True)
    hdr = "epoch,metrics/recall(B),metrics/mAP50(B)\n"
    (w / "depth/runs/EXP-003/results.csv").write_text(hdr + "1,0.3,0.2\n2,0.4,0.3\n")
    (w / "depth/runs/EXP-003f/results.csv").write_text(hdr + "1,0.2,0.1\n")
    (w / "depth/runs/EXP-003/model_meta.json").write_text(json.dumps(
        {"train_minutes": 1.0, "selection": {"picked": {"checkpoint": "epoch1.pt", "ghost_ap50": 0.5}}}))
    zipfile.ZipFile(w / "depth/runs/EXP-003_complete.zip", "w").close()
    (w / "logs/EXP-003f.log").write_text("1/30\rTraceback (most recent call last):\nRuntimeError: boom\n")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    ns = {}
    exec(cell.replace('WORK = "/kaggle/working"', f'WORK = "{w.as_posix()}"'), ns)
    assert (w / "EXP-003_complete.zip").exists() and not (w / "EXP-003f_complete.zip").exists()

    # the live process list: a training, its data-loader worker (same args, parent = the training), a
    # duplicate launch, the other run, a zombie and a one-GPU launcher shell
    t = "src/detection/train.py --data d.yaml --name"
    ps = "\n".join([f"  100     1 Sl  3600 python {t} EXP-003 --imgsz 1024",
                    f"  101   100 Sl  3590 python {t} EXP-003 --imgsz 1024",
                    f"  200     1 Sl   600 python -u {t} EXP-003 --imgsz 1024",
                    f"  300   400 Rl  3000 /usr/bin/python3 {t} EXP-003f --imgsz 640",
                    f"  301   300 Z      5 python {t} EXP-003f --imgsz 640",
                    f"  400     1 S   3600 bash -c python {t} EXP-003 ; python {t} EXP-003f",
                    "  500     1 S   9999 python other_script.py --name EXP-003"])
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout=ps if "ps -eo" in cmd[-1] else "", stderr=""))
    runs, any_alive = ns["processes"]()
    assert any_alive and runs == {"EXP-003": [(100, 3600), (200, 600)], "EXP-003f": [(300, 3000)]}

    # a live run in its checkpoint-selection phase is reported as such, not as "training"
    (w / "depth/runs/EXP-003f/weights").mkdir()
    for p in ("epoch1.pt", "epoch2.pt", "best.pt", "last.pt"):
        (w / "depth/runs/EXP-003f/weights" / p).write_bytes(b"")
    (w / "depth/runs/EXP-003f/select/epoch1").mkdir(parents=True)
    assert ns["phase"]("EXP-003f") == "CHOOSING THE BEST CHECKPOINT (validating 1 of 4)"
    assert ns["phase"]("EXP-003") == "TRAINING"
    assert ns["status"]() is True


def test_tiles_never_leave_an_unlabelled_partial_object():
    """EXP-002 post-mortem: a box cut by a tile lost its label but kept its pixels. Now frames with a
    large box are not tiled, and a cut small box is inpainted away (no label, no object)."""
    import numpy as np
    im = np.full((640, 640, 3), 60, np.uint8)
    assert bt.tile_frame(im, [(1, 0.5, 0.5, 0.40, 0.30)]) == []               # a large wreck: full frame only
    im[250:262, 370:400] = 255                                                  # a bright pot at x 370..400
    lab = [(0, 385 / 640, 256 / 640, 30 / 640, 12 / 640)]                        # tile 0 (x 0..384) sees 14/30
    tiles = {k: (crop, tl, n) for k, crop, tl, n in bt.tile_frame(im, lab)}
    crop, tl, n = tiles[0]
    assert tl == [] and n == 1 and crop[250:262, 370:384].max() < 200          # cut -> inpainted, unlabelled
    crop, tl, n = tiles[1]                                                      # tile 1 (x 256..640) sees all of it
    assert len(tl) == 1 and n == 0 and crop[250:262, 114:144].min() == 255


def test_ghost_tracker_keeps_the_best_ghost_epoch(tmp_path):
    """The callback reads ultralytics' per-class val AP (a numpy class index - the smoke test caught
    an 'or' on it) and copies last.pt to best_ghost.pt only when ghost AP improves."""
    import numpy as np
    from types import SimpleNamespace as NS
    import src.detection.train as tr
    last = tmp_path / "last.pt"
    tk = tr.GhostTracker(0)
    for ep, (ghost, wreck) in enumerate([(0.2, 0.9), (0.5, 0.1), (0.4, 0.95)]):
        last.write_text(f"epoch{ep}")
        box = NS(ap50=np.array([ghost, wreck]), map50=(ghost + wreck) / 2)
        trainer = NS(epoch=ep, last=last, wdir=tmp_path,
                     validator=NS(metrics=NS(ap_class_index=np.array([0, 1]), box=box)))
        tk.on_model_save(trainer)
    assert (tmp_path / "best_ghost.pt").read_text() == "epoch1" and tk.best == 0.5
    assert [r["ghost_ap50"] for r in tk.rows] == [0.2, 0.5, 0.4]
    opt = NS(param_groups=[{"lr": 0.0001, "initial_lr": 0.01}])
    tk.on_train_start(NS(optimizer=opt, args=NS(nbs=64), accumulate=4))
    assert tk.optimizer["lr"] == 0.01 and tk.optimizer["type"] == "SimpleNamespace"


def test_diagnose_sources_and_verdicts():
    from src.detection import diagnose as dg
    assert dg.group("Rec07_Sensor_x.jpg") == "crabpot_Rec7" and dg.group("BC_POST_T2_01.jpg") == "crabpot_bc_post"
    assert dg.group("seabed_train_natform_3.jpg") == "seabed_natform" and dg.group("TI0040_png.jpg") == "crabpot_other"
    rec = lambda tp, fp, n: {"group": "g", "pc": {"0": {"pts": [(0.9, 1)] * tp + [(0.8, 0)] * fp, "n_gt": n},
                                                  "1": {"pts": [], "n_gt": 0}}}
    m = dg.Model("M", Path("x.onnx"), 640, ["ghost_gear", "wreck_debris"])
    under = {("M", "train"): [rec(4, 0, 10)], ("M", "val"): [rec(3, 0, 10)]}
    gap = {("M", "train"): [rec(9, 0, 10)], ("M", "val"): [rec(3, 0, 10)]}
    good = {("M", "train"): [rec(9, 0, 10)], ("M", "val"): [rec(8, 0, 10)]}
    assert "UNDERFIT" in dg.verdict(m, under) and "GAP" in dg.verdict(m, gap) and "ready" in dg.verdict(m, good)
    assert dg.Model("B", Path("x"), 640, ["fishing_gear", "pipe_cylinder", "structural_fragment",
                                          "natural_formation"]).cmap == {0: 0, 1: 1, 2: 1}
