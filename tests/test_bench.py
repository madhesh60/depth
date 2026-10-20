"""
Tests for the COOL benchmark tooling (src/bench): survey-hour cost model, manifest hashing, the
provenance fingerprint and the A/B/C comparison report — no model needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bench.product_bench import survey_hour, load_frames
from src.bench.fingerprint import opencv_provenance, host
from src.bench import compare


def test_survey_hour_model():
    # 15 Hz × 3600 s / 640 pings per frame × 2 channels = 168.75 frames per survey-hour
    sh = survey_hour(mean_ms=200.0, price_hour=0.18, ping_rate_hz=15, pings_per_frame=640, channels=2)
    assert sh["frames"] == 168.8 and abs(sh["compute_s"] - 33.75) < 0.01
    assert abs(sh["real_time_factor"] - 33.75 / 3600) < 1e-4
    assert abs(sh["usd"] - 33.75 * 0.18 / 3600) < 1e-6
    assert survey_hour(200.0, 0.0, 15, 640, 2)["usd"] is None


def test_manifest_hash_is_content_based(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"one")
    (tmp_path / "b.jpg").write_bytes(b"two")
    frames, h1 = load_frames(tmp_path)
    assert [f for f, _ in frames] == ["a", "b"]
    (tmp_path / "b.jpg").write_bytes(b"TWO")
    assert load_frames(tmp_path)[1] != h1


def test_provenance_fields():
    o = opencv_provenance()
    assert o["version"] and isinstance(o["is_cool_path"], bool) and "cv2_file" in o
    assert host()["machine"]


def _run(label, total, cool, usd):
    st = {k: {"p50": v, "p95": v * 1.2, "mean": v} for k, v in (("decode", 3), ("see", total - 3), ("total", total))}
    return {"label": label, "workload": "product", "stages_ms": st, "throughput_fps": round(1000 / total, 2),
            "runtime": {"threads": 4},
            "fingerprint": {"host": {"machine": "aarch64" if "graviton" in label else "x86_64", "cpu_model": "x"},
                            "opencv": {"version": "5.0.0", "is_cool_path": cool}, "ec2": {"instance-type": "c8g.xlarge"}},
            "cost": {"usd_per_1k_frames": 0.01, "survey_hour": {"frames": 168.8, "compute_s": total * 0.1688,
                                                                  "real_time_factor": 0.01, "usd": usd}}}


def test_compare_report_has_both_comparisons(tmp_path):
    for r in (_run("baseline_x86", 300, False, 0.002), _run("graviton_stock", 250, False, 0.0018),
              _run("graviton_cool", 200, True, 0.0015)):
        (tmp_path / f"{r['label']}__product.json").write_text(json.dumps(r))
    md = compare.report(compare.load(tmp_path))
    assert "**COOL**" in md and "(B → C)" in md and "1.25× faster p50" in md
    assert "(A → C)" in md and "1.50× faster p50" in md and "cheaper per survey-hour" in md


def _parity_doc(dets: dict, manifest="m1") -> dict:
    return {"workload": "parity", "input_manifest_sha256": manifest, "frames": dets}


def test_parity_tolerates_float_noise_but_catches_real_differences():
    a = {"f1": [{"cls": "ghost_gear", "bbox": [10, 10, 50, 50], "conf": 0.61, "tier": "review"}],
         "f2": [{"cls": "wreck_debris", "bbox": [100, 100, 200, 180], "conf": 0.33, "tier": "review"}]}
    noisy = {"f1": [{"cls": "ghost_gear", "bbox": [10, 11, 50, 50], "conf": 0.6131, "tier": "review"}],
             "f2": [{"cls": "wreck_debris", "bbox": [100, 100, 200, 180], "conf": 0.3302, "tier": "review"}]}
    p = compare.parity(_parity_doc(a), _parity_doc(noisy))
    assert p["match"] == p["detections"] == 2 and p["extra"] == 0 and p["same_frames"]
    tier_flip = json.loads(json.dumps(noisy))
    tier_flip["f2"][0]["tier"] = "low_risk"
    assert compare.parity(_parity_doc(a), _parity_doc(tier_flip))["match"] == 1
    moved = json.loads(json.dumps(noisy))
    moved["f1"][0]["bbox"] = [30, 30, 70, 70]
    p2 = compare.parity(_parity_doc(a), _parity_doc(moved))
    assert p2["missing"] == 1 and p2["extra"] == 1 and p2["first_problem"]
    assert not compare.parity(_parity_doc(a), _parity_doc(noisy, "m2"))["same_frames"]


def test_report_sections_survey_ops_parity_and_receipts(tmp_path):
    r = _run("graviton_cool", 200, True, 0.0015)
    r["ops_ms"] = {k: {"p50": v, "p95": v, "mean": v} for k, v in
                   (("decode", 3), ("stage1", 5), ("dnn_pre", 4), ("dnn_forward", 170), ("dnn_post", 1),
                    ("evidence", 12), ("render", 5))}
    s = {**_run("graviton_cool", 200, True, 0.0015), "workload": "survey",
         "config": {"recording": "DATASET/external/pingmapper_large/Test-Large-DS.DAT"},
         "survey": {"sonar_duration_s": 3606.9, "frames": 280, "real_time_factor_median": 0.04,
                    "real_time_factor_range": [0.039, 0.041], "compute_s_per_survey_hour": 144.0,
                    "usd_per_survey_hour": 0.0068}}
    for x, name in ((r, "graviton_cool__product.json"), (s, "graviton_cool__survey.json")):
        (tmp_path / name).write_text(json.dumps(x))
    (tmp_path / "graviton_cool__parity.json").write_text(json.dumps(_parity_doc({})))
    (tmp_path / "graviton_cool__receipt.json").write_text("{}")
    runs = compare.load(tmp_path)
    assert [x["workload"] for x in runs] == ["product", "survey"]       # parity/receipt files are not runs
    md = compare.report(runs, compare.load_parity(tmp_path), tmp_path)
    assert "Where the time goes" in md and "85%" in md                  # 170 of 200 ms is the network
    assert "A real survey, measured" in md and "Test-Large-DS.DAT" in md and "$0.00680" in md
    assert "Machines" in md and "graviton_cool__receipt.json" in md


def test_claim_rule_needs_separate_repeat_ranges():
    def runs(vals):
        return {"workload": "product", "repeats": [{"total_ms": {"p50": v}} for v in vals]}
    assert compare.separation(runs([200, 205, 210]), runs([150, 152, 155])) == "repeat ranges separate"
    assert "overlap" in compare.separation(runs([200, 205, 210]), runs([190, 202, 208]))
    assert "slower" in compare.separation(runs([150, 151, 152]), runs([200, 201, 202]))
    assert compare.separation(runs([200]), runs([150])) == "single repeat - no claim"
    surv = lambda v: {"workload": "survey", "repeats": [{"real_time_factor": x} for x in v]}
    assert compare.separation(surv([0.07, 0.071]), surv([0.05, 0.051])) == "repeat ranges separate"


def _run_all():
    import tempfile
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            if fn.__code__.co_argcount:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  ok  {name}")


if __name__ == "__main__":
    _run_all()
