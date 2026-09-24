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
    (tmp_path / "a.jpg").write_bytes(b"one"); (tmp_path / "b.jpg").write_bytes(b"two")
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
