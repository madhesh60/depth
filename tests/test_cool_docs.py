"""cool_docs fills the COOL placeholders only from runs whose provenance says COOL."""
import json
import shutil
from pathlib import Path

import pytest

from src.bench import cool_docs

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"


def _fake_cool(tmp: Path, src: str, dst: str, kind: str = "product"):
    r = json.loads((BENCH / f"{src}__{kind}.json").read_text())
    r["fingerprint"]["opencv"]["is_cool_path"] = True
    r["label"] = dst
    (tmp / f"{dst}__{kind}.json").write_text(json.dumps(r))


def test_refuses_without_cool(tmp_path):
    for f in BENCH.glob("graviton_stock__*.json"):
        shutil.copy(f, tmp_path)
    with pytest.raises(SystemExit):
        cool_docs.collect(tmp_path)


def test_rows_and_headline(tmp_path):
    for f in BENCH.glob("*__*.json"):
        shutil.copy(f, tmp_path)
    _fake_cool(tmp_path, "graviton_stock", "graviton_cool")
    _fake_cool(tmp_path, "graviton_stock", "graviton_cool", "survey")
    c = cool_docs.collect(tmp_path)
    rows = cool_docs.report_rows(c)
    assert len(rows) == 2 and all("**COOL**" in r for r in rows)
    h = cool_docs.headline(c)
    assert "1.00× stock Graviton" in h and "cool_verification.md" in h
    assert cool_docs.apply(c, dry=True) == ["docs/technical_report.md", "README.md"]
