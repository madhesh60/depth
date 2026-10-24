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
    assert cool_docs.apply(c, dry=True) == ["docs/technical_report.md", "README.md", "JUDGES.md"]


def _story(texts: dict) -> set:
    """Every cool:status block of every document must say the same thing, either still pending or ran."""
    said = set()
    for name, text in texts.items():
        blocks = cool_docs.status_blocks(text)
        assert blocks, name
        for b in blocks:                                # the marker never starts a line (GitHub: raw HTML)
            assert f"**COOL:** {cool_docs.STATUS_START}{b}" in text, name
        outside = cool_docs.swap_status(text, "")
        assert "COOL has not run" not in outside and "waiting on a payment method" not in outside, name
        said |= set(blocks)
    return said


def test_every_document_tells_the_same_cool_story():
    """One wording of COOL's status, now and after a COOL run (so golive's pushed docs keep CI green)."""
    texts = {d.name: d.read_text(encoding="utf-8") for d in cool_docs.STATUS_DOCS}
    said = _story(texts)
    assert said == {cool_docs.PENDING_STATUS} or all(s.startswith("ran - ") for s in said)
    if said == {cool_docs.PENDING_STATUS}:
        assert cool_docs.PENDING_ROW in texts["technical_report.md"]   # the row apply() fills


def test_a_cool_run_replaces_every_status_block(tmp_path):
    for f in BENCH.glob("*__*.json"):
        shutil.copy(f, tmp_path)
    _fake_cool(tmp_path, "graviton_stock", "graviton_cool")
    c = cool_docs.collect(tmp_path)
    done = {d.name: cool_docs.swap_status(d.read_text(encoding="utf-8"), cool_docs.status_done(c, d))
            for d in cool_docs.STATUS_DOCS}
    said = _story(done)                                  # the go-live state passes the same story test
    assert all(s.startswith("ran - ") for s in said) and not any(cool_docs.PENDING_STATUS in t for t in done.values())
    for name in ("technical_report.md",):               # docs/*.md: links stay relative to the document
        b = cool_docs.status_blocks(done[name])[0]
        assert "](docs/" not in b and "](reports/cool_benchmark.md)" in b
