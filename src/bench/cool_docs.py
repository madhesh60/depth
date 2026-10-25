"""cool_docs.py — COOL's status in the docs: one sentence while it is pending, the measured numbers once it ran.

**One COOL story.** :data:`PENDING_STATUS` is the only wording of COOL's status. Every document that states it
(:data:`STATUS_DOCS`) holds it as ``**COOL:** <!-- cool:status -->…<!-- /cool:status -->`` - the marker never starts
the line, or GitHub would show the line as raw HTML;
``tests/test_cool_docs.py`` checks that every block says exactly this and that no other "COOL has not run"
wording is left beside them.

Run by `PHASE=cool ./infra/golive.sh` after the COOL benchmarks land, so switching to COOL needs no hand
edits: it fills the technical report's pending row, the README's COOL line and every status block. Every
number comes from benchmarks/*.json (product workload, survey workload, load test) of runs whose provenance
says COOL.

    python -m src.bench.cool_docs [--dir benchmarks] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs" / "technical_report.md"
README = ROOT / "README.md"
JUDGES = ROOT / "JUDGES.md"

PENDING_ROW = ("| **⏳ pending:** Graviton **COOL** (c8g.xlarge + c8g.large) | Marketplace subscription blocked on payment "
               "| — | — | — |")
PENDING_TARGET = "Graviton COOL pending. |"
README_START, README_END = "<!-- cool:readme -->", "<!-- /cool:readme -->"
STATUS_START, STATUS_END = "<!-- cool:status -->", "<!-- /cool:status -->"
STATUS_DOCS = (README, JUDGES, REPORT)
PENDING_STATUS = ("has not run yet: the AWS Marketplace subscription is blocked on payment. The demo server runs "
                  "stock OpenCV 5 and says so; the COOL benchmark and its signed receipt run in one command once it is "
                  "active.")


def _load(d: Path, name: str) -> dict | None:
    p = d / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _is_cool(r: dict | None) -> bool:
    return bool(r and r["fingerprint"]["opencv"].get("is_cool_path"))


def _itype(r: dict) -> str:
    ec2 = r["fingerprint"].get("ec2") or {}
    return ec2.get("instance_type") or r["label"]


def collect(d: Path) -> dict:
    """The COOL runs and their stock references (same size). Raises if no COOL run is present."""
    runs = {k: _load(d, f"{k}__product.json") for k in ("graviton_cool", "graviton_cool_large",
                                                         "graviton_stock", "baseline_x86")}
    surveys = {k: _load(d, f"{k}__survey.json") for k in ("graviton_cool", "graviton_stock", "baseline_x86")}
    if not _is_cool(runs["graviton_cool"]) and not _is_cool(runs["graviton_cool_large"]):
        raise SystemExit("no COOL product run in benchmarks/ (provenance must say cv2 loads from /opt/cool)")
    load = _load(d / "load", "load_test__c8g_large_cool.json")
    return {"runs": runs, "surveys": surveys, "load": load}


def _p50(r: dict) -> float:
    return r["stages_ms"]["total"]["p50"]


def report_rows(c: dict) -> list[str]:
    rows = []
    for key in ("graviton_cool", "graviton_cool_large"):
        r = c["runs"][key]
        if not _is_cool(r):
            continue
        sh = (r.get("cost") or {}).get("survey_hour") or {}
        usd = f" · ${sh['usd']:.4f}" if sh.get("usd") is not None else ""
        rows.append(f"| full product path, {r['runtime']['threads']} threads (EC2, COOL) | {_itype(r)} Graviton4, "
                    f"**COOL** OpenCV 5 | {_p50(r):.0f} ms | {r['throughput_fps']} | {sh.get('compute_s', 0):.1f} s{usd} |")
    s = c["surveys"]["graviton_cool"]
    if _is_cool(s):
        rows.append(f"| real 1-h recording end to end (EC2, COOL) | {_itype(s)} Graviton4, **COOL** | — | — | "
                    f"RTF {s['survey']['real_time_factor_median']:.4f} |")
    lt = c["load"]
    if lt:
        a = lt["summary"]["analyze"]
        rows.append(f"| load test, 4 clients + 1 survey, 120 s | c8g.large demo server (**COOL**) | "
                    f"p95 {a['client_s']['p95']} s (mark 3 s) | {a['ok']}/{a['requests']} ok | "
                    f"**{'passed' if lt.get('passed') else 'FAILED'}** |")
    return rows


def headline(c: dict) -> str:
    cool, stock, x86 = c["runs"]["graviton_cool"], c["runs"]["graviton_stock"], c["runs"]["baseline_x86"]
    parts = []
    if _is_cool(cool):
        parts.append(f"**COOL on c8g.xlarge: {_p50(cool):.0f} ms/frame**")
        if stock:
            parts.append(f"{_p50(stock) / _p50(cool):.2f}× stock Graviton ({_p50(stock):.0f} ms)")
        if x86:
            parts.append(f"{_p50(x86) / _p50(cool):.2f}× x86 at equal size ({_p50(x86):.0f} ms)")
    large = c["runs"]["graviton_cool_large"]
    if _is_cool(large):
        parts.append(f"the c8g.large demo size on COOL: {_p50(large):.0f} ms")
    return "; ".join(parts) + " (×>1 = COOL faster; [`docs/reports/cool_benchmark.md`](docs/reports/cool_benchmark.md), " \
        "receipt verified: [`docs/reports/cool_verification.md`](docs/reports/cool_verification.md))."


def status_blocks(text: str) -> list[str]:
    """What each ``cool:status`` block of a document says, in order."""
    return [part.split(STATUS_END, 1)[0] for part in text.split(STATUS_START)[1:] if STATUS_END in part]


def swap_status(text: str, line: str) -> str:
    """Every ``cool:status`` block of a document -> ``line`` (the markers stay, so it can be swapped again)."""
    out, rest = [], text
    while STATUS_START in rest and STATUS_END in rest.split(STATUS_START, 1)[1]:
        before, rest = rest.split(STATUS_START, 1)
        rest = rest.split(STATUS_END, 1)[1]
        out.append(f"{before}{STATUS_START}{line}{STATUS_END}")
    return "".join(out) + rest


def status_done(c: dict, doc: Path) -> str:
    """What the status blocks say once a COOL run is in benchmarks/ (its links relative to ``doc``)."""
    h = headline(c)
    if doc.parent != ROOT:                                     # docs/*.md link to reports/ beside them
        h = h.replace("](docs/", "](")
    return f"ran - {h}"


def apply(c: dict, dry: bool = False) -> list[str]:
    old = {p: p.read_text(encoding="utf-8") for p in (REPORT, README, *STATUS_DOCS)}
    new = dict(old)
    if PENDING_ROW in new[REPORT]:
        rep = new[REPORT].replace(PENDING_ROW, "\n".join(report_rows(c)))
        cool = c["runs"]["graviton_cool"] or c["runs"]["graviton_cool_large"]
        new[REPORT] = rep.replace(PENDING_TARGET, f"Graviton COOL: {_p50(cool):.0f} ms ({_itype(cool)}). |")
    rd = new[README]
    if README_START in rd and README_END in rd:
        a, rest = rd.split(README_START, 1)
        _, b = rest.split(README_END, 1)
        new[README] = f"{a}{README_START}\n{headline(c)}\n{README_END}{b}"
    for doc in STATUS_DOCS:
        new[doc] = swap_status(new[doc], status_done(c, doc))
    changed = [p for p in old if new[p] != old[p]]
    if not dry:
        for p in changed:
            p.write_text(new[p], encoding="utf-8")
    return [p.relative_to(ROOT).as_posix() for p in changed]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(ROOT / "benchmarks"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    c = collect(Path(a.dir))
    print(headline(c))
    for row in report_rows(c):
        print(row)
    print("updated:", apply(c, a.dry_run) or "nothing (already filled?)")


if __name__ == "__main__":
    main()
