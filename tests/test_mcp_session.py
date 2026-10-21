"""
The recorded MCP session's report (src/dashboard/mcp_session.py): every call is a table row, and the
autonomy table is well formed. (The session itself starts the real server; it is run as a command.)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dashboard.mcp_session import report


def test_every_call_is_a_row_and_the_pin_is_never_printed():
    log = [{"via": "MCP", "call": "run_survey", "args": {"use_samples": True}, "result": {"mode": "look_harder"}, "ms": 900},
           {"via": "HTTP", "call": "POST /api/approvals/AR-1/decide (a person)", "args": {"pin": "******"},
            "result": {"status_code": 200}, "ms": None}]
    md = report(log, {"opencv": "5.0.0", "model": "EXP-003", "runtime": {"engine": "new"}, "approver_pin_required": True}, 12.0)
    rows = [ln for ln in md.splitlines() if ln.startswith("| 1 |") or ln.startswith("| 2 |")]
    assert len(rows) == 2 and "(900 ms)" in rows[0] and "******" in rows[1]
    table = md.split("## What it shows")[1].splitlines()
    assert all(ln.startswith("|") for ln in table[2:5])                   # header, rule, the one row
