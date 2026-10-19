"""
load_test.py — can the live demo serve several judges at once? A scripted, repeatable concurrency test run
THROUGH the public URL (viewer → CloudFront → origin), with the pass mark fixed in ``PASS`` before any run.

    python -m src.bench.load_test https://<cloudfront-domain> --label c8g_large \
        [--clients 4] [--duration 120] [--think 0] [--admin-token T | $DEPTH_ADMIN_TOKEN]

* ``--clients`` workers loop ``POST /api/analyze?sample=<id>`` over the shipped frames, back to back
  (``--think`` seconds apart; 0 = worst case: every worker always has a request in flight);
* one survey job runs meanwhile (``POST /api/jobs/survey?use_samples=1``, then polled to the end);
* a probe calls ``GET /api/health`` every 2 s — the server must stay responsive during inference.

All requests come from one machine, i.e. one client address, so they carry the operator header
``X-DEPTH-Admin`` (the server's admin token: ``sudo cat /etc/depth/admin.env`` over SSM) — otherwise the
per-client rate limits, not the server, would decide the result. Python standard library only.

Writes ``benchmarks/load_<label>.json`` with every latency, the server-side timings and the verdict.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# the pass mark, fixed before running (4 clients back to back + 1 survey on the demo instance)
PASS = {"server_errors_5xx": 0, "transport_errors": 0, "analyze_p95_s": 3.0, "health_p95_s": 1.0,
        "survey_completes": True}


def _call(url: str, method: str = "GET", headers: dict | None = None, timeout: float = 90):
    t0 = time.perf_counter()
    req = urllib.request.Request(url, data=b"" if method == "POST" else None, method=method,
                                 headers={"User-Agent": "depth-load-test/1", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), time.perf_counter() - t0, None
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.perf_counter() - t0, None
    except Exception as e:                                   # timeouts, resets, DNS …
        return None, b"", time.perf_counter() - t0, f"{type(e).__name__}: {e}"


def pct(v: list[float], q: float) -> float | None:
    if not v:
        return None
    s = sorted(v)
    return round(s[min(len(s) - 1, max(0, int(round(q / 100 * len(s) + 0.5)) - 1))], 3)


def run(base: str, clients: int, duration: float, think: float, token: str) -> dict:
    hdr = {"X-DEPTH-Admin": token} if token else {}
    code, body, _, err = _call(f"{base}/api/samples")
    if err or code != 200:
        raise SystemExit(f"cannot reach {base}/api/samples: {err or code}")
    ids = [s["id"] for s in json.loads(body)["samples"]]
    stop = time.time() + duration
    lock = threading.Lock()
    analyze, health, survey = [], [], {}
    cycle = itertools.cycle(ids)

    def worker():
        while time.time() < stop:
            with lock:
                sid = next(cycle)
            c, b, dt, e = _call(f"{base}/api/analyze?sample={sid}", "POST", hdr)
            rec = {"t": round(time.time(), 2), "code": c, "s": round(dt, 3), "error": e}
            if c == 200:
                try:
                    d = json.loads(b)
                    rec["server_wall_ms"] = d.get("wall_ms")
                    rec["stage_ms"] = {k: round(v, 1) for k, v in (d.get("stage_ms") or {}).items()
                                       if isinstance(v, (int, float))}
                except ValueError:
                    pass
            with lock:
                analyze.append(rec)
            if think:
                time.sleep(think)

    def probe():
        while time.time() < stop:
            c, _, dt, e = _call(f"{base}/api/health")
            with lock:
                health.append({"t": round(time.time(), 2), "code": c, "s": round(dt, 3), "error": e})
            time.sleep(2)

    def surveyor():
        t0 = time.perf_counter()
        c, b, _, e = _call(f"{base}/api/jobs/survey?use_samples=1", "POST", hdr)
        if c != 200:
            survey.update(status="not_submitted", code=c, error=e or b[:200].decode(errors="replace"))
            return
        jid = json.loads(b)["job_id"]
        while time.perf_counter() - t0 < max(duration, 60) + 300:
            c, b, _, e = _call(f"{base}/api/jobs/{jid}")
            st = json.loads(b).get("status") if c == 200 else None
            if st in ("done", "error"):
                survey.update(status=st, seconds=round(time.perf_counter() - t0, 1), job_id=jid)
                return
            time.sleep(2)
        survey.update(status="timeout", seconds=round(time.perf_counter() - t0, 1), job_id=jid)

    ts = [threading.Thread(target=worker) for _ in range(clients)] + [threading.Thread(target=probe),
                                                                       threading.Thread(target=surveyor)]
    t_start = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t_start

    ok = [r["s"] for r in analyze if r["code"] == 200]
    srv = [r["server_wall_ms"] for r in analyze if r.get("server_wall_ms") is not None]
    h_ok = [r["s"] for r in health if r["code"] == 200]
    codes: dict = {}
    for r in analyze + health:
        codes[str(r["code"])] = codes.get(str(r["code"]), 0) + 1
    summary = {
        "analyze": {"requests": len(analyze), "ok": len(ok), "per_minute": round(len(ok) / wall * 60, 1),
                    "client_s": {"p50": pct(ok, 50), "p95": pct(ok, 95), "max": max(ok) if ok else None},
                    "server_wall_ms": {"p50": pct(srv, 50), "p95": pct(srv, 95)}},
        "health": {"requests": len(health), "client_s": {"p50": pct(h_ok, 50), "p95": pct(h_ok, 95)}},
        "survey": survey, "status_codes": codes,
        "server_errors_5xx": sum(1 for r in analyze + health if r["code"] and r["code"] >= 500),
        "transport_errors": sum(1 for r in analyze + health if r["error"]),
        "rate_limited_429": sum(1 for r in analyze + health if r["code"] == 429),
    }
    checks = {
        "server_errors_5xx": summary["server_errors_5xx"] <= PASS["server_errors_5xx"],
        "transport_errors": summary["transport_errors"] <= PASS["transport_errors"],
        "analyze_p95_s": (summary["analyze"]["client_s"]["p95"] or 1e9) <= PASS["analyze_p95_s"],
        "health_p95_s": (summary["health"]["client_s"]["p95"] or 1e9) <= PASS["health_p95_s"],
        "survey_completes": survey.get("status") == "done",
    }
    return {"summary": summary, "checks": checks, "passed": all(checks.values()),
            "samples": {"analyze": analyze, "health": health}, "wall_s": round(wall, 1)}


def main():
    ap = argparse.ArgumentParser(description="Concurrency test of a live DEPTH server (pass mark fixed in PASS).")
    ap.add_argument("url")
    ap.add_argument("--label", required=True, help="e.g. c8g_large")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--duration", type=float, default=120)
    ap.add_argument("--think", type=float, default=0.0)
    ap.add_argument("--admin-token", default=os.environ.get("DEPTH_ADMIN_TOKEN", ""))
    ap.add_argument("--out", default=str(REPO / "benchmarks"))
    a = ap.parse_args()
    base = a.url.rstrip("/")
    h = json.loads(_call(f"{base}/api/health")[1] or b"{}")
    res = run(base, a.clients, a.duration, a.think, a.admin_token)
    out = {"label": a.label, "url": base, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "config": {"clients": a.clients, "duration_s": a.duration, "think_s": a.think,
                      "operator_header": bool(a.admin_token)},
           "pass_mark": PASS, "server": {"cool": h.get("cool"), "opencv": h.get("opencv"),
                                         "is_cool_path": h.get("is_cool_path"), "runtime": h.get("runtime")},
           **res}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fp = Path(a.out) / f"load_{a.label}.json"
    fp.write_text(json.dumps(out, indent=2))
    s = res["summary"]
    print(f"[{a.label}] {a.clients} clients + 1 survey, {res['wall_s']} s via {base}")
    print(f"  analyze: {s['analyze']['ok']}/{s['analyze']['requests']} ok, {s['analyze']['per_minute']}/min, "
          f"client p50 {s['analyze']['client_s']['p50']} s p95 {s['analyze']['client_s']['p95']} s, "
          f"server wall p50 {s['analyze']['server_wall_ms']['p50']} ms p95 {s['analyze']['server_wall_ms']['p95']} ms")
    print(f"  health p95 {s['health']['client_s']['p95']} s | survey {s['survey']} | codes {s['status_codes']}")
    for k, v in res["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k} (mark: {PASS[k]})")
    print(("PASSED" if res["passed"] else "FAILED") + f" -> {fp}")
    raise SystemExit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
