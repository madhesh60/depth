"""
ui_smoke.py — the studio in a real headless Chrome: every mode driven, every JS error caught.

Starts the server on a free local port (fresh approvals log), opens the studio in headless Chrome over the
DevTools protocol and drives it: boot → the Terms glossary → Analyze a sample → a Survey → the Agent tab → Connect → Study →
Audit → the 3D twin → the guided demo. It records every JS exception, console error and failed or 4xx/5xx
request, and exits 1 if there was any. Needs the model (``python -m src.detection.fetch_model``), Chrome or
Chromium (``$CHROME`` or on the usual paths) and ``websockets`` (requirements-dev.txt). Not collected by
pytest (it takes ~2 min and a browser); run it by hand:

    python tests/ui_smoke.py
    python tests/ui_smoke.py --full-demo                          # play the guided demo to its last step
    python tests/ui_smoke.py --url http://127.0.0.1:8000          # drive a server that is already running

CI runs it the Codespaces way (``.github/workflows/codespaces.yml``): the devcontainer image, its own setup
and start commands, 2 CPUs, ``--url`` and ``--full-demo``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = [os.environ.get("CHROME", ""), r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              shutil.which("google-chrome") or "", shutil.which("chromium") or "", shutil.which("chromium-browser") or ""]

STEPS = [
    ("boot", "(async () => { await new Promise(r => setTimeout(r, 2500)); "
             "return {samples: document.querySelectorAll('.sample-card').length}; })()"),
    ("terms", "(async () => { document.querySelector('[data-pop=\"termsPop\"]').click(); "
              "await new Promise(r => setTimeout(r, 300)); const p = document.querySelector('#termsPop'); "
              "const n = p.hidden ? 0 : p.querySelectorAll('dt').length; Pop.closeAll(); return {terms: n}; })()"),
    ("analyze", "(async () => { document.querySelector('.sample-card').click(); await runAnalyze(); "
                "await new Promise(r => setTimeout(r, 800)); "
                "return {boxes: document.querySelectorAll('.bx').length, cards: document.querySelectorAll('.ev-card').length}; })()"),
    ("survey", "(async () => { setMode('survey'); await new Promise(r => setTimeout(r, 400)); const t0 = Date.now(); "
               "await runSurvey(); const s = Math.round((Date.now() - t0) / 100) / 10, run = (state.survey || {}).agent_run || {}; "
               "await new Promise(r => setTimeout(r, 1500)); return {hazards: document.querySelectorAll('#hazBody tr').length, "
               "seconds: s, compute_budget_s: (run.goal || {}).compute_s, mode: (run.mode || {}).name, "
               "downgraded_at: run.downgraded_at || null}; })()"),
    ("agent_tab", "(async () => { const b = [...document.querySelectorAll('button, .tab')].find(x => /^\\s*Agent\\s*$/.test(x.textContent)); "
                  "if (b) b.click(); await new Promise(r => setTimeout(r, 600)); return {clicked: !!b}; })()"),
    *[(m, f"(async () => {{ setMode('{m}'); await new Promise(r => setTimeout(r, 1500)); "
          f"const a = document.querySelector('.mode-btn.is-active'); return {{mode: a && a.dataset.mode}}; }})()")
      for m in ("connect", "study", "audit")],
    ("twin", "(async () => { setMode('analyze'); await new Promise(r => setTimeout(r, 400)); "
             "const b = document.querySelector('.dim-btn[data-dim=\"3d\"]'); if (b) b.click(); "
             "await new Promise(r => setTimeout(r, 2500)); return {canvas: document.querySelectorAll('canvas').length}; })()"),
    ("demo", "(async () => { document.querySelector('#demoBtn').click(); await new Promise(r => setTimeout(r, 6000)); "
             "const on = Demo.on; Demo.stop(); return {started: on}; })()"),
]
FULL_DEMO = ("demo", "(async () => { document.querySelector('#demoBtn').click(); const t0 = Date.now(); "
                     "await new Promise(r => setTimeout(r, 1000)); const on = Demo.on; "
                     "while (Demo.on && Date.now() - t0 < 600000) await new Promise(r => setTimeout(r, 500)); "
                     "return {started: on, finished: !Demo.on, step: document.querySelector('#demoStep').textContent, "
                     "seconds: Math.round((Date.now() - t0) / 1000)}; })()")
EXPECT = {"boot": lambda v: v["samples"] > 0, "terms": lambda v: v["terms"] >= 8, "analyze": lambda v: v["cards"] > 0, "survey": lambda v: v["hazards"] > 0,
          "agent_tab": lambda v: v["clicked"], "connect": lambda v: v["mode"] == "connect",
          "study": lambda v: v["mode"] == "study", "audit": lambda v: v["mode"] == "audit",
          "twin": lambda v: v["canvas"] > 0, "demo": lambda v: v["started"] and v.get("finished", True)
          and v.get("step", "7/7") == "7/7"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ready(base: str, timeout: float = 300) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if json.load(urllib.request.urlopen(base + "/api/health", timeout=5)).get("model_loaded"):
                return
        except Exception:
            pass
        time.sleep(1)
    raise SystemExit("the server did not load the model in time (python -m src.detection.fetch_model?)")


async def _drive(base: str, dbg: int, steps: list) -> tuple[dict, list]:
    import websockets
    url = None
    for _ in range(60):
        try:
            tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{dbg}/json/list", timeout=2))
            url = next(t["webSocketDebuggerUrl"] for t in tabs if t.get("type") == "page")
            break
        except Exception:
            await asyncio.sleep(0.5)
    events, results, pending, n = [], {}, {}, [0]
    async with websockets.connect(url, max_size=None) as ws:
        async def reader():
            async for raw in ws:
                m = json.loads(raw)
                if m.get("id") in pending:
                    pending.pop(m["id"]).set_result(m)
                elif m.get("method") in ("Runtime.exceptionThrown", "Runtime.consoleAPICalled", "Log.entryAdded",
                                         "Network.loadingFailed", "Network.responseReceived"):
                    events.append(m)
        task = asyncio.create_task(reader())

        async def send(method, params=None, timeout=900):
            n[0] += 1
            fut = asyncio.get_running_loop().create_future()
            pending[n[0]] = fut
            await ws.send(json.dumps({"id": n[0], "method": method, "params": params or {}}))
            return await asyncio.wait_for(fut, timeout)

        for d in ("Runtime.enable", "Log.enable", "Page.enable", "Network.enable"):
            await send(d)
        await send("Page.navigate", {"url": base + "/"})
        await asyncio.sleep(4)
        for name, js in steps:
            r = (await send("Runtime.evaluate", {"expression": js, "awaitPromise": True, "returnByValue": True}))["result"]
            results[name] = ({"error": r["exceptionDetails"].get("text")} if "exceptionDetails" in r
                             else r["result"].get("value"))
        task.cancel()
    errors = []
    for e in events:
        m, p = e["method"], e["params"]
        if m == "Runtime.exceptionThrown":
            d = p["exceptionDetails"]
            errors.append({"exception": ((d.get("exception") or {}).get("description") or d.get("text"))[:300]})
        elif m == "Runtime.consoleAPICalled" and p.get("type") == "error":
            errors.append({"console.error": " ".join(str(a.get("value", a.get("description", ""))) for a in p["args"])[:300]})
        elif m == "Log.entryAdded" and p["entry"].get("level") == "error":
            errors.append({"log": p["entry"].get("text", "")[:300]})
        elif m == "Network.responseReceived" and p["response"]["status"] >= 400:
            errors.append({"http": p["response"]["status"], "url": p["response"]["url"][:200]})
        elif m == "Network.loadingFailed" and not p.get("canceled"):
            errors.append({"net": p.get("errorText")})
    return results, errors


def main() -> int:
    ap = argparse.ArgumentParser(description="drive the DEPTH studio in headless Chrome; 0 JS errors to pass")
    ap.add_argument("--url", help="drive a server that is already running (default: start one on a free port)")
    ap.add_argument("--full-demo", action="store_true", help="play the guided demo to its last step")
    a = ap.parse_args()
    chrome = next((c for c in CANDIDATES if c and Path(c).exists()), None)
    if not chrome:
        print("no Chrome / Chromium found - set $CHROME")
        return 2
    steps = [FULL_DEMO if name == "demo" and a.full_demo else (name, js) for name, js in STEPS]
    port, dbg = _free_port(), _free_port()
    env = {**os.environ, "DEPTH_APPROVALS_DIR": tempfile.mkdtemp(prefix="depth-ui-"), "DEPTH_PUBLIC": "0",
           "PYTHONPATH": str(REPO)}
    for k in ("DEPTH_APPROVER_PIN", "DEPTH_MCP_TOKEN", "DEPTH_TRUST_PROXY", "DEPTH_ORIGIN_SECRET"):
        env.pop(k, None)
    srv = None if a.url else subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.dashboard.app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = (a.url or f"http://127.0.0.1:{port}").rstrip("/")
    # as root (a container: CI, Codespaces) Chromium needs --no-sandbox, and /dev/shm is small there
    root = ["--no-sandbox", "--disable-dev-shm-usage"] if hasattr(os, "geteuid") and os.geteuid() == 0 else []
    browser = None
    try:
        _ready(base)
        browser = subprocess.Popen([chrome, "--headless=new", f"--remote-debugging-port={dbg}",
                                    f"--user-data-dir={tempfile.mkdtemp(prefix='depth-ui-chrome-')}", "--no-first-run",
                                    "--no-default-browser-check", "--disable-gpu", "--window-size=1600,1000", *root,
                                    "about:blank"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        results, errors = asyncio.run(_drive(base, dbg, steps))
    finally:
        if browser:
            browser.kill()
        if srv:
            srv.kill()
    failed = [k for k, ok in EXPECT.items() if not (isinstance(results.get(k), dict) and "error" not in results[k]
                                                   and ok(results[k]))]
    print(json.dumps({"steps": results, "failed_steps": failed, "errors": errors}, indent=1, ensure_ascii=False))
    print("UI smoke:", "PASS" if not failed and not errors else "FAIL")
    return 1 if failed or errors else 0


if __name__ == "__main__":
    sys.exit(main())
