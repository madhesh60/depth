"""
ui_video.py — records a captioned walkthrough of the real studio as an MP4 (the submission video's footage).

Starts the server and headless Chrome like ``ui_smoke.py``, plays a fixed story through the studio — Analyze
(one card per object, the trace, the promise) → Survey (the agent's choices, one card per object, "without
Stage 1") → the raw recording with real GPS (the promise withdrawn, a full-review request) → Approvals →
Connect → Study — and records the screen with the DevTools screencast. Captions are part of the page, so they
are in the frames; the same lines are written as an ``.srt`` file for the video host. A wait longer than
:data:`WAIT_S` (a survey running on a laptop) plays in :data:`WAIT_S` seconds, marked "sped up N×" on screen.
Every number in a caption comes from a regenerated report (README "At a glance"). Not collected by pytest;
run it by hand:

    python tests/ui_video.py [--out runs/video] [--fps 15]

Writes ``<out>/depth_walkthrough.mp4`` (1920×1080, no sound — add a voice-over if you like) and
``<out>/depth_walkthrough.srt``. Needs what ``ui_smoke.py`` needs, the 8 shipped samples and, for the
real-GPS scene, the recording (``python -m src.cv_pipeline.humminbird fetch``).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui_smoke as U  # noqa: E402  (same folder: the server and browser helpers)

REPO = Path(__file__).resolve().parents[1]
VIEW = (1600, 900)            # the studio's viewport; the video is upscaled to 1080p so the text stays readable
OUT_SIZE = (1920, 1080)
WAIT_S = 2.5                  # an action that runs longer than this plays in this many seconds (marked)

OVERLAY = r"""(() => {
  const st = document.createElement('style');
  st.textContent = `#vcap{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);width:max-content;max-width:1240px;
      padding:11px 20px;background:rgba(4,9,15,.92);color:#eaf2f9;font:500 19px/1.42 var(--sans),'Segoe UI',sans-serif;
      border:1px solid rgba(127,214,240,.28);border-radius:12px;z-index:2147483646;box-shadow:0 10px 40px rgba(0,0,0,.55);
      text-align:center;opacity:0;transition:opacity .3s}
    #vcap.on{opacity:1} #vcap b{color:#7fd6f0;font-weight:600}
    #vcard{position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;
      background:radial-gradient(1200px 600px at 50% 40%,#0d2230 0%,#05090e 70%);color:#eaf2f9;z-index:2147483647;
      font-family:var(--sans),'Segoe UI',sans-serif;text-align:center;opacity:0;pointer-events:none;transition:opacity .5s}
    #vcard.on{opacity:1}
    #vcard h1{font-size:72px;letter-spacing:.06em;margin:0;font-weight:700}
    #vcard p{font-size:25px;margin:0;color:#c9d6e2;max-width:1150px;line-height:1.45}
    #vcard p.s{font-size:19px;color:#8fa3b5}`;
  document.head.appendChild(st);
  const c = document.createElement('div'); c.id = 'vcap'; document.body.appendChild(c);
  const k = document.createElement('div'); k.id = 'vcard'; document.body.appendChild(k);
  window.__cap = html => { c.innerHTML = html || ''; c.classList.toggle('on', !!html); };
  window.__card = html => { if (html) k.innerHTML = html; k.classList.toggle('on', !!html); };
  window.__wait = ms => new Promise(r => setTimeout(r, ms));
  window.__tab = name => { const t = [...document.querySelectorAll('.tabs .tab')].find(x => x.textContent.trim() === name);
    if (t) t.click(); return !!t; };
  window.__scrollTo = (sel, text) => { const el = [...document.querySelectorAll(sel)].find(x => !text || x.textContent.includes(text));
    if (el) el.scrollIntoView({block: 'center', behavior: 'smooth'}); return !!el; };
  window.__quiet = () => document.querySelectorAll('.notice .nt-x').forEach(b => b.click());
  return true;
})()"""

# (caption shown while the action runs, JavaScript action, seconds to hold afterwards)
STORY = [
    (None, "__card(`<h1>DEPTH</h1><p>Side-scan sonar in — a cleanup plan a person approves out.</p>"
           "<p class=s>Detect · Evidence · Prove · Triage · Hazard-map &nbsp;·&nbsp; OpenCV 5 on a CPU &nbsp;·&nbsp; "
           "AWS Graviton + COOL &nbsp;·&nbsp; Team Syndicate</p>`)", 5.0),
    (None, "__card(`<p style='font-size:34px;color:#eaf2f9'>DEPTH does not claim a better detector.<br>"
           "<b style='color:#7fd6f0'>It makes any detector dependable.</b></p>"
           "<p class=s>A promise of how many ghost pots reach a person · a check that the promise applies · "
           "a plan that waits for a named person</p>`)", 5.5),
    ("One raw sonar frame. <b>Stage 1</b> (OpenCV) tracks the seabed — the sonar's altitude, measured ping by ping.",
     "__card(null); setMode('analyze'); await __wait(400); "
     "(document.querySelector('.sample-card[data-id=\"sample-02\"]') || document.querySelector('.sample-card')).click(); "
     "await runAnalyze();", 4.0),
    ("The detector put <b>four boxes on one pot</b>. The agent shows them as <b>one card</b> — one decision per "
     "object, no box dropped. On unseen test frames: 3.86 → 2.32 cards per frame.",
     "const m = (state._analyze.candidates || []).findIndex(x => (x.card_members || []).length); "
     "if (m >= 0) selectCandidate(m);", 8.0),
    ("Every card carries its evidence and the agent's trace: shadow, calibrated P(pot), the tier and its promise, "
     "and who acts next.",
     "__scrollTo('.ev-card.is-linked .ev-chain, .ev-card.is-linked .atrace');", 7.0),
    ("The promise: <b>≥ 78%</b> of pots reach a person with one look, <b>≥ 84%</b> when the agent looks harder — "
     "fit on held-out recordings, held on unseen test (81.1% · 86.0%).",
     "const g = [...document.querySelectorAll('.panel.collapsible .panel-title')].find(x => x.textContent.trim() === "
     "'Guarantee'); if (g && g.closest('.panel').classList.contains('is-collapsed')) g.closest('.panel-head').click(); "
     "if (g) { await __wait(250); g.closest('.panel').scrollIntoView({block: 'center', behavior: 'smooth'}); }",
     8.0),
    ("A survey of 8 frames. The agent <b>checks that its promise applies</b>, <b>chooses how hard to look</b> "
     "(no recall goal → the strongest promise) and plans for people and the boat.",
     "__quiet(); setMode('survey'); await __wait(400); $('#gpsMode').value = 'synthetic'; await runSurvey(); "
     "await __wait(600); __tab('Agent');", 8.0),
    ("<b>43 review boxes → 26 cards</b>, one per object. Every option the agent weighed is logged.",
     "__scrollTo('#agentPanel li', 'one per object');", 6.0),
    ("Withhold OpenCV's seabed measurement and the pins move: on 80 test frames <b>95 of 142</b> leave their own "
     "error circle. OpenCV decides where the boat goes.",
     "const b = $('#cfBtn'); if (b && !b.hidden) b.click();", 7.5),
    ("A raw Humminbird recording with <b>real per-ping GPS</b> (Colorado River). It comes from another sonar …",
     "const b = $('#cfBtn'); if (b && b.classList.contains('is-on')) b.click(); "
     "if (!$('#gpsRecording').hidden) { $('#gpsMode').value = 'recording'; "
     "$('#gpsMode').dispatchEvent(new Event('change')); } await runSurvey(); await __wait(800); __tab('Agent'); "
     "__scrollTo('#agentPanel li', 'promise');", 3.0),
    ("… so the agent <b>withdraws its promise</b> and asks a person to approve a <b>full review</b>. "
     "On another sonar it did so in 22 of 22 surveys.", "", 7.5),
    ("Nothing is dispatched: <b>one approval request per action</b>. Agents — MCP clients too — can only ask; "
     "a named person decides.", "__quiet(); __tab('Approvals');", 8.0),
    ("It plugs into what agencies run: <b>OGC API</b> (QGIS · ArcGIS), signed webhooks, a drop-in panel and an "
     "<b>MCP server</b>.", "__quiet(); setMode('connect');", 7.0),
    ("Analyst minutes at <b>assumed</b> timings: DEPTH 43.2 vs manual 55.7 with one look. The Study tab measures "
     "it with people — not run yet, and said so.",
     "setMode('study'); await __wait(500); __tab('Effort');", 8.0),
    (None, "__cap(''); __card(`<h1>DEPTH</h1><p>A promise · a proof · a plan — and a person decides every action.</p>"
           "<p class=s>github.com/madhesh60/depth &nbsp;·&nbsp; every number in this video comes from a re-runnable "
           "report</p><p class=s>Not done yet: the COOL run, a live link, the timed user study.</p>`)", 7.0),
]


def _srt_time(s: float) -> str:
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _plain(html: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", html).replace("&nbsp;", " ")


async def _record(port: int, dbg: int, frames_dir: Path) -> tuple[list, list, float, float]:
    import websockets
    url = None
    for _ in range(60):
        try:
            tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{dbg}/json/list", timeout=2))
            url = next(t["webSocketDebuggerUrl"] for t in tabs if t.get("type") == "page")
            break
        except Exception:
            await asyncio.sleep(0.5)
    frames: list[tuple[float, Path]] = []
    captions: list[tuple[float, str]] = []
    spans: list[tuple[float, float]] = []                 # (start, end) of every action, for the time-lapse
    pending: dict = {}
    n = [0]
    async with websockets.connect(url, max_size=None) as ws:
        async def reader():
            async for raw in ws:
                m = json.loads(raw)
                if m.get("id") in pending:
                    pending.pop(m["id"]).set_result(m)
                elif m.get("method") == "Page.screencastFrame":
                    p = m["params"]
                    f = frames_dir / f"f{len(frames):05d}.jpg"
                    f.write_bytes(base64.b64decode(p["data"]))
                    frames.append((float(p["metadata"]["timestamp"]), f))
                    n[0] += 1
                    await ws.send(json.dumps({"id": n[0], "method": "Page.screencastFrameAck",
                                              "params": {"sessionId": p["sessionId"]}}))
        task = asyncio.create_task(reader())

        async def send(method, params=None, timeout=900):
            n[0] += 1
            fut = asyncio.get_running_loop().create_future()
            pending[n[0]] = fut
            await ws.send(json.dumps({"id": n[0], "method": method, "params": params or {}}))
            return await asyncio.wait_for(fut, timeout)

        async def js(expr: str):
            r = (await send("Runtime.evaluate", {"expression": f"(async () => {{ {expr} }})()", "awaitPromise": True,
                                                 "returnByValue": True}))["result"]
            if "exceptionDetails" in r:
                print("  step error:", r["exceptionDetails"].get("text"), (r["exceptionDetails"].get("exception") or {})
                      .get("description", "")[:200])

        for d in ("Runtime.enable", "Page.enable"):
            await send(d)
        await send("Emulation.setDeviceMetricsOverride", {"width": VIEW[0], "height": VIEW[1], "deviceScaleFactor": 1,
                                                          "mobile": False})
        await send("Page.navigate", {"url": f"http://127.0.0.1:{port}/"})
        await asyncio.sleep(5)
        await send("Runtime.evaluate", {"expression": OVERLAY, "returnByValue": True})
        await send("Page.startScreencast", {"format": "jpeg", "quality": 85, "maxWidth": VIEW[0], "maxHeight": VIEW[1],
                                            "everyNthFrame": 1})
        await asyncio.sleep(0.5)
        t_start = time.time()
        for cap, action, hold in STORY:
            if cap is not None:
                await js(f"__cap({json.dumps(cap)});")
                captions.append((time.time(), cap))
            if action:
                a0 = time.time()
                await js(action)
                spans.append((a0, time.time()))
            await asyncio.sleep(hold)
        t_end = time.time()
        await send("Page.stopScreencast")
        await asyncio.sleep(0.3)
        task.cancel()
    return frames, captions, spans, t_start, t_end


def _timeline(t0: float, t1: float, spans: list) -> list[tuple[float, float, float]]:
    """Source-time segments ``(start, end, speed)`` covering t0..t1: an action that ran longer than WAIT_S plays
    in WAIT_S seconds; everything else plays in real time."""
    segs, t = [], t0
    for a, b in sorted(spans):
        if b - a <= WAIT_S or b <= t:
            continue
        a = max(a, t)
        if a > t:
            segs.append((t, a, 1.0))
        segs.append((a, b, (b - a) / WAIT_S))
        t = b
    if t < t1:
        segs.append((t, t1, 1.0))
    return segs


def _to_out(ts: float, segs: list) -> float:
    """Where a source time lands in the video."""
    out = 0.0
    for a, b, v in segs:
        if ts <= b:
            return out + max(0.0, ts - a) / v
        out += (b - a) / v
    return out


def _badge(img, speed: float) -> None:
    """'sped up N×' in the top-right corner of a time-lapsed frame."""
    txt = f"sped up {speed:.0f}x"
    (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    x, y = OUT_SIZE[0] - w - 48, 40
    cv2.rectangle(img, (x - 16, y - 12), (x + w + 16, y + h + 14), (20, 14, 8), -1)
    cv2.putText(img, txt, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 214, 127), 2, cv2.LINE_AA)


def _encode(frames: list, segs: list, fps: int, out: Path) -> int:
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, OUT_SIZE)
    if not vw.isOpened():
        raise SystemExit("cv2.VideoWriter could not open an mp4v writer")
    frames = sorted(frames)
    k, last, img, written = 0, None, None, 0
    for a, b, speed in segs:
        n = max(1, int(round((b - a) / speed * fps)))
        for i in range(n):
            t = a + i * speed / fps
            while k < len(frames) and frames[k][0] <= t:
                k += 1
            if k and frames[k - 1][1] != last:
                last = frames[k - 1][1]
                src = cv2.imread(str(last))
                img = cv2.resize(src, OUT_SIZE, interpolation=cv2.INTER_CUBIC) if src is not None else img
            if img is not None:
                frame = img.copy() if speed > 1.01 else img
                if speed > 1.01:
                    _badge(frame, speed)
                vw.write(frame)
                written += 1
    vw.release()
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=str(REPO / "runs" / "video"))
    ap.add_argument("--fps", type=int, default=15)
    a = ap.parse_args()
    chrome = next((c for c in U.CANDIDATES if c and Path(c).exists()), None)
    if not chrome:
        print("no Chrome / Chromium found - set $CHROME")
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(tempfile.mkdtemp(prefix="depth-video-"))
    port, dbg = U._free_port(), U._free_port()
    env = {**os.environ, "DEPTH_APPROVALS_DIR": tempfile.mkdtemp(prefix="depth-video-appr-"), "DEPTH_PUBLIC": "0",
           "PYTHONPATH": str(REPO)}
    for k in ("DEPTH_APPROVER_PIN", "DEPTH_MCP_TOKEN", "DEPTH_TRUST_PROXY", "DEPTH_ORIGIN_SECRET"):
        env.pop(k, None)
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.dashboard.app:app", "--host", "127.0.0.1",
                            "--port", str(port)], cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    browser = None
    try:
        U._ready(f"http://127.0.0.1:{port}")
        browser = subprocess.Popen([chrome, "--headless=new", f"--remote-debugging-port={dbg}",
                                    f"--user-data-dir={tempfile.mkdtemp(prefix='depth-video-chrome-')}", "--no-first-run",
                                    "--no-default-browser-check", "--hide-scrollbars", f"--window-size={VIEW[0]},{VIEW[1]}",
                                    "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        frames, captions, spans, t0, t1 = asyncio.run(_record(port, dbg, frames_dir))
    finally:
        if browser:
            browser.kill()
        srv.kill()
    if not frames:
        print("no frames captured")
        return 1
    mp4 = out / "depth_walkthrough.mp4"
    segs = _timeline(t0, t1, spans)
    n = _encode(frames, segs, a.fps, mp4)
    srt = []
    for i, (ts, cap) in enumerate(captions):
        end = captions[i + 1][0] if i + 1 < len(captions) else t1
        srt += [str(i + 1), f"{_srt_time(_to_out(ts, segs))} --> {_srt_time(_to_out(end, segs))}", _plain(cap), ""]
    (out / "depth_walkthrough.srt").write_text("\n".join(srt), encoding="utf-8")
    shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"{mp4} — {n} frames at {a.fps} fps = {n / a.fps:.0f} s, {mp4.stat().st_size / 1e6:.1f} MB "
          f"({len(frames)} screencast frames); captions: {out / 'depth_walkthrough.srt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
