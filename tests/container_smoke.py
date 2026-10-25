"""container_smoke.py — does a running DEPTH container (Docker or Kubernetes) do real work end to end?

    python tests/container_smoke.py --url http://localhost:8000

Checks, in order: liveness (/healthz), readiness (/readyz: the detector loaded and warmed), /api/health (the
model, OpenCV 5, whether the build is COOL), one Analyze on a held-out sample (boxes, and the provenance stamp
names the code commit), and, when the image carries the raw recording, a Survey job on it (real GPS).
Exit 0 only if every check passes. Standard library only, so it runs anywhere Python does.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # localhost: never through a proxy


def call(url: str, method: str = "GET", timeout: float = 120) -> tuple[int, dict]:
    req = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)
    try:
        with OPENER.open(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--wait", type=float, default=180, help="seconds to wait for /readyz")
    ap.add_argument("--commit", help="the provenance stamp must name this commit")
    a = ap.parse_args()
    u = a.url.rstrip("/")
    fails: list[str] = []

    def check(ok: bool, what: str) -> None:
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    t0 = time.time()
    code = 0
    while time.time() - t0 < a.wait:
        try:
            code, _ = call(f"{u}/readyz", timeout=5)
        except OSError:
            code = 0
        if code == 200:
            break
        time.sleep(2)
    check(code == 200, f"/readyz 200 after {time.time() - t0:.0f} s (detector loaded and warmed)")
    check(call(f"{u}/healthz")[0] == 200, "/healthz 200")

    _, h = call(f"{u}/api/health")
    check(bool(h.get("model_loaded")), f"model {h.get('model')} loaded")
    check(str(h.get("opencv", "")).startswith("5."), f"OpenCV {h.get('opencv')} (COOL: {h.get('is_cool_path')})")

    code, r = call(f"{u}/api/analyze?sample=sample-02", "POST")
    stamp = json.dumps(r)
    check(code == 200 and bool(r.get("candidates")), f"analyze sample-02: {len(r.get('candidates') or [])} box(es) for review")
    if a.commit:
        check(f'"code_commit": "{a.commit[:7]}' in stamp, f"provenance names commit {a.commit[:7]}")

    if (h.get("recording") or {}).get("available"):
        code, j = call(f"{u}/api/jobs/survey?gps=recording", "POST")
        status = "submit failed"
        for _ in range(150):
            if code != 200:
                break
            status = call(f"{u}/api/jobs/{j['job_id']}")[1].get("status", "?")
            if status not in ("queued", "running"):
                break
            time.sleep(2)
        check(status == "done", f"survey of the raw recording (real GPS): {status}")
    else:
        print("skip survey: no raw recording in this image")

    print("container smoke:", "PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
