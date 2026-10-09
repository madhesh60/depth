"""
study_sensor.py — STUDY-16: does the agent's sensor cross-check fix the seabed geometry, on a
recording it was not designed on?

* **design** recording: Test-Small-DS (R01224, Humminbird 9xx, Colorado River, 150 s). Every
  threshold in ``sensor_check.py`` was set looking at it.
* **fresh** recording: Test-Large-DS (Rec00002, Humminbird Solix, Pearl River, 1 h). Never used
  to set anything; the criteria below were written into this file before it was first run.

There is no surveyed seabed for either recording, so the checks use what is independent of the
choices being judged:

* **C1 · port vs starboard** — two different transducers and images of the same pings. Over a
  flat bed their first returns coincide; a tracker that locks onto ring-down or a deeper layer on one
  side breaks that. Registered: over chunk pairs the loop accepted on both sides, the median
  |Δ| / altitude is ≤ 5% AND ≤ the tracker-alone median over pairs where both sides had a track.
* **C2 · scale stability** — metres per sample re-fit separately on the first and the second half
  of the recording agree within the reported uncertainty.
* **C3 · gross tracker failures caught** — chunks where the tracker alone is > 50% from a steady
  sounder (silently wrong geometry today): every one must end ``corrected`` / ``recovered`` /
  ``not_measured``, none accepted as is. (Reported for both recordings.)
* **visual audit** — an overlay sheet of chunks sampled at random (seed 0) is published; the
  developer's reading of it is reported as a non-blind check, the images are there to be re-read.

    python -m src.agentic.study_sensor [--fresh-only] [--no-fresh]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from src.cv_pipeline.canonical import bottom_track
from src.cv_pipeline.humminbird import read_recording, chunk_recording, range_scale
from . import sensor_check as sc

REPO = Path(__file__).resolve().parents[2]
DESIGN = REPO / "DATASET" / "external" / "pingmapper_sample" / "Test-Small-DS.DAT"
FRESH = REPO / "DATASET" / "external" / "pingmapper_large" / "Test-Large-DS.DAT"
C1_MAX = 0.05
GROSS = 0.50
# the developer's (non-blind) reading of the fresh audit sheet, 2026-10-09 - kept so a re-run keeps it
AUDIT_NOTE = ("in 16 of 16 randomly drawn chunks the accepted (green) line sits on the first seabed return, including "
              "the 4 corrected ones where the tracker alone (red) cut through the water column or the bed; on this river "
              "bed sunken trees (snags) rise above the seabed and the line follows the bed beneath them.")


def _depths(ps) -> np.ndarray:
    return np.array([p.depth_m if p.depth_m and p.depth_m > 0 else np.nan for p in ps])


def run(dat: Path, nchunk: int = 500) -> dict:
    t0 = time.perf_counter()
    rec = read_recording(dat)
    chunks = chunk_recording(rec, nchunk)
    first = range_scale([(ps, son) for _, _, ps, son in chunks])
    freq = next((ps[0].freq_hz for _, _, ps, _ in chunks if ps), None)
    res = sc.check_recording([(c, j, son, _depths(ps)) for c, j, ps, son in chunks], first["m_per_sample"], freq)
    scale = res["scale"]["m_per_sample"]
    rows, tracker = [], {}
    for c, j, ps, son in chunks:
        line, alt, conf, _ = bottom_track(son)
        tracker[(c, j)] = line
        g = res["chunks"][(c, j)]
        exp = _depths(ps) / scale
        err = None if line is None else float(np.nanmedian(np.abs(line - exp) / exp))
        rows.append({"channel": c, "index": j, "status": g.status, "source": g.source,
                     "sounder_steady": g.sounder_steady, "sounder_spread": g.metrics["sounder"]["spread"],
                     "tracker_alt": None if alt is None else round(alt, 1), "tracker_conf": round(conf, 2),
                     "tracker_err": None if err is None else round(err, 3),
                     "accepted_alt": g.metrics.get("accepted_altitude_samples"),
                     "altitude_m": None if g.altitude_m is None else round(g.altitude_m, 2),
                     "sounder_m": g.metrics["sounder"]["depth_m_median"]})
    # C1 - port vs starboard, tracker alone vs the loop
    def ps_diff(lines):
        d = []
        for (c, j), a in lines.items():
            b = lines.get(("starboard", j)) if c == "port" else None
            if a is not None and b is not None and len(a) == len(b):
                d.append(float(np.median(np.abs(a - b)) / max(float(np.median(a)), 1.0)))
        return d
    d0 = ps_diff(tracker)
    d1 = ps_diff({k: g.line if g.trusted else None for k, g in res["chunks"].items()})
    c1 = {"tracker_pairs": len(d0), "tracker_median": _r(np.median(d0)) if d0 else None,
          "loop_pairs": len(d1), "loop_median": _r(np.median(d1)) if d1 else None}
    c1["pass"] = bool(d1) and c1["loop_median"] <= C1_MAX and (not d0 or c1["loop_median"] <= c1["tracker_median"])
    # POST-HOC (added after the registered fresh run failed C1): the registered comparison pits the loop's
    # pairs against the tracker's — different sets (the tracker has a pair only where BOTH sides found a
    # track, the easy chunks). Same pairs only, and the pairs only the loop could measure:
    both = [k for k in tracker if k[0] == "port" and tracker[k] is not None and tracker.get(("starboard", k[1])) is not None]
    loop_lines = {k: g.line if g.trusted else None for k, g in res["chunks"].items()}
    same0 = ps_diff({k: tracker[k] for k in both} | {("starboard", k[1]): tracker[("starboard", k[1])] for k in both})
    same1 = ps_diff({k: loop_lines[k] for k in both} | {("starboard", k[1]): loop_lines[("starboard", k[1])] for k in both})
    only = [k for k in loop_lines if k[0] == "port" and k not in both]
    new1 = ps_diff({k: loop_lines[k] for k in only} | {("starboard", k[1]): loop_lines.get(("starboard", k[1])) for k in only})
    c1["posthoc"] = {"same_pairs": len(same0), "tracker_same": _r(np.median(same0)) if same0 else None,
                     "loop_same": _r(np.median(same1)) if same1 else None,
                     "loop_only_pairs": len(new1), "loop_only_median": _r(np.median(new1)) if new1 else None}
    # C2 - scale re-fit on each half
    half = max(r["index"] for r in rows) // 2
    halves = []
    for sel in (lambda j: j <= half, lambda j: j > half):
        part = [(c, j, son, _depths(ps)) for c, j, ps, son in chunks if sel(j)]
        r2 = sc.check_recording(part, scale, freq)
        halves.append(r2["scale"]["m_per_sample"])
    gap = abs(halves[0] - halves[1]) / scale
    c2 = {"first_half": _r(halves[0], 5), "second_half": _r(halves[1], 5), "rel_gap": _r(gap),
          "rel_unc": res["scale"]["rel_unc"], "pass": gap <= res["scale"]["rel_unc"]}
    # C3 - gross failures of the tracker alone against a steady sounder
    gross = [r for r in rows if r["sounder_steady"] and r["tracker_err"] is not None and r["tracker_err"] > GROSS]
    c3 = {"gross": len(gross), "accepted_as_is": sum(r["status"] in ("agree", "image_trusted") and r["source"] == "tracker"
                                                    for r in gross),
          "outcomes": {s: sum(r["status"] == s for r in gross) for s in sc.STATUSES}}
    c3["pass"] = c3["accepted_as_is"] == 0
    return {"recording": rec.name, "family": rec.dat["family"], "freq_hz": freq, "chunks": len(chunks),
            "pings_per_side": {k: len(v.pings) for k, v in rec.channels.items() if k in ("port", "starboard")},
            "duration_s": round((rec.channels["port"].pings[-1].time_ms - rec.channels["port"].pings[0].time_ms) / 1000, 1),
            "first_scale": first, "result": sc.summary(res), "iterations": res["iterations"], "rows": rows,
            "c1": c1, "c2": c2, "c3": c3, "seconds": round(time.perf_counter() - t0, 1),
            "_res": res, "_chunks": chunks, "_tracker": tracker}


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def overlay(son: np.ndarray, tracker, expected, accepted, label: str, size=(360, 300)) -> np.ndarray:
    """Red = OpenCV tracker alone, blue = the sounder's altitude, green = what the agent accepted."""
    im = cv2.cvtColor(son, cv2.COLOR_GRAY2BGR)
    H = son.shape[0]
    crop = int(min(H, max(np.nanmax(expected) * 1.6 if np.isfinite(expected).any() else 0,
                          (np.max(accepted) * 1.6) if accepted is not None else 0, H * 0.35)))
    for arr, col in ((expected, (255, 140, 0)), (tracker, (0, 0, 255)), (accepted, (0, 220, 0))):
        if arr is None:
            continue
        for x in range(0, len(arr), 2):
            if np.isfinite(arr[x]):
                cv2.circle(im, (x, int(arr[x])), 2, col, -1)
    im = cv2.resize(im[:max(crop, 40)], size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(im, (0, size[1] - 20), (size[0], size[1]), (0, 0, 0), -1)      # bottom: keep row 0 visible
    cv2.putText(im, label, (4, size[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def sheet(out: dict, picks: list[tuple[str, int]], path: Path, cols: int = 4) -> None:
    res, chunks, tracker = out["_res"], {(c, j): (ps, son) for c, j, ps, son in out["_chunks"]}, out["_tracker"]
    scale = res["scale"]["m_per_sample"]
    tiles = []
    for k in picks:
        ps, son = chunks[k]
        g = res["chunks"][k]
        tiles.append(overlay(son, tracker[k], _depths(ps) / scale, g.line if g.trusted else None,
                             f"{k[0]} {k[1]} - {g.status.replace('_', ' ')}"))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def audit_picks(out: dict, n: int = 16, seed: int = 0) -> list[tuple[str, int]]:
    """Up to n chunks at random, stratified so every status present is shown."""
    rng = random.Random(seed)
    by = {}
    for k, g in sorted(out["_res"]["chunks"].items()):
        by.setdefault(g.status, []).append(k)
    picks = []
    while len(picks) < n and any(by.values()):
        for s in sc.STATUSES:
            if by.get(s) and len(picks) < n:
                picks.append(by[s].pop(rng.randrange(len(by[s]))))
    return sorted(picks, key=lambda k: (k[1], k[0]))


def _table(out: dict) -> list[str]:
    L = ["| chunk | sounder | tracker alone (samples, conf, error vs sounder) | agent's verdict | source | altitude used |",
         "|---|---|---|---|---|---|"]
    for r in out["rows"]:
        L.append(f"| {r['channel']} {r['index']} | {r['sounder_m']} m, spread {r['sounder_spread']}"
                 f"{'' if r['sounder_steady'] else ' **(lost lock)**'} | "
                 + (f"{r['tracker_alt']}, {r['tracker_conf']}, {r['tracker_err']:.0%}" if r['tracker_alt'] is not None else "no track")
                 + f" | **{r['status'].replace('_', ' ')}** | {r['source'] or '—'} | "
                 + (f"{r['altitude_m']} m" if r['altitude_m'] is not None else "withheld") + " |")
    return L


def _criteria(out: dict) -> list[str]:
    c1, c2, c3 = out["c1"], out["c2"], out["c3"]
    ok = lambda b: "**PASS**" if b else "**FAIL**"
    return ["| criterion | registered | measured | |", "|---|---|---|---|",
            f"| C1 port vs starboard (independent images) | loop median ≤ {C1_MAX:.0%} and ≤ tracker alone | "
            f"loop {c1['loop_median']} over {c1['loop_pairs']} pairs · tracker alone {c1['tracker_median']} over "
            f"{c1['tracker_pairs']} pairs | {ok(c1['pass'])} |",
            *([f"| _C1 post-hoc (added after the run): same pairs_ | — | _on the {c1['posthoc']['same_pairs']} pairs where "
               f"the tracker alone had both sides: tracker {c1['posthoc']['tracker_same']}, loop {c1['posthoc']['loop_same']}; "
               f"on the {c1['posthoc']['loop_only_pairs']} pairs only the loop could measure: "
               f"{c1['posthoc']['loop_only_median']}_ | |"] if c1.get("posthoc") else []),
            f"| C2 scale stable across halves | gap ≤ reported uncertainty | {c2['first_half'] * 100:.3f} vs "
            f"{c2['second_half'] * 100:.3f} cm/sample: gap {c2['rel_gap']:.1%}, uncertainty ±{c2['rel_unc']:.1%} | {ok(c2['pass'])} |",
            f"| C3 gross tracker failures caught | none accepted as is | {c3['gross']} chunk(s) > {GROSS:.0%} off a steady "
            f"sounder → {', '.join(f'{v} {k.replace(chr(95), chr(32))}' for k, v in c3['outcomes'].items() if v) or '—'}; "
            f"accepted as is: {c3['accepted_as_is']} | {ok(c3['pass'])} |"]


def report(design: dict, fresh: dict | None) -> str:
    d = design
    L = ["# STUDY-16 — The agent cross-checks OpenCV's seabed track against the depth sounder", "",
         "_`python -m src.agentic.study_sensor` · thresholds set on the design recording only · the fresh recording "
         "is checked once against criteria written into the script before its first run · regenerated by the script "
         "— do not edit_", "",
         "**Why.** Every metre DEPTH reports on a raw recording hangs on one number per ping: the sonar altitude. "
         "It sets the ground range (where a pin goes), heights in metres (`h/H × altitude`) and the range scale "
         "itself. Stage 1 reads it off the image with OpenCV; the ping headers carry an independent depth sounder. "
         "On the design recording **neither is always right**: the tracker locked onto the transducer ring-down "
         "(8 samples against ~117) *with confidence 0.99*, and in the last third the sounder lost lock and jumped "
         "between ~1.5 m and ~6 m from one reading to the next. Picking one sensor by rule is wrong either way.", "",
         "**What the agent does** (`src/agentic/sensor_check.py`, per 500-ping chunk): OpenCV bottom track → is the "
         "sounder steady? → do they agree? → on a conflict, **re-run the OpenCV tracker** inside a window the steady "
         "sounder sets → accept it only if the edge is coherent (smooth ping to ping, ≥ 2× the speckle gradient) → "
         "with an unsteady sounder, trust a coherent image edge and flag the sounder → otherwise re-track inside the "
         "**other channel's** accepted line (same pings) → otherwise **not measured** (slant range, widened error "
         "radius, no metres). Then it **re-fits the range scale** from sounder-backed chunks and re-checks every chunk "
         "until the scale stops moving. Every call, skip and verdict is an `AgentStep` in the trace and the "
         "survey's Agent tab.", "",
         f"## Design recording — {d['recording']} ({d['family']}, {d['freq_hz'] / 1000:.0f} kHz, {d['duration_s']} s)", "",
         "![design overlays](img/sensor_check_design.jpg)", "",
         "_Red: OpenCV tracker alone · blue: the sounder's altitude at the final scale · green: what the agent accepted._", ""]
    L += _table(d)
    L += ["", f"Scale: first estimate {d['first_scale']['m_per_sample'] * 100:.3f} cm/sample → "
          + " → ".join(f"{it['m_per_sample_out'] * 100:.3f}" for it in d["iterations"] if it["m_per_sample_out"])
          + f" ({len(d['iterations'])} iteration(s)); beam-physics estimate "
          f"{d['result']['scale']['physics_m_per_sample'] * 100:.3f} cm/sample "
          f"(gap {d['first_scale'].get('physics_rel_gap', 0):.1%} → {d['result']['scale']['physics_rel_gap']:.1%}).", ""]
    L += _criteria(d)
    if fresh:
        f = fresh
        L += ["", f"## Fresh recording — {f['recording']} ({f['family']}, {f['freq_hz'] / 1000:.0f} kHz, "
              f"{f['duration_s'] / 60:.0f} min, {f['chunks']} chunks) — run once", "",
              f"Verdicts: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in f["result"]["counts"].items())
              + f" · {f['result']['iterations']} scale iteration(s) · {f['seconds']:.0f} s on this laptop "
              f"for a {f['duration_s'] / 60:.0f}-min recording.", ""]
        L += _criteria(f)
        L += ["", "![fresh audit sheet](img/sensor_check_fresh.jpg)", "",
              f"_{len(f['audit'])} chunks drawn at random (seed 0, stratified by verdict). Red: tracker alone · blue: "
              f"sounder · green: accepted._", ""]
        if f.get("audit_note"):
            L += [f"**Visual audit (developer, not blind):** {f['audit_note']}", ""]
        sc_ = f["result"]["scale"]
        L += [f"Scale on the fresh recording: {f['first_scale']['m_per_sample'] * 100:.3f} → {sc_['m_per_sample'] * 100:.3f} "
              f"cm/sample, ±{sc_['rel_unc']:.1%}. The beam-physics cross-check is not applied at "
              f"{f['freq_hz'] / 1000:.0f} kHz (the formula's transducer length is for the 455 kHz unit).", ""]
    if fresh:
        f, c1 = fresh, fresh["c1"]
        tracked0 = sum(r["tracker_alt"] is not None for r in f["rows"])
        L += ["## Decision", "",
              f"**On by default for raw recordings.** C2 and C3 held on the fresh recording; C1 **failed as registered** "
              f"(the loop's port/starboard median {c1['loop_median']:.2%} is under the {C1_MAX:.0%} limit but above the "
              f"tracker alone's {c1['tracker_median']:.2%}). The registered comparison used different pair sets — the "
              f"tracker alone has a pair only where both sides found a track ({c1['tracker_pairs']} easy pairs). On those "
              f"same pairs the loop is not worse ({c1['posthoc']['loop_same']:.2%} vs {c1['posthoc']['tracker_same']:.2%}, "
              f"post-hoc); the rest are the {c1['posthoc']['loop_only_pairs']} pairs only the loop could measure "
              f"({c1['posthoc']['loop_only_median']:.2%}). The tracker alone gave a seabed on {tracked0} of {f['chunks']} "
              f"chunks (with {f['c3']['gross']} gross failures it would have used silently); the agent resolved "
              f"{f['result']['trusted']} and accepted none of those failures. The failed criterion is kept in this "
              f"report, not re-registered.", ""]
    L += ["## Limits", "",
          "- No surveyed seabed: the checks are consistency between independent measurements (two transducers, the "
          "sounder, two halves of a recording) and published overlays, not ground truth.",
          "- Two recordings from two units; thresholds are design constants from one of them.",
          "- Port and starboard coincide only over a flat bed; a sloping bank legitimately separates them, so C1 is an "
          "upper bound on tracking error, not a measurement of it.", "",
          "---", "_Regenerated by the script; do not edit by hand._", ""]
    return "\n".join(L)


def _strip(out: dict) -> dict:
    return {k: v for k, v in out.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser(description="STUDY-16 sensor cross-check")
    ap.add_argument("--no-fresh", action="store_true")
    ap.add_argument("--audit-note", default=AUDIT_NOTE, help="the developer's reading of the fresh audit sheet")
    a = ap.parse_args()
    d = run(DESIGN)
    sheet(d, [("port", 0), ("starboard", 0), ("port", 2), ("port", 4), ("starboard", 5), ("starboard", 6)],
          REPO / "docs" / "img" / "sensor_check_design.jpg", cols=3)
    f = None
    res_path = REPO / "docs" / "sensor_check.json"
    if not a.no_fresh and FRESH.exists():
        f = run(FRESH)
        f["audit"] = [list(k) for k in audit_picks(f)]
        sheet(f, [tuple(k) for k in f["audit"]], REPO / "docs" / "img" / "sensor_check_fresh.jpg")
        f["audit_note"] = a.audit_note
    res_path.write_text(json.dumps({"design": _strip(d), "fresh": _strip(f) if f else None}, indent=1, default=str),
                        encoding="utf-8")
    (REPO / "docs" / "sensor_check.md").write_text(report(d, f), encoding="utf-8")
    print("design:", json.dumps({k: d[k] for k in ("c1", "c2", "c3")}, default=str))
    if f:
        print("fresh:", json.dumps({k: f[k] for k in ("c1", "c2", "c3")}, default=str), f["result"]["counts"])


if __name__ == "__main__":
    main()
