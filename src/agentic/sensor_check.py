"""
sensor_check.py — the agent cross-checks OpenCV's seabed track against an independent sensor and
re-measures when they disagree (STUDY-16).

A raw recording carries two measurements of the same thing: the **sonar altitude** that Stage 1's
OpenCV bottom tracker reads off the image, and the **depth sounder** in every ping header. Neither is
always right (STUDY-16, Test-Small-DS): the tracker can lock onto the transducer ring-down or a
deeper layer *with high confidence*, and the sounder can lose lock and jump between two depths. Every
downstream number hangs on this altitude — ground range (where a pin goes), heights in metres, the
range scale itself — so the agent does not pick one sensor by rule. Per chunk of pings it loops:

1. ``bottom_track``   — OpenCV's tracker on the native sonogram (Stage 1).
2. ``sounder_check``  — is the sounder steady here? (windowed p10–p90 spread / median)
3. ``crosscheck``     — do the two agree (≥ AGREE_SHARE of pings within AGREE_TOL)? → **agree**.
4. ``retrack_guided`` — on a conflict (or no track) with a steady sounder, the tracker is RE-RUN with
   a search window the sounder sets (an OpenCV tool call with new parameters). A coherent edge there
   → **corrected**; an incoherent one is not forced.
5. ``image_check``    — with an unsteady sounder, a coherent image edge wins → **image_trusted** (the
   sounder is flagged for that chunk).
6. ``retrack_cross_channel`` — still nothing: port and starboard share every ping, so the other
   channel's accepted line sets the window for one more re-track.
7. ``resolve``        — what was accepted, from which source, and what it means downstream; nothing
   coherent → **not_measured**: no metres, no ground range, a widened error radius (never guessed).

Then, over the recording: ``estimate_scale`` re-fits metres per sample from the sounder-backed
chunks only, and if it moved by more than SCALE_CONVERGED the whole loop runs again with the new
scale (at most MAX_ITERS times) — the agent revising its own measurement model.

The thresholds below were set on Test-Small-DS (R01224, Colorado River) and are design constants;
the fresh check is the 1-h Solix recording Test-Large-DS (Pearl River) — ``python -m
src.agentic.sensor_check`` writes the report (``docs/sensor_check.md``).
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.cv_pipeline.canonical import (TRACK_WIN, bottom_track, edge_picks, line_quality, smooth_line)
from .types import AgentStep

# -- design constants (set on Test-Small-DS; not fit on the fresh recording) ------------------------
AGREE_TOL = 0.15            # a ping agrees if |image altitude − sounder altitude| ≤ 15% of the sounder's
AGREE_SHARE = 0.60          # a chunk agrees if ≥ 60% of its pings do
SOUNDER_SPREAD_MAX = 0.50   # sounder steady if the median windowed (p90 − p10) / median ≤ 0.50 (real depth
                            # change reached 0.30 on Test-Small-DS; a sounder that lost lock 0.89–0.93)
SOUNDER_WIN = 100           # pings per sounder-spread window
GUIDE_WIN = 0.30            # guided re-track searches ±30% around the sounder's altitude
CROSS_WIN = 0.25            # cross-channel re-track searches ±25% around the other channel's line
ROUGH_MAX = 0.02            # a coherent edge: picks within 2% of the altitude of their rolling median
STRENGTH_MIN = 2.0          # ... and a step ≥ 2× the speckle gradient
MAX_ITERS = 3
SCALE_CONVERGED = 0.02

STATUSES = ("agree", "corrected", "recovered", "image_trusted", "not_measured")
TRUSTED = {"agree", "corrected", "recovered", "image_trusted"}
SOUNDER_BACKED = {"agree", "corrected", "recovered"}

_MEANING = {
    "agree": "image and sounder agree — ground range and heights in metres are measured",
    "corrected": "the tracker was wrong; re-tracked where the sounder pointed, and a coherent seabed edge was there",
    "recovered": "the tracker found no seabed; re-tracked where the sounder pointed, and a coherent edge was there",
    "image_trusted": "the sounder lost lock here; the image's seabed edge is coherent, so the image is trusted "
                     "and the sounder reading is flagged",
    "not_measured": "no source gives a coherent seabed here — ground range and metres are withheld; "
                    "positions use slant range with the error radius widened by the altitude bound",
}


@dataclass
class ChunkGeometry:
    channel: str
    index: int
    n_pings: int
    samples: int
    status: str = "not_measured"
    source: Optional[str] = None             # tracker | sounder-guided re-track | cross-channel re-track
    line: Optional[np.ndarray] = None        # accepted per-ping first-return row (native samples)
    sounder_steady: bool = False
    altitude_m: Optional[float] = None       # the chunk's altitude in metres (from the trusted source)
    metrics: dict = field(default_factory=dict)
    steps: list[AgentStep] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.channel}:{self.index}"

    @property
    def trusted(self) -> bool:
        return self.status in TRUSTED

    def line_resampled(self, size: int) -> Optional[list[float]]:
        """The accepted line in a ``size`` × ``size`` resized frame's pixels (columns = pings)."""
        if self.line is None:
            return None
        cols = np.linspace(0, len(self.line) - 1, size)
        return [round(float(v) * size / self.samples, 2) for v in np.interp(cols, np.arange(len(self.line)), self.line)]

    def to_dict(self) -> dict:
        return {"channel": self.channel, "index": self.index, "status": self.status, "source": self.source,
                "meaning": _MEANING[self.status], "sounder_steady": self.sounder_steady,
                "altitude_m": None if self.altitude_m is None else round(self.altitude_m, 2),
                "metrics": self.metrics, "steps": [s.to_dict() for s in self.steps]}


# ------------------------------------------------------------------------------------- measures
def sounder_spread(depths: np.ndarray, win: int = SOUNDER_WIN) -> Optional[float]:
    """Median over ``win``-ping windows of (p90 − p10) / median depth. A sounder that lost lock jumps
    between two depths (STUDY-16: 1.5 m ↔ 6 m from one reading to the next) — a large spread.
    Ping-to-ping differences miss it: the sounder holds each reading for many pings."""
    d = np.asarray(depths, float)
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) < 10:
        return None
    vals = []
    for s in range(0, len(d), win):
        w = d[s:s + win]
        if len(w) >= 10:
            vals.append((np.percentile(w, 90) - np.percentile(w, 10)) / max(np.median(w), 1e-6))
    return float(np.median(vals)) if vals else None


def coherent(q: dict) -> bool:
    return q.get("roughness") is not None and q["roughness"] <= ROUGH_MAX and q["strength"] >= STRENGTH_MIN


def agree_share(line: np.ndarray, expected: np.ndarray) -> Optional[float]:
    ok = np.isfinite(expected) & (expected > 0)
    if not ok.any():
        return None
    return float(np.mean(np.abs(line[ok] - expected[ok]) / expected[ok] <= AGREE_TOL))


def _step(tool, why, ms, status="done", **detail) -> AgentStep:
    return AgentStep(tool=tool, rationale=why, latency_ms=round(ms, 2), status=status, detail=detail)


# ------------------------------------------------------------------------------------- one chunk
def check_chunk(channel: str, index: int, son: np.ndarray, depths: np.ndarray, m_per_sample: float,
                other: Optional["ChunkGeometry"] = None) -> ChunkGeometry:
    """The per-chunk loop (steps 1–7). ``son``: native sonogram, rows = range samples (nadir first),
    columns = pings. ``other``: the paired channel's result for the same pings, if already decided."""
    H, W = son.shape
    g = ChunkGeometry(channel=channel, index=index, n_pings=W, samples=H)
    depths = np.asarray(depths, float)
    expected = np.where(np.isfinite(depths) & (depths > 0), depths / m_per_sample, np.nan)

    t = time.perf_counter()
    line, alt, conf, step = bottom_track(son)
    tq = None
    if line is not None:
        rows, st = edge_picks(son, line - TRACK_WIN, line + TRACK_WIN)
        tq = line_quality(rows, st)
    ms = (time.perf_counter() - t) * 1000
    if line is None:
        g.steps.append(_step("bottom_track", "OpenCV tracker: no water-column → seabed step found", ms, "failed",
                             track_conf=round(conf, 2)))
    else:
        g.steps.append(_step("bottom_track", f"OpenCV tracker: seabed at {alt:.0f} samples (tracker confidence "
                             f"{conf:.2f}; roughness {tq['roughness']}, edge {tq['strength']}× speckle)", ms,
                             altitude_samples=round(alt, 1), track_conf=round(conf, 2), **tq))
    g.metrics["tracker"] = {"altitude_samples": None if alt is None else round(alt, 1),
                            "track_conf": round(conf, 3), **(tq or {})}

    t = time.perf_counter()
    spread = sounder_spread(depths)
    g.sounder_steady = spread is not None and spread <= SOUNDER_SPREAD_MAX
    g.metrics["sounder"] = {"spread": None if spread is None else round(spread, 3),
                            "depth_m_median": None if not np.isfinite(depths).any() else round(float(np.nanmedian(depths)), 2),
                            "expected_samples": None if not np.isfinite(expected).any() else round(float(np.nanmedian(expected)), 1)}
    g.steps.append(_step("sounder_check", ("sounder steady" if g.sounder_steady else
                                           "sounder unsteady (lost lock / jumping)" if spread is not None else
                                           "no sounder depth in these pings") +
                         (f": windowed spread {spread:.0%} (limit {SOUNDER_SPREAD_MAX:.0%})" if spread is not None else ""),
                         (time.perf_counter() - t) * 1000, steady=g.sounder_steady))

    share = agree_share(line, expected) if line is not None else None
    g.metrics["tracker"]["agree_share"] = None if share is None else round(share, 3)
    if line is not None and share is not None:
        ok = g.sounder_steady and share >= AGREE_SHARE
        g.steps.append(_step("crosscheck", f"{share:.0%} of pings agree with the sounder within {AGREE_TOL:.0%}"
                             + (" → agree" if ok else " → CONFLICT" if g.sounder_steady else
                                " (sounder unsteady: no verdict from it)"), 0.0, agree_share=round(share, 3)))
        if ok:
            return _accept(g, line, "agree", "tracker", depths, m_per_sample)
    else:
        g.steps.append(_step("crosscheck", "nothing to compare (no track or no sounder)", 0.0, "skipped"))

    # 4 - re-run the tracker where the steady sounder points
    if g.sounder_steady:
        t = time.perf_counter()
        rows, st = edge_picks(son, expected * (1 - GUIDE_WIN), expected * (1 + GUIDE_WIN))
        q = line_quality(rows, st)
        gl = smooth_line(rows)
        ok = gl is not None and coherent(q)
        g.metrics["guided"] = q
        g.steps.append(_step("retrack_guided", f"OpenCV re-track within ±{GUIDE_WIN:.0%} of the sounder's altitude: "
                             f"roughness {q['roughness']}, edge {q['strength']}× speckle → "
                             + ("coherent seabed edge" if ok else "incoherent (speckle) — not forced"),
                             (time.perf_counter() - t) * 1000, window=GUIDE_WIN, **q))
        if ok:
            return _accept(g, gl, "corrected" if line is not None else "recovered", "sounder-guided re-track",
                           depths, m_per_sample)
    else:
        g.steps.append(_step("retrack_guided", "skipped — an unsteady sounder cannot guide a re-track", 0.0, "skipped"))

    # 5 - unsteady sounder: a coherent image edge wins
    if line is not None and not g.sounder_steady:
        ok = tq is not None and coherent(tq)
        g.steps.append(_step("image_check", f"tracker edge roughness {tq['roughness'] if tq else None}, "
                             f"{tq['strength'] if tq else 0}× speckle → " + ("coherent: trust the image" if ok else "incoherent"),
                             0.0, **(tq or {})))
        if ok:
            return _accept(g, line, "image_trusted", "tracker", depths, m_per_sample)

    # 6 - the other channel saw the same pings
    if other is not None and other.trusted and other.line is not None and other.n_pings == W:
        t = time.perf_counter()
        rows, st = edge_picks(son, other.line * (1 - CROSS_WIN), other.line * (1 + CROSS_WIN))
        q = line_quality(rows, st)
        cl = smooth_line(rows)
        ok = cl is not None and coherent(q)
        g.metrics["cross"] = q
        g.steps.append(_step("retrack_cross_channel", f"OpenCV re-track within ±{CROSS_WIN:.0%} of the {other.channel} "
                             f"line (same pings): roughness {q['roughness']}, edge {q['strength']}× speckle → "
                             + ("coherent" if ok else "incoherent — not forced"), (time.perf_counter() - t) * 1000, **q))
        if ok:
            return _accept(g, cl, "image_trusted" if not g.sounder_steady else "recovered", "cross-channel re-track",
                           depths, m_per_sample)
    else:
        g.steps.append(_step("retrack_cross_channel", "skipped — the other channel has no accepted line for these pings",
                             0.0, "skipped"))

    g.status = "not_measured"
    g.steps.append(_step("resolve", _MEANING["not_measured"], 0.0, status_="not_measured"))
    return g


def _accept(g: ChunkGeometry, line: np.ndarray, status: str, source: str, depths: np.ndarray,
            m_per_sample: float) -> ChunkGeometry:
    g.status, g.source, g.line = status, source, line
    if status in SOUNDER_BACKED and np.isfinite(depths).any():
        g.altitude_m = float(np.nanmedian(depths))                   # the steady sounder, as before
    else:
        g.altitude_m = float(np.median(line)) * m_per_sample          # the image × the recording's scale
    g.metrics["accepted_altitude_samples"] = round(float(np.median(line)), 1)
    g.steps.append(_step("resolve", f"{status.replace('_', ' ')} ({source}): {_MEANING[status]}", 0.0,
                         status_=status, source=source, altitude_m=round(g.altitude_m, 2)))
    return g


# ------------------------------------------------------------------------------------- recording
def check_recording(chunks: list[tuple[str, int, np.ndarray, np.ndarray]], m_per_sample: Optional[float],
                    freq_hz: Optional[int] = None) -> dict:
    """``chunks``: ``(channel, index, native sonogram, per-ping sounder depth m)``. ``m_per_sample``:
    the first scale estimate (``humminbird.range_scale``). Runs the per-chunk loop over every chunk,
    re-fits the scale from sounder-backed chunks, and repeats while the scale moves."""
    from src.cv_pipeline.humminbird import physics_spacing
    phys = physics_spacing(freq_hz or 455000)
    scale = m_per_sample or phys
    iters, log = [], []
    t0 = time.perf_counter()
    for it in range(MAX_ITERS):
        res: dict[tuple[str, int], ChunkGeometry] = {}
        order = sorted(chunks, key=lambda c: (c[1], c[0]))
        for ch, i, son, dep in order:                                  # first pass: sensors only
            res[(ch, i)] = check_chunk(ch, i, son, dep, scale)
        for ch, i, son, dep in order:                                  # second pass: borrow the paired channel
            if res[(ch, i)].status == "not_measured":
                other = next((g for (c2, i2), g in res.items() if i2 == i and c2 != ch and g.trusted), None)
                if other is not None:
                    res[(ch, i)] = check_chunk(ch, i, son, dep, scale, other=other)
        new, rel = _fit_scale(res, chunks)
        counts = {s: sum(g.status == s for g in res.values()) for s in STATUSES}
        moved = None if new is None else abs(new - scale) / scale
        iters.append({"iteration": it + 1, "m_per_sample_in": round(scale, 5),
                      "m_per_sample_out": None if new is None else round(new, 5),
                      "moved": None if moved is None else round(moved, 4), "counts": counts})
        log.append(_step("estimate_scale", f"iteration {it + 1}: scale {scale * 100:.3f} → "
                         + (f"{new * 100:.3f} cm/sample from {counts['agree'] + counts['corrected'] + counts['recovered']} "
                            f"sounder-backed chunks ({moved:.1%} change)" if new else "not re-fit (too few sounder-backed chunks)")
                         + (" → re-check every chunk with the new scale" if moved is not None and moved > SCALE_CONVERGED
                            and it + 1 < MAX_ITERS else " → converged"),
                         0.0, **iters[-1]).to_dict())
        if new is None or moved <= SCALE_CONVERGED:
            break
        scale = new
    final = new or scale
    rel_unc = rel if rel is not None else 0.25
    if freq_hz in (None, 455000):                                      # the physics formula's setting
        rel_unc = max(rel_unc, abs(final - phys) / final)
    return {"chunks": res, "iterations": iters, "log": log,
            "scale": {"m_per_sample": final, "rel_unc": round(max(rel_unc, 0.05), 3),
                      "physics_m_per_sample": round(phys, 5), "physics_rel_gap": round(abs(final - phys) / final, 3),
                      "source": "re-fit by the sensor cross-check from sounder-backed chunks"},
            "consistency": _port_starboard(res), "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}


def _fit_scale(res: dict, chunks) -> tuple[Optional[float], Optional[float]]:
    """Metres per sample = median over sounder-backed chunks of median(depth / accepted line)."""
    by = {(c, i): dep for c, i, _, dep in chunks}
    ratios = []
    for k, g in res.items():
        if g.status in SOUNDER_BACKED and g.line is not None:
            d = np.asarray(by[k], float)
            ok = np.isfinite(d) & (d > 0) & (g.line > 3)
            if ok.sum() >= 10:
                ratios.append(float(np.median(d[ok] / g.line[ok])))
    if len(ratios) < 3:
        return None, None
    r = np.array(ratios)
    med = float(np.median(r))
    return med, float(1.4826 * np.median(np.abs(r - med)) / med)


def _port_starboard(res: dict) -> dict:
    """Independent check: port and starboard are different images of the same pings — over a flat
    bed their first returns coincide. Median |Δ| / altitude over chunks where both were accepted."""
    diffs = []
    for (ch, i), g in res.items():
        if ch != "port" or not g.trusted:
            continue
        o = res.get(("starboard", i))
        if o is None or not o.trusted or o.n_pings != g.n_pings:
            continue
        diffs.append(float(np.median(np.abs(g.line - o.line)) / max(float(np.median(g.line)), 1.0)))
    return {"pairs": len(diffs), "median_rel_diff": round(float(np.median(diffs)), 4) if diffs else None}


def summary(result: dict) -> dict:
    gs = list(result["chunks"].values())
    counts = {s: sum(g.status == s for g in gs) for s in STATUSES}
    return {"chunks": len(gs), "counts": counts, "trusted": sum(g.trusted for g in gs),
            "iterations": len(result["iterations"]), "scale": result["scale"],
            "consistency": result["consistency"], "elapsed_ms": result["elapsed_ms"]}
