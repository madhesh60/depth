"""
stitch.py — chunk-boundary stitching: one physical object split across two adjacent sonar chunks.

This **replaces** the old ``match_other_pass`` "cross-pass corroboration", which was physically
wrong: it matched detections in the *same recording and channel* within 40 chunks at the same image
position — but adjacent chunks image **different seabed**, so those "matches" were coincidences, not
a second pass (review §2, §3.5). It also added +0.10 to the ranking score. Both are gone.

What *is* valid for PINGMapper sonograms: a waterfall is cut into consecutive chunks along-track, so
an object lying on the cut appears at the trailing edge of chunk *k* and the leading edge of chunk
*k+1*, at the **same range**. Stitching merges those two sightings into one hazard so the route and
counts don't double-count it (failure scenario 7). It never changes a verdict or a score.

Along-track axis: with the nadir at the top/bottom edge, x is along-track (ping order) and y is range;
with the nadir at the left/right edge it is the other way round. Unknown orientation ⇒ no stitching.

A true *second pass* (a different survey line over the same spot) can only be recognised with real
GPS — that is ``geo.cluster_hazards`` (geo-clustering within the position error radius).
"""
from __future__ import annotations

from typing import Optional

from .geo import parse_ping, parse_recording, parse_side
from .types import AgentStep, Candidate, FrameResult

EDGE_FRAC = 0.04          # a box "touches" a chunk edge if within this fraction of the frame size
MIN_EDGE_PX = 6
RANGE_TOL_FRAC = 0.5      # range centres must agree within this × the larger box's range extent
MIN_RANGE_TOL_PX = 10


def _chunk_key(frame_id: str) -> Optional[tuple[str, str, int]]:
    rec, side, ping = parse_recording(frame_id), parse_side(frame_id), parse_ping(frame_id)
    if rec is None or side is None or ping is None:
        return None
    return rec, side, ping


def _axes(nadir: str) -> Optional[tuple[int, int]]:
    """(along-track axis index into bbox, range axis index) — 0 = x, 1 = y."""
    if nadir in ("top", "bottom"):
        return 0, 1
    if nadir in ("left", "right"):
        return 1, 0
    return None


def _touch(c: Candidate, fr: FrameResult, along: int) -> tuple[bool, bool]:
    """Does the box touch the (leading, trailing) edge of its chunk along-track?"""
    size = fr.width if along == 0 else fr.height
    m = max(MIN_EDGE_PX, int(EDGE_FRAC * size))
    lo, hi = c.bbox[along], c.bbox[along + 2]
    return lo <= m, hi >= size - m


def stitch_boundaries(frames: list[FrameResult]) -> list[tuple[Candidate, Candidate, AgentStep]]:
    """Link candidates split across adjacent chunks. Mutates candidates (``continues_in`` on the
    earlier chunk's sighting, ``continuation_of`` on the later one) and appends a
    ``stitch_boundary`` trace step to both. Returns ``[(earlier, later, step)]`` links; the caller
    decides which sighting represents the hazard (see ``pipeline.run_survey``)."""
    by_key: dict[tuple[str, str, int], FrameResult] = {}
    for fr in frames:
        k = _chunk_key(fr.frame_id)
        if k is not None:
            by_key[k] = fr

    links: list[tuple[Candidate, Candidate, AgentStep]] = []
    for (rec, side, ping), fa in sorted(by_key.items(), key=lambda kv: kv[0]):
        fb = by_key.get((rec, side, ping + 1))
        if fb is None or fa.nadir != fb.nadir:
            continue
        ax = _axes(fa.nadir)
        if ax is None:
            continue
        along, rng = ax
        used_b: set[int] = set()
        for a in fa.candidates:
            if a.continues_in or not _touch(a, fa, along)[1]:           # must touch trailing edge
                continue
            a_rc = 0.5 * (a.bbox[rng] + a.bbox[rng + 2])
            a_ext = a.bbox[rng + 2] - a.bbox[rng]
            best: Optional[tuple[float, int]] = None
            for j, b in enumerate(fb.candidates):
                if j in used_b or b.cls_name != a.cls_name or not _touch(b, fb, along)[0]:
                    continue
                b_rc = 0.5 * (b.bbox[rng] + b.bbox[rng + 2])
                tol = max(MIN_RANGE_TOL_PX, RANGE_TOL_FRAC * max(a_ext, b.bbox[rng + 2] - b.bbox[rng]))
                d = abs(a_rc - b_rc)
                if d <= tol and (best is None or d < best[0]):
                    best = (d, j)
            if best is None:
                continue
            b = fb.candidates[best[1]]
            used_b.add(best[1])
            a.continues_in, b.continuation_of = fb.frame_id, fa.frame_id
            why = (f"same {a.cls_name} continues across the chunk boundary "
                   f"({fa.frame_id} -> {fb.frame_id}, range offset {best[0]:.0f}px) -> counted once")
            step = AgentStep(tool="stitch_boundary", rationale=why, latency_ms=0.0,
                             detail={"primary": fa.frame_id, "continuation": fb.frame_id,
                                     "range_offset_px": round(best[0], 1)})
            a.trace.append(step)
            b.trace.append(step)
            links.append((a, b, step))
    return links
