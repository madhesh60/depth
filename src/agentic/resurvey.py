"""
resurvey.py — the **Act** stage closes the loop physically (review A-1 + X-3).

1. :func:`merge_repeat_sightings` — the same object seen on another pass (or another line) is ONE
   hazard. Two detections of the same class from DIFFERENT frames merge when their geotag error
   circles substantially overlap (distance ≤ max(3 m, ½·√(e₁² + e₂²))) — tight on purpose: crab
   pots are laid in strings 10–30 m apart and must not be fused. The strongest sighting represents
   the object; ``sightings`` counts the passes that saw it. Never changes a tier or a score.

2. :func:`plan_resurvey` — for what is still uncertain, plan the **second look** that can settle it.
   An object that stands up off the seabed casts its acoustic shadow AWAY from the sonar. Seen again
   from the **opposite side**, a real 3-D object's shadow must flip direction; speckle and clutter do
   not do that consistently. So for each REVIEW target the agent plans a straight line on the other
   side of it with the target at **mid-swath** (best resolution), groups targets whose lines align
   into one pass, and ranks passes by **uncertainty resolved per metre of boat travel**
   (Σ p(1−p) over the targets, p = calibrated P(pot) — zero for near-certain cards). With a boat-time
   budget it keeps the densest passes that fit (greedy knapsack) — the boat-side twin of the
   analyst's budget mode. Every target carries a falsifiable prediction: the compass bearing its
   shadow must point on the new pass.

Plans need GPS (a heading and a position per frame). With the synthetic demo track every line is
marked synthetic; nothing is ever executed or dispatched — the plan is for a human to approve.
"""
from __future__ import annotations

import math
from typing import Optional

from .geo import PingFix, parse_side, _EARTH_R
from .types import TrackedObject, Verdict

SWATH_M = 640 * 0.05          # across-track range of one channel at the demo scale (m)
LEAD_M = 15.0                 # run-in / run-out so the target is imaged on a straight, steady line
SPEED_KN = 3.0                # typical side-scan survey speed
TURN_S = 90.0                 # time to turn onto each line
MAX_LINE_M = 300.0


# ---- local flat-earth frame (metres east / north of a reference point) --------------------------
def _enu(lat, lon, lat0, lon0):
    return (math.radians(lon - lon0) * _EARTH_R * math.cos(math.radians(lat0)),
            math.radians(lat - lat0) * _EARTH_R)


def _latlon(x, y, lat0, lon0):
    return (lat0 + math.degrees(y / _EARTH_R),
            lon0 + math.degrees(x / (_EARTH_R * math.cos(math.radians(lat0)))))


def _unit(bearing_deg):
    b = math.radians(bearing_deg)
    return math.sin(b), math.cos(b)             # (east, north)


def _bearing360(b):
    return round(b % 360.0, 1)


# ================================================================ 1. repeat sightings
_RANK = {Verdict.CONFIRMED: 2, Verdict.REVIEW: 1}


def merge_repeat_sightings(tracked: list[TrackedObject], min_gate_m: float = 3.0) -> tuple[list[TrackedObject], int]:
    geo = [t for t in tracked if t.lat is not None and t.lon is not None and t.geo_error_m]
    parent = {t.oid: t.oid for t in tracked}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # frames already in each cluster: union-find is transitive, so without this two distinct
    # detections in ONE frame could fuse through a third sighting from another pass
    frames = {t.oid: {t.frame_id, *t.also_in} for t in tracked}
    pairs = sorted(((_dist(a, b), a, b) for i, a in enumerate(geo) for b in geo[i + 1:]
                    if a.frame_id != b.frame_id and a.cls_name == b.cls_name), key=lambda x: x[0])
    merges = 0
    for d, a, b in pairs:                                     # closest pairs first
        ra, rb = find(a.oid), find(b.oid)
        if ra == rb or frames[ra] & frames[rb]:
            continue
        if d <= max(min_gate_m, 0.5 * math.hypot(a.geo_error_m, b.geo_error_m)):
            parent[rb] = ra
            frames[ra] |= frames[rb]
            merges += 1
    groups: dict[str, list[TrackedObject]] = {}
    for t in tracked:
        groups.setdefault(find(t.oid), []).append(t)
    out = []
    for members in groups.values():
        keep = max(members, key=lambda t: (_RANK.get(t.verdict, 0), t.p_pot or 0.0, t.evidence_score, t.conf))
        for m in members:
            if m is not keep:
                keep.also_in.append(m.frame_id)
                keep.sightings += m.sightings
        out.append(keep)
    order = {t.oid: i for i, t in enumerate(tracked)}
    return sorted(out, key=lambda t: order[t.oid]), merges


def _dist(a: TrackedObject, b: TrackedObject) -> float:
    from .geo import haversine_m
    return haversine_m((a.lat, a.lon), (b.lat, b.lon))


# ================================================================ 2. re-survey planner
def _target_line(t: TrackedObject, fix: PingFix, side: str, mid_m: float, lat0, lon0):
    """Where the second pass must run so ``t`` is on the OTHER side at mid-swath (same heading)."""
    h = fix.heading_deg
    # seen to starboard (bearing h+90 from the boat) → new boat on the target's starboard-ward... i.e.
    # the new boat sits on the far side so the target is now on its PORT side, and vice versa.
    away = h + 90.0 if side == "starboard" else h - 90.0          # original boat → target direction
    ux, uy = _unit(away)
    tx, ty = _enu(t.lat, t.lon, lat0, lon0)
    cx, cy = tx + ux * mid_m, ty + uy * mid_m                        # new boat position abeam of the target
    new_side = "port" if side == "starboard" else "starboard"
    # on the new pass the sonar looks back towards the old boat; the shadow falls beyond the target,
    # i.e. in the direction new-boat → target = away + 180
    old_shadow, new_shadow = _bearing360(away), _bearing360(away + 180.0)
    return (cx, cy), h, new_side, old_shadow, new_shadow


def plan_resurvey(tracked: list[TrackedObject], track: dict[str, PingFix], boat_minutes: Optional[float] = None,
                  swath_m: float = SWATH_M, speed_kn: float = SPEED_KN) -> dict:
    """Second-look passes for the REVIEW targets, ranked by uncertainty resolved per metre."""
    targets = []
    for t in tracked:
        fix = track.get(t.frame_id) if track else None
        side = parse_side(t.frame_id)
        if t.verdict is not Verdict.REVIEW or t.lat is None or fix is None or side is None:
            continue
        p = t.p_pot if t.p_pot is not None else 0.5
        targets.append((t, fix, side, p * (1 - p)))
    if not targets:
        return {"lines": [], "note": "no REVIEW targets with GPS + known side - nothing to re-survey"}
    lat0, lon0 = targets[0][0].lat, targets[0][0].lon
    mid = swath_m / 2.0
    info = []
    for t, fix, side, w in targets:
        c, h, new_side, old_sh, new_sh = _target_line(t, fix, side, mid, lat0, lon0)
        info.append({"t": t, "c": c, "h": h, "side": new_side, "w": w, "old_shadow": old_sh, "new_shadow": new_sh})

    # greedy grouping: seed = highest-VoI unassigned target; add targets whose required line is the same
    # straight line (heading within 10°, cross-track offset within ¼ swath, along-track within the cap)
    lines, used = [], set()
    for seed in sorted(info, key=lambda d: -d["w"]):
        if id(seed) in used:
            continue
        ux, uy = _unit(seed["h"])
        members = []
        for d in info:
            if id(d) in used or abs(((d["h"] - seed["h"] + 180) % 360) - 180) > 10 or d["side"] != seed["side"]:
                continue
            dx, dy = d["c"][0] - seed["c"][0], d["c"][1] - seed["c"][1]
            along, cross = dx * ux + dy * uy, -dx * uy + dy * ux
            if abs(cross) <= swath_m / 4 and abs(along) <= MAX_LINE_M / 2:
                members.append((along, d))
        for _, d in members:
            used.add(id(d))
        a_min = min(a for a, _ in members) - LEAD_M
        a_max = max(a for a, _ in members) + LEAD_M
        sx, sy = seed["c"][0] + ux * a_min, seed["c"][1] + uy * a_min
        ex, ey = seed["c"][0] + ux * a_max, seed["c"][1] + uy * a_max
        length = a_max - a_min
        voi = sum(d["w"] for _, d in members)
        secs = length / (speed_kn * 0.514444) + TURN_S
        lines.append({
            "start": [round(v, 6) for v in _latlon(sx, sy, lat0, lon0)],
            "end": [round(v, 6) for v in _latlon(ex, ey, lat0, lon0)],
            "heading_deg": _bearing360(seed["h"]), "length_m": round(length, 1),
            "targets": [d["t"].oid for _, d in sorted(members, key=lambda m: m[0])],
            "targets_on": seed["side"], "voi": round(voi, 3), "voi_per_100m": round(100 * voi / max(length, 1), 3),
            "boat_min": round(secs / 60, 1),
            "predictions": [{"id": d["t"].oid, "p_pot": d["t"].p_pot, "shadow_was": d["old_shadow"],
                             "shadow_must_point": d["new_shadow"]} for _, d in members],
            "synthetic": any(d["t"].frame_id and track[d["t"].frame_id].synthetic for _, d in members),
        })

    lines.sort(key=lambda L: -L["voi_per_100m"])
    chosen, spent = [], 0.0
    for L in lines:
        if boat_minutes is not None and spent + L["boat_min"] > boat_minutes:
            continue
        chosen.append(L); spent += L["boat_min"]
    for i, L in enumerate(chosen, 1):
        L["id"] = f"RS{i}"
        L["status"] = "PLANNED - not executed; needs human approval"
        L["why"] = (f"{len(L['targets'])} uncertain target(s) (Σp(1-p) = {L['voi']}) re-imaged from the other side "
                    f"at mid-swath (~{mid:.0f} m); a real object's shadow must flip "
                    f"(e.g. {L['predictions'][0]['id']}: {L['predictions'][0]['shadow_was']:.0f}° → "
                    f"{L['predictions'][0]['shadow_must_point']:.0f}°)")
    total_voi = sum(d["w"] for d in info)
    return {"lines": chosen, "boat_minutes_budget": boat_minutes, "boat_minutes_planned": round(spent, 1),
            "voi_covered": round(sum(L["voi"] for L in chosen), 3), "voi_total": round(total_voi, 3),
            "targets_covered": sum(len(L["targets"]) for L in chosen), "targets_total": len(info),
            "assumptions": {"swath_m": swath_m, "mid_range_m": mid, "speed_kn": speed_kn, "turn_s": TURN_S}}
