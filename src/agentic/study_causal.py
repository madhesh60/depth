"""
study_causal.py — STUDY-12: does OpenCV 5 output change what the agent *does*? (counterfactual)

The Agentic-Vision rules ask for "a trace showing OpenCV 5 output changing a later decision/action".
A single trace shows correlation; a **counterfactual** shows cause. The same survey (same frames, same
synthetic track, same model and calibration) is run three times:

* ``full``        — the product as shipped;
* ``no_stage1``   — Stage-1 geometry (bottom tracking → sonar altitude, slant→ground range, the
  object's own ping, water-column mask; all OpenCV) is withheld from Prove/Act. The detector input is
  unchanged, so every detection and tier is identical — only what Stage 1 *measured* is removed;
* ``no_shadow``   — the thin-line acoustic-shadow measurement is withheld (reported as "none").

and every downstream decision is diffed: tiers, the REVIEW queue, the analyst-budget picks, the
inspection route, repeat-sighting merges, the opposite-side re-survey passes and the geotags.

Expected, and reported either way: ``no_shadow`` changes **no decision** (the shadow is evidence for the
human card, never a gate — STUDY-03/04); EXP-001 spends no re-looks (``relook_mode: null``,
STUDY-07), so a re-look arm would be identical by construction and is not run.

Also prints the **causal chain of one hazard** (cv2.dnn confidence → tier → P(pot) → queue rank →
budget → inspection stop → re-survey pass + predicted shadow flip) and how Stage 1 moved it.

    python -m src.agentic.study_causal [--frames DIR --pattern GLOB --limit N] [--budget 2]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Optional

import cv2

from . import agent as agent_mod
from .geo import haversine_m, offset_latlon, synthetic_track
from .pipeline import AgenticPipeline
from .types import ShadowProof, ShadowQuality, SurveyResult

REPO = Path(__file__).resolve().parents[2]
SAMPLES = REPO / "webui" / "samples"


# ---- the arms ----------------------------------------------------------------------------------
class _Arm:
    """Context manager that withholds one OpenCV output from the agent (restored on exit)."""

    def __init__(self, pipe: AgenticPipeline, name: str):
        self.pipe, self.name, self._undo = pipe, name, []

    def __enter__(self):
        if self.name == "no_stage1":
            a, g = agent_mod._attach_geometry, agent_mod._altitude_for
            agent_mod._attach_geometry = lambda c, cf: None           # no ping / ground range / water column
            agent_mod._altitude_for = lambda cf, bbox: None           # no tracked altitude for heights
            self._undo.append(lambda: (setattr(agent_mod, "_attach_geometry", a), setattr(agent_mod, "_altitude_for", g)))
        elif self.name == "no_shadow":
            tools = self.pipe.agent.tools
            real = tools.shadow_check

            def none_shadow(gray, bbox, nadir, altitude_px=None):
                proof, step = real(gray, bbox, nadir, altitude_px)
                blank = ShadowProof(quality=ShadowQuality.NONE, contrast=0.0, run_px=0, strength=0.0, height_m=None,
                                    height_rel=0.0, echo_ratio=proof.echo_ratio, echo_xy=proof.echo_xy,
                                    strip=proof.strip, orientation_known=proof.orientation_known)
                step.rationale, step.detail = "shadow withheld (counterfactual arm)", {"quality": "none"}
                return blank, step
            tools.shadow_check = none_shadow
            self._undo.append(lambda: setattr(tools, "shadow_check", real))
        return self

    def __exit__(self, *exc):
        for u in reversed(self._undo):
            u()


def run_arm(pipe: AgenticPipeline, frames, track, name: str, budget: float) -> SurveyResult:
    with _Arm(pipe, name):
        return pipe.run_survey(frames, track=track, survey_id=f"causal-{name}", budget_minutes=budget)


# ---- diffing two surveys -----------------------------------------------------------------------
def _key(t) -> tuple:
    return (t.frame_id, tuple(t.bbox))


def _kendall(a: list, b: list) -> Optional[float]:
    common = [x for x in a if x in set(b)]
    if len(common) < 2:
        return None
    pos = {x: i for i, x in enumerate(b)}
    conc = disc = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            if pos[common[i]] < pos[common[j]]:
                conc += 1
            else:
                disc += 1
    return round((conc - disc) / max(1, conc + disc), 3)


def diff(A: SurveyResult, B: SurveyResult) -> dict:
    ka = {_key(t): t for t in A.tracked}
    kb = {_key(t): t for t in B.tracked}
    common = [k for k in ka if k in kb]
    va = {(fr.frame_id, tuple(c.bbox)): c.verdict for fr in A.frames for c in fr.candidates}
    vb = {(fr.frame_id, tuple(c.bbox)): c.verdict for fr in B.frames for c in fr.candidates}
    flips = sum(1 for k in va if k in vb and va[k] is not vb[k])
    ida = {t.oid: _key(t) for t in A.tracked}
    idb = {t.oid: _key(t) for t in B.tracked}
    qa = [ida[i] for i in A.mission.review_queue]
    qb = [idb[i] for i in B.mission.review_queue]
    ba = {ida[i] for i in (A.mission.budget or {}).get("review_ids", [])}
    bb = {idb[i] for i in (B.mission.budget or {}).get("review_ids", [])}
    ia = [ida[i] for i in A.mission.inspection_route]
    ib = [idb[i] for i in B.mission.inspection_route]
    pairs = [(k, haversine_m((ka[k].lat, ka[k].lon), (kb[k].lat, kb[k].lon))) for k in common
             if ka[k].lat is not None and kb[k].lat is not None]
    disp = [d for _, d in pairs]
    outside = sum(1 for k, d in pairs if ka[k].geo_error_m is not None and d > ka[k].geo_error_m)
    pa = [frozenset(ida[x] for x in L["targets"]) for L in (A.mission.resurvey_plan or {}).get("lines", [])]
    pb = [frozenset(idb[x] for x in L["targets"]) for L in (B.mission.resurvey_plan or {}).get("lines", [])]
    ra, rb = A.mission.resurvey_plan or {}, B.mission.resurvey_plan or {}
    wc = lambda S: sum(1 for fr in S.frames for c in fr.candidates if c.in_water_column)
    hr = lambda S: sum(1 for t in S.tracked if t.height_rel and t.shadow_quality != "none")
    q = lambda xs, p: round(sorted(xs)[min(len(xs) - 1, int(p * len(xs)))], 1) if xs else None
    return {
        "hazards": [len(A.tracked), len(B.tracked)],
        "verdict_flips": flips,
        "queue_kendall_tau": _kendall(qa, qb),
        "queue_positions_changed": sum(1 for i, k in enumerate(qa) if i >= len(qb) or qb[i] != k),
        "budget_picks_changed": len(ba ^ bb) // 2 if len(ba) == len(bb) else len(ba ^ bb),
        "inspection_same_order": ia == ib,
        "inspection_stops_changed": sum(1 for x, y in zip(ia, ib) if x != y) + abs(len(ia) - len(ib)),
        "inspection_length_m": [A.mission.inspection_length_m, B.mission.inspection_length_m],
        "repeat_merges": [A.mission.repeat_merges, B.mission.repeat_merges],
        "resurvey_passes": [len(pa), len(pb)],
        "resurvey_passes_identical": sum(1 for s in pa if s in pb),
        "resurvey_boat_min": [ra.get("boat_minutes_planned"), rb.get("boat_minutes_planned")],
        "resurvey_targets": [ra.get("targets_covered"), rb.get("targets_covered")],
        "geotag_shift_m": {"n": len(disp), "median": q(disp, 0.5), "p90": q(disp, 0.9),
                           "max": round(max(disp), 1) if disp else None,
                           "outside_stated_error": outside},
        "geo_error_m_median": [statistics.median([t.geo_error_m for t in S.tracked if t.geo_error_m is not None] or [0])
                               for S in (A, B)],
        "water_column_flags": [wc(A), wc(B)],
        "relative_heights": [hr(A), hr(B)],
    }


def chain(A: SurveyResult, B: Optional[SurveyResult] = None) -> Optional[dict]:
    """The causal chain of one hazard: the budgeted REVIEW card that heads a re-survey pass."""
    m = A.mission
    lines = (m.resurvey_plan or {}).get("lines", [])
    picks = (m.budget or {}).get("review_ids") or m.review_queue
    idx = {t.oid: t for t in A.tracked}
    measured = {f.frame_id for f in A.frames if (f.stage1 or {}).get("measured")}
    in_pass = lambda i: any(i in L["targets"] for L in lines)
    pref = lambda i: (in_pass(i), idx[i].frame_id in measured, idx[i].shadow_quality != "none")
    oid = max(picks, key=lambda i: (pref(i), -picks.index(i))) if picks else None   # best-evidenced budget pick
    if oid is None:
        return None
    t = idx[oid]
    fr = next(f for f in A.frames if f.frame_id == t.frame_id)
    c = next(c for c in fr.candidates if tuple(c.bbox) == tuple(t.bbox))
    L = next((L for L in lines if oid in L["targets"]), None)
    pred = next((p for p in (L or {}).get("predictions", []) if p["id"] == oid), None)
    steps = [(s.tool, s.rationale) for s in c.trace]
    out = {"id": oid, "frame": t.frame_id, "bbox": list(t.bbox), "cls": t.cls_name, "conf": round(c.conf, 3),
           "verdict": t.verdict.value, "p_pot": t.p_pot, "queue_rank": m.review_queue.index(oid) + 1,
           "queue_len": len(m.review_queue), "in_budget": oid in (m.budget or {}).get("review_ids", []),
           "inspection_stop": (m.inspection_route.index(oid) + 1) if oid in m.inspection_route else None,
           "resurvey_pass": L["id"] if L else None, "pass_heading": L["heading_deg"] if L else None,
           "shadow_was": pred["shadow_was"] if pred else None, "shadow_must_point": pred["shadow_must_point"] if pred else None,
           "ground_range_px": None if c.ground_range_px is None else round(c.ground_range_px, 1),
           "ping_px": None if c.ping_px is None else round(c.ping_px, 1), "altitude_px": (fr.stage1 or {}).get("altitude_px"),
           "altitude_measured": bool((fr.stage1 or {}).get("measured")), "shadow": t.shadow_quality,
           "height_rel": t.height_rel,
           "latlon": [t.lat, t.lon], "geo_error_m": t.geo_error_m, "trace": steps}
    if B is not None:
        kb = {_key(x): x for x in B.tracked}
        tb = kb.get(_key(t))
        if tb is not None:
            out["no_stage1"] = {"latlon": [tb.lat, tb.lon], "shift_m": round(haversine_m((t.lat, t.lon), (tb.lat, tb.lon)), 1)
                                if t.lat is not None and tb.lat is not None else None,
                                "geo_error_m": tb.geo_error_m,
                                "inspection_stop": (B.mission.inspection_route.index(tb.oid) + 1) if tb.oid in B.mission.inspection_route else None,
                                "resurvey_pass": next((L2["id"] for L2 in (B.mission.resurvey_plan or {}).get("lines", [])
                                                       if tb.oid in L2["targets"]), None)}
    return out


# ---- frames + track ----------------------------------------------------------------------------
_REC = re.compile(r"(Rec\d+)", re.I)


def survey_track(frame_ids: list[str], spacing_m: float = 500.0, heading_deg: float = 20.0) -> dict:
    """Synthetic demo track. One recording = one survey line (``synthetic_track``: chunk k at k frame
    lengths, port/starboard of a chunk share its fix); several recordings are laid on parallel lines
    ``spacing_m`` apart, so objects of different recordings never share a fix by accident."""
    groups: dict[str, list[str]] = {}
    for f in frame_ids:
        m = _REC.search(f)
        groups.setdefault(m.group(1).lower() if m else "", []).append(f)
    track = {}
    for i, (_, ids) in enumerate(sorted(groups.items())):
        start = offset_latlon(37.8000, -76.1500, i * spacing_m, heading_deg + 90.0)
        track.update(synthetic_track(ids, start=start, heading_deg=heading_deg))
    return track



def load_frames(frames_dir: Optional[Path], pattern: str, limit: int) -> list[tuple[str, "cv2.Mat"]]:
    if frames_dir is None:
        from src.dashboard import samples as samples_mod
        paths = [samples_mod.sample_path(s["id"]) for s in samples_mod.list_samples()]
    else:
        from src.detection.frames import unique_frames
        paths = unique_frames(sorted(frames_dir.glob(pattern)))[:limit]
    out = []
    for p in paths:
        img = cv2.imread(str(p)) if p and p.exists() else None
        if img is not None:
            out.append((p.stem, img))
    return out


def report(res: dict) -> str:
    L = ["# STUDY-12 — Does OpenCV 5 output change what the agent does? (counterfactual)", "",
         f"_`{res.get('cmd', 'python -m src.agentic.study_causal')}` · EXP-001 · same frames, synthetic track, model and calibration in "
         "every arm · analyst budget "
         f"{res['budget_min']} min · regenerated by the script — do not edit by hand_", "",
         "One survey, three runs. **full** = the product; **no_stage1** = Stage-1 geometry (bottom track → altitude, "
         "slant→ground range, the object's own ping, water-column mask) withheld from Prove/Act — detections and "
         "tiers are identical, only what Stage 1 *measured* is removed; **no_shadow** = the thin-line shadow "
         "measurement withheld. EXP-001's calibrated policy spends no re-looks (STUDY-07), so there is no re-look arm.", "",
         "Caveats: the track is a synthetic demo (one recording = one line; recordings on parallel lines 500 m apart) and "
         "metres use the demo range scale (0.05 m/px) — shifts scale with it, the error circles (≥ 3 m, 25% of ground "
         "range) mostly do too. Frame sets are small; this measures *whether and where* OpenCV output drives the plan, "
         "not accuracy.", ""]
    for name, r in res["sets"].items():
        L += [f"## {name} — {r['frames']} frames, {r['full_hazards']} hazards", "",
              "| downstream decision | full vs **no_stage1** | full vs **no_shadow** |", "|---|---|---|"]
        s, h = r["no_stage1"], r["no_shadow"]
        row = lambda lab, f: L.append(f"| {lab} | {f(s)} | {f(h)} |")
        row("tier flips (any candidate)", lambda d: d["verdict_flips"])
        row("REVIEW queue order (Kendall τ; 1 = same)", lambda d: d["queue_kendall_tau"])
        row("analyst-budget picks changed", lambda d: d["budget_picks_changed"])
        row("inspection route: stops in a different position", lambda d: f"{d['inspection_stops_changed']} "
            f"({d['inspection_length_m'][0]} → {d['inspection_length_m'][1]} m)")
        row("repeat-sighting merges", lambda d: f"{d['repeat_merges'][0]} → {d['repeat_merges'][1]}")
        row("re-survey passes (identical target sets)", lambda d: f"{d['resurvey_passes'][0]} → {d['resurvey_passes'][1]} "
            f"({d['resurvey_passes_identical']} identical); boat {d['resurvey_boat_min'][0]} → {d['resurvey_boat_min'][1]} min")
        row("geotag shift (median / p90 / max, m)", lambda d: f"{d['geotag_shift_m']['median']} / {d['geotag_shift_m']['p90']} / "
            f"{d['geotag_shift_m']['max']}")
        row("hazards pushed outside their own stated error circle", lambda d: f"{d['geotag_shift_m']['outside_stated_error']} "
            f"of {d['geotag_shift_m']['n']}")
        row("water-column flags", lambda d: f"{d['water_column_flags'][0]} → {d['water_column_flags'][1]}")
        row("hazards with a relative height", lambda d: f"{d['relative_heights'][0]} → {d['relative_heights'][1]}")
        L.append("")
    c = res.get("chain")
    if c:
        L += ["## One hazard, end to end — the trace", "",
              f"**{c['id']}** ({c['cls']}) in `{c['frame']}`, box {c['bbox']}:", "",
              "| step | OpenCV 5 output | → decision / action |", "|---|---|---|",
              (f"| Stage 1 | bottom track: sonar altitude **{c['altitude_px']} px**; object on ping {c['ping_px']}, "
               f"slant → ground range **{c['ground_range_px']} px** | geotag {c['latlon']} ± {c['geo_error_m']} m |"
               if c["altitude_measured"] else
               f"| Stage 1 | bottom track not confident on this frame (altitude not measured → ground = slant); object on "
               f"ping {c['ping_px']} | geotag {c['latlon']} ± {c['geo_error_m']} m |"),
              f"| See | `cv2.dnn` YOLO11 confidence {c['conf']} | tier **{c['verdict'].upper()}**, calibrated P(pot) {c['p_pot']} |",
              f"| Prove | thin-line shadow **{c['shadow']}**" + (f", relative height h/H {c['height_rel']}" if c['shadow'] != 'none' else "")
              + " | evidence on the card (never a gate) |",
              f"| Decide | P(pot) ordering | queue rank {c['queue_rank']} of {c['queue_len']}; in the analyst budget: "
              f"{'yes' if c['in_budget'] else 'no'} |",
              f"| Act | its position + P(pot) | inspection stop {c['inspection_stop']}; re-survey pass **{c['resurvey_pass']}** "
              f"(heading {c['pass_heading']}°) — its shadow must flip {c['shadow_was']}° → {c['shadow_must_point']}° |"]
        if c.get("no_stage1"):
            n = c["no_stage1"]
            L += [f"| **without Stage 1** | slant range, frame-centre ping | geotag moves **{n['shift_m']} m** "
                  f"(± {n['geo_error_m']} m); inspection stop {n['inspection_stop']}; re-survey pass {n['resurvey_pass']} |"]
        L += ["", "Its logged tool calls:", ""] + [f"- `{t}` — {why}" for t, why in c["trace"]] + [""]
    L += ["## Reading", "", res["reading"], "", "---", "_Regenerated by the script; do not edit by hand._", ""]
    return "\n".join(L)


def _reading(res: dict) -> str:
    out = []
    for name, r in res["sets"].items():
        s, h = r["no_stage1"], r["no_shadow"]
        g = s["geotag_shift_m"]
        out.append(f"**{name}:** withholding Stage 1 flips {s['verdict_flips']} tiers (it never gates) but moves the "
                   f"median hazard by {g['median']} m (max {g['max']} m), reorders "
                   f"{s['inspection_stops_changed']} inspection stops, and leaves "
                   f"{s['resurvey_passes_identical']} of {s['resurvey_passes'][0]} re-survey passes unchanged — "
                   f"{g['outside_stated_error']} of {g['n']} pins would land outside their own stated error circle; "
                   f"withholding the shadow changes {h['verdict_flips']} tiers, "
                   f"{h['budget_picks_changed']} budget picks and {h['inspection_stops_changed']} route stops — "
                   f"{'none, as designed (evidence for the human, never a gate)' if not (h['verdict_flips'] or h['budget_picks_changed'] or h['inspection_stops_changed']) else 'NOT zero — investigate'}.")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--frames", type=Path, default=None, help="extra frame directory (default: shipped samples only)")
    ap.add_argument("--pattern", default="crabpot_*wcp_ss_*.jpg")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--budget", type=float, default=2.0)
    a = ap.parse_args()
    pipe = AgenticPipeline()
    sets = {"shipped samples": load_frames(None, a.pattern, a.limit)}
    if a.frames:
        sets[f"{a.frames.parents[1].name} / {a.frames.parent.name} (unique crab-pot frames)"] = load_frames(a.frames, a.pattern, a.limit)
    import sys
    res = {"budget_min": a.budget, "sets": {},
           "cmd": " ".join(["python -m src.agentic.study_causal"] + [x.replace("\\", "/") for x in sys.argv[1:]])}
    for name, frames in sets.items():
        track = survey_track([f for f, _ in frames])
        A = run_arm(pipe, frames, track, "full", a.budget)
        B = run_arm(pipe, frames, track, "no_stage1", a.budget)
        C = run_arm(pipe, frames, track, "no_shadow", a.budget)
        res["sets"][name] = {"frames": len(frames), "full_hazards": len(A.tracked),
                             "no_stage1": diff(A, B), "no_shadow": diff(A, C)}
        if "chain" not in res:
            res["chain"] = chain(A, B)
        print(name, json.dumps(res["sets"][name], indent=1))
    res["reading"] = _reading(res)
    out = REPO / "runs" / "causal"; out.mkdir(parents=True, exist_ok=True)
    (out / "study.json").write_text(json.dumps(res, indent=1, default=str))
    (REPO / "docs" / "causal_trace.md").write_text(report(res), encoding="utf-8")
    print("wrote docs/causal_trace.md")


if __name__ == "__main__":
    main()
