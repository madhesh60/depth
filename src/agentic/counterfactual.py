"""
counterfactual.py — the same survey WITH vs WITHOUT the OpenCV evidence (live, per survey).

Nothing is re-inferred: the detector output is identical in both arms. The WITHOUT arm removes what
OpenCV measured after detection —

* the active-vision evidence (shadow / re-look / geometry → belief → action): every hazard keeps the
  action the detector alone gave it, and re-survey passes fall back to Σ p(1-p) of the detector score;
* the Stage-1 geometry (bottom track → ground range, the object's own ping): pins move to the
  slant-range / frame-centre placement (``geo.stage1_counterfactual``);

— and re-plans the mission with the same budgets. The table is what changed downstream: decisions,
queue priority, geolocation, the re-survey route and what a person is asked to approve.
"""
from __future__ import annotations

import copy
from typing import Optional


def _kendall(a: list[str], b: list[str]) -> Optional[float]:
    common = [x for x in a if x in set(b)]
    if len(common) < 2:
        return None
    pb = {x: i for i, x in enumerate(b)}
    n = conc = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            n += 1
            conc += pb[common[i]] < pb[common[j]]
    return round((2 * conc - n) / n, 3)


def without_opencv(survey, stage1_cf: Optional[dict] = None) -> dict:
    from .mission import plan_mission, DEFAULT_SEC_PER_CARD
    tracked = survey.tracked
    active_on = any(t.action for t in tracked)
    a = survey.plan_args or {}
    pins = (stage1_cf or {}).get("pins") or {}
    bare = []
    for t in tracked:
        b = copy.copy(t)
        b.human, b.human_priority, b.fallback = None, None, None
        b.action, b.p_evidence, b.info_bits, b.request_resurvey, b.conflict = t.action0, None, None, False, None
        b.height_rel, b.height_m, b.shadow_quality = 0.0, None, "none"
        if t.oid in pins:
            b.lat, b.lon = pins[t.oid]["lat"], pins[t.oid]["lon"]
        bare.append(b)
    full = []
    for t in tracked:
        f = copy.copy(t)
        f.human, f.human_priority, f.fallback = None, None, None
        full.append(f)
    args = (survey.mission.guarantees, a.get("budget_minutes"), a.get("sec_per_card", DEFAULT_SEC_PER_CARD),
            a.get("boat_minutes"), survey.mission.repeat_merges)
    m1 = plan_mission(full, survey.track, *args)
    m0 = plan_mission(bare, survey.track, *args)

    changed = [t for t in tracked if t.action != t.action0]
    acc = lambda ts, key: sum(getattr(t, key) == "accept" for t in ts)
    L1, L0 = (m1.resurvey_plan or {}).get("lines") or [], (m0.resurvey_plan or {}).get("lines") or []
    s1, s0 = {frozenset(L["targets"]) for L in L1}, {frozenset(L["targets"]) for L in L0}
    head1, head0 = m1.review_queue[:10], m0.review_queue[:10]
    rows = [
        {"what": "Decision (agent action)", "without": f"{acc(tracked, 'action0')} accept" if active_on else "tier from the detector",
         "with": f"{acc(tracked, 'action')} accept" if active_on else "same tier (active vision off)",
         "detail": (f"{len(changed)} of {len(tracked)} hazards change action" if active_on else
                    "with active vision off the evidence is shown to people but does not change the action"),
         "changed": bool(changed)},
        {"what": "Priority (top 10 of the queue)", "without": ", ".join(head0[:5]) or "-",
         "with": ", ".join(head1[:5]) or "-",
         "detail": f"{len(set(head1) ^ set(head0)) // 2} of 10 differ; Kendall τ {_kendall(m1.review_queue, m0.review_queue)}",
         "changed": head1 != head0},
        {"what": "Geolocation", "without": "slant range, frame-centre ping", "with": "Stage-1 ground range, own ping",
         "detail": (f"median shift {stage1_cf.get('median_shift_m')} m; {stage1_cf.get('outside')} of {stage1_cf.get('n')} "
                    f"pins leave their own error circle") if (stage1_cf or {}).get("available") else
                   ((stage1_cf or {}).get("reason") or "no GPS"),
         "changed": bool((stage1_cf or {}).get("available") and (stage1_cf.get("median_shift_m") or 0) > 0)},
        {"what": "Re-survey route", "without": f"{len(L0)} pass(es), {(m0.resurvey_plan or {}).get('boat_minutes_planned', 0)} boat-min",
         "with": f"{len(L1)} pass(es), {(m1.resurvey_plan or {}).get('boat_minutes_planned', 0)} boat-min",
         "detail": f"{len(s1 & s0)} identical pass(es); ranked by {(m1.resurvey_plan or {}).get('info_unit', '-')} with "
                   f"vs {(m0.resurvey_plan or {}).get('info_unit', '-')} without",
         "changed": s1 != s0},
        {"what": "Human approval", "without": f"{len(m0.inspection_route)} inspection stop(s), {len(L0)} pass request(s), 0 conflicts",
         "with": f"{len(m1.inspection_route)} inspection stop(s), {len(L1)} pass request(s), "
                 f"{sum(bool(t.conflict) for t in tracked)} conflict(s) flagged",
         "detail": "every action still waits for a named person in both arms",
         "changed": (len(L1), m1.inspection_route) != (len(L0), m0.inspection_route) or any(t.conflict for t in tracked)},
    ]
    ex = None
    for t in sorted(changed, key=lambda t: (not t.conflict, -(t.p_pot or 0))):
        ex = {"id": t.oid, "frame": t.frame_id, "conf": t.conf, "p_detector": t.p_pot, "p_evidence": t.p_evidence,
              "without": t.action0, "with": t.action, "conflict": (t.conflict or {}).get("why"),
              "resurvey": t.request_resurvey,
              "queue_rank_with": m1.review_queue.index(t.oid) + 1 if t.oid in m1.review_queue else None,
              "queue_rank_without": m0.review_queue.index(t.oid) + 1 if t.oid in m0.review_queue else None}
        break
    return {"available": True, "active": active_on, "rows": rows, "example": ex, "hazards": len(tracked),
            "actions_changed": len(changed),
            "note": "same detections in both arms; WITHOUT = no shadow / re-look / geometry evidence and no Stage-1 "
                    "ground range. Recall is identical (every find still reaches a person)."}


# ---- CLI: the shipped samples, both modes -> docs/opencv_counterfactual.md -------------------------
def main():
    """python -m src.agentic.counterfactual  — one survey of the shipped samples (synthetic track,
    analyst budget 2 min, boat budget 30 min), active vision OFF (the default) and EXPERIMENTAL."""
    import sys
    from pathlib import Path
    import cv2
    from src.dashboard import samples as samples_mod
    from .geo import synthetic_track, stage1_counterfactual
    from .pipeline import AgenticPipeline
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    frames = []
    for s in samples_mod.list_samples():
        p = samples_mod.sample_path(s["id"])
        im = cv2.imread(str(p)) if p and p.exists() else None
        if im is not None:
            frames.append((p.stem, im))
    track = synthetic_track([f for f, _ in frames])
    pipe = AgenticPipeline()
    L = ["# Does OpenCV change the plan? — the same survey WITH vs WITHOUT the OpenCV evidence\n",
         f"_`python -m src.agentic.counterfactual` · the {len(frames)} shipped sample frames (v2b test, Rec9) · "
         f"synthetic demo track · analyst budget 2 min · boat budget 30 min · regenerated by the script — do not edit_\n",
         "Same detections in both arms. WITHOUT = the detector's tier and action only, no shadow / re-look / "
         "geometry evidence, and slant-range geotags instead of the Stage-1 ground range.\n"]
    for label, faults in (("Active vision OFF (the default)", set()),
                          ("Active vision EXPERIMENTAL (switched on; not validated)", {"opt:active"})):
        s = pipe.run_survey(frames, track=track, survey_id="cf", budget_minutes=2.0, boat_minutes=30.0, faults=faults)
        cf1 = stage1_counterfactual(s.frames, s.tracked, track, pipe.m_per_px)
        cf = without_opencv(s, cf1)
        L.append(f"## {label}\n")
        L.append(f"{cf['hazards']} hazards.\n")
        L.append("| downstream | WITHOUT OpenCV | WITH OpenCV | |")
        L.append("|---|---|---|---|")
        for r in cf["rows"]:
            L.append(f"| {r['what']} | {r['without']} | {r['with']} | {r['detail']} |")
        if cf.get("example"):
            e = cf["example"]
            L.append(f"\nExample **{e['id']}** (`{e['frame']}`, conf {e['conf']}): {e['conflict']} → "
                     f"{e['without'].upper()} → {e['with'].upper()}"
                     + (", opposite-side pass requested (a person approves it)." if e["resurvey"] else "."))
        def cnt(tool, status):
            return sum(1 for fr in s.frames for c in fr.candidates for st in c.trace
                       if st.tool == tool and st.status == status)
        L.append(f"\nOpenCV tools per candidate: shadow check run {cnt('shadow_check', 'done')} / skipped "
                 f"{cnt('shadow_check', 'skipped')}; zoom re-look run {cnt('zoom_relook', 'done')} / skipped "
                 f"{cnt('zoom_relook', 'skipped')} (skipped = value of information 0).\n")
    L.append("Validation status of the evidence-driven actions: see "
             "[`evidence_model_exp003.md`](evidence_model_exp003.md) — both registered checks failed, so they are "
             "off by default. The Stage-1 rows (geolocation, re-survey route) are on in both modes (STUDY-12).")
    out = Path(__file__).resolve().parents[2] / "docs" / "opencv_counterfactual.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
