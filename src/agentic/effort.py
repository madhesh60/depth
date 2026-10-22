"""
effort.py — the analyst-effort curve (review X-2): recall of pots vs human review minutes.

The impact claim of a triage tool is "minutes of analyst time per survey-hour at a stated recall".
This module computes it on the calibration VERIFICATION split (unseen frames, unique frames, every
pot counted — including the ones the detector never proposes) for three workflows:

1. **manual** — an analyst scans every frame in survey order (``sec_per_frame``) and finds each pot
   with probability ``manual_recall``;
2. **detector list** — every detection at the floor, reviewed in survey order as it comes
   (``sec_per_card``), i.e. a detector with no triage;
3. **DEPTH queue** — the REVIEW/CONFIRMED cards in calibrated P(pot) order, with the **recall
   promise** marked (reviewing the queue down to τ_review is what the guarantee is about) and the
   agent's **forecast** (Σ P(pot) of the cards reviewed so far) next to what actually happened. A card is
   what a person sees: **one card per object** when ``calibration.json`` carries STUDY-23's rule
   (``cards.ios``, ``cards.py``) — a frame's overlapping same-class boxes on one card, none dropped —
   for both perception modes: the standard look and **looking harder** (the default when no recall goal
   is set), each at its own promise.

Honesty notes (printed in the report):
* when the calibrated policy is ``baseline_conf`` (EXP-001, EXP-003) the order *is* confidence order (STUDY-07) — DEPTH's gain over a sorted list
  is the stop rule (the promise) and the forecast, not the ranking;
* the timings are ASSUMED until the timed user study (``src/agentic/study.py``) replaces them — the
  **break-even card time** (the card speed at which DEPTH beats manual review at the promised recall)
  is reported because it does not depend on the card timing at all;
* the verification split is pot-dense (most frames have a pot); real surveys are mostly empty
  seabed, where manual review still pays per frame — so the report also gives the **survey mix**
  (minutes per survey-hour vs the share of frames with a pot, from the calibration split's card rates
  on empty and pot frames) and, when ``--real-water`` was run, the cards per hour on a **real raw
  recording** through the Survey tab's own path (the core loop);
* the P(pot) forecast is fit on the calibration recordings; its bias on the verification split is
  reported, not hidden.

Writes ``models/<MODEL>/effort_curve.json`` (small, tracked — the UI recomputes curves live from it
with any timings) and ``docs/reports/effort_curve.md`` + ``docs/img/effort_curve.png``.

    python -m src.agentic.effort [--sec-per-frame 20 --sec-per-card 8]
    python -m src.agentic.effort --real-water [DAT]     # + a raw recording end to end (default: the 1-h one)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULTS = {"sec_per_frame": 20.0, "sec_per_card": 8.0, "manual_recall": 1.0, "card_accuracy": 1.0}
FRAMES_PER_SURVEY_HOUR = 15.0 * 3600 / 640 * 2        # 15 Hz ping rate, 640 pings/frame, port+starboard
RECORDINGS = (REPO / "DATASET" / "external" / "pingmapper_large" / "Test-Large-DS.DAT",
              REPO / "DATASET" / "external" / "pingmapper_sample" / "Test-Small-DS.DAT")
MIX = (0.0, 0.05, 0.20, 0.80)                         # shares of frames with a pot shown in the survey-mix table


def _p_pot(conf: float, bins: list[dict]) -> float:
    for b in bins:
        if b["lo"] <= conf < b["hi"] or (conf >= b["hi"] and b is bins[-1]):
            return float(b["p_pot"])
    return float(bins[0]["p_pot"]) if bins else 0.0


def match(boxes: list[dict], gts: list[tuple]) -> list[int]:
    """One-to-one greedy matching by confidence at IoU ≥ ``calibrate.MATCH_IOU`` (``calibrate.label``'s rule):
    the labelled pot each box takes, or -1."""
    from src.agentic.calibrate import MATCH_IOU
    from src.detection.infer import _iou
    used, out = [False] * len(gts), [-1] * len(boxes)
    for i in sorted(range(len(boxes)), key=lambda k: -boxes[k]["conf"]):
        best, bj = 0.0, -1
        for j, g in enumerate(gts):
            if not used[j]:
                v = _iou(tuple(boxes[i]["bbox"]), g)
                if v > best:
                    best, bj = v, j
        if bj >= 0 and best >= MATCH_IOU:
            used[bj], out[i] = True, bj
    return out


def frame_cards(boxes: list[dict], ios_min: Optional[float]) -> list[list[int]]:
    """A frame's review boxes as cards (box indices, head first): one card per object with STUDY-23's rule
    (``cards.group``; a cache holds one class), else one card per box."""
    if not ios_min:
        return [[i] for i in range(len(boxes))]
    from src.agentic.cards import group
    return group([b["bbox"] for b in boxes], ["one class"] * len(boxes), [b["conf"] for b in boxes], ios_min)


def _queue(frames: list[dict], tau: float, bins: list[dict], ios_min: Optional[float]) -> list[list[float]]:
    """The DEPTH queue on a split, in P(pot) order: per card [labelled pots its boxes took, P(pot) of its head]."""
    q = []
    for fr in frames:
        bx = [c for c in fr["cands"] if c["conf"] >= tau]
        tp = match(bx, [tuple(g) for g in fr["gt"]])
        for m in frame_cards(bx, ios_min):
            q.append((sum(tp[i] >= 0 for i in m), _p_pot(bx[m[0]]["conf"], bins), bx[m[0]]["conf"]))
    q.sort(key=lambda c: (-c[1], -c[2]))
    return [[tp, round(p, 4)] for tp, p, _ in q]


def _mix(frames: list[dict], tau: float, ios_min: Optional[float]) -> dict:
    """Cards per empty frame and per frame with a pot — the survey-mix inputs (calibration split)."""
    n = lambda f: len(frame_cards([c for c in f["cands"] if c["conf"] >= tau], ios_min))
    empty, pots = [f for f in frames if not f["gt"]], [f for f in frames if f["gt"]]
    return {"split": "calibration", "empty_frames": len(empty), "pot_frames": len(pots),
            "cards_per_empty_frame": round(sum(map(n, empty)) / max(1, len(empty)), 3),
            "cards_per_pot_frame": round(sum(map(n, pots)) / max(1, len(pots)), 3)}


def build(model: str = "EXP-001") -> dict:
    """Card/frame facts of the verification split → a compact JSON the UI can replay. The top level is the
    standard look; ``modes.look_harder`` the closer look (when calibrated and its tile pass is cached)."""
    from src.agentic.calibrate import label
    from src.detection.calibration import load_calibration
    cal = load_calibration(model)
    t = cal.tiers
    calib = REPO / "runs" / "calib"
    frames = json.loads((calib / f"{model}_test.json").read_text())["frames"]
    cal_frames = json.loads((calib / f"{model}_val.json").read_text())["frames"]
    tau = float(t.get("tau_review") or 0.0)
    ios_min = cal.card_ios
    survey = []                      # a detector with no triage: every detection at the floor, in survey order
    for fi, fr in enumerate(frames):
        s, tp, _, _ = label([fr], lambda c: c["conf"])
        survey += [[fi, int(ok)] for _, ok in sorted(zip(s, tp), key=lambda x: -x[0])]
    g = t.get("guarantees") or {}
    E = {"model": model, "split": (cal.data.get("fit") or {}).get("verified_on"),
         "frames": len(frames), "pots": int(sum(len(f["gt"]) for f in frames)),
         "pots_per_frame": [len(f["gt"]) for f in frames],
         "tau_review": tau, "recall_promise": g.get("recall_promise"), "mode": "standard",
         "cards": ({"one_per_object": True, "ios": ios_min, "study": cal.cards.get("study") or "STUDY-23"}
                   if ios_min else {"one_per_object": False}),
         "detector_list": survey, "depth_queue": _queue(frames, tau, t.get("p_pot_bins") or [], ios_min),
         "frames_per_survey_hour": round(FRAMES_PER_SURVEY_HOUR, 2), "survey_mix": _mix(cal_frames, tau, ios_min),
         "note": "verification split (unseen, unique frames); timings are parameters — see docs/reports/effort_curve.md",
         "modes": {}}
    lh = cal.modes.get("look_harder") or {}
    tiles = {k: calib / f"{model}_{k}_tiles.json" for k in ("test", "val")}
    if lh.get("tau_review") is not None and all(q.exists() for q in tiles.values()):
        from src.agentic.study_look_harder import merged_frames
        cls, tau_lh = cal.guaranteed_class, float(lh["tau_review"])
        merged = {k: merged_frames(f, json.loads(tiles[k].read_text())["frames"], cls)
                  for k, f in (("test", frames), ("val", cal_frames))}
        E["modes"]["look_harder"] = {
            "mode": "look_harder", "tau_review": tau_lh, "recall_promise": lh.get("recall_promise"),
            "depth_queue": _queue(merged["test"], tau_lh, lh.get("p_pot_bins") or [], ios_min),
            "survey_mix": _mix(merged["val"], tau_lh, ios_min)}
    return E


def for_mode(E: dict, mode: Optional[str]) -> dict:
    """The effort facts of one perception mode (``standard`` = the top level)."""
    m = (E.get("modes") or {}).get(mode or "standard")
    return {**E, **m} if m else E


def survey_mix(E: dict, sec_per_frame: float, sec_per_card: float) -> dict:
    """Analyst minutes per survey-hour vs the share of frames with a pot: manual review pays per frame,
    the queue per card (the calibration split's card rates on empty and pot frames)."""
    m = E.get("survey_mix") or {}
    c0, c1 = m.get("cards_per_empty_frame"), m.get("cards_per_pot_frame")
    if c0 is None or c1 is None:
        return {}
    F = E["frames_per_survey_hour"]
    rows = [{"pot_share": r, "manual_min": round(F * sec_per_frame / 60, 1),
             "depth_min": round(F * ((1 - r) * c0 + r * c1) * sec_per_card / 60, 1)} for r in MIX]
    even = (sec_per_frame / sec_per_card - c0) / (c1 - c0) if c1 != c0 else None
    return {"rows": rows, "breakeven_pot_share": None if even is None else round(even, 3)}


def real_water(dat: Path, model: str) -> dict:
    """A raw recording through the Survey tab's own path (the core loop): review cards per sonar hour.
    No labels exist for it, so every card is a false-alarm candidate as far as anyone knows."""
    from src.agentic.core import Goal, SurveyAgent
    from src.agentic.pipeline import AgenticPipeline
    from src.cv_pipeline.humminbird import frames_and_track, read_recording, validate
    rec = read_recording(dat)
    frames, track, meta = frames_and_track(rec)
    chans = validate(rec)["channels"]
    ch = chans.get("port") or next(iter(chans.values()))
    pipe = AgenticPipeline()
    pipe.agent.perceptor.warmup()
    t0 = time.perf_counter()
    s = SurveyAgent(pipe).run(frames, goal=Goal(), track=track, survey_id="effort", geometry=meta.get("geometry"))
    cards = sum(fr.counts.get("review", 0) + fr.counts.get("confirmed", 0) for fr in s.frames)
    sonar_s = float(ch["duration_s"])
    run = s.agent_run or {}
    return {"model": model, "recording": dat.name, "sha256": hashlib.sha256(dat.read_bytes()).hexdigest(),
            "sonar_s": round(sonar_s, 1), "frames": len(frames), "cards": int(cards),
            "frames_per_hour": round(len(frames) / sonar_s * 3600, 1),
            "cards_per_hour": round(cards / sonar_s * 3600, 1),
            "promise_applies": run.get("promise_applies"), "mode": (run.get("mode") or {}).get("name"),
            "compute_s": round(time.perf_counter() - t0, 1), "created": time.strftime("%Y-%m-%d")}


def _real_water_runs(model: str) -> list[dict]:
    d = REPO / "runs" / "effort"
    return [json.loads(q.read_text()) for q in sorted(d.glob(f"{model}_*.json"))] if d.is_dir() else []


def curves(E: dict, sec_per_frame: float, sec_per_card: float, manual_recall: float = 1.0,
           card_accuracy: float = 1.0) -> dict:
    """Recall-vs-minutes for the three workflows + the promise point + break-even card time."""
    N = E["pots"]
    ppf = np.asarray(E["pots_per_frame"], float)
    man_x = np.arange(len(ppf) + 1) * sec_per_frame / 60.0
    man_y = np.concatenate([[0], np.cumsum(ppf)]) * manual_recall / N
    dl = np.asarray(E["detector_list"], float)
    dl_x = np.arange(len(dl) + 1) * sec_per_card / 60.0
    dl_y = np.concatenate([[0], np.cumsum(dl[:, 1])]) * card_accuracy / N if len(dl) else np.zeros(1)
    dq = np.asarray(E["depth_queue"], float)
    dq_x = np.arange(len(dq) + 1) * sec_per_card / 60.0
    dq_y = np.concatenate([[0], np.cumsum(dq[:, 0])]) * card_accuracy / N if len(dq) else np.zeros(1)
    fc_y = np.concatenate([[0], np.cumsum(dq[:, 1])]) / N if len(dq) else np.zeros(1)

    def minutes_to(xs, ys, r):
        i = np.nonzero(ys >= r - 1e-9)[0]
        return float(xs[i[0]]) if i.size else None

    R = E.get("recall_promise") or 0.0
    m_R, d_R = minutes_to(man_x, man_y, R), minutes_to(dq_x, dq_y, R)
    cards_R = None if d_R is None else int(round(d_R * 60 / sec_per_card))
    frames_R = None if m_R is None else m_R * 60 / sec_per_frame
    breakeven = (frames_R * sec_per_frame / cards_R) if (frames_R and cards_R) else None
    scale = E["frames_per_survey_hour"] / E["frames"]
    return {
        "manual": [man_x.tolist(), man_y.tolist()], "detector_list": [dl_x.tolist(), dl_y.tolist()],
        "depth": [dq_x.tolist(), dq_y.tolist()], "forecast": [dq_x.tolist(), fc_y.tolist()],
        "promise": R, "minutes_to_promise": {"manual": m_R, "depth": d_R},
        "cards_to_promise": cards_R, "breakeven_sec_per_card": None if breakeven is None else round(breakeven, 2),
        "per_survey_hour_min": {"manual": None if m_R is None else round(m_R * scale, 1),
                                "depth": None if d_R is None else round(d_R * scale, 1)},
        "forecast_at_promise": None if d_R is None else round(float(fc_y[cards_R]) * N, 1),
        "actual_at_promise": None if d_R is None else round(float(dq_y[cards_R]) * N / max(card_accuracy, 1e-9), 1),
        "survey_mix": survey_mix(E, sec_per_frame, sec_per_card),
        "params": {"sec_per_frame": sec_per_frame, "sec_per_card": sec_per_card,
                   "manual_recall": manual_recall, "card_accuracy": card_accuracy},
    }


def measured_params() -> tuple[dict, str]:
    """Timings from the user study when it exists, else the ASSUMED defaults."""
    try:
        from src.agentic.study import summary
        s = summary()
        if s.get("sessions", 0) >= 1 and s.get("manual", {}).get("sec_per_frame") and s.get("cards", {}).get("sec_per_card"):
            return ({"sec_per_frame": s["manual"]["sec_per_frame"], "sec_per_card": s["cards"]["sec_per_card"],
                     "manual_recall": s["manual"]["recall"] or 1.0, "card_accuracy": s["cards"]["accuracy_on_real"] or 1.0},
                    f"measured - user study, {s['sessions']} session(s)")
    except Exception:
        pass
    return dict(DEFAULTS), "ASSUMED (no user study yet)"


def chart(E: dict, C: dict, out: Path, provenance: str, C_lh: Optional[dict] = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink2, surface = "#52514e", "#fcfcfb"
    col = {"manual": "#2a78d6", "detector_list": "#eb6834", "depth": "#1baf7a", "look_harder": "#7a5bd6"}
    fig, ax = plt.subplots(figsize=(7.6, 4.2), facecolor=surface)
    ax.set_facecolor(surface)
    per = "one card per object" if (E.get("cards") or {}).get("one_per_object") else "one card per box"
    lab = {"manual": "manual review (every frame)", "detector_list": "detector list (every box, survey order)",
           "depth": f"DEPTH, standard look ({per})"}
    for k in ("manual", "detector_list", "depth"):
        x, y = C[k]
        ax.plot(x, y, color=col[k], linewidth=2, label=lab[k])
    if C_lh:
        x, y = C_lh["depth"]
        ax.plot(x, y, color=col["look_harder"], linewidth=2, label="DEPTH, looking harder (the default)")
    fx, fy = C["forecast"]
    ax.plot(fx, fy, color=col["depth"], linewidth=1.2, linestyle=(0, (3, 3)), label="DEPTH forecast (Σ P(pot))")
    for CC, k_dep, nm, dy in ((C, "depth", "DEPTH", -15), (C_lh, "look_harder", "DEPTH, closer look", 9)):
        R = (CC or {}).get("promise")
        if not R:
            continue
        ax.axhline(R, color="#9a9890", linewidth=0.8, linestyle=":")
        ax.text(0.2, R + 0.012, f"promise ≥ {R:.0%} (95% conf.)", fontsize=8, color=ink2)
        pts = [(CC["minutes_to_promise"][k], colour, name) for k, colour, name in
               (("manual", col["manual"], "manual"), ("depth", col[k_dep], nm)) if CC["minutes_to_promise"][k] is not None]
        for i, (m, colour, name) in enumerate(sorted(pts)):        # left point: label on its left; right point: right
            ax.plot([m], [R], marker="o", markersize=8, color=colour, markeredgecolor=surface, markeredgewidth=2)
            ax.annotate(f"{name} {m:.1f} min", (m, R), textcoords="offset points", xytext=(-8 if i == 0 else 8, dy),
                        ha="right" if i == 0 else "left", fontsize=8, color=colour, fontweight="bold",
                        bbox={"boxstyle": "round,pad=0.15", "fc": surface, "ec": "none", "alpha": 0.9})
    ax.set_xlabel(f"analyst minutes for {E['frames']} frames  ·  timings {provenance}", fontsize=8, color=ink2)
    ax.set_ylabel("share of all labelled pots found", fontsize=8, color=ink2)
    ax.set_ylim(0, 1.0)
    ax.tick_params(colors=ink2, labelsize=8)
    ax.grid(color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c9c8c2")
    leg = ax.legend(fontsize=8, frameon=False, loc="lower right")
    for t in leg.get_texts():
        t.set_color(ink2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=surface)
    plt.close(fig)


def _even_text(even: Optional[float]) -> str:
    if even is None:
        return "—"
    return "at any share" if even >= 1 else "never" if even <= 0 else f"below {even:.0%}"


def report(E: dict, C: dict, provenance: str, C_lh: Optional[dict] = None) -> str:
    from src.detection.calibration import load_calibration
    policy = load_calibration(E["model"]).tiers.get("policy")
    with_pot = sum(1 for n in E["pots_per_frame"] if n > 0)
    p = C["params"]
    fa, fc = C.get("actual_at_promise"), C.get("forecast_at_promise")
    bias = (fc - fa) / fa if fa and fc is not None else None
    one = (E.get("cards") or {}).get("one_per_object")
    modes = [("standard look", E, C)] + ([("looking harder (the default)", for_mode(E, "look_harder"), C_lh)]
                                          if C_lh else [])
    rows = []
    for name, _, CC in modes:
        mt, pm = CC["minutes_to_promise"], CC["per_survey_hour_min"]
        rows.append(f"| {name} | ≥ {CC['promise']:.0%} | {CC['cards_to_promise']} | "
                    f"{'—' if mt['manual'] is None else round(mt['manual'], 1)} | "
                    f"**{'—' if mt['depth'] is None else round(mt['depth'], 1)}** | {pm['manual']} | {pm['depth']} | "
                    f"{CC['breakeven_sec_per_card']} s |")
    mix_lines = []
    if all((CC.get("survey_mix") or {}).get("rows") for _, _, CC in modes):
        sm = [EE.get("survey_mix") or {} for _, EE, _ in modes]
        mixes = [CC["survey_mix"] for _, _, CC in modes]
        mix_lines = [
            "## Survey mix — a real survey-hour is mostly empty seabed", "",
            "Manual review pays per frame whatever it holds (here: every frame, so it finds every pot under the "
            "assumed manual recall); the queue pays per card. On the calibration split the queue raises "
            + "; ".join(f"**{m['cards_per_empty_frame']:.2f} cards per empty frame** and {m['cards_per_pot_frame']:.2f} "
                        f"per frame with a pot ({name})" for (name, _, _), m in zip(modes, sm))
            + f". Per survey-hour (~{E['frames_per_survey_hour']:.0f} frames, modelled: 15 Hz, 640 pings per frame, "
            "both sides), at the timings above:", "",
            "| share of frames with a pot | manual review, min | " + " | ".join(f"DEPTH, {name}, min" for name, _, _ in modes)
            + " |", "|--:|--:|" + "--:|" * len(modes),
            *[f"| {r['pot_share']:.0%} | {r['manual_min']} | " + " | ".join(str(mx["rows"][i]["depth_min"]) for mx in mixes)
              + " |" for i, r in enumerate(mixes[0]["rows"])], "",
            "**Break-even:** DEPTH is faster when the share of frames with a pot is "
            + "; ".join(f"{_even_text(mx.get('breakeven_pot_share'))} ({name})" for (name, _, _), mx in zip(modes, mixes))
            + " — at these assumed timings; the recall promise is per pot, so it does not depend on the mix.", ""]
    runs = _real_water_runs(E["model"])
    water_lines = [] if not runs else [
        "## Real water — raw recordings through the Survey tab's own path", "",
        "No labels exist for these recordings (no known pots), so every card is a false-alarm candidate as far as "
        "anyone knows. Both come from another sonar than the calibration recordings: the agent's applicability "
        "check (STUDY-19) withdraws the recall promise there and asks a person for a full review, so these rows "
        "are the card load, not a recall-backed saving.", "",
        "| recording | sonar | frames / h | cards / h | promise applies | analyst min / h at "
        f"{p['sec_per_card']:.0f} s per card | manual min / h at {p['sec_per_frame']:.0f} s per frame |",
        "|---|--:|--:|--:|:--:|--:|--:|",
        *[f"| `{r['recording']}` | {r['sonar_s'] / 60:.1f} min | {r['frames_per_hour']:.0f} | "
          f"**{r['cards_per_hour']:.0f}** | {'yes' if r['promise_applies'] else 'no — withdrawn'} | "
          f"{r['cards_per_hour'] * p['sec_per_card'] / 60:.1f} | {r['frames_per_hour'] * p['sec_per_frame'] / 60:.1f} |"
          for r in runs], ""]
    L = ["# Analyst effort — recall of pots vs review minutes", "",
         f"_`python -m src.agentic.effort` · model {E['model']} · verification split: {E['split']}_", "",
         f"Timings: **{provenance}** — {p['sec_per_frame']:.0f} s per frame (manual), {p['sec_per_card']:.0f} s per "
         f"card, manual recall {p['manual_recall']:.2f}, card accuracy {p['card_accuracy']:.2f}.",
         ("Cards: **one card per object** — a frame's overlapping same-class boxes share one card and one decision; "
          "no box is dropped, so the pots that reach a person are unchanged "
          "([STUDY-23](card_grouping.md); one card per box before)." if one else "Cards: one per detector box."), "",
         "![effort curve](../img/effort_curve.png)", "",
         f"| perception mode | promise | cards to the promise | manual min ({E['frames']} frames) | DEPTH min | "
         f"manual min / survey-hour | DEPTH min / survey-hour | break-even card time |",
         "|---|--:|--:|--:|--:|--:|--:|--:|", *rows, "",
         "**Break-even card time:** DEPTH reaches the promise faster than manual review whenever reviewing one card "
         "takes less than this — it does not depend on the card timing, only on the manual seconds per frame and how "
         "many cards the promise needs. The timed study (`Study` mode) measures both.", "",
         f"**Forecast vs actual** (standard look): at the promise point the agent forecast **{C['forecast_at_promise']}** "
         f"real pots from its calibrated P(pot); **{C['actual_at_promise']}** were real — the forecast is "
         + (f"**biased {'low' if bias < 0 else 'high'} by {abs(bias):.0%}** on this split" if bias else "unbiased")
         + ". P(pot) is fit on the calibration recordings, so budget mode's forecast (what it tells an analyst "
           "*before* they spend the minutes) is an estimate from those recordings, not a measured rate for a "
           "new survey; on another survey it can err either way.", "",
         *mix_lines, *water_lines,
         "Honesty notes:",
         (f"- {E['model']}'s calibrated policy orders cards by detector confidence (`baseline_conf`, STUDY-07); "
          "DEPTH's gain over a sorted list is the stop rule (the promise) and the forecast, not the ranking."
          if policy == "baseline_conf" else
          f"- {E['model']}'s calibrated policy is `{policy}`; the queue order is P(pot) under that policy."),
         f"- The verification split is pot-dense ({with_pot} of {E['frames']} frames have at least one pot). A "
         "real survey-hour is mostly empty seabed: manual review still pays per frame, the queue pays per card.",
         "- Manual recall of 1.0 is an optimistic assumption for manual review; the study measures it.",
         *(["- A card with several boxes on one object is assumed to take as long as a one-box card; the study "
            "measures what a card takes."] if one else []),
         "", "---", "_Regenerated by the script; do not edit by hand._", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default=None)
    ap.add_argument("--sec-per-frame", type=float, default=None)
    ap.add_argument("--sec-per-card", type=float, default=None)
    ap.add_argument("--real-water", nargs="?", const="", default=None, metavar="DAT",
                    help="also run a raw recording end to end (default: the 1-h one if fetched, else the 150-s one)")
    a = ap.parse_args()
    from src.detection.calibration import DEFAULT_MODEL
    model = a.model or DEFAULT_MODEL
    E = build(model)
    params, prov = measured_params()
    if a.sec_per_frame:
        params["sec_per_frame"], prov = a.sec_per_frame, "set on the command line"
    if a.sec_per_card:
        params["sec_per_card"], prov = a.sec_per_card, "set on the command line"
    if a.real_water is not None:
        dat = Path(a.real_water) if a.real_water else next((r for r in RECORDINGS if r.exists()), None)
        if dat is None or not dat.exists():
            raise SystemExit("no raw recording: python -m src.cv_pipeline.humminbird fetch [--large]")
        rw = real_water(dat, model)
        out = REPO / "runs" / "effort" / f"{model}_{dat.stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rw, indent=1), encoding="utf-8")
        print(f"real water: {rw['cards']} cards in {rw['sonar_s']:.0f} s of sonar -> {rw['cards_per_hour']} cards/h -> {out}")
    C = curves(E, **params)
    C_lh = curves(for_mode(E, "look_harder"), **params) if (E.get("modes") or {}).get("look_harder") else None
    (REPO / "models" / model / "effort_curve.json").write_text(json.dumps(E), encoding="utf-8")
    chart(E, C, REPO / "docs" / "img" / "effort_curve.png", prov, C_lh)
    (REPO / "docs" / "reports" / "effort_curve.md").write_text(report(E, C, prov, C_lh), encoding="utf-8")
    keys = ("promise", "minutes_to_promise", "cards_to_promise", "breakeven_sec_per_card", "per_survey_hour_min",
            "forecast_at_promise", "actual_at_promise")
    print(json.dumps({name: {k: CC[k] for k in keys} for name, CC in (("standard", C), ("look_harder", C_lh)) if CC},
                     indent=1))


if __name__ == "__main__":
    main()
