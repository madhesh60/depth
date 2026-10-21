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
   agent's **forecast** (Σ P(pot) of the cards reviewed so far) next to what actually happened.

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


def build(model: str = "EXP-001") -> dict:
    """Card/frame facts of the verification split → a compact JSON the UI can replay."""
    from src.agentic.calibrate import label
    from src.detection.calibration import load_calibration
    cal = load_calibration(model)
    t = cal.tiers
    blob = json.loads((REPO / "runs" / "calib" / f"{model}_test.json").read_text())
    frames = blob["frames"]
    tau = float(t.get("tau_review") or 0.0)
    bins = t.get("p_pot_bins") or []
    # survey order: per frame, its cards (by conf) with one-to-one TP labels
    survey, depth = [], []
    for fi, fr in enumerate(frames):
        s, tp, _, _ = label([fr], lambda c: c["conf"])
        for sc, ok in sorted(zip(s, tp), key=lambda x: -x[0]):
            survey.append({"f": fi, "conf": round(float(sc), 4), "tp": int(ok)})
            if sc >= tau:
                depth.append({"f": fi, "conf": round(float(sc), 4), "tp": int(ok), "p": round(_p_pot(sc, bins), 4)})
    depth.sort(key=lambda c: (-c["p"], -c["conf"]))
    g = t.get("guarantees") or {}
    cal_frames = json.loads((REPO / "runs" / "calib" / f"{model}_val.json").read_text())["frames"]
    empty, pots = [f for f in cal_frames if not f["gt"]], [f for f in cal_frames if f["gt"]]
    cards = lambda fs: sum(sum(c["conf"] >= tau for c in f["cands"]) for f in fs)
    return {
        "model": model, "split": (cal.data.get("fit") or {}).get("verified_on"),
        "frames": len(frames), "pots": int(sum(len(f["gt"]) for f in frames)),
        "pots_per_frame": [len(f["gt"]) for f in frames],
        "tau_review": tau, "recall_promise": g.get("recall_promise"),
        "detector_list": [[c["f"], c["tp"]] for c in survey],
        "depth_queue": [[c["tp"], c["p"]] for c in depth],
        "frames_per_survey_hour": round(FRAMES_PER_SURVEY_HOUR, 2),
        "survey_mix": {"split": "calibration", "empty_frames": len(empty), "pot_frames": len(pots),
                       "cards_per_empty_frame": round(cards(empty) / max(1, len(empty)), 3),
                       "cards_per_pot_frame": round(cards(pots) / max(1, len(pots)), 3)},
        "note": "verification split (unseen, unique frames); timings are parameters — see docs/reports/effort_curve.md",
    }


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


def chart(E: dict, C: dict, out: Path, provenance: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    col = {"manual": "#2a78d6", "detector_list": "#eb6834", "depth": "#1baf7a"}   # validated slots 1-3
    fig, ax = plt.subplots(figsize=(7.6, 4.2), facecolor=surface)
    ax.set_facecolor(surface)
    lab = {"manual": "manual review (every frame)", "detector_list": "detector list (survey order)",
           "depth": "DEPTH queue (P(pot) order)"}
    for k in ("manual", "detector_list", "depth"):
        x, y = C[k]
        ax.plot(x, y, color=col[k], linewidth=2, label=lab[k])
    fx, fy = C["forecast"]
    ax.plot(fx, fy, color=col["depth"], linewidth=1.2, linestyle=(0, (3, 3)), label="DEPTH forecast (Σ P(pot))")
    R = C["promise"]
    if R:
        ax.axhline(R, color="#9a9890", linewidth=0.8, linestyle=":")
        ax.text(0.2, R + 0.012, f"recall promise ≥ {R:.0%} (95% conf.)", fontsize=8, color=ink2)
        offs = {"manual": (-74, 10), "depth": (8, -16)}          # keep the two labels apart when they coincide
        names = {"manual": "manual", "depth": "DEPTH"}
        for k in ("manual", "depth"):
            m = C["minutes_to_promise"][k]
            if m is not None:
                ax.plot([m], [R], marker="o", markersize=8, color=col[k], markeredgecolor=surface, markeredgewidth=2)
                ax.annotate(f"{names[k]} {m:.1f} min", (m, R), textcoords="offset points", xytext=offs[k],
                            fontsize=8, color=ink)
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


def report(E: dict, C: dict, provenance: str) -> str:
    from src.detection.calibration import load_calibration
    policy = load_calibration(E["model"]).tiers.get("policy")
    with_pot = sum(1 for n in E["pots_per_frame"] if n > 0)
    p = C["params"]
    mt, pm = C["minutes_to_promise"], C["per_survey_hour_min"]
    fa, fc = C.get("actual_at_promise"), C.get("forecast_at_promise")
    bias = (fc - fa) / fa if fa and fc is not None else None
    sm, mix = E.get("survey_mix") or {}, C.get("survey_mix") or {}
    even = mix.get("breakeven_pot_share")
    mix_lines = [] if not mix else [
        "## Survey mix — a real survey-hour is mostly empty seabed", "",
        f"On the calibration split the queue raises **{sm['cards_per_empty_frame']:.2f} cards per empty frame** "
        f"({sm['empty_frames']} frames without a pot) and {sm['cards_per_pot_frame']:.2f} per frame with a pot "
        f"({sm['pot_frames']} frames). Manual review pays per frame whatever it holds (here: every frame, so it finds "
        f"every pot under the assumed manual recall); the queue pays per card (every card: ≥ {E['recall_promise']:.0%} "
        f"of pots, the promise). At the timings above, per survey-hour (~{E['frames_per_survey_hour']:.0f} frames, "
        "modelled: 15 Hz, 640 pings per frame, both sides):", "",
        "| share of frames with a pot | manual review, min | DEPTH queue, min |", "|--:|--:|--:|",
        *[f"| {r['pot_share']:.0%} | {r['manual_min']} | {r['depth_min']} |" for r in mix["rows"]], "",
        (f"**Break-even: DEPTH is faster when fewer than {even:.0%} of frames hold a pot** (at these assumed "
         "timings; the recall promise is per pot, so it does not depend on the mix). On empty seabed the saving "
         "is small, because the detector at τ_review still raises about two cards per empty frame."
         if even is not None and 0 <= even <= 1 else "No break-even inside 0–100% at these timings."), ""]
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
         f"card, manual recall {p['manual_recall']:.2f}, card accuracy {p['card_accuracy']:.2f}.", "",
         "![effort curve](../img/effort_curve.png)", "",
         f"| at the recall promise (≥ {C['promise']:.0%}) | manual review | DEPTH queue |", "|---|--:|--:|",
         f"| analyst minutes for these {E['frames']} frames | {mt['manual'] if mt['manual'] is None else round(mt['manual'], 1)} | "
         f"{mt['depth'] if mt['depth'] is None else round(mt['depth'], 1)} |",
         f"| per survey-hour of sonar (~{E['frames_per_survey_hour']:.0f} frames) | {pm['manual']} | {pm['depth']} |",
         f"| cards reviewed | — | {C['cards_to_promise']} |",
         "",
         f"**Break-even card time: {C['breakeven_sec_per_card']} s.** DEPTH reaches the promise faster than "
         f"manual review whenever reviewing one card takes less than this — it does not depend on the card "
         f"timing, only on the manual seconds per frame and how many cards the promise needs. The timed "
         f"study (`Study` mode) measures both.", "",
         f"**Forecast vs actual:** at the promise point the agent forecast **{C['forecast_at_promise']}** real "
         f"pots from its calibrated P(pot); **{C['actual_at_promise']}** were real — the forecast is "
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
         "real survey-hour is mostly empty seabed: manual review still pays per frame, the queue pays per detection.",
         "- Manual recall of 1.0 is an optimistic assumption for manual review; the study measures it.",
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
    (REPO / "models" / model / "effort_curve.json").write_text(json.dumps(E), encoding="utf-8")
    chart(E, C, REPO / "docs" / "img" / "effort_curve.png", prov)
    (REPO / "docs" / "reports" / "effort_curve.md").write_text(report(E, C, prov), encoding="utf-8")
    print(json.dumps({k: C[k] for k in ("promise", "minutes_to_promise", "cards_to_promise",
                                        "breakeven_sec_per_card", "per_survey_hour_min",
                                        "forecast_at_promise", "actual_at_promise")}, indent=1))


if __name__ == "__main__":
    main()
