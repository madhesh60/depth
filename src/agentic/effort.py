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
* for EXP-001 the calibrated order *is* confidence order (STUDY-07) — DEPTH's gain over a sorted list
  is the stop rule (the promise) and the forecast, not the ranking;
* the timings are ASSUMED until the timed user study (``src/agentic/study.py``) replaces them — the
  **break-even card time** (the card speed at which DEPTH beats manual review at the promised recall)
  is reported because it does not depend on the card timing at all;
* the verification split is pot-dense (every frame has a pot); real surveys are mostly empty
  seabed, where manual review still pays per frame.

Writes ``models/<MODEL>/effort_curve.json`` (small, tracked — the UI recomputes curves live from it
with any timings) and ``docs/effort_curve.md`` + ``docs/img/effort_curve.png``.

    python -m src.agentic.effort [--sec-per-frame 20 --sec-per-card 8]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULTS = {"sec_per_frame": 20.0, "sec_per_card": 8.0, "manual_recall": 1.0, "card_accuracy": 1.0}
FRAMES_PER_SURVEY_HOUR = 15.0 * 3600 / 640 * 2        # 15 Hz ping rate, 640 pings/frame, port+starboard


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
    return {
        "model": model, "split": (cal.data.get("fit") or {}).get("verified_on"),
        "frames": len(frames), "pots": int(sum(len(f["gt"]) for f in frames)),
        "pots_per_frame": [len(f["gt"]) for f in frames],
        "tau_review": tau, "recall_promise": g.get("recall_promise"),
        "detector_list": [[c["f"], c["tp"]] for c in survey],
        "depth_queue": [[c["tp"], c["p"]] for c in depth],
        "frames_per_survey_hour": round(FRAMES_PER_SURVEY_HOUR, 2),
        "note": "verification split (unseen, unique frames); timings are parameters — see docs/effort_curve.md",
    }


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
    ax.grid(color="#e4e3df", linewidth=0.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c9c8c2")
    leg = ax.legend(fontsize=8, frameon=False, loc="lower right")
    for t in leg.get_texts():
        t.set_color(ink2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=160, facecolor=surface); plt.close(fig)


def report(E: dict, C: dict, provenance: str) -> str:
    p = C["params"]
    mt, pm = C["minutes_to_promise"], C["per_survey_hour_min"]
    L = ["# Analyst effort — recall of pots vs review minutes", "",
         f"_`python -m src.agentic.effort` · model {E['model']} · verification split: {E['split']}_", "",
         f"Timings: **{provenance}** — {p['sec_per_frame']:.0f} s per frame (manual), {p['sec_per_card']:.0f} s per "
         f"card, manual recall {p['manual_recall']:.2f}, card accuracy {p['card_accuracy']:.2f}.", "",
         "![effort curve](img/effort_curve.png)", "",
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
         f"pots from its calibrated P(pot); **{C['actual_at_promise']}** were real — this is what budget mode "
         f"tells an analyst *before* they spend the minutes.", "",
         "Honesty notes:",
         "- For EXP-001 the calibrated order equals confidence order (STUDY-07); DEPTH's gain over a sorted "
         "list is the stop rule (the promise) and the forecast, not the ranking.",
         "- The verification split is pot-dense (every frame has at least one pot). A real survey-hour is "
         "mostly empty seabed: manual review still pays per frame, the queue pays per detection.",
         "- Manual recall of 1.0 is an optimistic assumption for manual review; the study measures it.",
         "", "---", "_Regenerated by the script; do not edit by hand._", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default=None)
    ap.add_argument("--sec-per-frame", type=float, default=None)
    ap.add_argument("--sec-per-card", type=float, default=None)
    a = ap.parse_args()
    from src.detection.calibration import DEFAULT_MODEL
    model = a.model or DEFAULT_MODEL
    E = build(model)
    params, prov = measured_params()
    if a.sec_per_frame:
        params["sec_per_frame"], prov = a.sec_per_frame, "set on the command line"
    if a.sec_per_card:
        params["sec_per_card"], prov = a.sec_per_card, "set on the command line"
    C = curves(E, **params)
    (REPO / "models" / model / "effort_curve.json").write_text(json.dumps(E), encoding="utf-8")
    chart(E, C, REPO / "docs" / "img" / "effort_curve.png", prov)
    (REPO / "docs" / "effort_curve.md").write_text(report(E, C, prov), encoding="utf-8")
    print(json.dumps({k: C[k] for k in ("promise", "minutes_to_promise", "cards_to_promise",
                                        "breakeven_sec_per_card", "per_survey_hour_min",
                                        "forecast_at_promise", "actual_at_promise")}, indent=1))


if __name__ == "__main__":
    main()
