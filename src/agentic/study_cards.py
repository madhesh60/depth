"""
study_cards.py — STUDY-23: one review card per object. How many of the analyst's review cards are the same
object seen twice, and what does one card per object do to the review load — without touching the recall promise?

The judge audit: "about 5 review cards per frame in the default mode, a heavy load for an analyst"; at the
assumed timings DEPTH was slower than manual review on the pot-dense test split. Looking at the cards showed
why: the detector often puts two or three boxes on one pot (a tight box and a looser one around it; looking
harder, the tile pass's own box beside the first look's), and every box was its own card.

**Protocol (fixed in this file and committed before the verification split was read for this question)**

1. Splits, frames and predictions = STUDY-18's (the ``calibrate.py`` / ``study_look_harder.py`` caches):
   calibration = v2b ``val`` ``*wcp_ss_*`` (163 unique frames, Rec10/12/16); verification = v2b ``test``
   ``*wcp_ss_*`` (199 unique frames, Rec8/9/14). Both modes exactly as the product runs them: standard = the
   full-frame look; look harder = + the 2×2 tile pass, folded in by ``perception.merge_candidates``. A box is a
   review card at its mode's τ_review from ``calibration.json``. Nothing is refit and nothing new is inferred.
2. Rule = :func:`cards.group`, the function the product runs: same class, intersection over the smaller box
   ≥ :data:`IOS`, joined transitively; the head is the strongest box.
3. Measured per split and mode: review cards per frame before and after; cards that hold two or more labelled
   pots (any member box at IoU ≥ 0.3); the pots that reach a person before and after (an identity check: the rule
   keeps every box, so they must be the same); and, at the ASSUMED timings of ``effort.py`` (20 s per frame,
   8 s per card — and one card is one decision whatever its box count, an assumption the Study tab measures),
   the analyst minutes to each mode's promise, the break-even card time and the survey-mix break-even.
4. Gates. **G1** (calibration) — cards per frame fall by ≥ 25% in both modes, and ≤ 1% of cards hold two or
   more labelled pots. **G2** (verification, scored once) — the pots that reach a person are identical with and
   without the rule in both modes; cards per frame fall by ≥ 25% in both modes; ≤ 2% of cards hold two or more
   labelled pots. Only if G1 and G2 pass is the rule written to ``calibration.json`` (``cards``) and switched on
   in the product; otherwise the report says why and every box stays its own card.
5. Disclosed before the verification run: the rule was chosen on the calibration split, where these variants
   were explored — IoU ≥ 0.3 / 0.2 / 0.1 and intersection over the smaller box ≥ 0.8 / 0.6 / 0.5 / 0.3 / any
   overlap: 32–44% fewer cards with the standard look, 38–51% looking harder, and no card with two labelled pots
   in any variant. IoS ≥ 0.5 ("half of the smaller box lies inside the other") is the plainest rule on the
   plateau. Joining across classes changed 3 of about 400 cards there and would put a wreck (protected
   positions are generalised in public exports) on a gear card, so classes stay apart. The product path on the
   calibration split (both classes, evidence included) gave the same picture: 3.52 → 1.99 and 4.95 → 2.48
   cards per frame. The verification split's per-box predictions were scored for recall once, in STUDY-18; this
   study reuses them and refits nothing.

Usage:
    python -m src.agentic.study_cards                    # calibration split only
    python -m src.agentic.study_cards --verify           # + the verification split, once
    python -m src.agentic.study_cards --verify --write   # + calibration.json "cards" (only if G1 and G2 pass)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.detection.calibration import DEFAULT_MODEL, load_calibration, require_loaded_model, save_calibration
from src.detection.infer import DEFAULT_ONNX, _iou
from .calibrate import MATCH_IOU, collect
from .cards import group
from .effort import DEFAULTS, FRAMES_PER_SURVEY_HOUR, _p_pot, curves
from .study_look_harder import PATTERN, V2B, collect_tiles, merged_frames

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "docs" / "reports" / "card_grouping.md"
IOS = 0.5                                  # the rule: half of the smaller box lies inside the other
GATE_DROP = 0.25                           # G1 + G2: cards per frame fall at least this much, both modes
GATE_MULTI = {"cal": 0.01, "ver": 0.02}    # cards holding >= 2 labelled pots, at most this share
MODES = ("standard", "look_harder")


def _tp_per_box(boxes: list[dict], gts: list[tuple]) -> list[int]:
    """One-to-one greedy matching by confidence at IoU ≥ MATCH_IOU (``calibrate.label``'s rule): the pot each
    box takes, or -1."""
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


def measure(frames: list[dict], tau: float, ios_min: float) -> dict:
    """Cards with and without the rule on one split and mode, the pots each arrangement brings to a person, and
    the queue (pots matched, head confidence) per card for the effort curve."""
    boxes = cards = multi = 0
    reached_b: set = set()
    reached_c: set = set()
    q_boxes, q_cards = [], []
    per_frame = []
    for fi, fr in enumerate(frames):
        gts = [tuple(g) for g in fr["gt"]]
        bx = [c for c in fr["cands"] if c["conf"] >= tau]
        tp = _tp_per_box(bx, gts)
        hits = [{j for j, g in enumerate(gts) if _iou(tuple(b["bbox"]), g) >= MATCH_IOU} for b in bx]
        for i, b in enumerate(bx):
            reached_b |= {(fi, j) for j in hits[i]}
            q_boxes.append((int(tp[i] >= 0), b["conf"]))
        g = group([b["bbox"] for b in bx], ["one class"] * len(bx), [b["conf"] for b in bx], ios_min)
        for m in g:
            pots = set().union(*(hits[i] for i in m))
            reached_c |= {(fi, j) for j in pots}
            multi += len(pots) >= 2
            q_cards.append((sum(tp[i] >= 0 for i in m), bx[m[0]]["conf"]))
        boxes += len(bx)
        cards += len(g)
        per_frame.append((len(bx), len(g), bool(gts)))
    n = max(1, len(frames))
    return {"frames": len(frames), "pots": int(sum(len(f["gt"]) for f in frames)), "boxes": boxes, "cards": cards,
            "boxes_per_frame": round(boxes / n, 3), "cards_per_frame": round(cards / n, 3),
            "drop": round(1 - cards / boxes, 4) if boxes else 0.0,
            "multi_pot_cards": int(multi), "multi_share": round(multi / cards, 4) if cards else 0.0,
            "reached_boxes": len(reached_b), "reached_cards": len(reached_c), "same_pots": reached_b == reached_c,
            "_q_boxes": q_boxes, "_q_cards": q_cards, "_per_frame": per_frame}


def mix_rates(m: dict, grouped: bool) -> dict:
    """Cards per empty frame and per frame with a pot (the survey-mix inputs of ``effort.survey_mix``)."""
    k = 1 if grouped else 0
    empty = [f[k] for f in m["_per_frame"] if not f[2]]
    pots = [f[k] for f in m["_per_frame"] if f[2]]
    return {"split": "calibration", "empty_frames": len(empty), "pot_frames": len(pots),
            "cards_per_empty_frame": round(sum(empty) / max(1, len(empty)), 3),
            "cards_per_pot_frame": round(sum(pots) / max(1, len(pots)), 3)}


def effort_at(frames: list[dict], m: dict, grouped: bool, bins: list[dict], promise: float, mix: dict) -> dict:
    """``effort.curves`` on one arrangement of the queue (P(pot) order) at the assumed timings."""
    q = m["_q_cards"] if grouped else m["_q_boxes"]
    dq = sorted(([tp, _p_pot(conf, bins), conf] for tp, conf in q), key=lambda c: (-c[1], -c[2]))
    E = {"pots": m["pots"], "pots_per_frame": [len(f["gt"]) for f in frames], "frames": len(frames),
         "frames_per_survey_hour": round(FRAMES_PER_SURVEY_HOUR, 2), "recall_promise": promise,
         "detector_list": [], "depth_queue": [[tp, p] for tp, p, _ in dq], "survey_mix": mix}
    C = curves(E, DEFAULTS["sec_per_frame"], DEFAULTS["sec_per_card"])
    return {"minutes": C["minutes_to_promise"], "cards_to_promise": C["cards_to_promise"],
            "breakeven_sec_per_card": C["breakeven_sec_per_card"], "per_survey_hour_min": C["per_survey_hour_min"],
            "mix_breakeven": (C.get("survey_mix") or {}).get("breakeven_pot_share"),
            "mix_rows": (C.get("survey_mix") or {}).get("rows")}


def _public(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------------------- report
def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def _min(x) -> str:
    return "—" if x is None else f"{x:.1f}"


def report(model: str, res: dict) -> str:
    L = [f"# STUDY-23 — one review card per object ({model})\n",
         "_Generated by `python -m src.agentic.study_cards --verify` · protocol fixed in the module docstring and "
         "committed before the verification split was read for this question · unique frames · match IoU "
         f"{MATCH_IOU}_\n",
         f"**Rule** (`src/agentic/cards.py`, the function the product runs): within one frame, two review boxes of the "
         f"same class are one card when at least {IOS:.0%} of the smaller box lies inside the other, joined "
         "transitively. The strongest box heads the card (its tier, P(pot), position, place in the queue); every "
         "other box stays on the card. No box is dropped.\n"]
    for key, title in (("cal", "## 1. Calibration split (design)"), ("ver", "## 2. Verification split (scored once)")):
        s = res.get(key)
        if not s:
            L += ["## 2. Verification split\n", "_Not scored yet (`--verify`)._\n"]
            continue
        L += [title + "\n", f"{s['split']}\n",
              "| mode | τ_review | boxes / frame | **cards / frame** | fewer cards | cards with ≥ 2 pots | pots that "
              "reach a person (boxes → cards) |", "|---|--:|--:|--:|--:|--:|--:|"]
        for mode in MODES:
            m = s[mode]
            L.append(f"| {mode} | {m['tau']:.3f} | {m['boxes_per_frame']:.2f} | **{m['cards_per_frame']:.2f}** | "
                     f"{100 * m['drop']:.0f}% | {m['multi_pot_cards']} of {m['cards']} | {m['reached_boxes']} → "
                     f"{m['reached_cards']} of {m['pots']} ({'same pots' if m['same_pots'] else '**different**'}) |")
        gate = "G1" if key == "cal" else "G2"
        L += ["", f"**{gate}** — {'the same pots reach a person; ' if key == 'ver' else ''}cards per frame fall "
              f"≥ {GATE_DROP:.0%} in both modes; ≤ {GATE_MULTI[key]:.0%} of cards hold two or more labelled pots: "
              f"**{'PASS' if res['gates'][gate] else 'FAIL'}**\n"]
    e = res.get("effort")
    if e:
        L += ["## 3. What it does to the analyst's minutes (ASSUMED timings)\n",
              f"Verification split, {res['ver'][MODES[0]]['frames']} frames; 20 s per frame for manual review and 8 s per "
              "card; **one card is one decision whatever its box count** — an assumption until the Study tab measures "
              "it. Manual review reaches each promise in the same time either way.\n",
              "| mode (promise) | cards | cards to the promise | DEPTH min | manual min | break-even s / card | "
              "faster when frames with a pot < |", "|---|---|--:|--:|--:|--:|--:|"]
        for mode in MODES:
            for arr in ("boxes", "cards"):
                r = e[mode][arr]
                L.append(f"| {mode} (≥ {res['promise'][mode]:.0%}) | {'one per box (before)' if arr == 'boxes' else '**one per object**'} | "
                         f"{r['cards_to_promise']} | {_min(r['minutes']['depth'])} | {_min(r['minutes']['manual'])} | "
                         f"{r['breakeven_sec_per_card']} | {_pct(r['mix_breakeven'])} |")
        L += ["", "The last column is the survey-mix break-even: the calibration split's cards per empty frame and "
              "per frame with a pot, priced at the same timings (a real survey-hour is mostly empty seabed).\n"]
    L += ["## 4. Decision\n", res["decision"], "",
          "Caveats: labels are incomplete, so a card's 'pots' are the labelled ones; the time a person needs for a "
          "card with three boxes on one object is assumed equal to a one-box card; the calibration split is 3 "
          "recordings.", "", "---", "_Regenerated by the script; do not edit by hand._"]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--cal-root", default=str(V2B / "val"))
    ap.add_argument("--ver-root", default=str(V2B / "test"))
    ap.add_argument("--verify", action="store_true", help="score the verification split (once)")
    ap.add_argument("--write", action="store_true", help="write calibration.json cards if G1 and G2 pass")
    a = ap.parse_args()
    require_loaded_model(a.model, "src.agentic.study_cards")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cal = load_calibration(a.model)
    lh = cal.modes.get("look_harder") or {}
    if lh.get("tau_review") is None:
        raise SystemExit("no look-harder mode in calibration.json (STUDY-18 first)")
    cls = cal.guaranteed_class
    tau = {"standard": float(cal.tiers["tau_review"]), "look_harder": float(lh["tau_review"])}
    bins = {"standard": cal.tiers.get("p_pot_bins") or [], "look_harder": lh.get("p_pot_bins") or []}
    promise = {"standard": float(cal.guarantees["recall_promise"]), "look_harder": float(lh["recall_promise"])}
    cache, onnx = REPO / "runs" / "calib", Path(a.onnx)

    def split(root: Path, desc: str) -> tuple[dict, str]:
        base = collect(root, PATTERN, cls, 0, cache / f"{a.model}_{root.name}.json", onnx)
        tiles = collect_tiles(root, cls, cache / f"{a.model}_{root.name}_tiles.json", onnx)
        frames = base["frames"]
        recs = sorted({f["recording"] for f in frames if f["recording"]})
        return ({"standard": frames, "look_harder": merged_frames(frames, tiles["frames"], cls)},
                f"{desc}: `{root.relative_to(REPO).as_posix()}` {PATTERN} — {len(frames)} unique frames, recordings {recs}")

    def gate(s: dict, key: str) -> bool:
        return all(s[m]["drop"] >= GATE_DROP and s[m]["multi_share"] <= GATE_MULTI[key]
                   and (key == "cal" or s[m]["same_pots"]) for m in MODES)

    print("calibration split ...")
    c_frames, c_desc = split(Path(a.cal_root), "Calibration")
    c = {m: {**measure(c_frames[m], tau[m], IOS), "tau": tau[m]} for m in MODES}
    res: dict = {"cal": {"split": c_desc, **c}, "promise": promise, "ios": IOS,
                 "gates": {"G1": gate(c, "cal"), "G2": None}}
    if a.verify:
        print("verification split (scored once) ...")
        v_frames, v_desc = split(Path(a.ver_root), "Verification")
        v = {m: {**measure(v_frames[m], tau[m], IOS), "tau": tau[m]} for m in MODES}
        res["ver"] = {"split": v_desc, **v}
        res["gates"]["G2"] = gate(v, "ver")
        res["effort"] = {m: {arr: effort_at(v_frames[m], v[m], arr == "cards", bins[m], promise[m],
                                            mix_rates(c[m], arr == "cards"))
                             for arr in ("boxes", "cards")} for m in MODES}
    ok = bool(res["gates"]["G1"] and res["gates"]["G2"])
    res["decision"] = (
        f"**Ship** one card per object: the product folds a frame's overlapping same-class review boxes into one card "
        f"(`calibration.json` `cards.ios` = {IOS}). The recall promise is unchanged — the same pots reach a person — "
        "and the card counts the core loop plans with come from this study." if ok else
        ("**Not shipped**: a gate failed; every box stays its own card." if a.verify else
         "_Pending the verification split._"))
    REPORT.write_text(report(a.model, res), encoding="utf-8")
    out = REPO / "runs" / "cards" / f"{a.model}_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    clean = {**res, "cal": {k: (_public(v) if isinstance(v, dict) else v) for k, v in res["cal"].items()}}
    if "ver" in res:
        clean["ver"] = {k: (_public(v) if isinstance(v, dict) else v) for k, v in res["ver"].items()}
    out.write_text(json.dumps(clean, indent=1))
    print(json.dumps({"gates": res["gates"], **{m: {k: c[m][k] for k in ("boxes_per_frame", "cards_per_frame", "drop",
                                                                         "multi_pot_cards")} for m in MODES}}, indent=1))
    print(f"wrote {REPORT.relative_to(REPO)} and {out.relative_to(REPO)}")
    if a.write and ok:
        v = res["ver"]
        block = {"rule": "same class; at least ios of the smaller box inside the other; transitive; head = the "
                         "strongest box (src/agentic/cards.py)",
                 "ios": IOS, "study": "STUDY-23", "report": REPORT.relative_to(REPO).as_posix(),
                 "cards_per_frame_cal": {m: c[m]["cards_per_frame"] for m in MODES},
                 "boxes_per_frame_cal": {m: c[m]["boxes_per_frame"] for m in MODES},
                 "verified_on_test": {m: {k: v[m][k] for k in ("cards_per_frame", "boxes_per_frame", "drop",
                                                               "multi_pot_cards", "same_pots")} for m in MODES},
                 "fit": {"split": c_desc, "verified_on": v["split"], "created": time.strftime("%Y-%m-%d")}}
        p = save_calibration({"cards": block}, model=a.model)
        print(f"wrote cards -> {p}")


if __name__ == "__main__":
    main()
