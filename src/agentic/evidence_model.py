"""
evidence_model.py — how much each OpenCV measurement should move the agent's belief, MEASURED.

For each detector-confidence band, the likelihood ratio of an observation is
``P(observation | real pot) / P(observation | false alarm)``, counted on the **validation**
recordings (the same cache ``calibrate.py`` builds: one row per candidate with the shadow and
re-look measured by the runtime code). The belief update is Bayes in odds form, starting from the
calibrated detector-only P(pot) in ``calibration.json``:

    odds(P | evidence) = odds(P_calibrated(conf)) × LR_shadow(band) × LR_relook(band)

(conditional independence of the two observations given the truth within a band is an assumption;
the verification below is what tells us whether the update helps.) Jeffreys smoothing (+0.5) keeps
small bands honest.

**Verification on test, once:** log-loss and Brier score of the evidence-updated P vs the
detector-only P, and the replay of the active policy (``active.py``) on every test candidate —
which tools it chose, conflicts, actions and how real pots are ranked WITH vs WITHOUT the OpenCV
evidence. ``use`` is set only if the update does not worsen test log-loss; otherwise the agent
keeps the detector-only belief (and says so in every trace).

    python -m src.agentic.evidence_model            # fit (val) -> verify (test) -> write json + report
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date
from pathlib import Path
from typing import Optional

from src.detection.calibration import calibration_path, load_calibration, DEFAULT_MODEL
from . import active

REPO = Path(__file__).resolve().parents[2]
BAND_EDGES = (0.15, 0.30)                 # declared on validation: low / middle / high detector score
THRESHOLDS = {"accept": 0.60, "watch": 0.15, "resurvey_flip": 0.25}   # design constants (not fit)
RELOOK_KEY = "rl_single"                  # the agent's zoom re-look = one crop, one cv2.dnn pass


def model_path(model: Optional[str] = None) -> Path:
    return calibration_path(model).parent / "evidence_model.json"


class EvidenceModel:
    def __init__(self, data: dict):
        self.data = data
        self.bands = list(data.get("bands") or [])
        self.thresholds = dict(data.get("thresholds") or THRESHOLDS)
        self.use = bool(data.get("use"))
        self.costs = dict(data.get("costs_ms") or {})

    @classmethod
    def load(cls, model: Optional[str] = None) -> Optional["EvidenceModel"]:
        p = model_path(model)
        if not p.exists():
            return None
        return cls(json.loads(p.read_text(encoding="utf-8")))

    def band(self, conf: float) -> int:
        return sum(conf >= e for e in self.data.get("band_edges", BAND_EDGES))

    def lr_for(self, kind: str, conf: float) -> Optional[dict]:
        b = self.bands[self.band(conf)] if self.bands else None
        return (b or {}).get(kind)

    def cost_ms(self, tool: str) -> float:
        return float(self.costs.get(tool, 1e9))

    def p_after(self, p0: float, conf: float, shadow: Optional[bool] = None,
                relook: Optional[bool] = None) -> float:
        p = p0
        if shadow is not None:
            p = active.update(self, p, conf, "shadow", shadow)
        if relook is not None:
            p = active.update(self, p, conf, "relook", relook)
        return p


# ================================================================================ fit + verify
def _records(model: str, split: str) -> list[dict]:
    """One row per candidate with its truth (greedy one-to-one match, IoU >= calibrate.MATCH_IOU)."""
    from src.detection.infer import _iou
    from .calibrate import MATCH_IOU
    cache = REPO / "runs" / "calib" / f"{model}_{split}.json"
    blob = json.loads(cache.read_text())
    out = []
    for fr in blob["frames"]:
        gts = [tuple(g) for g in fr["gt"]]
        used = [False] * len(gts)
        for c in sorted(fr["cands"], key=lambda c: -c["conf"]):
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if not used[j]:
                    v = _iou(tuple(c["bbox"]), g)
                    if v > best:
                        best, bj = v, j
            hit = bj >= 0 and best >= MATCH_IOU
            if hit:
                used[bj] = True
            out.append({**c, "tp": hit, "frame": fr["name"], "orient": _orient_known(fr["name"])})
    return out, blob.get("timing", {})


def _orient_known(name: str) -> bool:
    """The runtime measures a shadow only when the source rule gives the orientation (never guessed);
    the replay must do the same (the first fresh-split replay guessed 'top' - a harness defect)."""
    from src.cv_pipeline.orientation import resolve_orientation
    return resolve_orientation(Path(name).stem).nadir is not None


def _obs(r: dict, kind: str) -> bool:
    return r["shadow"] != "none" if kind == "shadow" else r[RELOOK_KEY] > 0


def fit(rows: list[dict], tau_review: float) -> list[dict]:
    bands = []
    edges = (tau_review,) + BAND_EDGES + (math.inf,)
    for lo, hi in zip(edges, edges[1:]):
        B = [r for r in rows if lo <= r["conf"] < hi]
        tp = [r for r in B if r["tp"]]
        fp = [r for r in B if not r["tp"]]
        band = {"lo": lo, "hi": None if hi == math.inf else hi, "n": len(B), "n_tp": len(tp)}
        for kind in ("shadow", "relook"):
            a_tp = sum(_obs(r, kind) for r in tp)
            a_fp = sum(_obs(r, kind) for r in fp)
            ptp = (a_tp + 0.5) / (len(tp) + 1.0)                     # Jeffreys smoothing
            pfp = (a_fp + 0.5) / (len(fp) + 1.0)
            band[kind] = {"present": round(ptp / pfp, 4), "absent": round((1 - ptp) / (1 - pfp), 4),
                          "present_tp": round(ptp, 4), "present_fp": round(pfp, 4),
                          "absent_tp": round(1 - ptp, 4), "absent_fp": round(1 - pfp, 4),
                          "counts": {"present_tp": a_tp, "tp": len(tp), "present_fp": a_fp, "fp": len(fp)}}
        bands.append(band)
    return bands


def _ll(ps, ys):
    e = 1e-6
    return -sum(math.log(min(max(p, e), 1 - e)) if y else math.log(1 - min(max(p, e), 1 - e))
                for p, y in zip(ps, ys)) / max(1, len(ys))


def _brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / max(1, len(ys))


def _auc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y]
    neg = [p for p, y in zip(ps, ys) if not y]
    if not pos or not neg:
        return None
    s = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return s / (len(pos) * len(neg))


def replay(em: EvidenceModel, rows: list[dict], tiers) -> dict:
    """Run the active policy on every candidate (observations looked up from the cache) and compare
    WITH vs WITHOUT the OpenCV evidence. Only candidates >= tau_review (the REVIEW tier)."""
    R = [r for r in rows if r["conf"] >= tiers.tau_review]
    n_tp = sum(r["tp"] for r in R)
    out = {"candidates": len(R), "real_pots": n_tp, "tools": {"shadow_check": 0, "zoom_relook": 0},
           "conflicts": 0, "conflicts_resolved": 0, "resurvey_requested": 0}
    acts_with, acts_without, p_with, p_without, ys = [], [], [], [], []
    for r in R:
        p0 = tiers.p_pot(r["conf"])
        avail = active.IN_FRAME_TOOLS if r.get("orient", True) else ("zoom_relook",)
        res = active.run_policy(em, r["conf"], p0, lambda tool, r=r: _obs(r, "shadow" if tool == "shadow_check" else "relook"),
                                available=avail)
        for t in res["tools_run"]:
            out["tools"][t] += 1
        out["no_tool_needed"] = out.get("no_tool_needed", 0) + (not res["tools_run"])
        if res["conflict"]:
            out["conflicts"] += 1
            out["conflicts_resolved"] += bool(res["conflict_resolved"])
        out["resurvey_requested"] += res["request_resurvey"]
        acts_with.append(res["action"]); acts_without.append(res["action0"])
        p_with.append(res["p"]); p_without.append(p0); ys.append(r["tp"])
    out["relook_share"] = round(out["tools"]["zoom_relook"] / max(1, len(R)), 3)
    out["actions"] = {}
    for name, acts in (("with_opencv", acts_with), ("without_opencv", acts_without)):
        d = {}
        for a in ("accept", "review", "watch"):
            idx = [i for i, x in enumerate(acts) if x == a]
            k = sum(ys[i] for i in idx)
            d[a] = {"n": len(idx), "real": k, "precision": round(k / len(idx), 3) if idx else None}
        out["actions"][name] = d
    out["changed_action"] = sum(a != b for a, b in zip(acts_with, acts_without))
    # the human queue: how many real pots are in the first N cards (N = a quarter / half of the queue)
    for frac in (0.25, 0.5):
        k = max(1, int(round(frac * len(R))))
        order_w = sorted(range(len(R)), key=lambda i: (active.ACTION_RANK[acts_with[i]], -p_with[i]))
        order_0 = sorted(range(len(R)), key=lambda i: -p_without[i])
        out[f"pots_in_first_{int(frac * 100)}pct"] = {
            "cards": k, "with_opencv": sum(ys[i] for i in order_w[:k]),
            "without_opencv": sum(ys[i] for i in order_0[:k])}
    out["log_loss"] = {"with_opencv": round(_ll(p_with, ys), 4), "without_opencv": round(_ll(p_without, ys), 4)}
    out["brier"] = {"with_opencv": round(_brier(p_with, ys), 4), "without_opencv": round(_brier(p_without, ys), 4)}
    out["auc"] = {"with_opencv": _r(_auc(p_with, ys)), "without_opencv": _r(_auc(p_without, ys))}
    return out


def _r(x, nd=3):
    return None if x is None else round(x, nd)


def _shadow_cost_ms(model: str) -> float:
    """Median wall time of one shadow measurement on validation frames (no inference)."""
    import cv2
    import numpy as np
    from .shadow import ShadowProver, ShadowConfig
    from .calibrate import list_frames
    blob = json.loads((REPO / "runs" / "calib" / f"{model}_val.json").read_text())
    root = REPO / "DATASET" / "03_yolo_ready_dataset_v2b" / "val" / "images"
    prover, ts = ShadowProver(ShadowConfig(nadir="top")), []
    for fr in blob["frames"][:40]:
        im = cv2.imread(str(root / fr["name"]), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        for c in fr["cands"]:
            t = time.perf_counter()
            prover.prove(im, tuple(c["bbox"]))
            ts.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(ts)), 2) if ts else 1.0


def build(model: str = DEFAULT_MODEL) -> dict:
    from .policy import GuaranteedTiers
    tiers = GuaranteedTiers.from_calibration(load_calibration(model))
    val, timing = _records(model, "val")
    test, _ = _records(model, "test")
    bands = fit([r for r in val if r["conf"] >= tiers.tau_review], tiers.tau_review)
    costs = {"geometry_check": 0.0, "shadow_check": _shadow_cost_ms(model),
             "zoom_relook": float(timing.get("single_ms_per_cand") or 400.0)}
    data = {"model": model, "created": date.today().isoformat(), "fit_split": "val", "verify_split": "test",
            "band_edges": list(BAND_EDGES), "bands": bands, "thresholds": THRESHOLDS,
            "relook": RELOOK_KEY, "costs_ms": costs, "use": True,
            "assumptions": ["shadow and re-look are conditionally independent given the truth within a band",
                            "an opposite-side pass is modelled as one more shadow observation (no paired passes "
                            "in the data to measure it)",
                            "thresholds accept 0.60 / watch 0.15 are design constants, not fit"]}
    em = EvidenceModel(data)
    data["validation"] = replay(em, val, tiers)
    data["verification"] = replay(em, test, tiers)
    v = data["verification"]["log_loss"]
    # 1) the PROBABILITY shown to people: gated on test log-loss (registered first)
    data["use_for_belief"] = v["with_opencv"] <= v["without_opencv"]
    data["belief_reason"] = (f"test log-loss {v['without_opencv']} (detector only) -> {v['with_opencv']} (with the "
                             f"OpenCV evidence): " + ("better calibrated - people see the evidence-updated P(pot)"
                                                      if data["use_for_belief"] else
                                                      "not better calibrated - people keep seeing the detector-only "
                                                      "calibrated P(pot)"))
    # 2) the ACTIONS: registered after (1) failed, judged on a split never used before (FRESH_SPLIT)
    fresh = REPO / "runs" / "calib" / f"{model}_{FRESH_SPLIT}.json"
    if fresh.exists():
        fr, _ = _records(model, FRESH_SPLIT)
        data["fresh"] = replay(em, fr, tiers)
        data["fresh"]["split"] = FRESH_SPLIT
        ok, why = action_criteria(data["fresh"])
        data["use_for_actions"], data["actions_reason"] = ok, why
    else:
        data["use_for_actions"], data["actions_reason"] = False, f"fresh split {FRESH_SPLIT} not measured yet"
    data["fresh_history"] = FRESH_HISTORY
    data["use"] = data["use_for_actions"]
    return data


FRESH_SPLIT = "test_xsonar"     # another sonar, 555 frames, disjoint from train / val / test
# Every evaluation of the registered action criteria on the fresh split is kept - failures included.
FRESH_HISTORY = [
    {"run": 1, "date": "2026-10-07", "result": "FAILED",
     "numbers": "ACCEPT precision 0.86 -> 0.632; real targets left in WATCH 127 -> 147; AUC 0.648 -> 0.625",
     "harness": "the replay measured a shadow on every Contact_*_sslo crop with a GUESSED orientation ('top'); the "
                "runtime never does that (orientation unknown -> no shadow), so run 1 scored a measurement the agent "
                "cannot make. The replay was fixed to follow the runtime's orientation rule and re-run ONCE (run 2)."},
    {"run": 2, "date": "2026-10-07", "result": "FAILED",
     "numbers": "ACCEPT precision 0.86 -> 0.848; real targets left in WATCH 127 -> 161; AUC 0.648 -> 0.644",
     "harness": "runtime-faithful replay (only the zoom re-look is available on these crops). The re-look ratios fit "
                "on crab-pot sonograms do not transfer to this sonar. No further runs."},
]


def action_criteria(r: dict) -> tuple[bool, str]:
    """Registered BEFORE the fresh split was measured (2026-10-07): the evidence-driven actions are
    used only if, on the fresh split, (a) ACCEPT is at least as precise WITH the OpenCV evidence as
    without, and (b) no more real targets are left at the bottom of the queue (WATCH)."""
    a1, a0 = r["actions"]["with_opencv"]["accept"], r["actions"]["without_opencv"]["accept"]
    w1, w0 = r["actions"]["with_opencv"]["watch"]["real"], r["actions"]["without_opencv"]["watch"]["real"]
    pa = (a1["precision"] or 0) >= (a0["precision"] or 0)
    pw = w1 <= w0
    why = (f"on the fresh split: ACCEPT precision {a0['precision']} -> {a1['precision']} "
           f"({'ok' if pa else 'FAILED'}); real targets left in WATCH {w0} -> {w1} ({'ok' if pw else 'FAILED'})")
    return pa and pw, why


def report(d: dict) -> str:
    L = []
    A = L.append
    A(f"# Evidence model — {d['model']} (OpenCV evidence → agent belief → action)\n")
    A(f"_`python -m src.agentic.evidence_model` · fit on **{d['fit_split']}** (held-out recordings), verified once on "
      f"**{d['verify_split']}** (unique crab-pot frames) · {d['created']} · regenerated by the script — do not edit_\n")
    A("## Likelihood ratios (validation)\n")
    A("| detector band | n (real) | shadow present | shadow absent | re-look re-fires | re-look silent |")
    A("|---|---|---|---|---|---|")
    for b in d["bands"]:
        hi = f"{b['hi']:.2f}" if b["hi"] is not None else "1"
        A(f"| {b['lo']:.2f}–{hi} | {b['n']} ({b['n_tp']}) | ×{b['shadow']['present']:.2f} | ×{b['shadow']['absent']:.2f} "
          f"| ×{b['relook']['present']:.2f} | ×{b['relook']['absent']:.2f} |")
    A("\nA ratio > 1 raises the odds that the find is a real pot, < 1 lowers them.\n")
    for key, title in (("validation", "validation (fit)"), ("verification", "test (verified once)")):
        r = d[key]
        A(f"## Replay of the active policy — {title}\n")
        A(f"{r['candidates']} candidates ≥ τ_review, {r['real_pots']} real pots.\n")
        A("| | WITHOUT OpenCV evidence (detector only) | WITH OpenCV evidence |")
        A("|---|---|---|")
        for a in ("accept", "review", "watch"):
            w0, w1 = r["actions"]["without_opencv"][a], r["actions"]["with_opencv"][a]
            f = lambda x: f"{x['n']} ({x['real']} real{', ' + format(x['precision'], '.0%') if x['precision'] is not None else ''})"
            A(f"| {a.upper()} | {f(w0)} | {f(w1)} |")
        for k in ("pots_in_first_25pct", "pots_in_first_50pct"):
            q = r[k]
            A(f"| real pots in the first {q['cards']} cards of the queue | {q['without_opencv']} | {q['with_opencv']} |")
        A(f"| log-loss (lower = better) | {r['log_loss']['without_opencv']} | {r['log_loss']['with_opencv']} |")
        A(f"| Brier score | {r['brier']['without_opencv']} | {r['brier']['with_opencv']} |")
        A(f"| AUC, real vs false | {r['auc']['without_opencv']} | {r['auc']['with_opencv']} |")
        A("")
        A(f"- Candidates whose action changed because of OpenCV evidence: **{r['changed_action']} of {r['candidates']}**.")
        A(f"- Tools chosen by value of information: shadow check on {r['tools']['shadow_check']}, zoom re-look on "
          f"{r['tools']['zoom_relook']} ({r['relook_share']:.0%}); {r['no_tool_needed']} needed no tool at all.")
        A(f"- Evidence conflicts: {r['conflicts']} ({r['conflicts_resolved']} resolved in-frame); opposite-side pass "
          f"requested for {r['resurvey_requested']}.\n")
    if d.get("fresh"):
        r = d["fresh"]
        A(f"## Fresh split — `{r['split']}` (another sonar; never used to fit or tune anything)\n")
        A(f"{r['candidates']} candidates ≥ τ_review, {r['real_pots']} real targets.\n")
        A("| | WITHOUT OpenCV evidence | WITH OpenCV evidence |")
        A("|---|---|---|")
        for a in ("accept", "review", "watch"):
            w0, w1 = r["actions"]["without_opencv"][a], r["actions"]["with_opencv"][a]
            f = lambda x: f"{x['n']} ({x['real']} real{', ' + format(x['precision'], '.0%') if x['precision'] is not None else ''})"
            A(f"| {a.upper()} | {f(w0)} | {f(w1)} |")
        A(f"| AUC, real vs false | {r['auc']['without_opencv']} | {r['auc']['with_opencv']} |")
        A(f"\n- Actions changed by OpenCV evidence: **{r['changed_action']} of {r['candidates']}**; zoom re-look on "
          f"{r['tools']['zoom_relook']} ({r['relook_share']:.0%}); conflicts {r['conflicts']}.\n")
    A("## Fresh-split history (every run of the registered criteria)\n")
    for h in d.get("fresh_history") or []:
        A(f"- **Run {h['run']} ({h['date']}): {h['result']}** — {h['numbers']}. {h['harness']}")
    A("")
    A("## Decisions\n")
    A(f"1. **Probability shown to people** (gate registered first: test log-loss) — "
      f"**{'evidence-updated' if d['use_for_belief'] else 'detector-only'}**. {d['belief_reason']}.")
    A(f"2. **Agent actions** (gate registered after (1) failed, judged on the fresh split only) — "
      f"**{'evidence-driven' if d['use_for_actions'] else 'detector-only by default'}**. {d['actions_reason']}.\n")
    if not d["use_for_actions"]:
        A("So the evidence-driven actions are **off by default**. The studio can switch them on as **Active vision "
          "(experimental — not validated)**: the loop is real (every tool call, belief update and conflict is "
          "measured and logged), but its benefit is not established beyond the crab-pot test split. What stays on "
          "by default is validated or physical: Stage-1 geometry -> geolocation, routes and re-survey passes "
          "(STUDY-12), the water-column rule, information-ranked passes and human control.\n")
    A("Costs used to order the tools (measured on this machine): " + ", ".join(
        f"`{k}` {v:g} ms" for k, v in d["costs_ms"].items()) + ".\n")
    A("Assumptions: " + "; ".join(d["assumptions"]) + ".\n")
    A("The calibrated tiers are unchanged: every candidate ≥ τ_review is still a REVIEW card, so the recall "
      "promise in `calibration.json` holds exactly as before. The evidence changes the belief, the order of the "
      "queue, the action and the re-survey plan — never whether a person sees a find.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    d = build(a.model)
    model_path(a.model).write_text(json.dumps(d, indent=2), encoding="utf-8")
    doc = REPO / "docs" / f"evidence_model_{a.model.lower().replace('-', '')}.md"
    doc.write_text(report(d), encoding="utf-8")
    print(report(d))
    print(f"-> {model_path(a.model)}\n-> {doc}")


if __name__ == "__main__":
    main()
