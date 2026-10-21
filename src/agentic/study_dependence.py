"""
study_dependence.py — STUDY-21: does the recall promise survive correlated frames?

The Clopper-Pearson bound behind "≥ 79% of pots reach a person (95% confidence)" treats every pot as an
independent trial. Pots are not independent: port and starboard frames of the same pings show the same
seabed, consecutive frames of a recording share a site, a gain setting and sometimes an object, and a
recording shares one sonar set-up. Correlation makes the real uncertainty larger than the bound says.
(``guarantees.py`` claimed that ``calibrate.py`` reports a frame-group bootstrap; it did not — this study
is that check.)

**Protocol (fixed in this file and committed before any number below was computed)**

1. Data: the cached per-frame detections ``calibrate.py`` and STUDY-18 already scored
   (``runs/calib/<MODEL>_val.json``, ``_test.json`` and their ``_tiles`` twins) — no new inference. The
   test split was scored once (STUDY-18); here its cached predictions are only broken down by recording.
   No threshold is fit or changed by this study. Modes: ``standard`` (τ_review from ``calibration.json``)
   and ``look_harder`` (``modes.look_harder``, the STUDY-18 merge of the tile pass).
2. Unit of dependence: a **block** = the frames of one recording whose ping index (``_port_00012`` /
   ``_star_00012``) falls in the same run of ``b`` consecutive indices — port and starboard of the same
   pings always share a block. ``b`` ∈ {1, 4, 8}.
3. **A1 — block bootstrap (calibration split, the gate).** At each mode's shipped τ: resample blocks with
   replacement within each recording (each recording keeps its number of blocks), B = 20,000, seed 0;
   recall = reached pots ÷ all pots of the resample; the 5th percentile is the dependence-aware 95% lower
   bound. Also reported: the CP lower bound at the same τ and the design effect (bootstrap variance ÷
   binomial variance) → the effective number of independent pots.
4. **A2 — recordings (reported, no gate).** Recall at the shipped τ on each of the 3 calibration and 3
   test recordings with its own CP 95% lower bound, and leave-one-recording-out on the calibration
   split: the promise refit on two recordings (``best_recall_guarantee``, as ``calibrate.fit`` does) and
   checked on the third. Also reported: a two-stage bootstrap (recordings, then blocks within them) —
   with 3 recordings this bound is itself very uncertain, so it is a sensitivity check, not a gate.
5. **Decision rule R1 (fixed now).** For each mode: if the A1 lower bound with the largest block
   (``b`` = 8) is ≥ the promise, the promise stands as stated. If it is lower, the promise is restated
   to that bound rounded down to 0.01 — ``calibration.json`` keeps the Clopper-Pearson value as
   ``recall_promise_cp`` and ``recall_promise`` becomes the restated one, so the studio, the API, MCP
   and the mission plan all show it; τ is unchanged, and the test verification is re-compared against
   the restated promise (no test number is recomputed).
6. Wording rule (whatever R1 says): the headline says **promise**, not guarantee, and names its
   assumption — the pots of a new survey are exchangeable with the calibration pots; A2's per-recording
   spread is reported next to it, because the promise is a pooled rate over recordings like the
   calibration ones, not a per-survey guarantee.

    python -m src.agentic.study_dependence            # A1 + A2 → docs/reports/dependence.md
    python -m src.agentic.study_dependence --write    # + apply R1 to calibration.json (only if R1 restates)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from src.detection.calibration import DEFAULT_MODEL, load_calibration, save_calibration
from src.detection.infer import _iou
from .calibrate import FLOOR, MATCH_IOU
from .guarantees import best_recall_guarantee, cp_upper
from .study_look_harder import merged_frames

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "runs" / "calib"
REPORT = REPO / "docs" / "reports" / "dependence.md"
BLOCKS = (1, 4, 8)
B, SEED, DELTA = 20000, 0, 0.05
_IDX = re.compile(r"_(port|star)_(\d+)")


# ---------------------------------------------------------------------------------------- data
def load_split(model: str, split: str, cls_name: str) -> dict[str, list[dict]]:
    """The cached frames of one split for both modes (no inference: the caches must exist)."""
    base, tiles = CACHE / f"{model}_{split}.json", CACHE / f"{model}_{split}_tiles.json"
    for p in (base, tiles):
        if not p.exists():
            raise SystemExit(f"missing cache {p.relative_to(REPO)}: run python -m src.agentic.calibrate and "
                             "python -m src.agentic.study_look_harder --verify first (this study runs no inference)")
    frames = json.loads(base.read_text())["frames"]
    return {"standard": frames,
            "look_harder": merged_frames(frames, json.loads(tiles.read_text())["frames"], cls_name)}


def pot_rows(frames: list[dict]) -> list[dict]:
    """Per frame: recording, ping index and the best score of each true pot (0 = never proposed)."""
    out = []
    for fr in frames:
        m = _IDX.search(fr.get("frame") or fr["name"])
        s = [max([c["conf"] for c in fr["cands"] if _iou(tuple(c["bbox"]), tuple(g)) >= MATCH_IOU] or [0.0])
             for g in fr["gt"]]
        out.append({"recording": fr.get("recording") or "?", "index": int(m.group(2)) if m else None,
                    "name": fr["name"], "scores": s})
    return out


def blocks_of(rows: list[dict], tau: float, b: int) -> dict[str, np.ndarray]:
    """Per recording: an array of (reached, pots) per block of ``b`` consecutive ping indices."""
    per: dict[str, dict] = {}
    for r in rows:
        key = r["index"] // b if r["index"] is not None else r["name"]
        blk = per.setdefault(r["recording"], {}).setdefault(key, [0, 0])
        blk[0] += sum(s >= tau for s in r["scores"])
        blk[1] += len(r["scores"])
    return {rec: np.array(list(bl.values()), float) for rec, bl in per.items()}


def recall(rows: list[dict], tau: float) -> tuple[int, int]:
    s = [x for r in rows for x in r["scores"]]
    return int(sum(x >= tau for x in s)), len(s)


def cp_lcb(k: int, n: int) -> float:
    return 1.0 - cp_upper(n - k, n, DELTA) if n else float("nan")


# ---------------------------------------------------------------------------------------- A1
def block_bootstrap(blocks: dict[str, np.ndarray], reps: int = B, seed: int = SEED) -> np.ndarray:
    """Recall of ``reps`` resamples: blocks drawn with replacement within each recording."""
    rng = np.random.default_rng(seed)
    hit, tot = np.zeros(reps), np.zeros(reps)
    for rec in sorted(blocks):
        arr = blocks[rec]
        idx = rng.integers(0, len(arr), size=(reps, len(arr)))
        hit += arr[idx, 0].sum(axis=1)
        tot += arr[idx, 1].sum(axis=1)
    return hit / np.maximum(tot, 1)


def dependence_bound(frames: list[dict], tau: float, b: int = BLOCKS[-1]) -> float:
    """The R1 bound for frames in ``calibrate.py``'s format: the block-bootstrap 95% lower bound of
    recall at ``tau`` (blocks of ``b`` ping indices, resampled within recordings)."""
    return float(np.percentile(block_bootstrap(blocks_of(pot_rows(frames), tau, b)), 100 * DELTA))


def apply_r1(promise: float, frames: list[dict], tau: float) -> dict:
    """Rule R1 as the fitters use it (``calibrate.py``, ``study_look_harder.py``): the promise stands if the
    dependence-aware bound reaches it, else it is restated to that bound (rounded down to 0.01); τ stays.
    Returns ``recall_promise`` (the one in force) + the record of how it was set."""
    lcb = round(dependence_bound(frames, tau), 4)
    out = {"recall_promise": promise, "dependence_lcb": lcb,
           "recall_promise_basis": f"Clopper-Pearson; STUDY-21 block bootstrap (b = {BLOCKS[-1]}) {lcb:.3f} - stands"}
    if lcb < promise - 1e-9:
        out.update(recall_promise=float(np.floor(lcb * 100 + 1e-9) / 100), recall_promise_cp=promise,
                   recall_promise_basis=f"STUDY-21 block bootstrap (b = {BLOCKS[-1]}): {REPORT.relative_to(REPO).as_posix()}")
    return out


def two_stage(blocks: dict[str, np.ndarray], reps: int = B, seed: int = SEED) -> np.ndarray:
    """Recordings drawn with replacement, then blocks within each drawn recording (sensitivity check)."""
    rng = np.random.default_rng(seed)
    recs = sorted(blocks)
    out = np.empty(reps)
    for i in range(reps):
        hit = tot = 0.0
        for j in rng.integers(0, len(recs), len(recs)):
            arr = blocks[recs[j]]
            pick = arr[rng.integers(0, len(arr), len(arr))]
            hit += pick[:, 0].sum()
            tot += pick[:, 1].sum()
        out[i] = hit / max(tot, 1)
    return out


def a1(rows: list[dict], tau: float) -> dict:
    k, n = recall(rows, tau)
    p = k / n
    out = {"tau": tau, "pots": n, "reached": k, "recall": round(p, 4), "cp_lcb": round(cp_lcb(k, n), 4), "blocks": {}}
    for b in BLOCKS:
        bl = blocks_of(rows, tau, b)
        r = block_bootstrap(bl)
        var_b, var_bin = float(r.var()), p * (1 - p) / n
        deff = var_b / var_bin if var_bin else float("nan")
        out["blocks"][b] = {"n_blocks": int(sum(len(v) for v in bl.values())),
                            "lcb": round(float(np.percentile(r, 5)), 4), "design_effect": round(deff, 3),
                            "effective_pots": round(n / deff, 1) if deff else None}
    out["two_stage_b8_lcb"] = round(float(np.percentile(two_stage(blocks_of(rows, tau, BLOCKS[-1])), 5)), 4)
    return out


# ---------------------------------------------------------------------------------------- A2
def per_recording(rows: list[dict], tau: float, promise: float) -> list[dict]:
    out = []
    for rec in sorted({r["recording"] for r in rows}):
        k, n = recall([r for r in rows if r["recording"] == rec], tau)
        out.append({"recording": rec, "frames": sum(r["recording"] == rec for r in rows), "pots": n,
                    "recall": round(k / n, 4) if n else None, "cp_lcb": round(cp_lcb(k, n), 4) if n else None,
                    "promise": promise, "meets_promise": bool(n and k / n >= promise)})
    return out


def loro(rows: list[dict]) -> list[dict]:
    """Leave one calibration recording out: refit the promise (calibrate.fit's rule) on the other two."""
    out = []
    for rec in sorted({r["recording"] for r in rows}):
        fit_rows = [r for r in rows if r["recording"] != rec]
        pots = np.array([x for r in fit_rows for x in r["scores"]], float)
        promise, tau = best_recall_guarantee(pots, FLOOR, DELTA)
        tau = tau if tau is not None else FLOOR
        k, n = recall([r for r in rows if r["recording"] == rec], tau)
        out.append({"held_out": rec, "fit_on": sorted({r["recording"] for r in fit_rows}), "promise": promise,
                    "tau": tau, "held_out_recall": round(k / n, 4), "held_out_pots": n,
                    "held": bool(k / n >= promise)})
    return out


# ---------------------------------------------------------------------------------------- report
def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def report(model: str, res: dict) -> str:
    L = [f"# STUDY-21 — does the recall promise survive correlated frames? ({model})", "",
         "_`python -m src.agentic.study_dependence` · protocol fixed in the module docstring and committed "
         "before any number was computed · cached detections only (no inference, no threshold changed) · "
         f"B = {B:,} resamples, seed {SEED} · 95% one-sided bounds_", "",
         "The Clopper-Pearson (CP) bound behind each promise treats every pot as an independent trial. Here "
         "the frames are grouped into **blocks** — runs of `b` consecutive ping indices of one recording, port "
         "and starboard of the same pings together — and the blocks are resampled within each recording.", "",
         "## A1 — the gate: block bootstrap on the calibration split", "",
         "| mode | promise | τ | pots | recall at τ | CP bound | b = 1 | b = 4 | b = 8 (gate) | design effect "
         "(b = 8) | effective pots | two-stage (sensitivity) | R1 |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
    for name, m in res["modes"].items():
        x, bl = m["a1"], m["a1"]["blocks"]
        L.append(f"| {name} | ≥ {_pct(m['promise'])} | {x['tau']:.3f} | {x['pots']} | {_pct(x['recall'])} | "
                 f"{_pct(x['cp_lcb'])} | {_pct(bl[1]['lcb'])} | {_pct(bl[4]['lcb'])} | **{_pct(bl[8]['lcb'])}** | "
                 f"{bl[8]['design_effect']:.2f} | {bl[8]['effective_pots']} | {_pct(x['two_stage_b8_lcb'])} | "
                 f"**{m['r1']}** |")
    L += ["", "R1: the promise stands if the `b` = 8 bound is ≥ the promise; otherwise it is restated to that bound "
          "(rounded down to 0.01) everywhere, τ unchanged. The two-stage bound resamples the 3 recordings too; "
          "with 3 recordings it is itself very uncertain, so it is reported, not gated.", "",
          "## A2 — recordings", "",
          "Recall at each mode's shipped τ on every recording (the test predictions were scored once in STUDY-18; "
          "here they are only broken down):", "",
          "| mode | split | recording | frames | pots | recall | CP bound | meets the promise in force (after R1) |",
          "|---|---|---|--:|--:|--:|--:|:--:|"]
    for name, m in res["modes"].items():
        for split in ("calibration", "test"):
            for r in m["per_recording"][split]:
                L.append(f"| {name} | {split} | {r['recording']} | {r['frames']} | {r['pots']} | {_pct(r['recall'])} | "
                         f"{_pct(r['cp_lcb'])} | {'yes' if r['meets_promise'] else '**no**'} (≥ {_pct(r['promise'])}) |")
    L += ["", "Leave one calibration recording out — the promise refit on the other two (the `calibrate.fit` rule) "
          "and checked on the held-out recording:", "",
          "| mode | held out | fit on | promise refit | τ | held-out recall | held |", "|---|---|---|--:|--:|--:|:--:|"]
    for name, m in res["modes"].items():
        for r in m["loro"]:
            L.append(f"| {name} | {r['held_out']} ({r['held_out_pots']} pots) | {', '.join(r['fit_on'])} | "
                     f"≥ {_pct(r['promise'])} | {r['tau']:.3f} | {_pct(r['held_out_recall'])} | "
                     f"{'yes' if r['held'] else '**no**'} |")
    L += ["", "## What this means", ""] + [f"- {s}" for s in res["summary"]] + [
          "", "---", "_Regenerated by the script; do not edit by hand._"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="STUDY-21: does the recall promise survive correlated frames?")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--write", action="store_true", help="apply rule R1 to calibration.json (only if it restates)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cal = load_calibration(a.model)
    cls = cal.guaranteed_class
    g = cal.tiers["guarantees"]
    lh = cal.modes.get("look_harder") or {}
    shipped = {"standard": (float(g.get("recall_promise_cp") or g["recall_promise"]), float(cal.tiers["tau_review"])),
               "look_harder": (float(lh.get("recall_promise_cp") or lh["recall_promise"]), float(lh["tau_review"]))}
    data = {"calibration": load_split(a.model, "val", cls), "test": load_split(a.model, "test", cls)}
    res: dict = {"model": a.model, "modes": {}, "summary": []}
    for name, (promise, tau) in shipped.items():
        rows = {s: pot_rows(data[s][name]) for s in data}
        x = a1(rows["calibration"], tau)
        lcb = x["blocks"][BLOCKS[-1]]["lcb"]
        stands = lcb >= promise - 1e-9
        restated = promise if stands else float(np.floor(lcb * 100 + 1e-9) / 100)
        m = {"promise": promise, "a1": x, "r1": "stands" if stands else f"restated to ≥ {_pct(restated)}",
             "restated_promise": None if stands else restated,
             "per_recording": {s: per_recording(rows[s], tau, restated) for s in rows},
             "loro": loro(rows["calibration"])}
        res["modes"][name] = m
        below = [f"{r['recording']} {_pct(r['recall'])}" for s in rows for r in m["per_recording"][s]
                 if not r["meets_promise"]]
        held = sum(r["held"] for r in m["loro"])
        n_rec = sum(len(m["per_recording"][s]) for s in rows)
        res["summary"].append(
            f"**{name}** (≥ {_pct(promise)}, τ {tau:.3f}): with blocks of 8 ping indices the 95% bound on the "
            f"calibration split is {_pct(lcb)} (CP {_pct(x['cp_lcb'])}, design effect "
            f"{x['blocks'][8]['design_effect']:.2f}) → the promise **{m['r1']}**. Per recording at ≥ "
            f"{_pct(restated)}: {n_rec - len(below)} of {n_rec} meet it"
            + (f" (below: {', '.join(below)})" if below else "")
            + f"; leave-one-recording-out refits held on {held} of {len(m['loro'])}.")
    res["summary"].append("The promise is a pooled rate over pots from recordings like the calibration ones — "
                          "not a per-survey guarantee; the per-recording rows show the spread a single survey can see.")
    REPORT.write_text(report(a.model, res), encoding="utf-8")
    out = REPO / "runs" / "dependence" / f"{a.model}_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    print("\n".join(res["summary"]))
    print(f"wrote {REPORT.relative_to(REPO)} and {out.relative_to(REPO)}")
    if a.write:
        upd: dict = {}
        for name, m in res["modes"].items():
            if m["restated_promise"] is None:
                continue
            note = {"recall_promise_cp": m["promise"], "recall_promise": m["restated_promise"],
                    "recall_promise_basis": f"STUDY-21 block bootstrap (b = 8): {REPORT.relative_to(REPO).as_posix()}"}
            if name == "standard":
                upd.setdefault("tiers", {})["guarantees"] = note
            else:
                upd.setdefault("modes", {})["look_harder"] = note
        if upd:
            print(f"R1 restates: {json.dumps(upd)} -> {save_calibration(upd, model=a.model)}")
        else:
            print("R1: every promise stands - calibration.json unchanged")


if __name__ == "__main__":
    main()
