"""
study_look_harder.py — STUDY-18: can the agent keep a STRONGER recall promise by looking harder?
(Registered as "STUDY-17" in commit 94a3af5; renumbered because STUDY-17 is the COOL benchmark registered
on 2026-10-19 in ``docs/experiments.md``. The protocol text is unchanged.)

The recall promise is capped by the detector's **recall ceiling**: with EXP-003, 15% of pots are never
proposed at any score (``docs/calibration_exp003.md``), so no threshold can promise 90%. EXP-003 was
trained on full frames **and** 2×2 tiles (``build_tiles.py``), so a second look at the tile scale is
one the detector has seen. This study asks whether that second look raises the promise the agent can
keep, and at what cost in review cards and compute. The agent then chooses between the two
perception modes from the operator's goal (``core.py``).

**Modes**

* ``standard``    — the full-frame look (1 forward pass per frame): the shipped tiers.
* ``look_harder`` — the full-frame look + :meth:`Perceptor.look_harder` (2×2 tiles, ``TILE_FRAC``,
  4 more passes), folded in by :func:`perception.merge_candidates` (IoU ≥ ``MERGE_IOU``). The study
  scores exactly the functions the product runs.

**Protocol (fixed in this file and committed before the verification split was scored)**

1. Splits = ``calibrate.py``'s for EXP-003: calibration = v2b ``val`` ``*wcp_ss_*`` (163 unique frames,
   Rec10/12/16); verification = v2b ``test`` ``*wcp_ss_*`` (199 unique frames, Rec8/9/14). Unique
   frames only; match IoU 0.3; detector floor 0.05; δ = 0.05.
2. Fit on calibration only, as ``calibrate.fit`` does: the requested 90% first; if the ceiling makes
   it impossible, the strongest promise the Clopper-Pearson bound supports (``best_recall_guarantee``)
   and its highest threshold τ. P(pot) bins for the queue are fit the same way.
3. Gates:
   * **G1 (calibration)** — the look-harder promise must be strictly stronger than the standard
     promise in ``calibration.json``;
   * **G2 (verification, scored once)** — the look-harder promise must hold: the empirical recall at
     its τ on the verification split ≥ the promise.
   Only a mode that passes both is written to ``calibration.json`` (``modes.look_harder``); a failed
   gate is reported, and the mode stays off.
4. Disclosed before the run: the mode design (tiles on every frame, merge rule) was chosen after
   looking at the calibration split. Per-frame triggers (tile only frames whose first look found
   something, or nothing) were explored there and dropped: the pots only the tile pass proposed (9 in
   8 frames with an earlier merge variant, 8 in 7 frames with the shipped rule) had no first-look
   signal that predicted them, so a trigger would be fit to noise.

Usage:
    python -m src.agentic.study_look_harder                 # calibration split only (exploration)
    python -m src.agentic.study_look_harder --verify        # + the verification split, once
    python -m src.agentic.study_look_harder --verify --write   # + modes.look_harder (only if G1+G2 pass)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.detection.calibration import DEFAULT_MODEL, load_calibration, save_calibration
from src.detection.infer import DEFAULT_ONNX, Detection, _iou
from .calibrate import FLOOR, MATCH_IOU, collect, list_frames, p_pot_bins
from .guarantees import best_recall_guarantee, cp_upper, default_grid, tau_review
from .perception import Perceptor, merge_candidates

REPO = Path(__file__).resolve().parents[2]
V2B = REPO / "DATASET" / "03_yolo_ready_dataset_v2b"
PATTERN = "*wcp_ss_*.jpg"
TILE_FRAC = 0.6          # build_tiles.py --tile-frac default: the scale EXP-003 trained on
MERGE_IOU = 0.5          # a tile find on a first-look box of its class is the same object
ALPHA, DELTA = 0.10, 0.05
REPORT = REPO / "docs" / "reports" / "look_harder.md"


# ---------------------------------------------------------------------------------------- collect
def collect_tiles(root: Path, cls_name: str, cache: Path, onnx: Path = DEFAULT_ONNX) -> dict:
    """The tile pass on every unique frame of a split (cached): ``{name: [cand, ...]}`` + timing."""
    frames = list_frames(root, PATTERN)
    key = f"{onnx}|{root}|{PATTERN}|{len(frames)}|tiles{TILE_FRAC}|floor{FLOOR}|v1"
    if cache.exists():
        blob = json.loads(cache.read_text())
        if blob.get("key") == key:
            print(f"  [cache] {cache.name}: {len(blob['frames'])} frames")
            return blob
    perc = Perceptor(onnx)
    perc.warmup()
    out, ms, t0 = {}, [], time.time()
    for i, p in enumerate(frames):
        im = cv2.imread(str(p))
        if im is None:
            continue
        t = time.perf_counter()
        dets = perc.look_harder(im, TILE_FRAC, cls_name)
        ms.append((time.perf_counter() - t) * 1000)
        out[p.name] = [{"bbox": list(d.bbox), "conf": round(d.conf, 4)} for d in dets]
        if (i + 1) % 40 == 0:
            print(f"  {root.name}: {i + 1}/{len(frames)} frames ({time.time() - t0:.0f}s)")
    blob = {"key": key, "frames": out, "tile_pass_ms_median": round(float(np.median(ms)), 1) if ms else None}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(blob))
    return blob


def merged_frames(frames: list[dict], tiles: dict, cls_name: str) -> list[dict]:
    """Each frame's candidates after :func:`merge_candidates` — in ``calibrate.py``'s frame format."""
    out = []
    for fr in frames:
        base = [Detection(tuple(c["bbox"]), 0, cls_name, c["conf"]) for c in fr["cands"]]
        extra = [Detection(tuple(c["bbox"]), 0, cls_name, c["conf"]) for c in tiles.get(fr["name"], [])]
        dets, src = merge_candidates(base, extra, MERGE_IOU)
        out.append({**fr, "cands": [{"bbox": list(d.bbox), "conf": round(d.conf, 4), "source": s}
                                    for d, s in zip(dets, src)]})
    return out


# ---------------------------------------------------------------------------------------- measure
def pot_scores(frames: list[dict]) -> np.ndarray:
    """Best candidate score per true pot (IoU ≥ MATCH_IOU), 0 if never proposed."""
    s = []
    for fr in frames:
        for g in fr["gt"]:
            ov = [c["conf"] for c in fr["cands"] if _iou(tuple(c["bbox"]), tuple(g)) >= MATCH_IOU]
            s.append(max(ov) if ov else 0.0)
    return np.asarray(s, float)


def cards_per_frame(frames: list[dict], tau: float) -> float:
    return sum(sum(c["conf"] >= tau for c in fr["cands"]) for fr in frames) / max(1, len(frames))


def fit(frames: list[dict]) -> dict:
    """``calibrate.fit``'s recall side: the requested 1-α, else the strongest supportable promise."""
    pots = pot_scores(frames)
    grid = default_grid(FLOOR, 1.0, int(round((1 - FLOOR) / 0.005)) + 1)
    tau = tau_review(pots, alpha=ALPHA, delta=DELTA, grid=grid)
    ok = tau is not None
    promise = 1 - ALPHA
    if not ok:
        promise, tau = best_recall_guarantee(pots, FLOOR, DELTA)
        tau = tau if tau is not None else FLOOR
    return {"pots": int(len(pots)), "frames": len(frames), "ceiling": round(float(np.mean(pots >= FLOOR)), 4),
            "requested_achievable": ok, "recall_promise": round(float(promise), 4), "tau_review": float(tau),
            "recall_at_tau": round(float(np.mean(pots >= tau)), 4),
            "cards_per_frame": round(cards_per_frame(frames, tau), 3)}


def verify(frames: list[dict], tau: float, promise: float) -> dict:
    pots = pot_scores(frames)
    miss = int((pots < tau).sum())
    rec = 1 - miss / max(1, len(pots))
    return {"pots": int(len(pots)), "frames": len(frames), "ceiling": round(float(np.mean(pots >= FLOOR)), 4),
            "recall": round(rec, 4), "recall_lcb": round(1 - cp_upper(miss, len(pots), DELTA), 4),
            "promise_held": bool(rec >= promise), "cards_per_frame": round(cards_per_frame(frames, tau), 3)}


def ceiling_gain_ci(std: list[dict], lh: list[dict], reps: int = 2000, seed: int = 0) -> list[float]:
    """Paired frame bootstrap of the recall-ceiling gain (look_harder − standard), mean + 95% CI."""
    a = [pot_scores([f]) for f in std]
    b = [pot_scores([f]) for f in lh]
    rng = np.random.default_rng(seed)
    n, g = len(std), []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        sa = np.concatenate([a[i] for i in idx]) if n else np.array([])
        sb = np.concatenate([b[i] for i in idx]) if n else np.array([])
        if len(sa):
            g.append(float(np.mean(sb >= FLOOR) - np.mean(sa >= FLOOR)))
    return [round(float(np.mean(g)), 4), round(float(np.percentile(g, 2.5)), 4), round(float(np.percentile(g, 97.5)), 4)]


def recovered(std: list[dict], lh: list[dict]) -> dict:
    """Pots only the tile pass proposed, and the tile-only candidates that hit no pot."""
    rec = frames_hit = fa = 0
    for fs, fl in zip(std, lh):
        a, b = pot_scores([fs]), pot_scores([fl])
        k = int(((a < FLOOR) & (b >= FLOOR)).sum())
        rec += k
        frames_hit += k > 0
        fa += sum(c.get("source") == "tiles" and not any(_iou(tuple(c["bbox"]), tuple(g)) >= MATCH_IOU for g in fl["gt"])
                  for c in fl["cands"])
    return {"pots_recovered": rec, "frames_with_recovered": frames_hit, "tile_only_false_alarms": fa}


# ---------------------------------------------------------------------------------------- report
def _pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def report(model: str, res: dict) -> str:
    c, v = res["cal"], res.get("ver")
    L = [f"# STUDY-18 — look harder: a stronger recall promise from a second look ({model})\n",
         "_Generated by `python -m src.agentic.study_look_harder --verify` · protocol fixed in the module "
         "docstring and committed before the verification split was scored · unique frames · match IoU "
         f"{MATCH_IOU} · floor {FLOOR} · δ = {DELTA} (95% confidence)_\n",
         f"**Look harder** = the full-frame look + a 2×2 tile pass (tile side {TILE_FRAC:.0%} of the frame, the "
         f"scale EXP-003 was trained on; 4 more `cv2.dnn` passes), merged at IoU ≥ {MERGE_IOU} "
         "(`perception.merge_candidates`).\n",
         "## 1. Calibration split (fit)\n",
         f"{c['split']}\n",
         "| mode | recall ceiling | promise it supports | τ | recall at τ | cards / frame | forward passes / frame |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    for name, m in (("standard", c["standard"]), ("look_harder", c["look_harder"])):
        L.append(f"| {name} | {m['ceiling']:.3f} | **≥ {_pct(m['recall_promise'])}** | {m['tau_review']:.3f} | "
                 f"{m['recall_at_tau']:.3f} | {m['cards_per_frame']:.2f} | {1 if name == 'standard' else 5} |")
    g = c["ceiling_gain_ci"]
    r = c["recovered"]
    L += ["", f"Recall-ceiling gain, paired frame bootstrap: **{g[0]:+.3f}** (95% CI {g[1]:+.3f} .. {g[2]:+.3f}). "
          f"The tile pass proposed {r['pots_recovered']} pots the full-frame look never did, in "
          f"{r['frames_with_recovered']} frames; it also added {r['tile_only_false_alarms']} tile-only candidates "
          "that hit no labelled pot (some may be unlabelled pots).",
          "", f"**G1** — look-harder promise stronger than the standard promise ({_pct(c['standard']['recall_promise'])}): "
          f"**{'PASS' if res['gates']['G1'] else 'FAIL'}**\n"]
    if v:
        L += ["## 2. Verification split (scored once)\n", f"{v['split']}\n",
              "| mode | τ (from calibration) | recall ceiling | recall at τ (95% LCB) | promise | held | cards / frame |",
              "|---|--:|--:|--:|--:|:--:|--:|"]
        for name in ("standard", "look_harder"):
            m, f = v[name], c[name]
            L.append(f"| {name} | {f['tau_review']:.3f} | {m['ceiling']:.3f} | {m['recall']:.3f} ({m['recall_lcb']:.3f}) | "
                     f"≥ {_pct(f['recall_promise'])} | {'yes' if m['promise_held'] else '**no**'} | {m['cards_per_frame']:.2f} |")
        L += ["", f"**G2** — the look-harder promise holds on the verification split: "
              f"**{'PASS' if res['gates']['G2'] else 'FAIL'}**\n"]
    else:
        L += ["## 2. Verification split\n", "_Not scored yet (`--verify`)._\n"]
    L += ["## 3. Cost\n",
          f"- Tile pass, median on this machine: {res.get('tile_pass_ms')} ms per frame (4 forward passes), on top "
          "of the full-frame look.",
          "- Review cards per frame at each mode's own promise are in the tables; a stronger promise costs more cards.",
          "", "## 4. Decision\n", res["decision"], "",
          "Caveats: frames of one recording are correlated (the Clopper-Pearson bound assumes independence); "
          "the calibration split is 3 recordings. Labels are incomplete, so tile-only 'false alarms' are an upper "
          "bound.", "", "---", "_Regenerated by the script; do not edit by hand._"]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--cal-root", default=str(V2B / "val"))
    ap.add_argument("--ver-root", default=str(V2B / "test"))
    ap.add_argument("--verify", action="store_true", help="score the verification split (once)")
    ap.add_argument("--write", action="store_true", help="write modes.look_harder if G1 and G2 pass")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cal_json = load_calibration(a.model)
    cls = cal_json.guaranteed_class
    std_promise = float(cal_json.tiers["guarantees"]["recall_promise"])
    cache = REPO / "runs" / "calib"
    onnx = Path(a.onnx)

    def split(root: Path, desc: str) -> tuple[list[dict], list[dict], dict, str]:
        base = collect(root, PATTERN, cls, 0, cache / f"{a.model}_{root.name}.json", onnx)
        tiles = collect_tiles(root, cls, cache / f"{a.model}_{root.name}_tiles.json", onnx)
        frames = base["frames"]
        recs = sorted({f["recording"] for f in frames if f["recording"]})
        return frames, merged_frames(frames, tiles["frames"], cls), tiles, (
            f"{desc}: `{root.relative_to(REPO).as_posix()}` {PATTERN} — {len(frames)} unique frames, recordings {recs}")

    print("calibration split ...")
    c_std, c_lh, c_tiles, c_desc = split(Path(a.cal_root), "Calibration")
    fs, fl = fit(c_std), fit(c_lh)
    res = {"cal": {"split": c_desc, "standard": fs, "look_harder": fl,
                   "ceiling_gain_ci": ceiling_gain_ci(c_std, c_lh), "recovered": recovered(c_std, c_lh)},
           "tile_pass_ms": c_tiles.get("tile_pass_ms_median"),
           "gates": {"G1": fl["recall_promise"] > std_promise, "G2": None}}
    if a.verify:
        print("verification split (scored once) ...")
        v_std, v_lh, _, v_desc = split(Path(a.ver_root), "Verification")
        res["ver"] = {"split": v_desc, "standard": verify(v_std, fs["tau_review"], fs["recall_promise"]),
                      "look_harder": verify(v_lh, fl["tau_review"], fl["recall_promise"])}
        res["gates"]["G2"] = res["ver"]["look_harder"]["promise_held"]
    ok = bool(res["gates"]["G1"] and res["gates"]["G2"])
    res["decision"] = (
        f"**Ship** `look_harder` as a calibrated mode: by looking harder the agent may promise ≥ "
        f"{_pct(fl['recall_promise'])} instead of the standard ≥ {_pct(std_promise)}. `core.py` chooses it when the "
        "operator's goal and budgets allow (more review cards and 5 forward passes per frame)." if ok else
        ("**Not shipped**: a gate failed; the standard mode stays the only calibrated mode." if a.verify else
         "_Pending the verification split._"))
    REPORT.write_text(report(a.model, res), encoding="utf-8")
    out = REPO / "runs" / "look_harder" / f"{a.model}_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("gates", "decision")}, indent=1))
    print(f"wrote {REPORT.relative_to(REPO)} and {out.relative_to(REPO)}")
    if a.write and ok:
        mode = {
            "label": "look harder", "tile_frac": TILE_FRAC, "merge_iou": MERGE_IOU, "forward_passes_per_frame": 5,
            "tau_review": fl["tau_review"], "recall_promise": fl["recall_promise"],
            "recall_ceiling": fl["ceiling"], "cards_per_frame_cal": fl["cards_per_frame"],
            "p_pot_bins": p_pot_bins(c_lh, lambda c: c["conf"], fl["tau_review"]),
            "verified_on_test": {k: res["ver"]["look_harder"][k] for k in ("recall", "recall_lcb", "promise_held",
                                                                           "cards_per_frame")},
            "standard_cards_per_frame_test": res["ver"]["standard"]["cards_per_frame"],
            "report": REPORT.relative_to(REPO).as_posix(),
            "fit": {"split": c_desc, "verified_on": res["ver"]["split"], "created": time.strftime("%Y-%m-%d")}}
        p = save_calibration({"modes": {"look_harder": mode}}, model=a.model)
        print(f"wrote modes.look_harder -> {p}")


if __name__ == "__main__":
    main()
