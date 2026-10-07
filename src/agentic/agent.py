"""
agent.py — the **Decide** stage: a deterministic, logged, human-gated agent with a stated objective.

**Objective (calibrated mode — the product default):** meet the two promises in
``models/<MODEL>/calibration.json`` — "≥ R% of pots reach a human" and "≥ P% of CONFIRMED are real"
(``guarantees.py``) — while spending the **least compute and the fewest human cards**. For every
candidate the agent asks *"can more looking change this tier?"* (value of information):

* detector confidence ≥ τ_confirm → **CONFIRMED** with no extra compute (a re-look cannot change it);
* confidence < τ_review → **LOW-RISK** with no extra compute (the calibrated tier that holds at most
  the complementary share of pots — kept for audit, never deleted);
* only the **uncertain band** in between gets tools: a re-look (packed 4 crops per inference in
  mosaic mode) and, if calibration showed it helps, a CLAHE "try harder" pass. The re-look can lift
  the score over τ_confirm; otherwise the candidate becomes a **REVIEW** card ordered by the
  calibrated P(pot).
* **compute budget / stop rule:** at most ``max_relooks`` re-looks per frame (``$DEPTH_MAX_RELOOKS``),
  spent on the band candidates most likely to be real; the rest are *skipped* (recorded) and go to a
  person on their detector score — the agent stops when more computation is unlikely to pay.

Every candidate's trace reads as one chain: ``voi_check`` ("can another observation change the
decision?") → evidence tools — each **done**, **skipped** (with the reason) or **failed** (with the
fallback) → ``update_belief`` (P(pot) before → after the evidence) → ``decide`` (tier) → ``handoff``
(what happens next and who must approve it). A failed tool never raises: the agent falls back to
the detector's own score, which can only send a find to a person (``incidents.py``). The rule core is
the sole decision authority (reproducible + safe); an LLM never decides.

**Legacy mode** (no fitted calibration, and the unit tests): the old re-look ladder with the
thresholds in ``policy.TriageConfig`` (tuned on test — superseded by the calibrated tiers).

CLI:  python -m src.agentic.agent <image|dir> --out runs/agent [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.cv_pipeline.canonical import Canonicaliser, CanonicalFrame
from src.detection.calibration import load_calibration
from src.detection.infer import Detection
from .perception import Perceptor
from .shadow import ShadowProver
from .tools import Toolbox
from .evidence import fuse_score, evidence_notes
from . import active
from .evidence_model import EvidenceModel
from .incidents import incident
from .policy import TriageConfig, GuaranteedTiers, decide
from .types import AgentStep, Candidate, Evidence, FrameResult, RelookResult, ShadowProof, ShadowQuality, Verdict

REPO = Path(__file__).resolve().parents[2]

_VERDICT_BGR = {
    Verdict.CONFIRMED: (0, 210, 0),      # green
    Verdict.REVIEW: (0, 170, 235),       # amber
    Verdict.LOW_RISK: (130, 130, 130),   # grey
}


@dataclass
class AgentConfig:
    triage: TriageConfig = field(default_factory=TriageConfig)
    # legacy ladder: try an enhanced (CLAHE) re-look only when uncertain AND conf is low
    enhance_when_relook_below: float = 0.40
    enhance_when_conf_below: float = 0.35
    # calibrated mode (None ⇒ legacy rules)
    tiers: Optional[GuaranteedTiers] = None
    # compute budget: re-looks per frame (the stop rule); the rest are skipped + recorded
    max_relooks: int = field(default_factory=lambda: int(os.environ.get("DEPTH_MAX_RELOOKS", "16")))

    @classmethod
    def from_calibration(cls) -> "AgentConfig":
        return cls(tiers=GuaranteedTiers.from_calibration(load_calibration()))


class ReLookAgent:
    """Runs See → Prove → Decide on a frame, emitting per-candidate tool-call traces."""

    def __init__(self, perceptor: Optional[Perceptor] = None,
                 shadow_prover: Optional[ShadowProver] = None,
                 cfg: Optional[AgentConfig] = None):
        self.perceptor = perceptor or Perceptor()
        self.shadow = shadow_prover or ShadowProver()
        self.tools = Toolbox(self.perceptor, self.shadow)
        self.cfg = cfg if cfg is not None else AgentConfig.from_calibration()
        self.stage1 = Canonicaliser()                       # Stage 1: measured sonar geometry
        self.detector_input = load_calibration().detector_input
        self._faults: set[str] = set()                      # failure drills for this call (?simulate=)
        self._active_mode = "off"
        # active vision: the measured evidence model decides which OpenCV tool runs next (active.py).
        # Used only when its fresh-split check passed (evidence_model.json "use"); DEPTH_ACTIVE=0 = off.
        # Default: on only if its validation gates passed ("use"); a caller can switch it on per run as
        # EXPERIMENTAL (faults/options token "opt:active", the studio's switch); DEPTH_ACTIVE=off disables it.
        self.evidence_model = EvidenceModel.load() if self.cfg.tiers is not None else None
        self.evidence = None                                # the model in force for the current run

    @property
    def mode(self) -> str:
        return "calibrated" if self.cfg.tiers is not None else "legacy"

    def run_frame(self, frame: np.ndarray, frame_id: str = "frame",
                  nadir: str | None = None,
                  progress_cb: Optional[Callable[[str, dict], None]] = None,
                  altitude_m: Optional[float] = None, faults: Optional[set] = None) -> FrameResult:
        """``altitude_m``: a MEASURED sonar altitude in metres (a real recording's depth field) — only
        then are shadow heights given in metres. ``nadir`` is an explicit override; otherwise orientation comes from the source rule in
        ``src.cv_pipeline.orientation`` (or the prover's configured default) - never guessed.

        Stage 1 (``src.cv_pipeline.canonical``) runs first: palette -> luminance, orientation,
        bottom tracking (sonar altitude in px), and the detector input ``calibration.json`` selects."""
        H, W = frame.shape[:2]
        self._faults = set(faults or ())
        em, env = self.evidence_model, os.environ.get("DEPTH_ACTIVE", "auto").lower()
        want = env != "off" and em is not None and (em.use or "opt:active" in self._faults or env == "on")
        self.evidence = em if want else None
        self._active_mode = ("off" if not want else "validated" if em.use else "experimental")
        t0 = time.perf_counter()
        cf = self.stage1.process(frame, frame_id=frame_id, nadir=nadir or self.shadow.cfg.nadir)
        det_in = self.stage1.detector_input(frame, cf, self.detector_input)
        stage1_ms = (time.perf_counter() - t0) * 1000
        orient, gray = cf.orientation, cf.gray
        if progress_cb:
            progress_cb("stage1", {"altitude_px": cf.altitude_px, "nadir": orient.label})

        t0 = time.perf_counter()
        dets, _see = self.tools.detect(det_in)
        if "no_detection" in self._faults:                  # failure drill: an empty frame
            dets = []
        see_ms = (time.perf_counter() - t0) * 1000
        if progress_cb:
            progress_cb("see", {"candidates": len(dets), "nadir": orient.label})

        t0 = time.perf_counter()
        if self.cfg.tiers is not None:
            candidates, n_inf = self._decide_calibrated(det_in, gray, dets, cf, altitude_m)
        else:
            candidates = [self._decide_legacy(det_in, gray, d, cf) for d in dets]
            n_inf = 1 + sum(sum(s.tool in ("zoom_relook", "enhance_relook") for s in c.trace)
                            for c in candidates)
        for c in candidates:                                     # Stage-1 geometry for Act (geotag)
            _attach_geometry(c, cf)
        prove_decide_ms = (time.perf_counter() - t0) * 1000
        if progress_cb:
            progress_cb("decide", {"counts": _counts(candidates)})

        s1 = cf.to_dict()
        s1["detector_input"] = self.detector_input
        return FrameResult(
            frame_id=frame_id, width=W, height=H, candidates=candidates,
            stage_ms={"stage1": stage1_ms, "see": see_ms, "prove_decide": prove_decide_ms,
                      "inferences": float(n_inf)},
            nadir=orient.label, orientation=orient.to_dict(), stage1=s1,
            incidents=_frame_incidents(frame_id, candidates, cf, self._faults),
            active=self._active_info(),
        )

    def _active_info(self) -> dict:
        em = self.evidence_model
        if em is None:
            return {"mode": "unavailable", "reason": "no evidence model for this detector (python -m src.agentic.evidence_model)"}
        d = em.data
        return {"mode": self._active_mode, "validated": bool(em.use),
                "status": d.get("actions_reason"), "belief": d.get("belief_reason"),
                "report": f"docs/evidence_model_{str(d.get('model', '')).lower().replace('-', '')}.md",
                "thresholds": em.thresholds, "costs_ms": em.costs}

    # ======================================================================= calibrated mode
    def _decide_calibrated(self, frame, gray, dets: list[Detection], cf: CanonicalFrame,
                           altitude_m: Optional[float] = None):
        tiers = self.cfg.tiers
        nadir = cf.orientation.nadir
        traces: list[list[AgentStep]] = [[] for _ in dets]
        relooks: list[Optional[RelookResult]] = [None] * len(dets)
        scores = [d.conf for d in dets]
        failed: list[bool] = [False] * len(dets)

        # 0) value of information: can another observation change this candidate's outcome?
        band = []
        for i, d in enumerate(dets):
            yes, why = self._voi(d)
            traces[i].append(AgentStep(tool="voi_check", rationale=why, latency_ms=0.0, conf_before=round(d.conf, 4),
                                       detail={"question": "can another observation change the decision?",
                                               "answer": "yes" if yes else "no"}))
            if yes:
                band.append(i)
        # compute budget (stop rule): the likeliest band candidates first; the rest are skipped
        band.sort(key=lambda i: -dets[i].conf)
        cap = max(0, int(self.cfg.max_relooks))
        for i in band[cap:]:
            traces[i].append(AgentStep(
                tool="zoom_relook", status="skipped", latency_ms=0.0, conf_before=round(dets[i].conf, 4),
                rationale=f"compute budget spent ({cap} re-looks this frame, used on the likelier finds) -> "
                          f"stop: goes to a person on its detector score",
                detail={"reason": "compute_budget", "budget": cap}))
        band = sorted(band[:cap])

        n_inf = 1                                                          # the detect pass
        escalated: set[int] = set()
        if band:
            items = [dets[i] for i in band]
            try:
                if "tool_failed" in self._faults:
                    raise RuntimeError("simulated re-look failure (failure drill)")
                res, steps, n = self.tools.relook_band(frame, items, tiers.relook_mode, enhance=False)
                n_inf += n
                for i, r, s in zip(band, res, steps):
                    relooks[i], scores[i] = r, tiers.score(dets[i].conf, r.conf)
                    traces[i].append(s)
            except Exception as e:                                         # contained: fall back, tell a person
                for i in band:
                    failed[i] = True
                    traces[i].append(_failed_step("zoom_relook", e, dets[i].conf))
            # optional CLAHE escalation - only for band candidates still below tau_confirm
            ok = [i for i in band if not failed[i]]
            still = [i for i in ok if tiers.tau_confirm is None or scores[i] < tiers.tau_confirm]
            if tiers.escalate_clahe and still:
                try:
                    res2, steps2, n2 = self.tools.relook_band(frame, [dets[i] for i in still],
                                                               tiers.relook_mode, enhance=True)
                    n_inf += n2
                    for i, r, s in zip(still, res2, steps2):
                        traces[i].append(s)
                        escalated.add(i)
                        if r.conf > relooks[i].conf:
                            relooks[i] = r
                        scores[i] = tiers.score(dets[i].conf, relooks[i].conf)
                except Exception as e:
                    for i in still:
                        failed[i] = True
                        traces[i].append(_failed_step("enhance_relook", e, dets[i].conf))
            for i in ok:
                if i in escalated or failed[i]:
                    continue
                why = ("calibration showed a CLAHE pass does not improve the calibrated score -> not spent"
                       if not tiers.escalate_clahe else "score already at/above tau_confirm -> no further looking")
                traces[i].append(AgentStep(tool="enhance_relook", status="skipped", rationale=why, latency_ms=0.0,
                                           detail={"reason": "no_value"}))
        for i in range(len(dets)):
            if not any(s.tool == "zoom_relook" for s in traces[i]):
                traces[i].append(AgentStep(tool="zoom_relook", status="skipped", latency_ms=0.0,
                                           rationale="value of information 0: a re-look could not change this tier",
                                           detail={"reason": "voi_zero"}))

        out: list[Candidate] = []
        for i, det in enumerate(dets):
            trace = traces[i]
            rl = relooks[i] or RelookResult(found=False, conf=0.0, gain=0.0, scale=1.0)
            if relooks[i] is not None:
                rl.gain = round(rl.conf - det.conf, 4)
            if self._active_ok(det):
                # the active loop asks its own value-of-information question per tool (below)
                trace[:] = [s for s in trace if s.tool != "voi_check" and not (
                    s.tool in ("zoom_relook", "enhance_relook") and s.status == "skipped")]
                proof, wc, rl_a, res, n_a = self._active(i, det, frame, gray, cf, altitude_m, trace, failed)
                n_inf += n_a
                out.append(self._finish_active(det, trace, proof, wc, rl_a or rl, res, failed[i]))
                continue
            # physical evidence for the card (cheap, no inference): Stage-1 water column, shadow +
            # relative height against the TRACKED altitude
            wc = None
            try:
                wc, s = self.tools.water_column_check(cf, det.bbox)
                trace.append(s if s is not None else AgentStep(
                    tool="water_column_check", status="skipped", latency_ms=0.0,
                    rationale="seabed not tracked on this frame -> water-column check not possible (not guessed)",
                    detail={"reason": "not_measured"}))
            except Exception as e:
                failed[i] = True
                trace.append(_failed_step("water_column_check", e, None))
            try:
                if "tool_failed" in self._faults and i == 0:
                    raise RuntimeError("simulated shadow-tool failure (failure drill)")
                proof, s = self.tools.shadow_check(gray, det.bbox, nadir, altitude_px=_altitude_for(cf, det.bbox),
                                                   altitude_m=altitude_m)
                trace.append(s)
            except Exception as e:
                failed[i] = True
                proof = _no_proof(det.bbox)
                trace.append(_failed_step("shadow_check", e, None))
            if proof.has_shadow and proof.orientation_known:
                _, s = self.tools.estimate_height(proof)
                trace.append(s)
            else:
                trace.append(AgentStep(tool="estimate_height", status="skipped", latency_ms=0.0,
                                       rationale="no measurable shadow -> height not estimable (never assumed)",
                                       detail={"reason": "no_shadow"}))

            # probability update: P(pot) from the detector alone -> after the evidence
            hazard = det.cls_name == tiers.guaranteed_class and det.conf >= tiers.tau_review
            p0 = tiers.p_pot(det.conf) if hazard else None
            p1 = tiers.p_pot(scores[i]) if hazard else None
            if p0 is not None and p1 is not None:
                if abs(p1 - p0) < 1e-9:
                    why = (f"P(pot) stays {p1:.0%} (calibrated on held-out recordings from the detector score); "
                           f"the shadow / height / water-column evidence is shown to the person, not folded into "
                           f"the calibrated number")
                else:
                    why = (f"P(pot) {p0:.0%} from the detector alone -> {p1:.0%} after the re-look "
                           f"({'raised' if p1 > p0 else 'lowered'}; calibrated on held-out recordings)")
                trace.append(AgentStep(
                    tool="update_belief", latency_ms=0.0, conf_before=round(det.conf, 4), conf_after=round(scores[i], 4),
                    rationale=why, detail={"p_prior": p0, "p_post": p1}))

            verdict, why, p = self._tier(det, scores[i], relooks[i] is not None)
            if failed[i] and verdict is Verdict.CONFIRMED and scores[i] > det.conf:
                verdict, p = Verdict.REVIEW, tiers.p_pot(det.conf)        # never auto-confirm on partial evidence
                why += " -> but an evidence tool failed, so a person decides (REVIEW)"
            notes = evidence_notes(det.conf, det.cls_name, rl, proof)
            if wc:
                notes.append("sits in the water column above the tracked seabed (fish / bubbles / surface "
                             "return?) - not on the seabed; a human should check it")
            if relooks[i] is None and det.cls_name == tiers.guaranteed_class and not failed[i]:
                notes[0] = "re-look: not needed - it could not change this tier (value of information 0)"
            if failed[i]:
                notes.insert(0, "an evidence tool failed on this find - the agent used the detector's own score "
                                "and sent it to a person (see the trace)")
            evidence = Evidence(relook=rl, shadow=proof, echo_ratio=proof.echo_ratio,
                                evidence_score=round(scores[i], 4), notes=notes, p_pot=p)
            trace.append(AgentStep(
                tool="decide", rationale=why, latency_ms=0.0, conf_before=round(det.conf, 4),
                conf_after=round(scores[i], 4),
                detail={"verdict": verdict.value, "score": round(scores[i], 4), "p_pot": p,
                        "tau_review": tiers.tau_review, "tau_confirm": tiers.tau_confirm,
                        "relooked": relooks[i] is not None, "mode": "calibrated", "tool_failed": failed[i]},
            ))
            trace.append(_handoff(verdict, p))
            out.append(Candidate(bbox=det.bbox, cls_id=det.cls_id, cls_name=det.cls_name,
                                 conf=det.conf, evidence=evidence, verdict=verdict, trace=trace))
        return out, n_inf

    # ======================================================================= active vision
    def _active_ok(self, det: Detection) -> bool:
        t = self.cfg.tiers
        return self.evidence is not None and det.cls_name == t.guaranteed_class and det.conf >= t.tau_review

    def _active(self, i, det: Detection, frame, gray, cf: CanonicalFrame, altitude_m, trace, failed):
        """OpenCV result -> belief -> the next tool, chosen by value of information (``active.py``).
        Returns (shadow proof or None, water-column flag, re-look or None, policy result, inferences)."""
        t, em = self.cfg.tiers, self.evidence
        th = em.thresholds
        p0 = t.p_pot(det.conf)
        a0 = active.action_of(p0, th)
        state = {"proof": None, "rl": None, "n": 0}
        # geometry first: Stage 1 already measured it, so it costs nothing
        wc = None
        try:
            wc, s = self.tools.water_column_check(cf, det.bbox)
            if s is not None:
                s.tool = "geometry_check"
                trace.append(s)
            else:
                trace.append(AgentStep(tool="geometry_check", status="skipped", latency_ms=0.0,
                                       rationale="seabed not tracked on this frame -> no geometry to check (not guessed)",
                                       detail={"reason": "not_measured"}))
        except Exception as e:
            failed[i] = True
            trace.append(_failed_step("geometry_check", e, None))
        trace.append(AgentStep(
            tool="assess", latency_ms=0.0, conf_before=round(det.conf, 4),
            rationale=(f"detector alone: conf {det.conf:.2f} -> P(pot) {p0:.0%} (calibrated) -> {a0.upper()}"
                       + ("; but the box sits in the WATER COLUMN - a seabed object cannot be there: "
                          "EVIDENCE CONFLICT, it is never accepted" if wc else "")),
            detail={"p": p0, "action": a0, "thresholds": th, "water_column": wc}))

        def choose(tool, opts, p, act):
            o = next(x for x in opts if x["tool"] == tool)
            why = (f"P(pot) {p:.0%} -> {act.upper()}. The {tool.replace('_', ' ')} could change that "
                   f"({o['p_flip']:.0%} chance: " + ", ".join(
                       f"{x['outcome']} -> {x['action_after'].upper()} at P {x['p_after']:.0%}" for x in o["outcomes"])
                   + f"); cost ~{em.cost_ms(tool):.0f} ms")
            for x in opts:
                if x["tool"] != tool:
                    why += (f"; {x['tool'].replace('_', ' ')} could too, but costs ~{em.cost_ms(x['tool']):.0f} ms"
                            if x["p_flip"] > 0 else f"; {x['tool'].replace('_', ' ')} cannot change it")
            trace.append(AgentStep(tool="choose_tool", latency_ms=0.0, rationale=why,
                                   detail={"chosen": tool, "options": opts, "p": round(p, 4)}))

        def observe(tool):
            try:
                if "tool_failed" in self._faults and i == 0:
                    raise RuntimeError(f"simulated {tool} failure (failure drill)")
                if tool == "shadow_check":
                    proof, s = self.tools.shadow_check(gray, det.bbox, cf.orientation.nadir,
                                                       altitude_px=_altitude_for(cf, det.bbox), altitude_m=altitude_m)
                    trace.append(s)
                    state["proof"] = proof
                    return proof.has_shadow if proof.orientation_known else None
                rl, s = self.tools.zoom_relook(frame, det.bbox, det.cls_name, det.conf)
                trace.append(s)
                state["rl"], state["n"] = rl, state["n"] + 1
                return rl.conf > 0
            except Exception as e:
                failed[i] = True
                trace.append(_failed_step(tool, e, det.conf))
                return None

        def updated(st, conflict):
            moved = st["action_after"] != st["action_before"]
            why = (f"{st['tool'].replace('_', ' ')}: {st['result']} -> P(pot) {st['p_before']:.0%} -> "
                   f"{st['p_after']:.0%} (likelihood ratio measured on validation) -> "
                   + (f"decision CHANGED {st['action_before'].upper()} -> {st['action_after'].upper()}" if moved
                      else f"still {st['action_after'].upper()}"))
            trace.append(AgentStep(tool="update_belief", latency_ms=0.0, rationale=why,
                                   detail={"p_prior": st["p_before"], "p_post": st["p_after"], "changed": moved,
                                           "from": st["action_before"], "to": st["action_after"],
                                           "tool": st["tool"], "result": st["result"]}))
            if conflict:
                trace.append(AgentStep(tool="conflict", latency_ms=0.0,
                                       rationale=f"EVIDENCE CONFLICT - {conflict['why']} -> another observation is needed",
                                       detail=conflict))

        avail = active.IN_FRAME_TOOLS if cf.orientation.nadir else ("zoom_relook",)   # no orientation: no shadow
        res = active.run_policy(em, det.conf, p0, observe, th, wc, available=avail, on_choose=choose, on_update=updated)
        if wc and res["conflict"] is None:
            res["conflict"] = {"between": "detector vs geometry", "why": "the box sits above the tracked seabed"}
        ran = set(res["tools_run"])
        last = res["steps"][-1].get("options", []) if res["steps"] else []
        for tool in active.IN_FRAME_TOOLS:
            if tool in ran or any(s.tool == tool for s in trace):
                continue
            if tool not in avail:
                trace.append(AgentStep(tool=tool, status="skipped", latency_ms=0.0, detail={"reason": "orientation_unknown"},
                                       rationale="frame orientation unknown -> the shadow cannot be measured (never guessed)"))
                continue
            o = next((x for x in last if x["tool"] == tool), None)
            why = ("not run: value of information 0 - " + (", ".join(
                f"{x['outcome']} -> P {x['p_after']:.0%} ({x['action_after'].upper()})" for x in o["outcomes"])
                if o else "no measured likelihood ratio") + f" - neither result changes {res['action'].upper()}")
            trace.append(AgentStep(tool=tool, status="skipped", latency_ms=0.0, rationale=why,
                                   detail={"reason": "voi_zero", "saved_ms": em.cost_ms(tool)}))
        if res["request_resurvey"]:
            g = res["resurvey"]
            unsettled = res["conflict"] and not res["conflict_resolved"]
            trace.append(AgentStep(
                tool="request_observation", latency_ms=0.0,
                rationale=(("the conflict is not settled by the in-frame tools" if unsettled
                            else "still uncertain after the in-frame tools")
                           + f" -> asks for an OPPOSITE-SIDE PASS (a real object's shadow must flip): "
                             f"{g['p_flip']:.0%} chance it changes the action, {g['info_bits']:.2f} bits expected. "
                             f"A person approves it."),
                detail={"observation": "opposite_side_pass", **{k: g[k] for k in ("p_flip", "info_bits")}}))
        proof = state["proof"]
        if proof is not None and proof.has_shadow and proof.orientation_known:
            _, s = self.tools.estimate_height(proof)
            trace.append(s)
        return proof, wc, state["rl"], res, state["n"]

    def _finish_active(self, det, trace, proof, wc, rl, res, failed_any) -> Candidate:
        t = self.cfg.tiers
        measured = proof is not None
        proof = proof or _no_proof(det.bbox)
        verdict, why, p = self._tier(det, det.conf, False)
        act = res["action"]
        if failed_any and act == "accept":
            act = "review"
            why += " -> an evidence tool failed, so it is not accepted on partial evidence"
        notes = evidence_notes(det.conf, det.cls_name, rl, proof)
        if not measured:
            notes = [n for n in notes if not n.startswith("acoustic shadow")]
            notes.append("acoustic shadow: not measured - it could not change the decision (value of information 0)")
        if "zoom_relook" not in res["tools_run"]:
            notes = [n for n in notes if not n.startswith("re-look")]
            notes.insert(0, "re-look: not run - it could not change the decision (value of information 0)")
        if wc:
            notes.append("sits in the water column above the tracked seabed - not on the seabed; never accepted")
        if res["conflict"]:
            notes.insert(0, "evidence conflict: " + res["conflict"]["why"])
        words = {"accept": "ACCEPT - priority inspection target (dispatch needs a person)",
                 "review": "REVIEW - evidence card for a person", "watch": "WATCH - low-priority card (still shown)"}
        ev = Evidence(relook=rl, shadow=proof, echo_ratio=proof.echo_ratio, evidence_score=round(res["p"], 4),
                      notes=notes, p_pot=p, p_evidence=res["p"], action=act, action0=res["action0"],
                      conflict=res["conflict"], request_resurvey=res["request_resurvey"],
                      resurvey_gain={k: res["resurvey"][k] for k in ("p_flip", "info_bits")})
        changed = act != res["action0"]
        if self._active_mode == "experimental":
            notes.insert(0, "active vision is EXPERIMENTAL here: its benefit failed the registered validation "
                            "checks (see the evidence-model report) - the action is a suggestion for a person")
        trace.append(AgentStep(
            tool="decide", latency_ms=0.0, conf_before=round(det.conf, 4), conf_after=round(det.conf, 4),
            rationale=(f"{why}. OpenCV evidence: P {res['p0']:.0%} -> {res['p']:.0%}; action "
                       + (f"{res['action0'].upper()} -> {act.upper()} (changed by the evidence)" if changed
                          else f"{act.upper()} (unchanged)")),
            detail={"verdict": verdict.value, "score": round(det.conf, 4), "p_pot": p, "p_evidence": res["p"],
                    "action": act, "action0": res["action0"], "changed": changed, "mode": "active",
                    "tau_review": t.tau_review, "tau_confirm": t.tau_confirm, "tool_failed": failed_any,
                    "active_mode": self._active_mode, "relooked": "zoom_relook" in res["tools_run"]}))
        trace.append(AgentStep(tool="handoff", latency_ms=0.0, rationale=words[act] + (
            "; opposite-side pass requested (a person approves it)" if res["request_resurvey"] else ""),
            detail={"next": {"accept": "inspection_target", "review": "human_review", "watch": "human_review_low"}[act],
                    "resurvey": res["request_resurvey"]}))
        return Candidate(bbox=det.bbox, cls_id=det.cls_id, cls_name=det.cls_name, conf=det.conf,
                         evidence=ev, verdict=verdict, trace=trace)

    def _voi(self, det: Detection) -> tuple[bool, str]:
        """Can another observation change this candidate's outcome? (the agent's first question)"""
        t = self.cfg.tiers
        q = f"{det.cls_name} conf {det.conf:.2f}: can another observation change the decision? "
        if det.cls_name in t.non_hazard_classes:
            return False, q + "no - natural seabed class: LOW-RISK whatever a re-look shows"
        if det.cls_name != t.guaranteed_class:
            return False, q + "no - class outside the calibrated guarantee: a person decides either way"
        if t.relook_mode is None:
            return False, q + "no - calibration found a re-look does not move the calibrated score"
        if det.conf < t.tau_review:
            return False, (q + f"no - conf < tau_review {t.tau_review:.2f}: the tier is set by the detector "
                               f"score; a re-look cannot lift it into review")
        if not t.needs_relook(det.conf):
            return False, q + f"no - already above tau_confirm {t.tau_confirm:.2f}: CONFIRMED either way"
        if t.tau_confirm is None:
            return True, (q + "yes - no auto-confirm exists, but a re-look moves P(pot) and so this card's place "
                              "in the human queue")
        return True, q + f"yes - in the uncertain band; a re-look can lift it over tau_confirm {t.tau_confirm:.2f}"

    def _tier(self, det: Detection, score: float, relooked: bool) -> tuple[Verdict, str, Optional[float]]:
        t = self.cfg.tiers
        if det.cls_name in t.non_hazard_classes:
            return (Verdict.LOW_RISK, f"class '{det.cls_name}' is natural seabed, not debris -> "
                                      f"kept for audit, not a hazard", None)
        if det.cls_name != t.guaranteed_class:
            return (Verdict.REVIEW, f"class '{det.cls_name}' is not covered by the calibrated guarantee "
                                    f"-> a human decides (never auto-confirmed)", None)
        v = t.tier(det.conf, score)
        p = t.p_pot(score) if v is Verdict.REVIEW else None
        rp = t.recall_promise
        pp = t.precision_promise
        if v is Verdict.CONFIRMED:
            if relooked and score > det.conf:
                how = f"re-look lifted the score {det.conf:.2f} -> {score:.2f}"
            elif relooked:
                how = f"re-look re-fired (no veto), score {score:.2f}"
            else:
                how = f"detector confidence {det.conf:.2f} already above tau_confirm, no re-look spent"
            return v, (f"{how} >= tau_confirm {t.tau_confirm:.2f} -> CONFIRMED "
                       f"(tier promise: >= {pp:.0%} real, 95% conf.)" if pp else f"{how} -> CONFIRMED"), None
        if v is Verdict.LOW_RISK:
            return v, (f"confidence {det.conf:.2f} < tau_review {t.tau_review:.2f}: a re-look cannot lift it "
                       f"into review -> LOW-RISK, no compute spent; kept for audit"
                       + (f" (this tier holds <= {1 - rp:.0%} of pots)" if rp else "")), None
        why = (f"uncertain band ({t.tau_review:.2f} <= conf {det.conf:.2f}"
               + (f" < {t.tau_confirm:.2f}" if t.tau_confirm is not None else "") + ")")
        if relooked and score < det.conf:
            why += f", did NOT re-fire when zoomed -> score vetoed {det.conf:.2f} -> {score:.2f}"
        elif relooked:
            why += f", re-look score {score:.2f}" + (" still below tau_confirm" if t.tau_confirm is not None else "")
        why += " -> REVIEW card" + (f", P(pot) ~{p:.0%}" if p is not None else "")
        if t.tau_confirm is None:
            why += " (no precision promise is achievable with this model, so nothing is auto-confirmed)"
        return v, why, p

    # ======================================================================= legacy mode
    def _decide_legacy(self, frame, gray, det: Detection, cf: CanonicalFrame) -> Candidate:
        """The old adaptive ladder with the test-tuned thresholds (fallback + unit tests)."""
        cfg = self.cfg
        nadir = cf.orientation.nadir
        tri = cfg.triage
        trace: list[AgentStep] = []

        relook, s = self.tools.zoom_relook(frame, det.bbox, det.cls_name, det.conf)
        trace.append(s)
        confident = relook.conf >= tri.confirm_relook or det.conf >= tri.confirm_conf

        proof, s = self.tools.shadow_check(gray, det.bbox, nadir, altitude_px=_altitude_for(cf, det.bbox))
        trace.append(s)
        if proof.has_shadow and proof.orientation_known:
            _, s = self.tools.estimate_height(proof)
            trace.append(s)

        escalated = False
        if (not confident and relook.conf < cfg.enhance_when_relook_below
                and det.conf < cfg.enhance_when_conf_below):
            relook2, s = self.tools.zoom_relook(frame, det.bbox, det.cls_name, det.conf, enhance=True)
            trace.append(s)
            escalated = True
            if relook2.conf > relook.conf:
                relook = relook2

        evidence = Evidence(
            relook=relook, shadow=proof, echo_ratio=proof.echo_ratio,
            evidence_score=round(fuse_score(det.conf, relook, proof), 3),
            notes=evidence_notes(det.conf, det.cls_name, relook, proof),
        )
        verdict, path = decide(det.conf, evidence, tri)
        trace.append(AgentStep(
            tool="decide", rationale=self._why_legacy(verdict, path, det.conf, relook, escalated),
            latency_ms=0.0, conf_before=round(det.conf, 4), conf_after=round(relook.conf, 4),
            detail={"verdict": verdict.value, "confirm_path": path,
                    "evidence_score": evidence.evidence_score, "escalated": escalated, "mode": "legacy"},
        ))
        return Candidate(bbox=det.bbox, cls_id=det.cls_id, cls_name=det.cls_name,
                         conf=det.conf, evidence=evidence, verdict=verdict, trace=trace)

    def _why_legacy(self, verdict: Verdict, path: str, conf: float, relook, escalated: bool) -> str:
        t = self.cfg.triage
        if verdict is Verdict.CONFIRMED:
            if path == "high_confidence":
                return (f"detector already highly confident ({conf:.2f} >= {t.confirm_conf:.2f}) -> "
                        f"auto-confirmed; escalation skipped (legacy rule)")
            if path == "both":
                return (f"detector confident ({conf:.2f}) and re-look persisted ({relook.conf:.2f}) -> "
                        f"confirmed on two signals (legacy rule)")
            return (f"re-look persisted ({relook.conf:.2f} >= {t.confirm_relook:.2f}) -> confirmed (legacy rule)"
                    + (" after an enhanced try-harder pass" if escalated else ""))
        if verdict is Verdict.LOW_RISK:
            return (f"low confidence ({conf:.2f}) and no re-look ({relook.conf:.2f})"
                    + (" even after enhancement" if escalated else "")
                    + " -> LOW-RISK, kept for audit (not deleted)")
        return (f"uncertain (conf {conf:.2f}, re-look {relook.conf:.2f})"
                + (" even after an enhanced pass" if escalated else "")
                + " -> routed to human review")


def _altitude_for(cf: CanonicalFrame, bbox) -> Optional[float]:
    """Tracked sonar altitude (px) at the ping under the box centre; None if not measured."""
    if not cf.measured:
        return None
    c = cf.to_canonical(0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))
    return cf.altitude_at(c[0]) if c else cf.altitude_px


def _attach_geometry(c: Candidate, cf: CanonicalFrame) -> None:
    """Ping index + across-track GROUND range of the object centre + water-column flag - what the
    Act stage geotags from. Nothing is attached when the orientation is unknown."""
    x1, y1, x2, y2 = c.bbox
    cc = cf.to_canonical(0.5 * (x1 + x2), 0.5 * (y1 + y2))
    if cc is None:
        return
    c.ping_px = float(cc[0])
    c.n_pings = cf.width if cf.orientation.nadir in ("top", "bottom") else cf.height
    c.ground_range_px = cf.ground_range_px(0.5 * (x1 + x2), 0.5 * (y1 + y2))
    c.in_water_column = cf.in_water_column(c.bbox)


def _failed_step(tool: str, e: Exception, conf: Optional[float]) -> AgentStep:
    return AgentStep(tool=tool, status="failed", latency_ms=0.0, conf_before=None if conf is None else round(conf, 4),
                     rationale=f"tool failed ({type(e).__name__}: {str(e)[:120]}) -> fell back to the detector's "
                               f"own score; this find goes to a person",
                     detail={"error": f"{type(e).__name__}: {str(e)[:200]}", "fallback": "detector_score"})


def _no_proof(bbox) -> ShadowProof:
    x1, y1, x2, y2 = bbox
    return ShadowProof(ShadowQuality.NONE, 0.0, 0, 0.0, None, 0.0, 1.0, (int(0.5 * (x1 + x2)), int(y1)),
                       (x1, y2, x2, y2), orientation_known=False)


def _handoff(v: Verdict, p: Optional[float]) -> AgentStep:
    """The act the decision leads to - and who has to approve it. Nothing is dispatched here."""
    if v is Verdict.CONFIRMED:
        why = "eligible for the recovery route; dispatch needs a named person's approval"
    elif v is Verdict.REVIEW:
        why = ("evidence card for a person" + (f", queued by P(pot) {p:.0%}" if p is not None else "")
               + "; analyst + boat budgets are checked when the survey is planned")
    else:
        why = "kept for audit, not queued; a person can override it"
    return AgentStep(tool="handoff", rationale=why, latency_ms=0.0,
                     detail={"next": {"confirmed": "recovery_route", "review": "human_review"}.get(v.value, "audit")})


def _frame_incidents(frame_id: str, cands: list[Candidate], cf: CanonicalFrame, faults: set) -> list[dict]:
    out = []
    if not cands:
        out.append(incident("no_detection", f"{frame_id}: the detector found nothing above its calibrated floor",
                            frame_id=frame_id, simulated="no_detection" in faults))
    elif all(c.verdict is Verdict.LOW_RISK for c in cands):
        out.append(incident("low_confidence_only", f"{frame_id}: {len(cands)} find(s), all below tau_review",
                            frame_id=frame_id))
    if not cf.orientation.nadir:
        out.append(incident("orientation_unknown", f"{frame_id}: orientation not known - no ground range / shadow",
                            frame_id=frame_id))
    nf = sum(any(s.status == "failed" for s in c.trace) for c in cands)
    if nf:
        out.append(incident("tool_failed", f"{frame_id}: an evidence tool failed on {nf} find(s)",
                            frame_id=frame_id, affected=nf, simulated="tool_failed" in faults))
    return out


def _counts(cands: list[Candidate]) -> dict:
    return {v.value: sum(c.verdict is v for c in cands) for v in Verdict}


# ---- rendering + CLI ----------------------------------------------------------------------------
def render(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    """Annotate a frame with each candidate's box coloured by verdict (for overlays/thumbnails)."""
    vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for c in result.candidates:
        x1, y1, x2, y2 = c.bbox
        colour = _VERDICT_BGR[c.verdict]
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
        sc = c.evidence.evidence_score if c.evidence else c.conf
        cv2.putText(vis, f"{c.verdict.value} {c.conf:.2f}->{sc:.2f}", (x1, max(11, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    cc = result.counts
    cv2.putText(vis, f"CONFIRMED {cc['confirmed']}  REVIEW {cc['review']}  LOW-RISK {cc['low_risk']}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def _iter_images(path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if path.is_dir():
        yield from sorted(p for p in path.rglob("*") if p.suffix.lower() in exts)
    else:
        yield path


def main():
    ap = argparse.ArgumentParser(description="Run the See->Prove->Decide agent on image(s).")
    ap.add_argument("source", help="image file or directory")
    ap.add_argument("--out", default=str(REPO / "runs" / "agent"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    agent = ReLookAgent()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = inf = 0
    for ip in _iter_images(Path(a.source)):
        frame = cv2.imread(str(ip))
        if frame is None:
            continue
        res = agent.run_frame(frame, frame_id=ip.stem)
        cv2.imwrite(str(out / f"{ip.stem}.jpg"), render(frame, res))
        (out / f"{ip.stem}.json").write_text(json.dumps(res.to_dict(), indent=2))
        cc = res.counts
        inf += res.stage_ms.get("inferences", 0)
        print(f"{ip.name}: {len(res.candidates)} cand | CONFIRMED {cc['confirmed']} "
              f"REVIEW {cc['review']} LOW-RISK {cc['low_risk']} | {res.stage_ms.get('inferences', 0):.0f} inferences")
        n += 1
        if a.limit and n >= a.limit:
            break
    print(f"\n{n} frames -> {out}  ({agent.mode} mode, {inf / max(1, n):.2f} inferences/frame)")


if __name__ == "__main__":
    main()
