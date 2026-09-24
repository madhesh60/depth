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

Every tool call is an :class:`AgentStep`, so each decision carries a full audit trail — including
the calls the agent chose *not* to make and why. The rule core is the sole decision authority
(reproducible + safe).

**Legacy mode** (no fitted calibration, and the unit tests): the old re-look ladder with the
thresholds in ``policy.TriageConfig`` (tuned on test — superseded by the calibrated tiers).

CLI:  python -m src.agentic.agent <image|dir> --out runs/agent [--limit N]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.cv_pipeline.orientation import resolve_orientation
from src.detection.calibration import load_calibration
from src.detection.infer import Detection
from .perception import Perceptor
from .shadow import ShadowProver
from .tools import Toolbox
from .evidence import fuse_score, evidence_notes
from .policy import TriageConfig, GuaranteedTiers, decide
from .types import AgentStep, Candidate, Evidence, FrameResult, RelookResult, Verdict

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

    @property
    def mode(self) -> str:
        return "calibrated" if self.cfg.tiers is not None else "legacy"

    def run_frame(self, frame: np.ndarray, frame_id: str = "frame",
                  nadir: str | None = None,
                  progress_cb: Optional[Callable[[str, dict], None]] = None) -> FrameResult:
        """``nadir`` is an explicit override; otherwise orientation comes from the source rule in
        ``src.cv_pipeline.orientation`` (or the prover's configured default) — never guessed."""
        H, W = frame.shape[:2]
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        orient = resolve_orientation(frame_id, override=nadir or self.shadow.cfg.nadir, gray=gray)

        t0 = time.perf_counter()
        dets, _see = self.tools.detect(frame)
        see_ms = (time.perf_counter() - t0) * 1000
        if progress_cb:
            progress_cb("see", {"candidates": len(dets), "nadir": orient.label})

        t0 = time.perf_counter()
        if self.cfg.tiers is not None:
            candidates, n_inf = self._decide_calibrated(frame, gray, dets, orient.nadir)
        else:
            candidates = [self._decide_legacy(frame, gray, d, orient.nadir) for d in dets]
            n_inf = 1 + sum(sum(s.tool in ("zoom_relook", "enhance_relook") for s in c.trace)
                            for c in candidates)
        prove_decide_ms = (time.perf_counter() - t0) * 1000
        if progress_cb:
            progress_cb("decide", {"counts": _counts(candidates)})

        return FrameResult(
            frame_id=frame_id, width=W, height=H, candidates=candidates,
            stage_ms={"see": see_ms, "prove_decide": prove_decide_ms, "inferences": float(n_inf)},
            nadir=orient.label, orientation=orient.to_dict(),
        )

    # ======================================================================= calibrated mode
    def _decide_calibrated(self, frame, gray, dets: list[Detection], nadir: Optional[str]):
        tiers = self.cfg.tiers
        traces: list[list[AgentStep]] = [[] for _ in dets]
        relooks: list[Optional[RelookResult]] = [None] * len(dets)
        scores = [d.conf for d in dets]

        # 1) value of information: which candidates can a re-look move (tier or queue position)?
        band = [i for i, d in enumerate(dets)
                if d.cls_name == tiers.guaranteed_class and tiers.needs_relook(d.conf)]
        n_inf = 1                                                          # the detect pass
        if band:
            items = [dets[i] for i in band]
            res, steps, n = self.tools.relook_band(frame, items, tiers.relook_mode, enhance=False)
            n_inf += n
            for i, r, s in zip(band, res, steps):
                relooks[i], scores[i] = r, tiers.score(dets[i].conf, r.conf)
                traces[i].append(s)
            # 2) optional CLAHE escalation — only for band candidates still below τ_confirm
            if tiers.escalate_clahe:
                still = [i for i in band if tiers.tau_confirm is None or scores[i] < tiers.tau_confirm]
                if still:
                    res2, steps2, n2 = self.tools.relook_band(frame, [dets[i] for i in still],
                                                               tiers.relook_mode, enhance=True)
                    n_inf += n2
                    for i, r, s in zip(still, res2, steps2):
                        traces[i].append(s)
                        if r.conf > relooks[i].conf:
                            relooks[i] = r
                        scores[i] = tiers.score(dets[i].conf, relooks[i].conf)

        out: list[Candidate] = []
        for i, det in enumerate(dets):
            trace = traces[i]
            rl = relooks[i] or RelookResult(found=False, conf=0.0, gain=0.0, scale=1.0)
            if relooks[i] is not None:
                rl.gain = round(rl.conf - det.conf, 4)
            # physical evidence for the card (cheap, no inference): shadow + relative height
            proof, s = self.tools.shadow_check(gray, det.bbox, nadir)
            trace.append(s)
            if proof.has_shadow and proof.orientation_known:
                _, s = self.tools.estimate_height(proof)
                trace.append(s)

            verdict, why, p = self._tier(det, scores[i], relooks[i] is not None)
            notes = evidence_notes(det.conf, det.cls_name, rl, proof)
            if relooks[i] is None and det.cls_name == tiers.guaranteed_class:
                notes[0] = "re-look: not needed - it could not change this tier (value of information 0)"
            evidence = Evidence(relook=rl, shadow=proof, echo_ratio=proof.echo_ratio,
                                evidence_score=round(scores[i], 4), notes=notes, p_pot=p)
            trace.append(AgentStep(
                tool="decide", rationale=why, latency_ms=0.0, conf_before=round(det.conf, 4),
                conf_after=round(scores[i], 4),
                detail={"verdict": verdict.value, "score": round(scores[i], 4), "p_pot": p,
                        "tau_review": tiers.tau_review, "tau_confirm": tiers.tau_confirm,
                        "relooked": relooks[i] is not None, "mode": "calibrated"},
            ))
            out.append(Candidate(bbox=det.bbox, cls_id=det.cls_id, cls_name=det.cls_name,
                                 conf=det.conf, evidence=evidence, verdict=verdict, trace=trace))
        return out, n_inf

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
    def _decide_legacy(self, frame, gray, det: Detection, nadir: Optional[str]) -> Candidate:
        """The old adaptive ladder with the test-tuned thresholds (fallback + unit tests)."""
        cfg = self.cfg
        tri = cfg.triage
        trace: list[AgentStep] = []

        relook, s = self.tools.zoom_relook(frame, det.bbox, det.cls_name, det.conf)
        trace.append(s)
        confident = relook.conf >= tri.confirm_relook or det.conf >= tri.confirm_conf

        proof, s = self.tools.shadow_check(gray, det.bbox, nadir)
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
