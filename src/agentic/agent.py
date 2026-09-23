"""
agent.py — the **Decide** stage: a deterministic, logged, human-gated re-look agent.

For each candidate the agent gathers evidence by *calling tools* (shadow_check → zoom_relook →,
if still uncertain, an enhanced re-look) and then triages it into CONFIRMED / REVIEW / REJECTED via
the pre-registered rules in ``policy.py``. Every tool call is recorded as an :class:`AgentStep`, so
each decision carries a full audit trail — including cases where a re-look *changed* the outcome
(the evidence the Agentic-Vision award asks for: perception results must change the agent's next
step). The rule core is the sole decision authority (reproducible + safe); an LLM narrator could
sit on top later without touching this logic.

Recall-safe by design: REJECTED means "low evidence — deprioritised, retained for audit", never
deleted; the only auto-*trusted* tier is CONFIRMED (precision ~0.77 vs ~0.60 raw, STUDY-03).

CLI:  python -m src.agentic.agent <image|dir> --out runs/agent [--limit N]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.detection.infer import Detection
from .perception import Perceptor
from .shadow import ShadowProver
from .tools import Toolbox
from .evidence import fuse_score, evidence_notes
from .policy import TriageConfig, triage
from .types import AgentStep, Candidate, Evidence, FrameResult, Verdict

REPO = Path(__file__).resolve().parents[2]

_VERDICT_BGR = {
    Verdict.CONFIRMED: (0, 210, 0),      # green
    Verdict.REVIEW: (0, 170, 235),       # amber
    Verdict.REJECTED: (130, 130, 130),   # grey
}


@dataclass
class AgentConfig:
    triage: TriageConfig = field(default_factory=TriageConfig)
    # try an enhanced (CLAHE) re-look only when the first look is uncertain AND conf is low
    enhance_when_relook_below: float = 0.40
    enhance_when_conf_below: float = 0.35


class ReLookAgent:
    """Runs See → Prove → Decide on a frame, emitting per-candidate tool-call traces."""

    def __init__(self, perceptor: Optional[Perceptor] = None,
                 shadow_prover: Optional[ShadowProver] = None,
                 cfg: Optional[AgentConfig] = None):
        self.perceptor = perceptor or Perceptor()
        self.shadow = shadow_prover or ShadowProver()
        self.tools = Toolbox(self.perceptor, self.shadow)
        self.cfg = cfg or AgentConfig()

    def run_frame(self, frame: np.ndarray, frame_id: str = "frame",
                  nadir: str | None = None,
                  progress_cb: Optional[Callable[[str, dict], None]] = None) -> FrameResult:
        import time
        H, W = frame.shape[:2]
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nadir = nadir or self.shadow.calibrate_nadir(gray)

        t0 = time.perf_counter()
        dets, _see = self.tools.detect(frame)
        see_ms = (time.perf_counter() - t0) * 1000
        if progress_cb:
            progress_cb("see", {"candidates": len(dets), "nadir": nadir})

        candidates: list[Candidate] = []
        t0 = time.perf_counter()
        for det in dets:
            candidates.append(self._decide_candidate(frame, gray, det, nadir))
        prove_decide_ms = (time.perf_counter() - t0) * 1000
        if progress_cb:
            progress_cb("decide", {"counts": _counts(candidates)})

        return FrameResult(
            frame_id=frame_id, width=W, height=H, candidates=candidates,
            stage_ms={"see": see_ms, "prove_decide": prove_decide_ms}, nadir=nadir,
        )

    # -- per-candidate decision procedure ----------------------------------------------------
    def _decide_candidate(self, frame, gray, det: Detection, nadir: str) -> Candidate:
        cfg = self.cfg
        trace: list[AgentStep] = []

        proof, s_shadow = self.tools.shadow_check(gray, det.bbox, nadir)
        trace.append(s_shadow)

        relook, s_relook = self.tools.zoom_relook(frame, det.bbox, det.cls_name, det.conf)
        trace.append(s_relook)

        # try harder once, only when genuinely uncertain
        if relook.conf < cfg.enhance_when_relook_below and det.conf < cfg.enhance_when_conf_below:
            relook2, s_enh = self.tools.zoom_relook(frame, det.bbox, det.cls_name, det.conf, enhance=True)
            trace.append(s_enh)
            if relook2.conf > relook.conf:
                relook = relook2

        evidence = Evidence(
            relook=relook, shadow=proof, echo_ratio=proof.echo_ratio,
            evidence_score=round(fuse_score(det.conf, relook, proof), 3),
            notes=evidence_notes(det.conf, det.cls_name, relook, proof),
        )
        verdict = triage(det.conf, evidence, cfg.triage)
        trace.append(AgentStep(
            tool="decide", rationale=self._why(verdict, det.conf, relook),
            latency_ms=0.0, conf_before=round(det.conf, 4), conf_after=round(relook.conf, 4),
            detail={"verdict": verdict.value, "evidence_score": evidence.evidence_score},
        ))
        return Candidate(bbox=det.bbox, cls_id=det.cls_id, cls_name=det.cls_name,
                         conf=det.conf, evidence=evidence, verdict=verdict, trace=trace)

    def _why(self, verdict: Verdict, conf: float, relook) -> str:
        t = self.cfg.triage
        if verdict is Verdict.CONFIRMED:
            return (f"re-look persisted ({relook.conf:.2f} >= {t.confirm_relook:.2f}) -> auto-confirmed "
                    f"(precision ~0.77 at this tier)")
        if verdict is Verdict.REJECTED:
            return (f"low confidence ({conf:.2f}) and no re-look ({relook.conf:.2f}) -> deprioritised, "
                    f"kept for audit (not deleted)")
        return f"uncertain (re-look {relook.conf:.2f}) -> routed to human review, ranked by evidence"


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
        rl = c.evidence.relook.conf if c.evidence else 0.0
        cv2.putText(vis, f"{c.verdict.value} {c.conf:.2f}->{rl:.2f}", (x1, max(11, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    cc = result.counts
    cv2.putText(vis, f"CONFIRMED {cc['confirmed']}  REVIEW {cc['review']}  REJECTED {cc['rejected']}",
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
    results = []
    flips = 0
    n = 0
    for ip in _iter_images(Path(a.source)):
        frame = cv2.imread(str(ip))
        if frame is None:
            continue
        res = agent.run_frame(frame, frame_id=ip.stem)
        cv2.imwrite(str(out / f"{ip.stem}.jpg"), render(frame, res))
        (out / f"{ip.stem}.json").write_text(json.dumps(res.to_dict(), indent=2))
        # highlight decisions a re-look flipped: CONFIRMED despite a low raw confidence
        for c in res.candidates:
            if c.verdict is Verdict.CONFIRMED and c.conf < 0.25:
                flips += 1
        cc = res.counts
        print(f"{ip.name}: {len(res.candidates)} cand | "
              f"CONFIRMED {cc['confirmed']} REVIEW {cc['review']} REJECTED {cc['rejected']}")
        n += 1
        if a.limit and n >= a.limit:
            break
    print(f"\n{n} frames -> {out}  ({flips} decisions where a re-look confirmed a low-confidence candidate)")


if __name__ == "__main__":
    main()
