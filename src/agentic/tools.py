"""
tools.py — the agent's toolbox.

Each tool wraps one primitive perception/physics operation and returns ``(payload, AgentStep)``:
the payload the agent reasons over, plus a timed, human-readable :class:`AgentStep` for the trace.
Keeping the tools thin and uniform is what makes the agent's decisions auditable (Agentic-Vision:
"the picture results must change the agent's next step") — every step the agent takes is one tool
call recorded here.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from src.detection.infer import Detection
from .perception import Perceptor
from .shadow import ShadowProver
from .types import AgentStep, RelookResult, ShadowProof


class Toolbox:
    """Timed, trace-emitting wrappers around the See/Prove primitives."""

    def __init__(self, perceptor: Perceptor, shadow_prover: Optional[ShadowProver] = None):
        self.perceptor = perceptor
        self.shadow = shadow_prover or ShadowProver()

    # -- See -------------------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> tuple[list[Detection], AgentStep]:
        t0 = time.perf_counter()
        dets = self.perceptor.perceive(frame)
        ms = (time.perf_counter() - t0) * 1000
        n_fg = sum(d.cls_name == "fishing_gear" for d in dets)
        return dets, AgentStep(
            tool="detect", rationale=f"full-frame scan found {len(dets)} candidates ({n_fg} fishing_gear)",
            latency_ms=round(ms, 1), detail={"n": len(dets)},
        )

    # -- Prove: physical shadow ------------------------------------------------------------
    def shadow_check(self, gray: np.ndarray, bbox, nadir: str | None,
                     altitude_px: Optional[float] = None) -> tuple[ShadowProof, AgentStep]:
        t0 = time.perf_counter()
        proof = self.shadow.prove(gray, bbox, nadir=nadir, altitude_px=altitude_px)
        ms = (time.perf_counter() - t0) * 1000
        if not proof.orientation_known:
            why = "frame orientation unknown -> shadow not measured (never guessed)"
        elif proof.has_shadow:
            why = (f"acoustic shadow {proof.quality.value} (contrast {proof.contrast:.2f}, "
                   f"{proof.run_px}px thin line) - evidence for the card, not a filter")
        else:
            why = "no measurable acoustic shadow (common for small pots in speckle)"
        return proof, AgentStep(
            tool="shadow_check", rationale=why, latency_ms=round(ms, 1),
            detail={"quality": proof.quality.value, "contrast": proof.contrast,
                    "run_px": proof.run_px, "height_rel": proof.height_rel,
                    "height_m": proof.height_m, "orientation_known": proof.orientation_known},
        )

    # -- Prove: re-look consistency (the agent's core action) ------------------------------
    def zoom_relook(self, frame, bbox, cls_name: str, conf_before: float,
                    enhance: bool = False) -> tuple[RelookResult, AgentStep]:
        t0 = time.perf_counter()
        rl = self.perceptor.zoom_relook(frame, bbox, cls_name, enhance=enhance)
        rl.gain = round(rl.conf - conf_before, 4)
        ms = (time.perf_counter() - t0) * 1000
        label = "enhanced re-look (CLAHE)" if enhance else "re-look"
        if rl.found:
            verb = "persisted" if rl.conf >= conf_before else "weakened"
            why = f"{label}: {verb} at {rl.conf:.2f} on {rl.scale:.1f}x zoom (was {conf_before:.2f})"
        else:
            why = f"{label}: did not re-fire when zoomed in (was {conf_before:.2f})"
        return rl, AgentStep(
            tool="enhance_relook" if enhance else "zoom_relook", rationale=why,
            latency_ms=round(ms, 1), conf_before=round(conf_before, 4), conf_after=round(rl.conf, 4),
            detail={"scale": round(rl.scale, 2), "found": rl.found},
        )

    # -- Prove: height from the shadow geometry (a named, traced step) ---------------------
    def estimate_height(self, proof: ShadowProof) -> tuple[float, AgentStep]:
        """Report the shadow-derived height as an explicit tool step. It is RELATIVE to the sonar
        altitude (h/H); metres appear only when the altitude was measured, never assumed."""
        t0 = time.perf_counter()
        ms = (time.perf_counter() - t0) * 1000
        if proof.height_m is not None:
            why = (f"height from shadow geometry ~{proof.height_m:.2f} m "
                   f"({100 * proof.height_rel:.0f}% of the measured sonar altitude)")
        elif proof.run_px > 0:
            why = (f"relative height ~{100 * proof.height_rel:.0f}% of sonar altitude "
                   f"(metres need a measured altitude - not assumed)")
        else:
            why = "no measurable shadow -> height not estimable (common for small low-relief pots)"
        return proof.height_rel, AgentStep(
            tool="estimate_height", rationale=why, latency_ms=round(ms, 1),
            detail={"height_rel": proof.height_rel, "height_m": proof.height_m},
        )
