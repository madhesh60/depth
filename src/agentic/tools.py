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
from .geo import parse_recording, parse_side, parse_ping
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
    def shadow_check(self, gray: np.ndarray, bbox, nadir: str | None) -> tuple[ShadowProof, AgentStep]:
        t0 = time.perf_counter()
        proof = self.shadow.prove(gray, bbox, nadir=nadir)
        ms = (time.perf_counter() - t0) * 1000
        if proof.has_shadow:
            h = f", h~{proof.height_m:.1f}m" if proof.height_m is not None else ""
            why = f"acoustic shadow {proof.quality.value} (contrast {proof.contrast:.2f}{h})"
        else:
            why = "no measurable acoustic shadow (common for small pots in speckle)"
        return proof, AgentStep(
            tool="shadow_check", rationale=why, latency_ms=round(ms, 1),
            detail={"quality": proof.quality.value, "contrast": proof.contrast,
                    "run_px": proof.run_px, "height_m": proof.height_m},
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
    def estimate_height(self, proof: ShadowProof) -> tuple[Optional[float], AgentStep]:
        """Report the shadow-derived object height as an explicit tool step. The value is computed
        by ``shadow_check``; this surfaces it in the trace (and states honestly when it's absent)."""
        t0 = time.perf_counter()
        ms = (time.perf_counter() - t0) * 1000
        if proof.height_m is not None:
            why = (f"height from shadow geometry ~{proof.height_m:.1f} m "
                   f"(h_rel {proof.height_rel:.2f}; plausible for a crab-pot)")
        else:
            why = "no measurable shadow -> height not estimable (common for small low-relief pots)"
        return proof.height_m, AgentStep(
            tool="estimate_height", rationale=why, latency_ms=round(ms, 1),
            detail={"height_m": proof.height_m, "height_rel": proof.height_rel},
        )

    # -- Decide: cross-pass corroboration ("seen in another pass") -------------------------
    def match_other_pass(self, frame_id: str, cx_norm: float, cy_norm: float, cls_name: str,
                         others: list[dict], ping_window: int = 40, pos_tol: float = 0.12
                         ) -> tuple[bool, Optional[str], AgentStep]:
        """Look for the same object in an overlapping pass: a same-class detection in a *different*
        frame of the same recording+channel, within ``ping_window`` chunks and ``pos_tol`` normalised
        distance. Consecutive side-scan pings overlap, so a persistent target re-appears; speckle does
        not. Heuristic (no per-frame slant geometry here) and **non-gating** — it informs the human
        review ranking and the trace, never silently confirms."""
        t0 = time.perf_counter()
        rec, side, ping = parse_recording(frame_id), parse_side(frame_id), parse_ping(frame_id)
        match_frame = None
        if rec is not None and ping is not None:
            for o in others:
                if o["frame_id"] == frame_id or o.get("cls") != cls_name:
                    continue
                if parse_recording(o["frame_id"]) != rec or parse_side(o["frame_id"]) != side:
                    continue
                op = parse_ping(o["frame_id"])
                if op is None or not (0 < abs(op - ping) <= ping_window):
                    continue
                if abs(o["cx"] - cx_norm) <= pos_tol and abs(o["cy"] - cy_norm) <= pos_tol:
                    match_frame = o["frame_id"]
                    break
        ms = (time.perf_counter() - t0) * 1000
        if match_frame:
            why = f"corroborated: same {cls_name} at this across-track position in an overlapping pass"
        elif rec is None or ping is None:
            why = "no pass metadata in filename -> cross-pass corroboration not applicable"
        else:
            why = "no corroborating sighting in an overlapping pass (single-pass detection)"
        return bool(match_frame), match_frame, AgentStep(
            tool="match_other_pass", rationale=why, latency_ms=round(ms, 1),
            detail={"matched": bool(match_frame), "match_frame": match_frame},
        )
