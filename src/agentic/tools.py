"""
tools.py — the agent's toolbox.

Each tool wraps one primitive perception/physics operation and returns ``(payload, AgentStep)``:
the payload the agent reasons over, plus a timed, human-readable :class:`AgentStep` for the trace.
Keeping the tools thin and uniform is what makes the agent's decisions auditable (Agentic-Vision:
"the picture results must change the agent's next step") — every step the agent takes is one tool
call recorded here.

Every tool that does real work runs under a **deadline** (``$DEPTH_TOOL_TIMEOUT_S``, default 20 s;
``0`` = off). A call that overruns raises :class:`ToolTimeout`, which the agent handles like any
failed tool: the find falls back to the detector's score and goes to a person. Python cannot kill
a thread, so a hung call is left to finish on its own; until it does, the tools that share the
detector **fail fast** instead of queueing behind it (``cv2.dnn`` nets are not re-entrant).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional, TypeVar

import numpy as np

from src.detection.infer import Detection
from .perception import Perceptor, merge_candidates
from .shadow import ShadowProver
from .types import AgentStep, RelookResult, ShadowProof

T = TypeVar("T")
_NET_TOOLS = {"detect", "zoom_relook", "relook_band", "look_harder"}   # share the one cv2.dnn net


def default_timeout_s() -> float:
    try:
        return max(0.0, float(os.environ.get("DEPTH_TOOL_TIMEOUT_S", "20")))
    except ValueError:
        return 20.0


class ToolTimeout(RuntimeError):
    """A tool overran its deadline (or the detector is still busy with one that did)."""


class Toolbox:
    """Timed, trace-emitting wrappers around the See/Prove primitives."""

    def __init__(self, perceptor: Perceptor, shadow_prover: Optional[ShadowProver] = None,
                 timeout_s: Optional[float] = None):
        self.perceptor = perceptor
        self.shadow = shadow_prover or ShadowProver()
        self.timeout_s = default_timeout_s() if timeout_s is None else float(timeout_s)
        self._hung: list[threading.Thread] = []             # timed-out calls still running

    # -- deadline guard ----------------------------------------------------------------------
    def _guard(self, tool: str, fn: Callable[[], T], timeout_s: Optional[float] = None) -> T:
        """Run ``fn`` with a deadline; raise :class:`ToolTimeout` if it overruns."""
        limit = self.timeout_s if timeout_s is None else timeout_s
        self._hung = [t for t in self._hung if t.is_alive()]
        if tool in _NET_TOOLS and any(t.name.split(":", 1)[1] in _NET_TOOLS for t in self._hung):
            raise ToolTimeout(f"{tool} not started: the detector is still busy with a call that timed out")
        if not limit:
            return fn()
        box: dict = {}

        def work():
            try:
                box["v"] = fn()
            except BaseException as e:                       # re-raised in the caller's thread
                box["e"] = e

        th = threading.Thread(target=work, name=f"depth-tool:{tool}", daemon=True)
        th.start()
        th.join(limit)
        if th.is_alive():
            self._hung.append(th)
            raise ToolTimeout(f"{tool} did not answer within {limit:g} s")
        if "e" in box:
            raise box["e"]
        return box["v"]

    def wait_idle(self, timeout_s: float) -> bool:
        """Wait (at most ``timeout_s``) for timed-out calls still running on the detector to finish.
        True when the detector is free again — the agent then retries the frame instead of skipping it."""
        end = time.perf_counter() + max(0.0, timeout_s)
        for th in list(self._hung):
            th.join(max(0.0, end - time.perf_counter()))
        self._hung = [t for t in self._hung if t.is_alive()]
        return not self._hung

    def simulate_hang(self, tool: str, timeout_s: float = 0.3) -> None:
        """Failure drill: a tool that never answers, under a short deadline -> :class:`ToolTimeout`."""
        stop = threading.Event()
        try:
            self._guard(f"drill_{tool}", lambda: stop.wait(timeout_s * 10), timeout_s=timeout_s)
        finally:
            stop.set()                                         # release the drill thread at once

    # -- See -------------------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> tuple[list[Detection], AgentStep]:
        t0 = time.perf_counter()
        dets = self._guard("detect", lambda: self.perceptor.perceive(frame))
        ms = (time.perf_counter() - t0) * 1000
        g = getattr(self.perceptor, "guaranteed_class", "fishing_gear")
        n_fg = sum(d.cls_name == g for d in dets)
        return dets, AgentStep(
            tool="detect", rationale=f"full-frame scan found {len(dets)} candidates ({n_fg} {g})",
            latency_ms=round(ms, 1), detail={"n": len(dets)},
        )

    def look_harder(self, frame: np.ndarray, dets: list[Detection], tile_frac: float,
                    merge_iou: float) -> tuple[list[Detection], list[str], AgentStep]:
        """The second, closer look at the whole frame (2×2 tiles, 4 forward passes) folded into the
        first look. Returns the merged candidates, where each came from, and the step."""
        t0 = time.perf_counter()
        extra = self._guard("look_harder", lambda: self.perceptor.look_harder(frame, tile_frac))
        merged, src = merge_candidates(dets, extra, merge_iou)
        ms = (time.perf_counter() - t0) * 1000
        new, raised = src.count("tiles"), src.count("full+tiles")
        return merged, src, AgentStep(
            tool="look_harder", latency_ms=round(ms, 1),
            rationale=(f"2×2 tile pass ({tile_frac:.0%} tiles, the scale the detector was trained on): "
                       f"{len(extra)} finds -> {new} new candidate(s) the full-frame look did not propose, "
                       f"{raised} first-look score(s) raised"),
            detail={"tile_finds": len(extra), "new": new, "raised": raised, "passes": 4, "tile_frac": tile_frac})

    # -- Prove: Stage-1 water column (geometry, no inference) -------------------------------
    def water_column_check(self, cf, bbox) -> tuple[Optional[bool], Optional[AgentStep]]:
        """Is the box above the tracked seabed? Returns (flag, step); (None, None) when the bottom
        was not tracked - nothing to say, so no trace line."""
        t0 = time.perf_counter()
        wc = cf.in_water_column(bbox)
        if wc is None:
            return None, None
        ms = (time.perf_counter() - t0) * 1000
        why = ("box lies above the tracked seabed -> suspended in the water column (flagged for the human)"
               if wc else f"on the seabed (below the first bottom return at ~{cf.altitude_px:.0f} px slant range)")
        return wc, AgentStep(tool="water_column_check", rationale=why, latency_ms=round(ms, 2),
                             detail={"in_water_column": wc, "altitude_px": round(cf.altitude_px, 1)})

    # -- Prove: physical shadow ------------------------------------------------------------
    def shadow_check(self, gray: np.ndarray, bbox, nadir: str | None,
                     altitude_px: Optional[float] = None,
                     altitude_m: Optional[float] = None) -> tuple[ShadowProof, AgentStep]:
        t0 = time.perf_counter()
        proof = self._guard("shadow_check", lambda: self.shadow.prove(
            gray, bbox, nadir=nadir, altitude_px=altitude_px, altitude_m=altitude_m))
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
        rl = self._guard("zoom_relook", lambda: self.perceptor.zoom_relook(frame, bbox, cls_name, enhance=enhance))
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
        elif proof.height_rel > 0:
            why = (f"relative height ~{100 * proof.height_rel:.0f}% of sonar altitude "
                   f"(metres need a measured altitude - not assumed)")
        else:
            why = "no measurable shadow -> height not estimable (common for small low-relief pots)"
        return proof.height_rel, AgentStep(
            tool="estimate_height", rationale=why, latency_ms=round(ms, 1),
            detail={"height_rel": proof.height_rel, "height_m": proof.height_m},
        )

    # -- Prove: batched re-look of the uncertain band (value-of-information agent) ---------
    def relook_band(self, frame, dets: list[Detection], mode: Optional[str],
                    enhance: bool = False) -> tuple[list[RelookResult], list[AgentStep], int]:
        """Re-look every candidate the agent judged worth it, in ONE batch.

        ``mode="mosaic"`` packs up to 4 crops into a single detector pass (review I-6);
        ``mode="single"`` runs one pass per candidate. Returns (results, per-candidate steps,
        number of inferences spent)."""
        t0 = time.perf_counter()
        items = [(d.bbox, d.cls_name) for d in dets]
        batch = getattr(self.perceptor, "relook_batch", None)
        if mode == "mosaic" and batch is not None:
            results = self._guard("relook_band", lambda: batch(frame, items, enhance=enhance))
            n_inf = -(-len(items) // 4)
        else:
            results = self._guard("relook_band", lambda: [
                self.perceptor.zoom_relook(frame, b, c, enhance=enhance) for b, c in items])
            n_inf = len(items)
        ms = (time.perf_counter() - t0) * 1000
        label = ("enhanced " if enhance else "") + ("mosaic re-look" if mode == "mosaic" else "re-look")
        share = f"{n_inf} inference(s) for {len(items)} candidate(s)"
        steps = []
        for d, r in zip(dets, results):
            r.gain = round(r.conf - d.conf, 4)
            if r.found:
                verb = "persisted" if r.conf >= d.conf else "weakened"
                why = f"{label}: {verb} at {r.conf:.2f} (was {d.conf:.2f}); {share}"
            else:
                why = f"{label}: did not re-fire (was {d.conf:.2f}); {share}"
            steps.append(AgentStep(
                tool="enhance_relook" if enhance else ("mosaic_relook" if mode == "mosaic" else "zoom_relook"),
                rationale=why, latency_ms=round(ms / max(1, len(items)), 1),
                conf_before=round(d.conf, 4), conf_after=round(r.conf, 4),
                detail={"scale": round(r.scale, 2), "found": r.found, "batch": len(items), "inferences": n_inf}))
        return results, steps, n_inf
