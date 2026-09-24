"""
orientation.py — where is the sonar (nadir) in this frame? Decided by SOURCE RULES, never guessed.

The old ``ShadowProver.calibrate_nadir`` guessed the nadir edge from the darkest/flattest image
border. On real PINGMapper sonograms — whose nadir is *always* the top edge — that guess was right
only 26% of the time (NEEDTOIMPROVE §4.3), which pointed the shadow search and the geotag range the
wrong way on 3 of 4 frames. Orientation is now resolved in this order:

1. an explicit override from the caller (API ``?nadir=top|bottom|left|right``) — rule ``"user"``;
2. a **source rule** from the filename (verified by eye on real frames, ``runs/_look``):

   ============================  ======  ====================================================
   pattern                       nadir   why
   ============================  ======  ====================================================
   ``*_ss_port_*`` / ``*_ss_star*``  top   PINGMapper Humminbird sonogram (``wcp`` = water column
                                         present): water column at the top edge, slant range
                                         grows downward, shadows are thin vertical lines below
                                         each target.
   ============================  ======  ====================================================

3. otherwise **unknown** — mosaics (``BC_POST``/``baycove``), rotated/augmented frames (``TI00xx``),
   the orange ``Contact_*_sslo`` crops (range is horizontal but the crop hides which side the
   sonar track is on) and unrecognised uploads. With an unknown orientation the shadow is **not
   measured** and the geotag degrades to the boat position with a swath-wide error radius — an
   honest "don't know" instead of a confident wrong answer.

``"auto"`` is still accepted as an override for diagnostics only; it is labelled unreliable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

EDGES = ("top", "bottom", "left", "right")

_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"_ss_(port|star|starboard)(_|$)", re.I), "top",
     "PINGMapper sonogram: water column at the top edge, range grows downward"),
]


@dataclass(frozen=True)
class Orientation:
    """Resolved frame orientation + where the answer came from (shown in the UI/trace)."""
    nadir: Optional[str]          # "top" | "bottom" | "left" | "right" | None (unknown)
    rule: str                     # provenance, human-readable
    source: str                   # "user" | "source-rule" | "auto-guess" | "unknown"

    @property
    def known(self) -> bool:
        return self.nadir in EDGES

    @property
    def label(self) -> str:
        return self.nadir if self.known else "unknown"

    def to_dict(self) -> dict:
        return {"nadir": self.label, "rule": self.rule, "source": self.source}


def auto_guess_nadir(gray: np.ndarray) -> str:
    """The legacy heuristic (darkest + flattest border band). **Unreliable** — 26% correct on real
    wcp sonograms; kept only as an explicit diagnostic override."""
    g = gray if gray.ndim == 2 else gray.mean(axis=2)
    h, w = g.shape[:2]
    bh, bw = max(3, h // 12), max(3, w // 12)
    bands = {"top": g[:bh, :], "bottom": g[-bh:, :], "left": g[:, :bw], "right": g[:, -bw:]}
    scores = {e: float(b.mean()) + float(b.std()) for e, b in bands.items()}
    return min(scores, key=scores.get)


def resolve_orientation(frame_id: str = "", override: Optional[str] = None,
                        gray: Optional[np.ndarray] = None) -> Orientation:
    """Resolve the nadir edge for a frame (override → source rule → unknown)."""
    ov = (override or "").strip().lower()
    if ov in EDGES:
        return Orientation(ov, f"set by user (nadir={ov})", "user")
    if ov == "auto" and gray is not None:
        e = auto_guess_nadir(gray)
        return Orientation(e, "auto-guess from the darkest border (unreliable, diagnostic only)",
                           "auto-guess")
    for pat, nadir, why in _RULES:
        if pat.search(frame_id or ""):
            return Orientation(nadir, why, "source-rule")
    return Orientation(None, "unknown source orientation (mosaic / rotated / unrecognised) - "
                             "shadow and range not measured", "unknown")
