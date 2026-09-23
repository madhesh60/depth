"""
shadow.py — the **Prove** stage's physical acoustic-shadow evidence (pure OpenCV/NumPy).

A real object standing on the seabed reflects a bright echo and casts a dark **acoustic shadow**
on its far-range side (sound cannot pass through it) — like a person under a streetlight. Flat
seabed and speckle do not. This module measures that shadow for a candidate box and estimates the
object's height from the shadow geometry.

Honesty (STUDY-03): on this side-scan crab-pot data a *measurable* shadow exists only on a minority
of larger / higher-relief objects, so ``ShadowQuality.NONE`` is a common and legitimate result. The
shadow is therefore **evidence shown where it exists** (and a great visual for the report), never a
silent gate — ``quality == NONE`` never rejects a candidate on its own.

Geometry: the nadir / water-column band sits at one image edge; range increases away from it and
shadows fall on the far-range side. :meth:`ShadowProver.calibrate_nadir` finds that edge
automatically (the crab-pot ``wcp_ss`` exports put it at the top; verified on real frames). The core
measurement is written for "far range = down" and other orientations are handled by a tested 90°
rotation.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import ShadowProof, ShadowQuality

# nadir edge → number of CCW 90° rotations that make "far range" point down (see _rotate_*).
_NADIR_TO_K = {"top": 0, "right": 1, "bottom": 2, "left": 3}


@dataclass
class ShadowConfig:
    nadir: str = "auto"              # "auto" | "top" | "bottom" | "left" | "right"
    search_frac: float = 4.0        # search the shadow up to this × box-height beyond the echo
    col_width_frac: float = 0.25    # shadow column half-width as a fraction of box width
    shadow_ratio: float = 0.70      # a range step is "shadow" if its mean < ratio × local background
    clear_contrast: float = 0.25    # contrast ≥ this AND run ≥ clear_run ⇒ CLEAR
    clear_run_px: int = 6
    weak_contrast: float = 0.12     # contrast ≥ this OR run ≥ weak_run ⇒ WEAK
    weak_run_px: int = 3
    altitude_m: float = 10.0        # assumed sonar altitude above seabed (flagged; per-survey)


# ---- coordinate rotation (tested in tests/test_shadow.py) ---------------------------------------
def _rotate_point(x: int, y: int, k: int, h: int, w: int) -> tuple[int, int]:
    """Map (x,y) in an (h,w) image to its location in ``np.rot90(img, k)`` (CCW). Returns (x',y')."""
    k %= 4
    if k == 0:
        return x, y
    if k == 1:     # dest shape (w, h): (r,c)->(w-1-c, r)
        return y, (w - 1 - x)
    if k == 2:     # dest shape (h, w): (r,c)->(h-1-r, w-1-c)
        return (w - 1 - x), (h - 1 - y)
    return (h - 1 - y), x   # k == 3, dest shape (w, h): (r,c)->(c, h-1-r)


def _rotate_box(box, k: int, h: int, w: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    ax, ay = _rotate_point(x1, y1, k, h, w)
    bx, by = _rotate_point(x2, y2, k, h, w)
    return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))


class ShadowProver:
    """Measure the acoustic shadow + height for candidate boxes. Reuse one instance."""

    def __init__(self, cfg: ShadowConfig | None = None):
        self.cfg = cfg or ShadowConfig()

    # -- nadir / range calibration -----------------------------------------------------------
    def calibrate_nadir(self, gray: np.ndarray) -> str:
        """Return the nadir edge ("top"/"bottom"/"left"/"right"): the darkest, lowest-variance edge
        band (the water column / nadir). Range increases away from it; shadows fall to far range."""
        if self.cfg.nadir != "auto":
            return self.cfg.nadir
        h, w = gray.shape[:2]
        g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        bh, bw = max(3, h // 12), max(3, w // 12)
        bands = {
            "top": g[:bh, :], "bottom": g[-bh:, :],
            "left": g[:, :bw], "right": g[:, -bw:],
        }
        # nadir band is dark AND flat → score = mean + std (lower is more nadir-like)
        scores = {e: float(b.mean()) + float(b.std()) for e, b in bands.items()}
        return min(scores, key=scores.get)

    # -- main entry --------------------------------------------------------------------------
    def prove(self, gray: np.ndarray, box, nadir: str | None = None) -> ShadowProof:
        """Measure the shadow for ``box`` (x1,y1,x2,y2). ``gray`` may be BGR or single-channel."""
        g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        nadir = nadir or self.calibrate_nadir(g)
        k = _NADIR_TO_K.get(nadir, 0)
        if k == 0:
            return self._prove_down(g, box)
        # rotate so far-range points down, measure, then map overlay coords back
        grot = np.rot90(g, k)
        proof = self._prove_down(grot, _rotate_box(box, k, h, w))
        hr, wr = grot.shape
        ex, ey = _rotate_point(proof.echo_xy[0], proof.echo_xy[1], (4 - k) % 4, hr, wr)
        sx1, sy1, sx2, sy2 = _rotate_box(proof.strip, (4 - k) % 4, hr, wr)
        proof.echo_xy = (ex, ey)
        proof.strip = (sx1, sy1, sx2, sy2)
        return proof

    # -- core measurement: far range = down (+row) -------------------------------------------
    def _prove_down(self, g: np.ndarray, box) -> ShadowProof:
        cfg = self.cfg
        H, W = g.shape
        x1, y1, x2, y2 = (int(max(0, box[0])), int(max(0, box[1])),
                          int(min(W, box[2])), int(min(H, box[3])))
        bw, bh = x2 - x1, y2 - y1
        if bw < 3 or bh < 3:
            return self._empty(x1, y1, x2, y2)

        boxf = g[y1:y2, x1:x2].astype(np.float32)
        # echo = brightest column in the near-range (upper) 60% of the box
        upper = boxf[: max(1, int(bh * 0.6))]
        ex = int(np.argmax(upper.mean(axis=0)))
        peak_row = int(np.argmax(boxf[:, ex]))
        echo_x, echo_y = x1 + ex, y1 + peak_row

        # local background = flanks beside the box at the box rows
        flanks = [a.ravel() for a in
                  (g[max(0, y1 - 3):y2, max(0, x1 - bw):x1], g[max(0, y1 - 3):y2, x2:min(W, x2 + bw)])
                  if a.size]
        ring = max(float(np.median(np.concatenate(flanks))) if flanks else float(np.median(boxf)), 1.0)
        echo_val = float(boxf[max(0, peak_row - 1):peak_row + 2, max(0, ex - 1):ex + 2].mean())
        echo_ratio = echo_val / ring

        # thin shadow column starting just past the echo, extending to far range (down)
        cw = max(2, int(bw * cfg.col_width_frac))
        c1, c2 = max(0, echo_x - cw), min(W, echo_x + cw + 1)
        sy0 = min(H - 1, echo_y + max(2, int(bh * 0.2)))
        sy1 = min(H, sy0 + int(bh * cfg.search_frac) + 4)
        col = g[sy0:sy1, c1:c2].astype(np.float32)
        if col.size == 0:
            return self._empty(x1, y1, x2, y2, echo_ratio, (echo_x, echo_y))

        col_flanks = [a.ravel() for a in
                      (g[sy0:sy1, max(0, c1 - bw):c1], g[sy0:sy1, c2:min(W, c2 + bw)]) if a.size]
        local_bg = max(float(np.median(np.concatenate(col_flanks))) if col_flanks else ring, 1.0)

        # walk to far range: skip the echo's bright tail / any neutral gap, then measure the first
        # contiguous dark run (the shadow starts near — but not always exactly at — the object edge).
        rowm = col.mean(axis=1)
        thr = cfg.shadow_ratio * local_bg
        run, start_i, started = 0, 0, False
        for i, rm in enumerate(rowm):
            if rm < thr:
                if not started:
                    started, start_i = True, i
                run += 1
            elif started:
                break
        dark_mean = float(rowm[start_i:start_i + run].mean()) if run else float(rowm.mean())
        contrast = (local_bg - dark_mean) / local_bg

        # quality classification (honest: NONE is common and OK)
        if contrast >= cfg.clear_contrast and run >= cfg.clear_run_px:
            quality = ShadowQuality.CLEAR
        elif contrast >= cfg.weak_contrast or run >= cfg.weak_run_px:
            quality = ShadowQuality.WEAK
        else:
            quality = ShadowQuality.NONE

        # height from shadow geometry: h = altitude · Ls / (ground_range + Ls); pixel scale cancels.
        ground_range_px = max(1.0, float(echo_y))     # distance from top nadir edge to the echo
        height_rel = run / (ground_range_px + run) if run > 0 else 0.0
        height_m = cfg.altitude_m * height_rel if run > 0 else None

        # soft strength score for ranking (0 when NONE)
        strength = 0.0
        if quality is not ShadowQuality.NONE:
            strength = float(np.clip(0.6 * max(contrast, 0.0) / cfg.clear_contrast
                                     + 0.4 * min(run, cfg.clear_run_px) / cfg.clear_run_px, 0.0, 1.0))

        strip = (c1, sy0 + start_i, c2, min(H, sy0 + start_i + max(run, 1)))
        return ShadowProof(
            quality=quality, contrast=round(contrast, 3), run_px=int(run), strength=round(strength, 3),
            height_m=round(height_m, 2) if height_m is not None else None,
            height_rel=round(height_rel, 3), echo_ratio=round(echo_ratio, 3),
            echo_xy=(int(echo_x), int(echo_y)), strip=strip,
        )

    @staticmethod
    def _empty(x1, y1, x2, y2, echo_ratio: float = 1.0, echo_xy=None) -> ShadowProof:
        exy = echo_xy or (int(0.5 * (x1 + x2)), int(y1))
        return ShadowProof(ShadowQuality.NONE, 0.0, 0, 0.0, None, 0.0, round(echo_ratio, 3),
                           exy, (x1, y2, x2, y2))

    # -- overlay -----------------------------------------------------------------------------
    def overlay(self, image: np.ndarray, box, proof: ShadowProof) -> np.ndarray:
        """Draw the evidence: detection box, echo point, and shadow strip with a short caption."""
        vis = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 1)
        cv2.circle(vis, proof.echo_xy, 3, (0, 0, 255), -1)
        colour = {ShadowQuality.CLEAR: (0, 220, 0), ShadowQuality.WEAK: (0, 170, 220),
                  ShadowQuality.NONE: (120, 120, 120)}[proof.quality]
        sx1, sy1, sx2, sy2 = [int(v) for v in proof.strip]
        cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), colour, 1)
        h_txt = f" h~{proof.height_m:.1f}m" if proof.height_m is not None else ""
        cv2.putText(vis, f"shadow:{proof.quality.value} c{proof.contrast:.2f}{h_txt}",
                    (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
        return vis
