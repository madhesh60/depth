"""
shadow.py — the **Prove** stage's physical acoustic-shadow evidence (pure OpenCV/NumPy).

A real object standing on the seabed reflects a bright echo and casts a dark **acoustic shadow**
on its far-range side (sound cannot pass through it) — like a person under a streetlight. Flat
seabed and speckle do not. This module measures that shadow for a candidate box.

Honesty (STUDY-03/04 + review §3.4):
* The shadow is **evidence shown where it exists**, never a filter: it separates real pots from
  random seabed only modestly and true from false *detections* barely at all (AUC ~0.52), because
  the detector's false positives are also real 3-D returns. ``quality == NONE`` never rejects.
* Orientation is **not guessed**. The caller passes the nadir edge resolved by
  ``src.cv_pipeline.orientation`` (source rule or user); with an unknown orientation nothing is
  measured (``orientation_known=False``). The old auto-nadir guess was right 26% of the time.
* PINGMapper shadows are **2-4 px-wide vertical lines**, so the measurement follows the darkest
  3-px line below the echo instead of averaging a wide column that dilutes it.
* Height is reported **relative to the sonar altitude** (``height_rel = h/H = Ls/(R+Ls)``). A value in
  metres is only produced when the altitude was *measured* (``altitude_m``); the old fixed 10 m
  assumption is gone (the crab-pot bays are a few metres deep, so it was off by 3-5x).

Geometry: the core measurement is written for "far range = down"; other orientations are handled
by a tested 90° rotation. If Stage 1 bottom-tracking supplies the altitude in pixels, slant range is
converted to ground range (``ground = sqrt(slant^2 - alt^2)``) before the height ratio.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .types import ShadowProof, ShadowQuality

# nadir edge → number of CCW 90° rotations that make "far range" point down (see _rotate_*).
_NADIR_TO_K = {"top": 0, "right": 1, "bottom": 2, "left": 3}


@dataclass
class ShadowConfig:
    nadir: Optional[str] = None     # default edge when the caller passes none (None = unknown)
    search_frac: float = 4.0        # search the shadow up to this × box-height beyond the echo
    line_px: int = 3                # width of the thin shadow line followed (PINGMapper: 2-4 px)
    shadow_ratio: float = 0.70      # a range step is "shadow" if its mean < ratio × local background
    clear_contrast: float = 0.25    # contrast ≥ this AND run ≥ clear_run ⇒ CLEAR
    clear_run_px: int = 8
    weak_contrast: float = 0.12     # contrast ≥ this AND run ≥ weak_run ⇒ WEAK
    weak_run_px: int = 4
    altitude_m: Optional[float] = None   # MEASURED sonar altitude (m); None ⇒ relative height only


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


def _darkest_line(g: np.ndarray, x1: int, x2: int, y0: int, y1: int, lp: int):
    """Darkest ``lp``-px-wide vertical line in ``g[y0:y1, x1:x2]`` → (value, left column). The per-
    column statistic is the 30th percentile over rows: robust to the echo's bright tail at the top
    of the window, yet still dark when a shadow fills only part of the search depth."""
    win = g[y0:y1, x1:x2]
    if win.shape[0] < 2 or win.shape[1] < 1:
        return None
    colmean = np.percentile(win.astype(np.float32), 30, axis=0)
    k = max(1, min(lp, colmean.size))
    sm = np.convolve(colmean, np.ones(k) / k, mode="valid")
    c = int(np.argmin(sm))
    return float(sm[c]), x1 + c


class ShadowProver:
    """Measure the acoustic shadow + relative height for candidate boxes. Reuse one instance."""

    def __init__(self, cfg: ShadowConfig | None = None):
        self.cfg = cfg or ShadowConfig()

    # -- nadir (diagnostic only) --------------------------------------------------------------
    def calibrate_nadir(self, gray: np.ndarray) -> Optional[str]:
        """Return the configured nadir edge. ``"auto"`` runs the legacy darkest-border guess — kept
        for diagnostics only (26% correct on real sonograms); the product resolves orientation with
        ``src.cv_pipeline.orientation.resolve_orientation`` instead."""
        if self.cfg.nadir == "auto":
            from src.cv_pipeline.orientation import auto_guess_nadir
            return auto_guess_nadir(gray)
        return self.cfg.nadir

    # -- main entry --------------------------------------------------------------------------
    def prove(self, gray: np.ndarray, box, nadir: Optional[str] = None,
              altitude_px: Optional[float] = None, altitude_m: Optional[float] = None) -> ShadowProof:
        """Measure the shadow for ``box`` (x1,y1,x2,y2). ``gray`` may be BGR or single-channel.

        ``nadir`` is the resolved edge (falls back to the config default); unknown ⇒ not measured.
        ``altitude_px`` (Stage-1 bottom track) enables the slant→ground correction; ``altitude_m``
        (a *measured* altitude) is the only way a height in metres is produced."""
        g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        nadir = nadir or self.calibrate_nadir(g)
        if nadir not in _NADIR_TO_K:
            x1, y1, x2, y2 = (int(v) for v in box)
            proof = self._empty(x1, y1, x2, y2)
            proof.orientation_known = False
            return proof
        alt_m = altitude_m if altitude_m is not None else self.cfg.altitude_m
        k = _NADIR_TO_K[nadir]
        if k == 0:
            return self._prove_down(g, box, altitude_px, alt_m)
        # rotate so far-range points down, measure, then map overlay coords back
        grot = np.rot90(g, k)
        proof = self._prove_down(grot, _rotate_box(box, k, h, w), altitude_px, alt_m)
        hr, wr = grot.shape
        ex, ey = _rotate_point(proof.echo_xy[0], proof.echo_xy[1], (4 - k) % 4, hr, wr)
        sx1, sy1, sx2, sy2 = _rotate_box(proof.strip, (4 - k) % 4, hr, wr)
        proof.echo_xy = (ex, ey)
        proof.strip = (sx1, sy1, sx2, sy2)
        return proof

    # -- core measurement: far range = down (+row) -------------------------------------------
    def _prove_down(self, g: np.ndarray, box, altitude_px: Optional[float] = None,
                    altitude_m: Optional[float] = None) -> ShadowProof:
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

        # search window just past the echo, extending to far range (down)
        sy0 = min(H - 1, echo_y + max(2, int(bh * 0.2)))
        sy1 = min(H, sy0 + int(bh * cfg.search_frac) + 4)
        half = max(cfg.line_px, bw // 2)
        wx1, wx2 = max(0, echo_x - half), min(W, echo_x + half + 1)
        win = g[sy0:sy1, wx1:wx2].astype(np.float32)
        if win.size == 0 or win.shape[0] < 2:
            return self._empty(x1, y1, x2, y2, echo_ratio, (echo_x, echo_y))

        # thin-line shadow: darkest ``line_px``-wide column below the echo (PINGMapper shadows are
        # 2-4 px wide vertical lines; a wide mean would dilute them).
        stat = _darkest_line(g, wx1, wx2, sy0, sy1, cfg.line_px)
        if stat is None:
            return self._empty(x1, y1, x2, y2, echo_ratio, (echo_x, echo_y))
        dark, c1 = stat
        c2 = min(W, c1 + cfg.line_px)

        # Selection-unbiased contrast: "the darkest line" always finds *something* in speckle, so
        # compare it with the darkest line found the SAME way in same-size flank windows (seabed
        # beside the object at the same range). Measured on 952 labelled pots vs random seabed
        # boxes (review sweep 2): the naive version called 30% of empty seabed a WEAK shadow.
        wwin = wx2 - wx1
        refs = []
        for off in (-3, -2, 2, 3):
            a = wx1 + off * wwin
            if 0 <= a and a + wwin <= W:
                r = _darkest_line(g, a, a + wwin, sy0, sy1, cfg.line_px)
                if r is not None:
                    refs.append(r[0])
        ref = max(float(np.median(refs)) if refs else ring, 1.0)
        contrast = (ref - dark) / ref

        # run length: longest contiguous stretch of the (3-row smoothed) line darker than
        # ``shadow_ratio`` × the local seabed brightness.
        col = g[sy0:sy1, c1:c2].astype(np.float32).mean(axis=1)
        if col.size >= 3:
            col = np.convolve(col, np.ones(3) / 3, mode="same")
        flank = [a.ravel() for a in (g[sy0:sy1, max(0, wx1 - wwin):wx1], g[sy0:sy1, wx2:min(W, wx2 + wwin)])
                 if a.size]
        local_bg = max(float(np.median(np.concatenate(flank))) if flank else ring, 1.0)
        dark_rows = col < cfg.shadow_ratio * local_bg
        run = start_i = cur = cur_start = 0
        for i, d in enumerate(dark_rows):
            if d:
                if cur == 0:
                    cur_start = i
                cur += 1
                if cur > run:
                    run, start_i = cur, cur_start
            else:
                cur = 0

        # quality classification (honest: NONE is common and OK) — BOTH contrast and length needed
        if contrast >= cfg.clear_contrast and run >= cfg.clear_run_px:
            quality = ShadowQuality.CLEAR
        elif contrast >= cfg.weak_contrast and run >= cfg.weak_run_px:
            quality = ShadowQuality.WEAK
        else:
            quality = ShadowQuality.NONE

        # height from shadow geometry, RELATIVE to the sonar altitude H:  h/H = Ls / (R + Ls).
        # With a bottom-tracked altitude (px) both ends are converted slant → ground range first.
        height_rel = 0.0
        if run > 0:
            near_px = float(sy0 + start_i)                 # slant range to the shadow start
            far_px = near_px + run                         # slant range to the shadow end
            if altitude_px and 0 < altitude_px < near_px:
                near_g = float(np.sqrt(near_px ** 2 - altitude_px ** 2))
                far_g = float(np.sqrt(far_px ** 2 - altitude_px ** 2))
                ls, rng = far_g - near_g, near_g
            else:
                ls, rng = float(run), max(1.0, float(echo_y))
            height_rel = ls / (rng + ls) if (rng + ls) > 0 else 0.0
        height_m = altitude_m * height_rel if (run > 0 and altitude_m) else None

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
        if not proof.orientation_known:
            cv2.putText(vis, "shadow: orientation unknown", (x1, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
            return vis
        cv2.circle(vis, proof.echo_xy, 3, (0, 0, 255), -1)
        colour = {ShadowQuality.CLEAR: (0, 220, 0), ShadowQuality.WEAK: (0, 170, 220),
                  ShadowQuality.NONE: (120, 120, 120)}[proof.quality]
        sx1, sy1, sx2, sy2 = [int(v) for v in proof.strip]
        cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), colour, 1)
        h_txt = (f" h~{proof.height_m:.1f}m" if proof.height_m is not None
                 else (f" h~{100 * proof.height_rel:.0f}%alt" if proof.run_px else ""))
        cv2.putText(vis, f"shadow:{proof.quality.value} c{proof.contrast:.2f}{h_txt}",
                    (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
        return vis
