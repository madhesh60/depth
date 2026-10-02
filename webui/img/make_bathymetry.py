"""Writes webui/img/bathymetry.svg — the studio's background: depth contours of a synthetic seabed, the
way a nautical chart draws them. Made with OpenCV (a smooth height field → cv2.findContours per depth
level → cv2.approxPolyDP), deterministic (seed 7). Every fifth line is an index contour, drawn a little
stronger, as on real charts.

    python webui/img/make_bathymetry.py
"""
from pathlib import Path

import cv2
import numpy as np

W, H, LEVELS, SEED = 800, 500, 26, 7
PAD = 40                     # computed larger, viewed cropped: edge-hugging contour segments fall outside


def seabed(seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-PAD:H + PAD, -PAD:W + PAD].astype(np.float32)
    z = 0.35 * (y / H)                                        # the seabed deepens away from the coast
    for _ in range(14):                                       # banks, holes and ridges
        cx, cy = rng.uniform(-0.1, 1.1) * W, rng.uniform(-0.1, 1.1) * H
        sx, sy = rng.uniform(60, 260), rng.uniform(50, 200)
        a = rng.uniform(-0.6, 0.6)
        z += a * np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
    noise = cv2.GaussianBlur(rng.normal(0, 1, (H + 2 * PAD, W + 2 * PAD)).astype(np.float32), (0, 0), 38)
    z += 0.9 * noise / (np.abs(noise).max() + 1e-6) * 0.25
    return cv2.normalize(z, None, 0, 1, cv2.NORM_MINMAX)


def main() -> None:
    z = seabed()
    paths = []
    for i, level in enumerate(np.linspace(0.04, 0.96, LEVELS)):
        mask = (z >= level).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        index = i % 5 == 0
        for c in contours:
            if cv2.arcLength(c, True) < 60:
                continue
            c = cv2.approxPolyDP(c, 1.1, True).reshape(-1, 2)
            d = "M" + " ".join(f"{px - PAD},{py - PAD}" for px, py in c) + "Z"
            paths.append(f'<path class="{"i" if index else "c"}" d="{d}"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid slice">'
           '<style>path{fill:none;stroke:#7fdcef;stroke-linejoin:round}'
           '.c{stroke-opacity:.16;stroke-width:.6}.i{stroke-opacity:.30;stroke-width:.9}</style>'
           + "".join(paths) + "</svg>")
    out = Path(__file__).with_name("bathymetry.svg")
    out.write_text(svg, encoding="utf-8")
    print(f"{out} · {len(paths)} contours · {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
