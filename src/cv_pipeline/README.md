# Stage 1 — Classical OpenCV pipeline (the COOL core workload)

CPU-only classical CV that turns a side-scan sonar frame into geometry-filtered candidate
ROIs for the Stage 2 YOLO verifier. This is the **workload we run on AWS Graviton via COOL**
and benchmark against x86 — the primary "Best Use of COOL" deliverable.

## Why this is the COOL workload

The COOL / KleidiCV build accelerates a specific set of ops on Arm/Graviton. Stage 1 is
deliberately composed of those ops, and every op is timed so we can prove where the compute
goes. Local x86 (stock OpenCV 4.12) baseline over 400 sonar frames:

| op | mean ms | COOL-accelerated? | share |
|---|---|---|---|
| to_gray | 0.003 | – | — |
| **resize** | 0.65 | ✅ | 2% |
| denoise (median) | 3.81 | – (cheap) | 13% |
| **threshold (adaptive-gaussian)** | 9.77 | ✅ | 34% |
| morphology | 0.82 | – | 3% |
| **contours (findContours)** | 11.58 | ✅ | 40% |
| geometry_filter | 2.54 | – | 9% |
| **TOTAL** | **29.2** | | 30.5 FPS |

**~76% of compute lives in the three COOL-accelerated ops** (resize + adaptive threshold +
contours). That concentration is what makes COOL's speedup attributable — not diluted by a
heavy non-accelerated op. (This is why `fastNlMeansDenoising` is OFF by default: it is slow
and *not* KleidiCV-accelerated; it lives in `QUALITY_PRESET` for the ablation only.)

Budget check: 29 ms/frame ≪ 300 ms target, 30.5 FPS ≫ 5 FPS target — with 10× headroom for
Graviton.

## The three-way benchmark (earns 30% + 20% of the COOL score)

Run the identical pipeline/config in three configs and diff the JSON:

```bash
# A) x86 + stock OpenCV
python -m src.cv_pipeline.benchmark --images <frames> --label baseline_x86      --out runs/bench
# B) Graviton + stock OpenCV
python -m src.cv_pipeline.benchmark --images <frames> --label graviton_stock    --out runs/bench
# C) Graviton + COOL
python -m src.cv_pipeline.benchmark --images <frames> --label graviton_cool     --out runs/bench
```

- **(B) → (C)** on identical hardware **isolates COOL's contribution** → the 30% "verified
  COOL integration" number.
- **(A) → (C)** is the headline Arm-vs-x86 story.
- Each JSON self-documents its OpenCV build (`cool_or_kleidicv_detected`), CPU arch, threads,
  config, and per-op latency — so the benchmark is reproducible (the 10% criterion).

## Usage

```python
from src.cv_pipeline import Stage1Pipeline, draw_candidates
import cv2
res = Stage1Pipeline().process(cv2.imread("frame.jpg"))
for c in res.candidates:          # -> Stage 2
    print(c.bbox, c.solidity, c.area_frac)
cv2.imwrite("vis.jpg", draw_candidates(cv2.imread("frame.jpg"), res))
```

All behaviour is set by `Stage1Config` (see `config.py`) — no magic numbers in the pipeline.

## Files

- `config.py` — every tunable + COOL-friendly defaults; `QUALITY_PRESET` for the ablation.
- `pipeline.py` — `Stage1Pipeline.process()` (per-op timed) + `draw_candidates()`.
- `benchmark.py` — the three-way latency/throughput/FPS harness (`python -m`).
