# Stage 1 — sonar canonicalisation (classical OpenCV)

> **Current role (2026-09-24, `experiments.md` STUDY-08).** Stage 1 in the product path is
> **`canonical.py` — sonar canonicalisation**: palette → luminance, orientation by source rule,
> **bottom tracking** (sonar altitude in px per ping; validated port-vs-starboard to a median
> 1.6 px on un-augmented frames), slant → ground range (`cv2.remap`), range-gain normalisation and
> a water-column mask (~5 ms/frame). Its outputs feed the shadow's relative height, the geotag's
> ground range + per-ping position, and a traced `water_column_check`. The detector keeps `raw`
> input (range gain measured: no gain for EXP-001). Report: `docs/stage1_canonical.md`
> (`python -m src.cv_pipeline.study_canonical`). The contour pipeline below (`pipeline.py`) is kept
> as the documented STUDY-01 negative result and a benchmark op-mix.


> **Where COOL is measured (2026-10-19).** Not here. COOL is benchmarked on the **product** workload —
> exactly what `/api/analyze` runs (Stage 1 canonicalisation → YOLO11 via `cv2.dnn` → evidence → render),
> split per op, plus a real 1-h recording end to end — by `src/bench/product_bench.py` on equal-size
> machines (`infra/bench_fleet.sh`; report `docs/cool_benchmark.md`). Everything below describes the
> retired contour pipeline (`pipeline.py`, the STUDY-01 negative result): its op table is history (stock
> OpenCV 4.12 on the x86 laptop) and **no Graviton-vs-x86 or COOL speed-up is claimed for it**.

CPU-only classical CV over a side-scan sonar frame (denoise → resize → adaptive threshold →
morphology → contours → geometry) — the retired ROI pipeline.

> **Role note (2026-09-22, see `experiments.md` STUDY-01).** This pipeline is **not** a
> Stage-2 ROI gate — classical CV has no discriminative power on this sonar (GT coverage caps
> ~72% at 141 ROIs/frame; it fires more on empty seafloor than on debris), so ROI-gating was
> retired. It is kept only to reproduce that result (and as an op-mix reference); it is not in the
> product path and was never measured on Graviton. YOLO
> (`src/detection/infer.py`) does the detection, full-frame. The `candidates` it emits are
> still useful for the transparency/demo view and as a benchmarkable tiling/region pass.

## History: why this was once meant as the COOL workload

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

Budget check: 29 ms/frame ≪ 300 ms target, 30.5 FPS ≫ 5 FPS target (x86 laptop).

## The legacy three-way harness (superseded by `src/bench/product_bench.py`)

Kept for the record; the COOL evidence comes from the product benchmark instead:

```bash
# A) x86 + stock OpenCV
python -m src.cv_pipeline.benchmark --images <frames> --label baseline_x86      --out benchmarks
# B) Graviton + stock OpenCV
python -m src.cv_pipeline.benchmark --images <frames> --label graviton_stock    --out benchmarks
# C) Graviton + COOL
python -m src.cv_pipeline.benchmark --images <frames> --label graviton_cool     --out benchmarks
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
for c in res.candidates:          # transparency/demo view (not a Stage-2 gate — see role note)
    print(c.bbox, c.solidity, c.area_frac)
cv2.imwrite("vis.jpg", draw_candidates(cv2.imread("frame.jpg"), res))
```

All behaviour is set by `Stage1Config` (see `config.py`) — no magic numbers in the pipeline.

## Files

- `config.py` — every tunable + COOL-friendly defaults; `QUALITY_PRESET` for the ablation.
- `pipeline.py` — `Stage1Pipeline.process()` (per-op timed) + `draw_candidates()`.
- `benchmark.py` — the three-way latency/throughput/FPS harness (`python -m`).
