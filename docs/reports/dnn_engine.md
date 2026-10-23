# STUDY-20 — OpenCV 5's new DNN engine on DEPTH's detector: speed and same answers

_`python -m src.bench.dnn_engine` · protocol in the module docstring (fixed before the first run). Every number names its machine; no number here is from a test split._

OpenCV 5 loads an ONNX model with one of two engines: the **classic** one (OpenCV 4's importer and layers) or the **new** graph engine. DEPTH asks for `ENGINE_AUTO`, which tries the new one first. A `cv2.dnn.Net` cannot say which engine built it, so the product now loads the engines explicitly and records the one that loaded (`infer.ENGINE` → `/api/health` `runtime.engine`, the provenance stamp on every result, and every benchmark JSON).

## Which engine does ENGINE_AUTO load?

| run | machine | OpenCV | ENGINE_AUTO = | bit-identical on | layers (classic · new) | new vs classic, max raw output gap |
|---|---|---|---|---|---|---|
| `baseline_x86` | x86_64 · c8i.xlarge | 5.0.0 stock | **new** | 8 of 8 frames | 250 · 212 | 0.0043 |
| `graviton_stock` | aarch64 · c8g.xlarge | 5.0.0 stock | **new** | 8 of 8 frames | 250 · 212 | 0.0028 |
| `graviton_stock_large` | aarch64 · c8g.large | 5.0.0 stock | **new** | 8 of 8 frames | 250 · 212 | 0.0028 |
| `laptop_x86` | AMD64 · Intel64 Family 6 Model 140 Ste | 5.0.0 stock | **new** | 8 of 8 frames | 250 · 212 | 0.0044 |

## Speed — one `cv2.dnn` forward of the 640² detector input

Interleaved rounds, rotated engine order, the same frames for every engine. Median (p90) per forward; the speed-up is claimed only if the per-round medians do not overlap.

| run | threads | classic | new | auto (default) | classic ÷ new | saved per forward | per-round ratio | claim |
|---|--:|--:|--:|--:|--:|--:|---|---|
| `baseline_x86` | 4 | 133.9 (134.6) ms | **148.9** (150.1) ms | 148.8 ms | **0.899×** | -15.0 ms | 0.897–0.901× | round ranges separate (slower) |
| `baseline_x86` | 1 | 341.8 (343.3) ms | **291.7** (293.2) ms | 291.2 ms | **1.172×** | 50.1 ms | 1.169–1.176× | round ranges separate |
| `graviton_stock` | 4 | 127.4 (127.7) ms | **187.3** (187.7) ms | 187.1 ms | **0.68×** | -59.9 ms | 0.679–0.682× | round ranges separate (slower) |
| `graviton_stock` | 1 | 430.5 (436.7) ms | **608.9** (615.1) ms | 607.9 ms | **0.707×** | -178.4 ms | 0.705–0.712× | round ranges separate (slower) |
| `graviton_stock_large` | 2 | 255.5 (256.1) ms | **376.7** (377.4) ms | 376.3 ms | **0.678×** | -121.2 ms | 0.678–0.679× | round ranges separate (slower) |
| `laptop_x86` | 8 | 411.3 (512.7) ms | **373.8** (519.8) ms | 359.8 ms | **1.1×** | 37.5 ms | 1.014–1.137× | round ranges overlap - within noise, no claim |
| `laptop_x86` | 1 | 764.9 (1196.2) ms | **886.0** (1380.8) ms | 1137.6 ms | **0.863×** | -121.1 ms | 0.622–1.429× | round ranges overlap - within noise, no claim |

## Same answers?

Both engines' outputs through the product's decoder at the calibrated thresholds. A detection matches when the class is equal, IoU ≥ 0.9, |Δscore| ≤ 0.01 and its tier at τ_review is equal (STUDY-17's parity rule).

| run | frames | detections (new) | identical | differ | missing | extra | tier flips | max \|Δscore\| |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| `baseline_x86` | 8 · webui/samples (the timing frames) | 31 | **31** | 0 | 0 | 0 | **0** | 0.0000 |
| `graviton_stock` | 8 · webui/samples (the timing frames) | 31 | **31** | 0 | 0 | 0 | **0** | 0.0000 |
| `graviton_stock_large` | 8 · webui/samples (the timing frames) | 31 | **31** | 0 | 0 | 0 | **0** | 0.0000 |
| `laptop_x86` | 163 · DATASET/03_yolo_ready_dataset_v2b/val *wcp_ss_* (calibration split, unique frames) | 634 | **634** | 0 | 0 | 0 | **0** | 0.0000 |

**Noise floor.** `auto` and `new` load the same engine (bit-identical outputs), so the gap between their medians is this machine's timing noise, not an engine effect:

| run | threads | auto vs new (same engine) | rounds where classic was slower |
|---|--:|--:|--:|
| `baseline_x86` | 4 | 0% | 0 of 5 |
| `baseline_x86` | 1 | 0% | 5 of 5 |
| `graviton_stock` | 4 | 0% | 0 of 5 |
| `graviton_stock` | 1 | 0% | 0 of 5 |
| `graviton_stock_large` | 2 | 0% | 0 of 5 |
| `laptop_x86` | 8 | 4% | 5 of 5 |
| `laptop_x86` | 1 | 28% | 1 of 5 |

## What it means for DEPTH

- `baseline_x86` (x86_64 · c8i.xlarge, 4 threads): ENGINE_AUTO loads the **new** engine; the **classic** engine is faster here: 133.9 vs 148.9 ms per forward (**1.11×**, 15.0 ms saved per forward; round ranges separate); the forward is 92% of the 146-ms product frame (`baseline_x86__product.json`); at 1 thread(s) the **new** engine is faster (341.8 classic vs 291.7 ms new), so the winner depends on the thread count too. Answers: 31 of 31 detections identical, 0 tier flip(s).
- `graviton_stock` (aarch64 · c8g.xlarge, 4 threads): ENGINE_AUTO loads the **new** engine; the **classic** engine is faster here: 127.4 vs 187.3 ms per forward (**1.47×**, 59.9 ms saved per forward; round ranges separate); the forward is 92% of the 139-ms product frame (`graviton_stock__product.json`). Answers: 31 of 31 detections identical, 0 tier flip(s).
- `graviton_stock_large` (aarch64 · c8g.large, 2 threads): ENGINE_AUTO loads the **new** engine; the **classic** engine is faster here: 255.5 vs 376.7 ms per forward (**1.47×**, 121.2 ms saved per forward; round ranges separate); the forward is 96% of the 268-ms product frame (`graviton_stock_large__product.json`). Answers: 31 of 31 detections identical, 0 tier flip(s).
- `laptop_x86` (AMD64 · Intel64 Family 6 Model 140 Ste, 8 threads): ENGINE_AUTO loads the **new** engine; classic ÷ new is 1.1×, but the rounds overlap: **no speed claim on this machine**. Answers: 634 of 634 detections identical, 0 tier flip(s).
- The look-harder mode runs 5 forwards per frame, so the per-forward saving counts five times there.
- Where the parity table shows 0 tier flips, the calibrated promise does not depend on the engine: forcing the classic one (`DEPTH_DNN_ENGINE=classic`) changes no decision on those frames.
- Which engine is faster depends on the machine, so the EC2 server does not assume it: `DEPTH_DNN_ENGINE=fastest` (`infra/depth.service`) loads both when the server starts, checks they give the same output on a fixed input, times them in interleaved rounds and keeps the classic engine only if it was faster in every round (`infer.faster_engine`); `/api/health` → `runtime.engine_choice` shows the evidence. The same rule runs on a COOL machine, whatever its build favours.
- More rows (x86 and Graviton at 4 vCPU, COOL) are added by the same command on each machine (`infra/bench_cool.sh`, run on every machine by `infra/bench_fleet.sh`).

---
_Regenerated by the script from `benchmarks/*__dnn_engine.json`; do not edit by hand._
