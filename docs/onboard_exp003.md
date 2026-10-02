# Onboarding EXP-003

_`python -m src.detection.onboard_model` · 2026-10-02 17:20_

- classes `['ghost_gear', 'wreck_debris']` · input 640px · ONNX sha256 `71339e99ad2b4b29…` · cv2.dnn 5.0.0: forward OK at 640px, output (1, 6, 8400)
- trained: best epoch 67 of 104 (val mAP50 0.1942), 134.9 min; data `v2b_tiles`
- optimizer built: {"type": "SGD", "lr": 0.01, "nominal_batch": 64, "accumulate": 4}
- **exported checkpoint:** `best_ghost.pt` — best ghost_gear AP@0.5 on the held-out val recordings (tie-break mAP50): ghost_gear AP50 0.3603 (ultralytics' fitness pick: 0.3418)

## Guarantees (calibrated on held-out val recordings, verified once on test)

| | EXP-001 (v1 val/test) | EXP-003 (v2b val/test) |
|---|--:|--:|
| recall ceiling at the detector floor (calibration) | 0.7239 | 0.8495 |
| recall promise (95%) | ≥ 0.65 | ≥ 0.79 |
| requested 90% achievable | False | False |
| precision promise (auto-confirm) | None | None |
| held on test | True | True |

_Different splits: EXP-001 is scored on its own unseen frames (v1); the fair model-vs-model comparison is the v2b test below, where EXP-001 is NOT leakage-free (v1 contains those recordings)._

## Speed (cv2.dnn forward on the onboarding machine, measured back to back)

| | EXP-001 @ 640 | EXP-003 @ 640 |
|---|--:|--:|
| forward ms p50 / p95 | 168.0 / 178.5 | 166.6 / 184.8 |
| compute per survey-hour of sonar (s) | 28.3 | 28.1 |

_OpenCV 5.0.0 · 8 threads · 169 frames per survey-hour. Re-measure on the deployment instance before switching (COOL benchmark)._

## Evaluation reports

- `test` → [docs/eval_exp003_test.md](docs/eval_exp003_test.md)
- `test_official398` → [docs/eval_exp003_test_official398.md](docs/eval_exp003_test_official398.md)
- `test_xsonar` → [docs/eval_exp003_test_xsonar.md](docs/eval_exp003_test_xsonar.md)

## Switch the product to this model

```bash
DEPTH_MODEL=EXP-003 python -m uvicorn src.dashboard.app:app --port 8000   # local
```
COOL server: `aws s3 cp models/EXP-003/best.onnx s3://<bucket>/models/EXP-003/best.onnx`, then set `DEPTH_MODEL=EXP-003` in `infra/depth.service` (or re-run `setup_cool_instance.sh` with `DEPTH_MODEL=EXP-003`).

Log every number above in `experiments.md` (EXP-002 entry) before switching the demo.
