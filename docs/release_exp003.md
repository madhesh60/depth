# Release `exp003-v1` — EXP-003 detector weights (prepared, not published)

**What it is:** DEPTH's default detector since 2026-10-06.

- YOLO11s at 640 px, 2 classes (`ghost_gear`, `wreck_debris`), trained on dataset v2b.
- Exported to ONNX and run by OpenCV 5 `cv2.dnn` on CPU.
- The thresholds and guaranteed tiers ship in the repo: [`models/EXP-003/calibration.json`](../models/EXP-003/calibration.json).
- The weights are the one file that is not in git.

| | |
|---|---|
| asset | `best.onnx` (from `models/EXP-003/best.onnx`) |
| bytes | 37,929,805 |
| SHA-256 | `71339e99ad2b4b29f8d516967ce02e1fb5895434bf6f483cb517efe6750ba2f2` |
| install | `python -m src.detection.fetch_model` (checks the hash; refuses a mismatch) |

## Numbers (each with its split)

| | value | split |
|---|--:|---|
| recall promise: share of pots that reach a human (95% confidence) | **≥ 79%** | fit on validation (Rec10/12/16) |
| the promise on unseen data | **held: 81.1%** (lower bound 76.7%) | v2b test, 264 pots, scored once |
| ghost gear AP@0.5 (95% CI) | 0.482 (0.43–0.55) | v2b test, 285 pots |
| wreck debris AP@0.5 | 0.156 (weak: see [`exp005_diagnosis.md`](exp005_diagnosis.md)) | v2b test, 70 boxes |
| GhostVision head-to-head, F1 | 0.41 (GhostVision reports 0.71–0.73) | official 398-frame split |
| cross-sonar ghost gear AP@0.5 | 0.34 | `test_xsonar` |
| `cv2.dnn` forward, p50 | 167 ms | laptop x86, 8 threads |

Why this model: it beat every later run on the held-out recordings (paired bootstrap; EXP-004 and
EXP-005 analyses). Records:

- [`onboard_exp003.md`](onboard_exp003.md)
- [`exp003_diagnosis.md`](exp003_diagnosis.md)
- [`exp005_diagnosis.md`](exp005_diagnosis.md)

## Licence — decide before publishing

- **The weights:** AGPL-3.0, from Ultralytics YOLO11s.
- **The training data (v2b, sonar only)** mixes sources with different terms
  ([`dataset_card.md`](dataset_card.md)):

  | source | terms |
  |---|---|
  | PINGEcosystem crab-pot | CC BY-SA 4.0 in the card metadata, "GPL" in its text; we have asked the authors |
  | SeabedObjects-KLSG | research use |
  | AI4Shipwrecks | check page |
  | Marine PULSE | check page |

  Publishing weights trained on "research use" data is a licence call for the owner, not a technical
  step. If in doubt, state "research use" in the release text, or keep the release private and use
  the S3 copy (`infra/deploy_aws.sh`).

## Publish (owner only)

```bash
gh release create exp003-v1 models/EXP-003/best.onnx --repo madhesh60/depth --title "EXP-003 detector weights" --notes-file docs/release_exp003.md
```

Then check: `python -m src.detection.fetch_model` on a fresh clone installs it and the hash matches.
