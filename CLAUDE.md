# DEPTH — Marine Debris Detection — Team Syndicate

Solo hackathon project (OpenCV AI Competition 2026 — targets **Agentic Vision** and **Best Use of
COOL**). The product is named **DEPTH** (Detect · Evidence · Prove · Triage · Hazard-map); the repo
stays `depth`. Proposal of record: [AGENT.md](AGENT.md) — the **as-built design is
[architecture.md](architecture.md)** and supersedes the proposal where they differ.

## What this is (as built, 2026-09-25)

Side-scan sonar frames → a human-approved cleanup plan for ghost fishing gear / wreck debris.
Runtime is torch-free **OpenCV 5** on CPU (built for AWS Graviton + COOL):

1. **Stage 1 — sonar canonicalisation** (`src/cv_pipeline/canonical.py`): palette→luminance,
   orientation by source rule (never guessed), **bottom tracking** (sonar altitude px per ping),
   slant→ground, water-column mask. It *measures geometry*; it is **not** an ROI gate (STUDY-01
   retired that). Validated: port/starboard altitude agree to 1.6 px (STUDY-08).
2. **See** — YOLO11 ONNX via `cv2.dnn` (`src/detection/infer.py`), class-aware NMS.
3. **Prove** — thin-line shadow, relative height vs the tracked altitude, water-column check,
   agent's-eye re-look view (display). Evidence, never a silent gate.
4. **Decide** — **guaranteed tiers** fit on held-out recordings (Clopper–Pearson / LTT,
   `src/agentic/calibrate.py`): EXP-001 promises ≥ 65% of pots reach a human (held on test);
   no precision promise → nothing auto-confirmed. Value-of-information tool use, P(pot), budgets.
5. **Act** — per-ping ground-range geotag, chunk stitching, repeat-sighting merge, recovery /
   inspection routes, **opposite-side re-survey passes**, GeoJSON/GPX/KML/CSV/JSON.
6. **Human loop** — labels (✓ / ✕ / ＋missed → fine-tune set), timed **Study** (effort curve,
   break-even card time), blinded false-alarm **Audit** with catch trials.

Every runtime threshold comes from **`models/<MODEL>/calibration.json`** (`$DEPTH_MODEL`, default
`EXP-001`): class names, input size, per-class floors, guaranteed class, tiers. Never hard-code a
class name or threshold.

Classes: EXP-001 (deployed) is 4-class `fishing_gear, pipe_cylinder, structural_fragment,
natural_formation`; EXP-002 (v2b) is 2-class `ghost_gear, wreck_debris`. The runtime is
model-agnostic.

## Cloud (scripted, not yet run — AWS work is scheduled for a dedicated day)

EC2 c8g.xlarge from the **COOL AMI** (OpenCV 5 under `/opt/cool`) + systemd + CloudFront (HTTPS) +
S3 + CloudWatch + Budgets + SSM. `infra/deploy_aws.sh` is **dry-run by default** (`APPLY=1` to
execute). The COOL venv is never modified (web deps via `pip --target`). The 3-way COOL benchmark
(`infra/bench_cool.sh`, `src/bench/`) times the **product workload** per stage; a run is COOL only if
`cv2.__file__` is under `/opt/cool`. Lambda / DynamoDB / Amplify / SageMaker are **retired**.
Details: [infra/README.md](infra/README.md).

## Dataset state

`DATASET/` is git-ignored (too large); only `DATASET/scripts/` builders are tracked.

- `03_yolo_ready_dataset_v1/` — 4-class, train 26,533 · val 1,204 · test 1,276; **EXP-001 trained
  on it**. Its unseen crab-pot sonograms (v1 val Rec19 = calibration, v1 test = verification,
  unique frames) host EXP-001's guarantees.
- **`03_yolo_ready_dataset_v2b/`** — **EXP-002 trains on this** (`build_dataset_v2b.py`): 2-class,
  sonar-only, one copy per Roboflow frame, val = held-out recordings Rec10/12/16, test = 214 unique
  crab-pot frames, `test_official398/` (GhostVision head-to-head), `test_xsonar/` (cross-sonar),
  `groups.json`. Its `data.yaml` is UTF-8 with **no `path:` key** (ultralytics resolves a relative
  `path` against the working directory).
- `03_yolo_ready_dataset_v2b_tiles/` — `build_tiles.py` output: full frames + 2×2 tiles (+ optional
  `--paste N` sonar-aware copy-paste); val/test point at v2b's full frames.
- The raw crab-pot archive is Roboflow-augmented (crops/rotations): measure pixel geometry only on
  single-copy originals; always evaluate on **unique frames** (`src/detection/frames.py`).

## Key commands

```bash
python -m uvicorn src.dashboard.app:app --port 8000        # the studio (Analyze/Survey/Study/Audit)
pytest -q                                                  # test suite
python -m src.agentic.calibrate                            # guaranteed tiers (val) → verified once (test)
python -m src.detection.evaluate [--model M --test-split S]  # deploy-faithful detector eval
python -m src.detection.onboard_model --zip EXP-002_complete.zip   # plug in a Kaggle-trained model
python -m src.agentic.effort                               # analyst-effort curve
python -m src.bench.product_bench --label <host>            # benchmark this machine
python -m src.agentic.feedback stats|export                 # human labels → fine-tune set
python -m src.detection.fp_audit build|summary              # blinded false-alarm audit
```

## Environment

Global Python 3.10 (no venv) with the runtime pinned in `requirements.txt` (OpenCV 5.0.0.93,
numpy 2.2.6, FastAPI) plus `ultralytics` + CPU torch for local smoke tests. The local GPU (MX330,
2 GB) cannot train YOLO11s — **training runs on Kaggle** (`docs/exp002_kaggle.md`).
AWS CLI v2 is installed user-scoped (`C:\Users\RAJ\AppData\Local\Programs\Amazon\AWSCLIV2\aws.exe`,
profile `hackathon`, us-east-1); login + the MCP wizard are deferred to the AWS day.

## Repo layout

```
src/cv_pipeline/  canonical.py (Stage 1) · orientation.py · study_canonical.py · pipeline.py (STUDY-01 record)
src/detection/    infer · calibration · evaluate · export_onnx · train · onboard_model · fp_audit · frames
src/agentic/      agent · perception · shadow · tools · policy · guarantees · calibrate · geo · stitch ·
                  resurvey · mission · pipeline · feedback · study · effort · types
src/bench/        product_bench · fingerprint · compare (COOL benchmark)
src/dashboard/    app (FastAPI) · jobs · metrics · samples
webui/            zero-build studio · samples/ (8 CC-BY-SA frames) · audit/ crops · vendor/leaflet
infra/            deploy_aws.sh · setup_cool_instance.sh · depth.service · bench_cool.sh
models/<MODEL>/   calibration.json · effort_curve.json · fp_audit_val.json (weights git-ignored)
tests/            pytest suite
```

## Git & delivery workflow

Repository: **https://github.com/madhesh60/depth.git** (branch `main`).

- **Ship components, not dumps.** When a component of the system is completed and verified,
  **push it to GitHub** as its own small, self-contained commit (or a short series) — don't
  batch unrelated work into one giant commit.
- Use typed, imperative commit subjects: `feat:`, `fix:`, `data:`, `docs:`, `infra:`, `exp:`,
  `chore:`, `test:`. One logical change per commit.
- Follow **PLAN → IMPLEMENT → TEST → REVIEW → DOCUMENT → COMMIT** for each component; only
  push once it's tested and the docs (`progress.md`, `experiments.md`) are updated.
- Version the *code*, never the *data*: `DATASET/` data stays git-ignored; only the
  processing scripts under `DATASET/scripts/` are tracked (see `.gitignore`).
- End every commit message with the co-author trailer:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Key constraints

- CPU-only runtime (Graviton, no GPU); OpenCV 5 must do real work end to end.
- Targets from the proposal: < 300 ms/frame, ≥ 5 FPS, mAP@0.5 ≥ 0.70 — and report the honest
  per-class **sonar crab-pot** numbers, never only the aggregate.
- Every detection traceable to its source frame; coordinates only with GPS (synthetic demo GPS is
  always labelled); metres of height only with a measured altitude in metres.
- Tune on validation, score test once; unique frames; name the split next to every number.
- Benchmarks for both Graviton (COOL) and x86 — the Best-Use-of-COOL criterion.
- Honesty-first: publish negative results; label assumptions until they are measured.
