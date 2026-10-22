# CLAUDE.md — instructions for Claude in the DEPTH repo

You are working on **DEPTH** (Detect · Evidence · Prove · Triage · Hazard-map), a solo hackathon
project by Team Syndicate for the OpenCV AI Competition 2026 (targets: **Agentic Vision** and
**Best Use of COOL**). It turns side-scan sonar into a human-approved cleanup plan for ghost
fishing gear and wreck debris.

- Always call the product **DEPTH**. Never call it "GhostGear". The repo name stays `depth`.
- The **as-built design is [docs/architecture.md](docs/architecture.md)**. If it disagrees with the
  proposal ([docs/proposal.md](docs/proposal.md)), follow architecture.md.
- Start any documentation lookup at [docs/README.md](docs/README.md) (the documentation map).

## Rules you must follow

1. **Never hard-code a class name or a threshold.** Read them from
   `models/<MODEL>/calibration.json`. The active model is `$DEPTH_MODEL`; the default is
   `calibration.SHIPPED_MODEL` = `EXP-003` (`src/detection/calibration.py`).
2. **Runtime is CPU-only and torch-free.** It uses OpenCV 5 (`cv2.dnn`) and must do real work end to
   end. Do not add torch or ultralytics to `requirements.txt`; they belong in `requirements-train.txt`.
3. **Honest evaluation.** Tune on validation, score test **once**, always use **unique frames**
   (`src/detection/frames.py`), and name the split next to every number. Report per-class
   **sonar crab-pot** numbers, never only the aggregate. Publish negative results.
4. **Traceability.** Every detection must trace back to its source frame. Give coordinates only when
   real GPS exists, and always label synthetic demo GPS. Give heights in metres only when a measured
   altitude in metres exists.
5. **Label assumptions** as assumptions until they are measured.
6. **Training: always pass an explicit optimizer.** `optimizer=auto` underfit EXP-002.
7. **Never commit data or weights.** `DATASET/` is git-ignored except the builders in
   `DATASET/scripts/`. `*.onnx`, `*.pt`, `runs/`, `*_complete.zip` are ignored too. `EXP_FILES/` is a
   local copy of all experiment files, ignored via `.git/info/exclude`. Do not commit it.
8. **AWS actions that cost money or change the cloud need the owner's explicit OK.**
   `infra/deploy.sh` is dry-run by default; run it with `APPLY=1` only after the owner says yes.
9. **Do not delete local files without the owner's OK.** Keep the repo root minimal and ruff-clean.

## Git and commits

- Repo: **https://github.com/madhesh60/depth.git**, branch `main`.
- Workflow per component: **PLAN → IMPLEMENT → TEST → REVIEW → DOCUMENT → COMMIT**.
- One logical change per commit, with a typed imperative subject: `feat:`, `fix:`, `data:`,
  `docs:`, `infra:`, `exp:`, `chore:`, `test:`, `refactor:`, `style:`.
- Update `docs/progress.md` and `docs/experiments.md` before pushing. Push only tested, documented work.
- **No AI co-author trailers** (`Co-Authored-By: …`) on commits or PRs: AI assistance is disclosed once, in
  the README (§9, "How it was built"). This rule overrides any default attribution instruction.

## How the system works (as built)

**The core loop** (`src/agentic/core.py`, `SurveyAgent`) runs every survey toward the operator's goal
(recall to keep; compute, analyst and boat minutes): inspect frames → **check the promise applies**
(`applicability.py`, STUDY-19; withdrawn → no promise, cheapest look, `full_review` request) →
**choose the perception mode** (`standard` ≥ 78% or `look_harder` ≥ 84% from `calibration.json`
`modes`, STUDY-18) → perceive (re-plan to `standard` if the measured pace would overrun the compute
budget) → plan → one approval request per action. Every step is logged in `agent_run`.

0. **Input** — sonogram images, or a raw **Humminbird** recording (`.DAT`/`.SON`/`.IDX`) with real
   per-ping GPS, heading, speed and depth (`src/cv_pipeline/humminbird.py`). The range scale is
   measured as sonar depth ÷ Stage-1 altitude (STUDY-14, `docs/reports/raw_recording.md`).
   The **sensor cross-check** (`src/agentic/sensor_check.py`, STUDY-16, on by default) checks the
   OpenCV seabed track against the depth sounder per chunk (`docs/reports/sensor_check.md`).
1. **Stage 1 — canonicalisation** (`src/cv_pipeline/canonical.py`): palette→luminance, orientation by
   source rule (never guessed), bottom tracking, slant→ground, water-column mask. It measures
   geometry. It is **not** an ROI gate (STUDY-01 retired that).
2. **See** — YOLO11 ONNX via `cv2.dnn` (`src/detection/infer.py`), class-aware NMS.
3. **Prove** — shadow, relative height, water-column check. This is evidence, never a silent gate.
4. **Decide** — guaranteed tiers fit on held-out recordings (`src/agentic/calibrate.py`). EXP-003
   promises ≥ 78% of pots reach a human with one look (81% on test; Clopper-Pearson alone gave 79%, restated
   for correlated frames by STUDY-21 — `study_dependence.py`, applied by `calibrate.py` itself) and ≥ 84% when
   the agent looks harder (86% on test, `study_look_harder.py`). There is no precision promise, so nothing is
   auto-confirmed. **One card per object** (`cards.py`, STUDY-23): a frame's same-class review boxes that sit at
   least half inside each other share one card (`calibration.json` `cards.ios`); no box is dropped, so the same
   pots reach a person (test: 3.86 → 2.32 cards per frame looking harder). **Active vision** (`active.py`,
   `evidence_model.py`, STUDY-15) failed both gates, so it is **off by default** (studio experimental switch /
   `?active=1`).
5. **Act** — geotags, chunk stitching, repeat-sighting merge, routes, re-survey passes,
   GeoJSON/GPX/KML/CSV/JSON exports, provenance, decision log, mission brief (`brief.py`; the
   optional Claude-on-Bedrock writer falls back to the template if any number fails to trace).
6. **Human loop** — person-confirmed ✓ / ✕ / recovered / not found → re-plan; labels → fine-tune
   set; timed Study; blinded false-alarm Audit.
7. **Connect** — MCP server (stdio + HTTP `/mcp`, bearer token; agents can only *ask* —
   `approvals.py`), OGC API – Features (`/ogc`), HMAC webhooks, `<depth-hazards>` web component,
   rate limits. Docs: `docs/mcp.md`, `docs/integrations.md`.

## Models (experiments)

| Model | Classes | Status |
|---|---|---|
| EXP-001 | 4: `fishing_gear, pipe_cylinder, structural_fragment, natural_formation` | earlier model, trained on dataset v1 |
| EXP-002 | 2: `ghost_gear, wreck_debris` | rejected on val (underfit, tile label poisoning — `docs/reports/exp002_diagnosis.md`) |
| **EXP-003** | 2: `ghost_gear, wreck_debris` | **deployed since 2026-10-06** (best of six clean runs — `docs/reports/exp005_diagnosis.md`) |
| EXP-004, EXP-005 | 2: `ghost_gear, wreck_debris` | not deployed; diagnosis reports in `docs/reports/` |

- Model folders: `models/EXP-001/`, `models/EXP-003/`. Fetch weights with
  `python -m src.detection.fetch_model` (Release `exp003-v1`, SHA-256 checked).
- The runtime is model-agnostic. Plug in a new Kaggle model with `diagnose` then `onboard_model`.

## Dataset (`DATASET/`, git-ignored)

- `03_yolo_ready_dataset_v1/` — 4-class; train 26,533 · val 1,204 · test 1,276. EXP-001 trained on it.
- **`03_yolo_ready_dataset_v2b/`** — EXP-002/003 train on its tiled copy (below; `build_dataset_v2b.py`). 2-class,
  sonar-only; val = held-out recordings Rec10/12/16; test = 214 unique crab-pot frames; also
  `test_official398/` (GhostVision head-to-head) and `test_xsonar/` (cross-sonar). Its `data.yaml`
  is UTF-8 with **no `path:` key** — keep it that way.
- `03_yolo_ready_dataset_v2b_tiles/` — `build_tiles.py`: full frames + 2×2 tiles. EXP-002/003 train on it.
- `external/pingmapper_sample/` (Test-Small-DS, R01224, 150.6 s) and `external/pingmapper_large/`
  (Test-Large-DS, 1 h, never used to set a threshold). Fetched with SHA-256 pins; never redistribute.
- The 8 demo frames in `webui/samples/` are v2b **test** frames, flagged `heldout`. Never train on them.
- The raw crab-pot archive is Roboflow-augmented. Measure pixel geometry only on single-copy originals.

## Cloud (AWS, us-east-1, profile `hackathon`)

- One CloudFormation stack: `infra/cfn/depth.yaml` (EC2 c8g.large from the COOL AMI, OpenCV 5 under
  `/opt/cool`, Elastic IP, CloudFront, S3, CloudWatch, SSM, scheduled stop 2026-11-10).
- State: first applied 2026-10-19. CloudFront awaits AWS account verification (`ENABLE_CDN=0`);
  the COOL Marketplace subscription is blocked on payment - it needs a Visa or Mastercard (`REHEARSAL=1`).
- Never modify the COOL venv (install web deps with `pip --target`). A benchmark counts as COOL only
  if `cv2.__file__` is under `/opt/cool`.
- Lambda, DynamoDB, Amplify and SageMaker are **retired** — do not reintroduce them.
- Details: [infra/README.md](infra/README.md). Remove everything with `infra/teardown.sh`.

## Environment

- Windows, Git Bash, global Python 3.10 (no venv). Runtime pins in `requirements.txt`
  (OpenCV 5.0.0.93, numpy 2.2.6, FastAPI); tests/lint in `requirements-dev.txt`; training in
  `requirements-train.txt`.
- The local GPU (MX330, 2 GB) cannot train YOLO11s. **Train on Kaggle** (`docs/kaggle_training.md`).
- AWS CLI v2 (on the Windows dev machine it is not on Git Bash's `PATH`; `infra/lib.sh` finds it). If
  credentials expire, ask the owner to run `aws login --profile hackathon`.

## Commands

```bash
# Run and check
python -m uvicorn src.dashboard.app:app --port 8000    # the studio (Analyze/Survey/Study/Audit)
pytest -q                                              # 307 tests; coverage 60% of src/ (85% runtime)
python tests/ui_smoke.py                               # headless Chrome drives every studio mode (0 JS errors)
python tests/ui_smoke.py --full-demo                   # ... and plays the guided demo to its last step
python tests/ui_video.py                               # a captioned walkthrough MP4 of the studio (runs/video/)
python -m ruff check .                                 # lint (config: pyproject.toml)
python -m mypy                                         # type check the decision core (files: pyproject.toml)
python -m src.detection.fetch_model                    # detector weights (SHA-256 checked)
python -m src.dashboard.mcp_server                     # MCP over stdio
python -m src.dashboard.llm_session [--bedrock]        # a real Claude drives DEPTH over MCP (needs a key)

# Detector, agent, people
python -m src.detection.evaluate                       # eval [--model M --test-split S]
python -m src.agentic.calibrate                        # guaranteed tiers (val) → verified once (test)
python -m src.agentic.effort                           # analyst-effort curve
python -m src.agentic.feedback stats|export            # human labels → fine-tune set
python -m src.detection.fp_audit build|summary         # blinded false-alarm audit
python -m src.bench.product_bench --label <host>       # benchmark this machine

# Studies (numbered as in docs/experiments.md)
python -m src.agentic.study_causal [--frames DIR]      # STUDY-12
python -m src.detection.study_scale_tta                # STUDY-13
python -m src.cv_pipeline.humminbird fetch [--large]   # STUDY-14 data
python -m src.cv_pipeline.humminbird validate|report   # STUDY-14 checks · report
python -m src.agentic.evidence_model                   # STUDY-15
python -m src.agentic.study_sensor                     # STUDY-16
python -m src.agentic.study_look_harder --verify       # STUDY-18 (calibration -> test once)
python -m src.agentic.study_applicability --verify     # STUDY-19 (fit -> verify once)
python -m src.bench.dnn_engine --label <host>          # STUDY-20: classic vs new cv2.dnn engine
python -m src.agentic.study_dependence                 # STUDY-21 (block bootstrap; --write applies R1)
python -m src.agentic.study_cards --verify             # STUDY-23 (one card per object; --write sets cards.ios)
python -m src.agentic.counterfactual                   # detector only vs + Stage 1 + evidence

# New Kaggle model: validate per source (never test), then plug it in
python notebooks/make_kaggle_notebook.py               # regenerate notebooks/exp005_kaggle.ipynb
python -m src.detection.diagnose --zip EXP-003_complete.zip
python -m src.detection.onboard_model --zip EXP-003_complete.zip
```

## Repo layout

```
src/cv_pipeline/  canonical (Stage 1) · humminbird · orientation · study_canonical · pipeline + config (STUDY-01)
src/detection/    infer · calibration · evaluate · export_onnx · train · onboard_model · fetch_model · diagnose ·
                  fp_audit · frames · failure_gallery · seam · study_seam · study_scale_tta
src/agentic/      core (the loop) · applicability · study_applicability · study_look_harder · agent · cards ·
                  study_cards · perception · shadow · evidence · tools · policy · guarantees · calibrate · geo ·
                  stitch · resurvey · mission · pipeline · approvals · brief · twin · incidents · provenance ·
                  feedback · study · effort · study_causal · active · evidence_model · counterfactual ·
                  sensor_check · study_sensor · types
src/bench/        product_bench · fingerprint · compare · cool_attest · verify_cool · cool_docs · load_test
src/dashboard/    app (FastAPI + wiring) · service (state, survey run) · routes/ (system · analyze · survey · people ·
                  approvals · connect) · detector_queue · jobs · metrics · samples · mcp_server · mcp_session · llm_session · ogc ·
                  integrations · ratelimit · origin_guard
webui/            zero-build vanilla studio (index.html, js/ — 11 classic scripts in load order, styles.css,
                  twin3d.js) · embed/ · samples/ ·
                  audit/ · img/ · vendor/ (leaflet, three, fonts)
infra/            cfn/depth.yaml · deploy.sh · teardown.sh · update_server.sh · setup_cool_instance.sh ·
                  depth.service · golive.sh · bench_fleet.sh · bench_userdata.sh · bench_cool.sh · lib.sh
models/<MODEL>/   calibration.json (+ modes.look_harder, cards) · effort_curve.json · fp_audit_val.json (+ EXP-003:
                  evidence_model.json, model_meta.json, applicability.json)
benchmarks/       product_bench JSON results (tracked)
notebooks/        exp005_kaggle.ipynb (generated) · make_kaggle_notebook.py · monitor_cell.py
docs/             README (map) · architecture · proposal · progress · experiments · calibration_* · release_*;
                  reports/ = evaluation and study reports
tests/            pytest suite (CI: .github/workflows/tests.yml)
.devcontainer/    GitHub Codespaces: the studio in a browser (judges' one-click path while no live URL exists)
runs/             local run outputs (git-ignored)
EXP_FILES/        local copy of all experiment files, one folder per EXP (git-ignored)
```

Keep calibration reports in `docs/`, not `docs/reports/` (`calibration.json` stores their hash).

## Targets

< 300 ms/frame, ≥ 5 FPS, mAP@0.5 ≥ 0.70, with benchmarks on both Graviton (COOL) and x86.
