# DEPTH documentation — where to find what

Every number in these documents names its split and the command that regenerates it. Reports marked
*generated* are written by a script; do not edit them by hand.

## Start here

| read | for |
|---|---|
| [`technical_report.md`](technical_report.md) | the whole system, its evaluation and its limits, in one document |
| [`agentic_vision.md`](agentic_vision.md) | the Agentic Vision entry: how perception changes the agent's next action |
| [`agentic_loop.md`](agentic_loop.md) | the loop step by step: SEE → PROVE → DECIDE → ACT, with a person in charge |
| [`../architecture.md`](../architecture.md) | the as-built design, module by module |
| [`model_card.md`](model_card.md) · [`dataset_card.md`](dataset_card.md) | the deployed detector (EXP-003) and the data it learned from |
| [`responsible_use.md`](responsible_use.md) | human control, protected wrecks, honest geotags, data |
| [`references.md`](references.md) | the published work behind specific design choices |

## Evidence — the claims and the measurements behind them

| claim | report |
|---|---|
| ≥ 79% of pots reach a person (95% confidence), held at 81.1% on unseen test | [`calibration_exp003.md`](calibration_exp003.md) *(generated)* |
| detector: crab-pot AP@0.5 0.482, mAP@0.5:0.95 0.106; confusion matrix + PR curve | [`eval_exp003_test.md`](eval_exp003_test.md) · [`…_official398`](eval_exp003_test_official398.md) · [`…_xsonar`](eval_exp003_test_xsonar.md) *(generated)* |
| why EXP-003 is the deployed model, and how it was plugged in | [`exp003_diagnosis.md`](exp003_diagnosis.md) · [tables](exp003_diagnosis_tables.md) *(generated)* · [`onboard_exp003.md`](onboard_exp003.md) *(generated)* |
| OpenCV's seabed measurement changes where the boat goes (STUDY-12) | [`causal_trace.md`](causal_trace.md) *(generated)* |
| the same survey, detector only vs + Stage 1 + OpenCV evidence | [`opencv_counterfactual.md`](opencv_counterfactual.md) *(generated)* |
| Stage 1 measures the sonar's altitude without labels (STUDY-08) | [`stage1_canonical.md`](stage1_canonical.md) |
| a raw Humminbird recording with real GPS and a measured range scale (STUDY-14) | [`raw_recording.md`](raw_recording.md) |
| the agent cross-checks its seabed track against the depth sounder (STUDY-16) | [`sensor_check.md`](sensor_check.md) |
| active vision: OpenCV evidence → belief → action (STUDY-15; validation failed, off by default) | [`evidence_model_exp003.md`](evidence_model_exp003.md) *(generated)* |
| analyst effort: break-even card time 6.88 s (timings assumed until the study runs) | [`effort_curve.md`](effort_curve.md) *(generated)* |
| per-stage latency, $ / survey-hour, COOL provenance | [`cool_benchmark.md`](cool_benchmark.md) *(generated)*, raw JSON in [`../benchmarks/`](../benchmarks) |
| the earlier EXP-001 baseline: deploy-faithful evaluation and its guaranteed tiers | [`eval_exp001.md`](eval_exp001.md) · [`calibration_exp001.md`](calibration_exp001.md) *(generated)* |

## What did not work (published on purpose)

| study / run | result |
|---|---|
| [`exp002_diagnosis.md`](exp002_diagnosis.md) ([tables](exp002_diagnosis_tables.md)) | EXP-002 underfit: `optimizer=auto` built AdamW 0.00167; tiles left cut objects unlabelled |
| [`exp004_diagnosis.md`](exp004_diagnosis.md) ([tables](exp004_diagnosis_tables.md)) · [`exp005_diagnosis.md`](exp005_diagnosis.md) ([tables](exp005_diagnosis_tables.md)) | cleaner data and stronger jitter did not beat EXP-003 |
| [`scale_tta.md`](scale_tta.md) | STUDY-13: larger input + flip, +0.10 AP on calibration, −0.06 on unseen data |
| [`seam_inference.md`](seam_inference.md) | STUDY-11b: seam inference, no recall gain |
| [`failure_gallery.md`](failure_gallery.md) | what the detector misses, with contact sheets (EXP-001) |
| [`../experiments.md`](../experiments.md) | every experiment, in order, including the negative ones |

## Run, connect, deploy

| read | for |
|---|---|
| [`../README.md`](../README.md) §3 | run the studio from a clean clone |
| [`release_exp003.md`](release_exp003.md) | the published weights (Release `exp003-v1`, SHA-256 checked by `fetch_model`) |
| [`mcp.md`](mcp.md) | let any MCP agent run DEPTH (it can only *ask* a person to approve) |
| [`integrations.md`](integrations.md) | OGC API – Features, signed webhooks, the drop-in web component |
| [`../infra/README.md`](../infra/README.md) | Graviton + COOL deployment (dry-run by default) and the 3-way benchmark |
| [`user_study.md`](user_study.md) | the timed user study protocol (≈ 10 min per person) |
| [`kaggle_training.md`](kaggle_training.md) | train a model on Kaggle and plug it in with one command |

## Submission material

| read | for |
|---|---|
| [`video_script.md`](video_script.md) | the demo video script (≤ 5 min) |
| [`lucidchart_diagram.md`](lucidchart_diagram.md) | the architecture diagram as Mermaid source, for Lucidchart |
| [`../AGENT.md`](../AGENT.md) | the original proposal; the as-built design supersedes it where they differ |

## Working notes (kept for transparency, not part of the entry)

| read | for |
|---|---|
| [`notes/initial_review.md`](notes/initial_review.md) | the 22 Sep strategic review whose findings drove the data, evaluation and honest-geotag fixes |
| [`notes/competition_rules.md`](notes/competition_rules.md) | a digest of the competition rules |
| [`notes/claude_code_playbook.md`](notes/claude_code_playbook.md) | the working method used to run the build with Claude Code |
| [`notes/colab_baseline.md`](notes/colab_baseline.md) | how the EXP-001 baseline was first set up (superseded by Kaggle) |
| [`notes/outreach.md`](notes/outreach.md) | email drafts asking for a user quote and a recording with real GPS |
| [`dataset_report.md`](dataset_report.md) | the audit of dataset v0 that led to v1 |
