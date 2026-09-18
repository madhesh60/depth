# TODO — Marine Debris Detection System

Phased, gated backlog. **Do not skip a gate** — each phase has an exit criterion that must
be true before the next begins. Priorities reflect the judging rubric: dataset quality, ML
validity, COOL benchmark, and a demoable product beat extra features every time.

**Related:** [`architecture.md`](architecture.md) · [`progress.md`](progress.md) ·
[`experiments.md`](experiments.md)

Legend: `[ ]` open · `[~]` in progress · `[x]` done · **(P0)** blocker · **(P1)** high · **(P2)** nice-to-have

---

## Phase 0 — Foundation & hygiene  *(this week)*

- [x] Read full context; establish architecture and plan.
- [x] Create documentation set (README, architecture, progress, experiments, TODO).
- [ ] **(P1)** Initial git commit of docs; adopt typed commit prefixes (`feat/fix/data/infra/docs/exp`).
- [x] Move + formalise stray reference docs into `docs/` (`competition_rules.md`,
      `claude_code_playbook.md`, `dataset_report.md`); links updated in README.
- [ ] **(P0)** Install & configure **AWS CLI** (blocks all cloud + COOL work). *(blocker B5)*
- [ ] **(P1)** Scaffold `src/`, `infra/`, `tests/` per architecture §5 (empty modules + `__init__`).

**Exit gate:** docs committed, repo scaffolded, AWS CLI authenticated.

---

## Phase 1 — Dataset v1 (fix blockers)  *(before ANY training)*

> Baselined on the verified audit — [`docs/dataset_report.md`](docs/dataset_report.md)
> (`python DATASET/scripts/audit_dataset.py`). Earlier polygon/5-class blockers are already
> resolved by prior reprocessing.

- [x] **Fix cross-split leakage** — grouped split by source frame; re-audited **0/0/0**. *(B1)*
- [x] **Taxonomy decision** — **4 classes**, `rope_line` merged into `fishing_gear`. *(B5)*
- [x] Drop 1,258 full-frame boxes (audit confirms 0 remain). *(B2)*
- [x] Handle degenerate boxes; keep plausibly-small ones for recall + flag for QA. *(B3)*
- [x] Corrupt-image + parity sweep (0 corrupt, 0 orphans, 0 malformed).
- [x] Re-audit v1 (`audit_dataset.py DATASET/03_yolo_ready_dataset_v1`) + `manifest.json`.
- [x] Add background negatives (1,099 empty-label images) for FP control.
- [x] Tag **dataset v1** (`VERSION` + `manifest.json`); reproducible via `build_dataset_v1.py`.
- [ ] **(P1)** Eyeball QA renders in `DATASET/exports/qa_train`, `qa_val` (via `visualize_labels.py`); fix any obviously-bad labels found.
- [ ] **(P1)** Decide augmentation policy for training (train-time aug on v1; the v0 baked augs are already de-duplicated).
- [ ] **(P2)** Push dataset v1 to `S3://.../<dataset>/` (needs AWS CLI — see Phase 4).

**Exit gate:** ✅ 0 leakage · full-frame boxes gone · taxonomy locked · v1 tagged & scripted.
Remaining before EXP-001: visual QA pass + class-imbalance training config.

---

## Phase 2 — Stage 1 classical CV (COOL core workload)

- [ ] **(P1)** Implement `src/cv_pipeline/` — preprocess, segment, contours, pipeline (architecture §3).
- [ ] **(P1)** Config-driven thresholds in `src/common/config.py` (no hardcoded constants).
- [ ] **(P1)** Unit tests with sonar fixtures; deterministic output per frame+config.
- [ ] **(P1)** ROI + candidate-mask output contract wired to feed Stage 2.
- [ ] **(P2)** Transparency artifacts: save raw → mask → ROI overlays for the demo view.

**Exit gate:** Stage 1 emits sensible ROIs on real frames, CPU-only, tested.

---

## Phase 3 — Stage 2 detection + evaluation

- [ ] **(P1)** `src/detection/train.py` reading `DATASET/03_yolo_ready_dataset/data.yaml`.
- [ ] **(P1)** Run **EXP-001** baseline (YOLO11s-seg, v1) → log in `experiments.md`.
- [ ] **(P1)** `src/detection/evaluate.py`: mAP@0.5, mAP@0.5:0.95, P/R/F1, per-class, confusion matrix.
- [ ] **(P1)** `src/detection/export_onnx.py` → ONNX + verify load via `cv2.dnn.readNetFromONNX()`.
- [ ] **(P1)** `src/detection/infer.py` with `full_frame` **and** `roi_guided` modes.
- [ ] **(P0 for award)** **Ablation:** measure FP reduction (full-frame vs ROI-guided) → target ≥60%.
- [ ] **(P1)** Error analysis on false negatives (esp. `fishing_gear` — ghost nets/rope) → feed next experiments.
- [ ] **(P2)** Experiment ladder EXP-002..006 as time allows.

**Exit gate:** a model meeting (or credibly approaching) mAP@0.5 ≥ 0.70 / P ≥ 0.80 / R ≥ 0.70, with ablation.

---

## Phase 4 — AWS + COOL benchmark (PRIMARY DELIVERABLE)

- [ ] **(P0)** S3 buckets/prefixes: `raw/ processed/ models/ reports/ benchmarks/`.
- [ ] **(P0)** Validate **COOL** on a small Stage-1 workload on Graviton before scaling.
- [ ] **(P0)** `infra/benchmarks/`: run Stage 1 over ≥1,000 fixed frames on **Graviton+COOL** and **x86 baseline**.
- [ ] **(P0)** Capture latency (p50/p95), throughput (FPS), CPU util, cost/1,000 frames → CSV + charts.
- [ ] **(P0)** Reproduction package: COOL version, instance type, deploy config, input manifest hash, one-command rerun.
- [ ] **(P1)** Lambda + API Gateway inference endpoint (Stage 1→2 → JSON).
- [ ] **(P1)** DynamoDB detections table (+ geo/temporal GSIs).
- [ ] **(P1)** SageMaker training job config + Model Registry versioning.
- [ ] **(P2)** CloudWatch dashboard for latency/throughput.
- [ ] **(P1)** Cost audit — tear down idle benchmark instances; keep within grant + Free Tier.

**Exit gate:** reproducible Graviton-vs-x86 benchmark with evidence package — the COOL award case.

---

## Phase 5 — Reporting + Dashboard (product)

- [ ] **(P1)** `src/reporting/geotag.py` — fuse ping metadata → geotagged detection record (schema in architecture §9).
- [ ] **(P1)** `src/reporting/report.py` — JSON / CSV / GeoJSON writers.
- [ ] **(P1)** `src/dashboard/app.py` (FastAPI): upload → run pipeline → overlays + downloadable report.
- [ ] **(P1)** Interactive map view of detections (lat/lon + class + confidence).
- [ ] **(P1)** **Transparency view:** raw frame → Stage 1 masks → Stage 2 classifications side-by-side.
- [ ] **(P2)** Deploy dashboard on **AWS Amplify** (or a warm container) for the live judge endpoint.

**Exit gate:** a judge can upload a sonar log and get overlays + a downloadable report end-to-end.

---

## Phase 6 — Agentic Vision (STRETCH — only if ahead)

- [ ] **(P2)** `src/agentic/tools.py` — expose OpenCV/COOL ops as MCP tools.
- [ ] **(P2)** `src/agentic/loop.py` — perception→decision→action; low-confidence → re-scan/gain-adjust.
- [ ] **(P2)** Capture a trace showing a visual result changing the next action.
- [ ] **(P2)** Agent workflow diagram + task-success/failure-handling eval.

---

## Phase 7 — Submission package  *(final week, by 2026-10-26)*

- [ ] **(P0)** Technical report: problem, users, architecture, OpenCV 5 impl, AWS deploy, evaluation, limitations, responsible use.
- [ ] **(P0)** ≤ 5-min video: team, app working, architecture, principal results, failure cases.
- [ ] **(P0)** Judge-accessible repo/archive; pinned deps + build/deploy/test instructions.
- [ ] **(P0)** Architecture diagram (OpenCV 5 + AWS + COOL components).
- [ ] **(P0)** Working web endpoint OR arranged live screen-share.
- [ ] **(P0)** Evaluation evidence incl. failure cases + COOL reproduction package.
- [ ] **(P1)** "Skeptical judge" self-review pass — fix top weaknesses (see `docs/claude_code_playbook.md` §13).

**Exit gate:** submitted before 2026-10-26 23:59 PT.

---

## Cross-cutting (ongoing)

- [ ] **(P1)** Tests for every stage before it's considered done; run after significant changes.
- [ ] **(P1)** Update `progress.md` each session; `experiments.md` after each run.
- [ ] **(P1)** Small, typed git commits; never a giant uncommitted tree.
- [ ] **(P2)** Weekly code review pass on new modules (bugs, leakage, cost, complexity).
