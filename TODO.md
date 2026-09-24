# TODO — DEPTH (updated 2026-09-25)

Deadline **2026-10-26 23:59 PT** (27 Oct 12:29 IST). Judging 27 Oct – 9 Nov (keep the demo up).
The review's roadmap lives in the local `NEEDTOIMPROVE.md`; this is the working backlog.

Legend: `[x]` done · `[ ]` open · **(YOU)** needs a person / an account / a GPU

## Needs you — highest leverage first

- [ ] **(YOU) Grant check-in** — due by **2 Oct** (unlocks the second half of the grant).
- [ ] **(YOU) EXP-002 on Kaggle** — the recall ceiling (0.72) is the binding limit. Follow
      `docs/exp002_kaggle.md` (upload v2b once, paste the cells, Save & Run All). Then locally:
      `python -m src.detection.onboard_model --zip EXP-002_complete.zip`. Optional EXP-002p (+copy-paste).
- [ ] **(YOU) AWS day** — `aws login`; subscribe to the COOL Graviton listing (note the AMI id);
      upload `best.onnx` to S3; `infra/deploy_aws.sh` (dry run → `APPLY=1`); check
      `/api/health` shows `is_cool_path: true`. Then the 3-way benchmark (`infra/bench_cool.sh` on
      c7i + stock c8g + COOL c8g; terminate the extra instances) → `python -m src.bench.compare`.
- [ ] **(YOU) Timed study** — 3+ people × ~10 min in the **Study** tab (`docs/user_study.md`).
- [ ] **(YOU) False-alarm audit** — 2+ people × ~15 min in the **Audit** tab.

## After those land (engineering)

- [ ] EXP-002 → if the onboarding report clears the gate: switch `DEPTH_MODEL`, regenerate
      `python -m src.agentic.effort`, re-run the audit build for EXP-002, update README numbers.
- [ ] Log EXP-002, STUDY-09 (study numbers), STUDY-10 (audit) in `experiments.md` — including misses.
- [ ] Put the measured study timings into the survey budget mode (`sec_per_card`).
- [ ] Cross-sonar table from `test_xsonar` (EXP-002 is the first leakage-free model for it).
- [ ] Fine-tune seed from human labels (`python -m src.agentic.feedback export`) — a small
      before/after if labels accumulate.

## Submission package (by 21 Oct freeze; 22–25 Oct polish)

- [ ] Technical report: problem → the two promises → architecture → OpenCV 5 + COOL → evaluation
      (guarantees, per-source, effort, audit, benchmark) → what didn't work → responsible use.
- [ ] ≤ 5-min video: problem, live demo on AWS (Analyze → Survey → Study), guarantees + effort
      curve, COOL chart, limits.
- [ ] Architecture diagram (from `architecture.md` §2), COOL benchmark chart + provenance JSONs.
- [ ] Failure gallery: 12 misses, 12 false alarms (with audit tags).
- [ ] 24 h soak test of the live link; CloudWatch alarm to email; daily check during judging.

## Optional (only if ahead)

- [ ] Bedrock mission brief written from the survey JSON only (never makes decisions).
- [ ] SQS + Graviton Spot worker scaling demo.
- [ ] Real per-ping GPS demo on a PINGMapper recording.

## Done (highlights — details in `progress.md`)

- [x] Dataset v1 → v2 → **v2b** (deduped, recording-level val, unique-frame + official + cross-sonar tests).
- [x] EXP-001 baseline; deploy-faithful `cv2.dnn` evaluation (per source, bootstrap CIs).
- [x] **Guaranteed tiers** (CP / LTT) fit on validation, verified on test; VoI agent; budgets.
- [x] **Stage 1 canonicalisation** in the product path (bottom tracking validated, STUDY-08).
- [x] Honesty fixes: orientation by rule, thin-line shadow, relative height, no fake cross-pass.
- [x] Backend hardening: jobs, limits, per-frame model lock, shipped samples, vendored map.
- [x] **COOL benchmark v3** (product workload) + Graviton/COOL deploy kit (dry-run).
- [x] **EXP-002 kit**: tiles, sonar-aware copy-paste, train → verify → zip → one-command onboarding.
- [x] Human labels (✓ / ✕ / ＋missed), **Study** mode + effort curve, **Audit** mode.
- [x] **Opposite-side re-survey** planner + repeat-sighting merge.
- [x] Docs truth pass (README, architecture, CLAUDE, TODO).
