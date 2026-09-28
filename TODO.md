# TODO — DEPTH (updated 2026-09-28)

Deadline **2026-10-26 23:59 PT** (27 Oct 12:29 IST). Judging 27 Oct – 9 Nov (keep the demo up).
The review's roadmap lives in the local `NEEDTOIMPROVE.md`; this is the working backlog.

Legend: `[x]` done · `[ ]` open · **(YOU)** needs a person / an account / a GPU

## Needs you — highest leverage first

- [ ] **(YOU) Grant check-in** — due by **2 Oct** (unlocks the second half of the grant).
- [ ] **(YOU) Publish the weights** — GitHub → Releases → Draft a new release, tag `exp001-v1`, attach
      `runs/EXP-001/weights/best.onnx` (37,932,951 bytes, SHA-256 `55f827db…`). `fetch_model` and CI
      pick it up automatically.
- [ ] **(YOU) EXP-002 on Kaggle** — the recall ceiling (0.72) is the binding limit. Follow
      `docs/exp002_kaggle.md` (upload v2b once, paste the cells, Save & Run All). Then locally:
      `python -m src.detection.onboard_model --zip EXP-002_complete.zip --max-ms 700`. Also the 640-px twin
      EXP-002s (STUDY-13: resolution must be chosen on held-out recordings). Optional EXP-002p (+copy-paste).
- [ ] **(YOU) AWS day** — `aws login`; subscribe to the COOL Graviton listing (note the AMI id);
      upload `best.onnx` to S3; `infra/deploy_aws.sh` (dry run → `APPLY=1`); check
      `/api/health` shows `is_cool_path: true`. Then the 3-way benchmark (`infra/bench_cool.sh` on
      c7i + stock c8g + COOL c8g; terminate the extra instances) → `python -m src.bench.compare`.
- [ ] **(YOU) Timed study** — 3+ people × ~10 min in the **Study** tab (`docs/user_study.md`).
- [ ] **(YOU) False-alarm audit** — 2+ people × ~15 min in the **Audit** tab.
- [ ] **(YOU) Outreach** — send the two drafts in `docs/outreach.md` (dataset authors: a quote, a
      recording with GPS **in a crab-pot area**, the licence question; a cleanup organisation / NIOT).
- [ ] **(YOU, optional)** OK to download PINGMapper's `Test-Large-DS` (Solix, 1 h, ~216 MB for the two
      side-scan channels) — would exercise the 152-byte Solix header path on real data.

## After those land (engineering)

- [ ] EXP-002 → if the onboarding report clears the gate: switch `DEPTH_MODEL`, regenerate
      `python -m src.agentic.effort`, re-run the audit build for EXP-002, update README numbers.
- [ ] Log EXP-002, STUDY-09 (study numbers), STUDY-10 (audit) in `experiments.md` — including misses.
- [ ] Put the measured study timings into the survey budget mode (`sec_per_card`).
- [ ] Cross-sonar table from `test_xsonar` (EXP-002 is the first leakage-free model for it).
- [ ] Fine-tune seed from human labels (`python -m src.agentic.feedback export`) — a small
      before/after if labels accumulate.

## Submission package (by 21 Oct freeze; 22–25 Oct polish)

- [ ] Technical report — **draft written** ([`docs/technical_report.md`](docs/technical_report.md)); fill the ⏳ items (COOL runs, EXP-002, study, audit, live URL): problem → the two promises → architecture → OpenCV 5 + COOL → evaluation
      (guarantees, per-source, effort, audit, benchmark) → what didn't work → responsible use.
- [ ] ≤ 5-min video — storyboard ready (`docs/video_script.md`): problem, live demo on AWS, guarantees,
      counterfactual, person-confirmed route, Connect, real-GPS recording, COOL chart, limits.
- [x] Architecture diagram (`docs/img/architecture.svg`). - [ ] COOL benchmark chart + provenance JSONs (after the EC2 runs).
- [x] Failure gallery: 12 misses, 12 false alarms (`docs/failure_gallery.md`). - [ ] audit tags once the audit runs.
- [ ] 24 h soak test of the live link; CloudWatch alarm to email; daily check during judging.

## Optional (only if ahead)

- [x] Mission brief (`brief.py`): template always; Bedrock writer built + tested with a fake client — enable on the AWS day (`infra/README.md`).
- [ ] SQS + Graviton Spot worker scaling demo.
- [x] Real per-ping GPS on a raw PINGMapper recording — `humminbird.py`, STUDY-14 (`docs/raw_recording.md`).

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
- [x] Docs truth pass (README, architecture, CLAUDE, TODO) — again 2026-09-28 after the sweeps below.
- [x] **Counterfactual trace** (STUDY-12) + live "⊘ without Stage 1" map toggle.
- [x] **Person-confirmed loop**: decisions re-plan the recovery route; impact ledger.
- [x] **MCP server** (stdio + `/mcp`), approvals inbox; **OGC API – Features**, signed webhooks,
      `<depth-hazards>` embed, Connect tab.
- [x] Calm studio redesign; architecture diagram; technical report draft; video script; outreach drafts.
- [x] Public-demo protection (rate limits, queue cap, admin + MCP tokens); CI; `fetch_model`.
- [x] STUDY-13 (larger input + flip TTA: negative) → speed-aware EXP-002 kit (640 + 1024, `--max-ms`).
- [x] **Raw recordings**: defensive Humminbird reader, physics checks, measured range scale, real
      per-ping geotags, heights in metres (STUDY-14).
