# TODO — DEPTH (updated 2026-10-02)

Deadline **2026-10-26 23:59 PT** (27 Oct 12:29 IST). Judging 27 Oct – 9 Nov (keep the demo up).
The review's roadmap lives in the local `NEEDTOIMPROVE.md`; this is the working backlog.

Legend: `[x]` done · `[ ]` open · **(YOU)** needs a person / an account / a GPU

**Order that unblocks the most:**

1. Grant check-in (due 2 Oct).
2. EXP-004 on Kaggle.
3. AWS day.
4. Study and audit with real people.
5. README numbers, diagram and screenshots last.

- [ ] **(YOU) Grant check-in** — due by **2 Oct** (unlocks the second half of the grant).

## 1) Best UI

- [x] Calm glass redesign: four type roles, one frosted-white primary action per screen, calm palette,
      glyph labels replaced by words (sweep 37).
- [ ] Screenshots of every mode (Analyze · Survey · Study · Audit · Connect) for the README and the
      submission. Take them after EXP-003, so they show the final model.
- [ ] Layout check at a laptop size (1366×768) and on a phone; fix anything cramped.
- [ ] Screen recording of the 60-s guided demo for the video.

## 2) COOL and agentic benchmarks

- [ ] Laptop baseline now: `python -m src.bench.product_bench --label laptop`.
- [ ] **(YOU, AWS) COOL 3-way benchmark.** Run `infra/bench_cool.sh` on x86 c7i, stock Graviton c8g and
      COOL Graviton c8g, then `python -m src.bench.compare`. This is the evidence for Best Use of
      COOL; add the chart and the provenance JSONs.
- [x] Agentic evidence so far: the counterfactual trace (STUDY-12), the analyst-effort curve, and the
      guaranteed tiers verified on test.
- [ ] **(YOU) Timed study.** 3+ people × ~10 min in the **Study** tab (`docs/user_study.md`); this
      measures the minutes DEPTH saves.
- [ ] **(YOU) Blinded false-alarm audit.** 2+ people × ~15 min in the **Audit** tab; this measures
      audited precision and the label noise.
- [ ] With the winning EXP-003:
  - re-run `python -m src.agentic.effort` and the audit build;
  - put the measured study timings into the survey budget (`sec_per_card`);
  - log STUDY-09 and STUDY-10 in `experiments.md`, including misses.

## 3) AWS (a dedicated day)

- [ ] **(YOU)** Budget alarm first, then `aws login` (profile `hackathon`, us-east-1).
- [ ] **(YOU)** Subscribe to the COOL Graviton listing (note the AMI id). Accepting the terms is yours to do.
- [ ] Upload `best.onnx` to S3. Run `infra/deploy_aws.sh` as a dry run, then `APPLY=1` only after an
      explicit yes.
- [ ] Check that `/api/health` shows `is_cool_path: true`, and that the CloudFront HTTPS link works.
- [ ] Optional: enable the Bedrock mission-brief writer (`infra/README.md`).
- [ ] Run the 3-way benchmark (section 2), then terminate the extra instances.
- [ ] 24 h soak test of the live link; CloudWatch alarm to email; daily check during judging.

## 4) Best training numbers

- [x] EXP-002 / EXP-002s ran and were **rejected on validation**. They were underfit: the
      auto-optimizer silently used AdamW at 0.00167, and tiles left cut objects unlabelled. Test was
      not scored ([`docs/exp002_diagnosis.md`](docs/exp002_diagnosis.md)).
- [x] Kit fixed:
  - explicit SGD, with the optimizer actually built recorded;
  - tiles without unlabelled partial objects;
  - ghost-AP checkpoint selection;
  - the `diagnose` gate with a pass bar calibrated before any result (AP@0.3 ≥ 0.65 on its own
    training frames).
  
  Full suite 150 passed; the exact EXP-003f flags were dry-run.
- [x] **EXP-003 on Kaggle**, diagnosed and onboarded. Recall promise 65% → **79%** (held on test at
      81%), fewer review cards. Wreck fails, and official-split F1 is 0.41 vs GhostVision 0.71–0.73
      ([`docs/exp003_diagnosis.md`](docs/exp003_diagnosis.md)).
- [ ] **(YOU) EXP-004 on Kaggle** (~2.5 h, T4 ×2). Import `notebooks/exp004_kaggle.ipynb`, add
      **depth-v2b**, Save & Run All ([`docs/kaggle_training.md`](docs/kaggle_training.md)). Download
      `EXP-004_complete.zip` + `EXP-004g_complete.zip`, then tell Claude "EXP-004 done".
- [ ] `python -m src.detection.diagnose --zip EXP-003_complete.zip --zip EXP-003f_complete.zip`.
- [ ] `onboard_model` for the models that pass:
  - guarantees fit on val, verified once on test;
  - GhostVision head-to-head on the official 398-frame split;
  - the cross-sonar `test_xsonar` table;
  - the speed gate.
- [ ] If it passes: switch `DEPTH_MODEL`, publish its weights as a GitHub Release (`fetch_model`), and
      log EXP-003 in `experiments.md`.
- [ ] If it fails: diagnose, then EXP-004. Candidates:
  - luminance-normalised training (one palette);
  - audit-confirmed missing labels (v2c, val only, never test);
  - dropping the rotated black-bordered copies.
- [ ] **(YOU) Publish the EXP-001 weights.** GitHub → Releases → tag `exp001-v1`, attach
      `runs/EXP-001/weights/best.onnx` (37,932,951 bytes, SHA-256 `55f827db…`).
- [ ] Fine-tune seed from human labels (`python -m src.agentic.feedback export`): a small
      before/after if labels accumulate.

## 5) Architecture diagram

- [x] `docs/img/architecture.svg` (as of 2026-09-28).
- [ ] Update it for:
  - raw-recording input with real GPS (STUDY-14);
  - the person-confirmed loop;
  - Connect (MCP, OGC API, webhooks, embed);
  - the `diagnose` gate in the training path;
  - the COOL deployment.
- [ ] A one-page "data flow + where the guarantees live" figure for the technical report.

## 6) Numbers in README.md

- [ ] Results table, with the split named next to every number:
  - EXP-001 per-class sonar crab-pot numbers (never only the aggregate);
  - the recall promise (≥ 65%, held at 86.2% on test);
  - EXP-003 once onboarded.
- [ ] Speed: laptop now; Graviton stock vs COOL and x86 after the AWS day (ms/frame, FPS against the
      < 300 ms / ≥ 5 FPS targets).
- [ ] Analyst minutes saved (study), audited precision (audit), GhostVision head-to-head, cross-sonar.
- [ ] A negative-results section, linked: STUDY-01, 11b, 13, and EXP-002.

## Submission package (by 21 Oct freeze; 22–25 Oct polish)

- [ ] Technical report — **draft written** ([`docs/technical_report.md`](docs/technical_report.md)).
      Fill the ⏳ items: COOL runs, EXP-003, study, audit, live URL.
- [ ] ≤ 5-min video — storyboard ready (`docs/video_script.md`).
- [x] Failure gallery: 12 misses, 12 false alarms (`docs/failure_gallery.md`).
- [ ] Add the audit tags to the failure gallery once the audit runs.
- [ ] **(YOU) Outreach** — send the two drafts in `docs/outreach.md`.
- [ ] **(YOU, optional)** OK to download PINGMapper's `Test-Large-DS` (~216 MB).

## Optional (only if ahead)

- [x] Mission brief (`brief.py`): template always; Bedrock writer built and tested with a fake client.
      Enable it on the AWS day.
- [x] Real per-ping GPS on a raw PINGMapper recording (`humminbird.py`, STUDY-14).
- [ ] SQS + Graviton Spot worker scaling demo.

## Done (highlights — details in `progress.md`)

- [x] Dataset v1 → v2 → **v2b**: deduped, recording-level val, unique-frame, official and cross-sonar
      tests. Measured quality is in the dataset card.
- [x] EXP-001 baseline; deploy-faithful `cv2.dnn` evaluation (per source, bootstrap CIs).
- [x] **Guaranteed tiers** (CP / LTT) fit on validation, verified on test; VoI agent; budgets.
- [x] **Stage 1 canonicalisation** in the product path (bottom tracking validated, STUDY-08).
- [x] Counterfactual trace (STUDY-12); person-confirmed loop; opposite-side re-survey planner.
- [x] MCP server, approvals inbox, OGC API – Features, signed webhooks, `<depth-hazards>` embed.
- [x] Public-demo protection (rate limits, queue cap, tokens); CI; `fetch_model`.
- [x] Raw recordings: Humminbird reader, physics checks, measured range scale (STUDY-14).
- [x] COOL benchmark v3 (product workload) + Graviton/COOL deploy kit (dry run).
- [x] EXP-002 post-mortem, fixed kit, `diagnose` gate (sweeps 36–38); calm glass studio (sweep 37).
