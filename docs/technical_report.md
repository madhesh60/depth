# DEPTH — Technical Report

**Side-scan sonar → a human-approved cleanup plan for ghost fishing gear and wreck debris**
OpenCV AI Competition 2026 · Team Syndicate (solo — Madhesh) · Tracks: **Agentic Vision**, **Best Use
of COOL** · Code: <https://github.com/madhesh60/depth> (AGPL-3.0)

> Status of 2026-10-12, deployed detector EXP-003. Every number names its split and links the script-generated report it comes
> from. Items marked **⏳ pending** are measurements that have not been taken yet; they are left
> empty rather than estimated.

---

## 0. Summary

A sonar debris detector alone is not a tool a cleanup crew can rely on: it misses things silently and
its false alarms cost boat time. DEPTH wraps one in an **OpenCV 5 agent** that measures the sonar
itself, proves or doubts each find with physical evidence, sorts finds into **tiers with calibrated
statistical promises**, and turns them into routes, second-look passes and reports. A human approves
everything that leaves the machine. It runs torch-free on CPU (OpenCV 5 `cv2.dnn`), built for **AWS
Graviton + COOL**.

| claim | evidence |
|---|---|
| **≥ 84% of pots reach a human when the agent looks harder** (95% confidence; ≥ 79% with one look), fit on held-out recordings and verified once on unseen test: **86.0% (LCB 82.0%)**, and 81.1% (LCB 76.7%) with one look — EXP-003, the deployed detector | [`look_harder.md`](reports/look_harder.md) · [`calibration_exp003.md`](calibration_exp003.md) |
| **The agent withdraws its promise where it does not apply**: on another sonar in 22 of 22 surveys (one look keeps 23.9% of the pots there), never on the 3 test recordings | [`applicability.md`](reports/applicability.md) |
| **A goal-driven core loop** chooses how hard to look from the operator's goal and compute budget, and re-plans mid-run on its measured pace | [`core.py`](../src/agentic/core.py) · §2 |
| Stage 1 measures the sonar's altitude without labels; port and starboard agree to **1.6 px** (random pairs 5.3 px) | [`stage1_canonical.md`](reports/stage1_canonical.md) |
| OpenCV output changes the agent's actions: without Stage 1, **151 of 222** pins leave their own error circle and 4 of 40 re-survey passes regroup (EXP-003, v2b test); the shadow changes no decision (evidence only) | [`causal_trace.md`](reports/causal_trace.md) |
| Break-even analyst time: DEPTH meets the promise faster than manual review if a card takes **< 6.88 s** (timings assumed until the study runs) | [`effort_curve.md`](reports/effort_curve.md) |
| Full product path **252 ms/frame p50** on a laptop CPU; one survey-hour of sonar in **51 s** | [`cool_benchmark.md`](reports/cool_benchmark.md) |
| A **raw sonar recording with real GPS** is read directly and checked against physics (GPS speed vs the speed field r = 0.95; course vs heading 2.4°), with a **measured range scale** (2.03 cm/sample ± 10.2% after the agent cross-checks its seabed track against the depth sounder) → pins and heights in real metres | [`raw_recording.md`](reports/raw_recording.md) |
| A person's decision makes the agent **re-plan** the recovery route, queue, budget and passes; an impact ledger measures the reviewed precision | [`agentic_vision.md`](agentic_vision.md) |

---

## 1. Problem and users

**Problem.** Lost crab pots and other derelict gear keep fishing ("ghost fishing"). Wreck debris is a
navigation and snag hazard. Both show up in **side-scan sonar**, which survey teams already collect.
Finding them means an analyst scrolling through hours of sonograms. Public detectors exist, but they
are imperfect: on the crab-pot sonar in our data, the recall ceiling of the best model is 0.72–0.86
depending on the recording.

**Users.**
- **Cleanup NGOs and fisheries agencies** plan recovery trips.
- **Survey analysts** review the sonar.
- **Boat crews** go to the coordinates.

Their question is not "what is the mAP?" but "how many real pots will we miss, how much of my time will
this take, and where exactly do I send the boat?"

**Design goal.** Make the detector **dependable** rather than just more accurate:

- promises that can be checked;
- evidence on every card;
- compute, analyst minutes and boat time spent only where they can change the outcome;
- no action without a person;
- every correction captured as a training label.

---

## 2. Architecture

![architecture](img/architecture.svg)

**The agent at the core** (`src/agentic/core.py`). A survey is run toward the operator's **goal** (the
recall to keep; compute, analyst and boat minutes). The agent (1) inspects the frames, (2) checks that
its promise applies — eight OpenCV statistics per frame, conformal p-values against the calibration
recordings; if not, it withdraws the promise and asks a person for a full review (STUDY-19), (3) chooses
how hard to look — the standard full-frame look (≥ 79%) or the look-harder mode with a 2×2 tile pass
(≥ 84%, STUDY-18), priced with its measured `cv2.dnn` pass time, (4) perceives every frame through the
stages below, re-planning to the standard look if the pace would overrun the compute budget, and (5) plans
and asks. Every decision, with the options weighed, is logged in `agent_run`.

The **runtime** processes each frame through five stages, all OpenCV 5 on CPU:

1. **Stage 1 — canonicalise** (`src/cv_pipeline/canonical.py`).
   - Palette → luminance.
   - Orientation from a source rule; unknown sources are *not measured*, never guessed.
   - **Bottom tracking** gives the sonar altitude per ping. It scans each ping for the first seabed return, then smooths across pings.
   - Slant → ground range via `cv2.remap`.
   - Water-column mask.
2. **See** — YOLO11 exported to ONNX and run by `cv2.dnn` (letterbox, class-aware NMS). The per-class floors come from `calibration.json`.
3. **Prove** — per candidate:
   - thin-line **acoustic shadow**;
   - **relative height** h/H against the *tracked* altitude;
   - water-column check;
   - a value-of-information re-look.

   Evidence goes on the human's card. It is never a silent gate (STUDY-03/04).
4. **Decide** — **guaranteed tiers** (§4.2): CONFIRMED / REVIEW / LOW-RISK. A calibrated P(pot) orders the review queue.
5. **Act** (survey level):
   - per-ping ground-range geotags with error radii;
   - chunk-boundary stitching and repeat-sighting merging;
   - nearest-neighbour recovery and inspection routes;
   - analyst-minute and boat-minute budgets;
   - **opposite-side re-survey passes**: each uncertain target predicts the bearing its shadow must flip to, a test that speckle cannot pass;
   - GeoJSON / GPX / KML / CSV / JSON exports with a **provenance stamp**, an agent **decision log** (JSONL) and a **mission brief**.

**Serving.** FastAPI with a background job queue, upload limits, per-client rate limits and per-stage
metrics. A zero-build web studio has five modes:

- **Analyze:** one frame with its evidence and the agent's-eye view.
- **Survey:** the shipped samples (synthetic GPS) or a **raw recording with real per-ping GPS**; map, routes, re-survey passes, the "without Stage 1" counterfactual, the person-confirmed recovery route, approvals, the mission brief, and the **3D digital twin** (three.js: seabed in true ground range, the sonar at its tracked altitude, the acoustic triangle each height is measured from).
- **Study:** timed user study.
- **Audit:** blinded false-alarm audit.
- **Connect:** MCP, OGC API – Features, signed webhooks and the drop-in panel, with live URLs.

A guided 60-second demo walks through the whole path.

**Integration — no DEPTH-specific code on the partner's side** ([`integrations.md`](integrations.md),
[`mcp.md`](mcp.md)):

- **MCP** (stdio, and Streamable HTTP at `/mcp` with a bearer token): any AI agent can run the loop and read the decision traces.
- **OGC API – Features** at `/ogc`: QGIS and ArcGIS read the hazards, re-survey passes, routes and **approved work orders**.
- **HMAC-signed webhooks**: mission-control and dispatch systems receive every survey and every decision.
- **`<depth-hazards>` web component**: any web console can embed the hazard panel.

Every channel can read and *ask*; only a named person approves, in the studio.

**Model onboarding.** Every threshold lives in `models/<MODEL>/calibration.json`, so a new model plugs
in with one command. `onboard_model` verifies the ONNX in `cv2.dnn`, gates weak models, calibrates on
validation recordings and verifies on test. Training runs on Kaggle; the runtime never imports torch.

---

## 3. OpenCV 5 implementation — where OpenCV does the work

| stage | OpenCV 5 operations | what the result changes |
|---|---|---|
| Stage 1 | `cvtColor`, `medianBlur` / `blur` + per-ping reductions for the bottom track, `remap` (slant → ground), `resize` | geotags, error radii, relative heights, water-column flags, re-survey geometry (§4.3) |
| See | `dnn.readNetFromONNX`, `blobFromImage`, `dnn.NMSBoxesBatched` | candidates, tiers, P(pot), the queue |
| Prove | crops + upscales for the re-look, `createCLAHE`, thin-line shadow statistics on the canonical view | card evidence; the re-look is spent only where it can change a tier or queue position |
| Act | (plain geometry on Stage-1 output: ground range → metres → lat/lon — not an OpenCV call) | routes, passes, exports |
| Training data | `seamlessClone` for sonar-aware copy-paste (`build_tiles.py --paste`) | EXP-002 training set |
| Display | agent's-eye view, overlays, JPEG encode, twin texture | what the human sees |

The runtime has **no torch dependency**. The same ONNX is evaluated in `cv2.dnn`, so deploy-time
numbers equal evaluation numbers ([`eval_exp001.md`](reports/eval_exp001.md)).

---

## 4. Evaluation

All numbers are on **unique frames** (Roboflow augmentation copies removed), and the split is named next to every number.

### 4.1 Detector — EXP-003 (YOLO11s, 640 px, dataset v2b tiles, 2 classes; deployed)

| | value | split |
|---|--:|---|
| **ghost gear (crab pots) — AP@0.5 (95% CI) / F1 at the val threshold** | **0.482 (0.43–0.55) / 0.49** | v2b test, 214 unique crab-pot frames |
| ghost gear — AP@0.5:0.95 | 0.165 | v2b test |
| wreck debris — AP@0.5 | 0.156 | v2b test (70 boxes, 8 wrecks) |
| mAP@0.5 / mAP@0.5:0.95, both classes | 0.319 / 0.106 | v2b test |
| official 398-frame GhostVision split — F1 / AP@0.5 | **0.41** / 0.39 (GhostVision: F1 0.71–0.73) | official test |
| cross-sonar — AP@0.5 / recall at the val threshold | 0.34 / 0.11 | `test_xsonar` |
| recall ceiling (any proposal ≥ 0.05) | 0.85 / 0.84 | calibration / verification |

We report the crab-pot number because it is the one that matters, not the aggregate. Reports with the
confusion matrix and PR curve: [`eval_exp003_test.md`](reports/eval_exp003_test.md),
[`…_official398`](reports/eval_exp003_test_official398.md), [`…_xsonar`](reports/eval_exp003_test_xsonar.md). The gap to
GhostVision is real; the diagnosis puts it in the data, not the recipe
([`exp003_diagnosis.md`](reports/exp003_diagnosis.md)). The earlier 4-class baseline EXP-001 (dataset v1) is in
[`eval_exp001.md`](reports/eval_exp001.md).

### 4.2 The agent's promises (Agentic Vision: task success)

Tiers are fit on the **calibration split** (held-out recordings Rec10 / 12 / 16: 163 unique frames,
186 pots) with Clopper–Pearson / Learn-Then-Test and **verified once** on the verification split (Rec8 /
9 / 14: 199 unique frames, 264 pots) ([`calibration_exp003.md`](calibration_exp003.md)):

- **Recall promise ≥ 79%** with one look (95% confidence; the requested 90% is above the detector's 0.85 ceiling). Verified: **81.1%, lower confidence bound 76.7% — held.** EXP-001's promise was ≥ 65%.
- **Recall promise ≥ 84% when the agent looks harder** (STUDY-18, [`look_harder.md`](reports/look_harder.md)): the full frame + a 2×2 tile pass at the scale EXP-003 trained on raises the ceiling to 0.892 (paired bootstrap +0.043, 95% CI +0.016..+0.076). Verified once: **86.0%, lower bound 82.0% — held**; cards per test frame 2.68 → 3.86; 4 more forward passes per frame. Protocol committed before the test split was scored.
- **The promise is checked before it is kept** (STUDY-19, [`applicability.md`](reports/applicability.md)): another sonar (`test_xsonar`) is flagged in 22 of 22 surveys, where one look keeps only 23.9% of the pots; the 3 test recordings and the demo frames keep it. The raw river recordings are flagged too, so they run without a promise and with a full-review request.
- **No precision promise is supportable** (best achievable 0.565 against the 0.85 target), so nothing is auto-confirmed and every find goes to a human. DEPTH says what it *cannot* promise.
- Re-look vs plain confidence: no re-look policy beat confidence on the calibration recordings (STUDY-07, repeated for EXP-003). The shipped policy therefore spends 1 inference per frame, and the value-of-information agent records the re-looks it chose not to make.

### 4.3 Does OpenCV output change what the agent does? (required evidence)

[`causal_trace.md`](reports/causal_trace.md) (STUDY-12) runs the same survey three ways: the full product,
with Stage-1 geometry withheld, and with the shadow withheld. It then diffs every downstream
decision. Results with EXP-003 on v2b test, 80 unique crab-pot frames, 223 hazards:

| downstream decision | without Stage 1 | without shadow |
|---|---|---|
| tier flips · queue order (Kendall τ) · budget picks | 0 · 1.0 · 0 | 0 · 1.0 · 0 |
| geotag shift, median / max | **9.1 / 15.7 m** | 0 |
| pins pushed outside their own error circle | **151 of 222** | 0 |
| re-survey passes unchanged | 36 of 40 | 40 of 40 |

(EXP-001 on v1 test gave the same picture: 193 of 264 pins outside their circle, 50 of 63 passes
unchanged.)

Stage 1 is not a gate, so it changes no tier. It does decide **where the boat goes**. The shadow
changes nothing, by design. The report also prints one hazard's full chain: bottom track → `cv2.dnn`
→ tier → P(pot) → queue rank → budget → inspection stop → re-survey pass with its predicted shadow
flip. The product shows this live with the map's **"⊘ without Stage 1"** toggle. Every survey also
exports its decision log.

### 4.4 Human effort

The effort curve ([`effort_curve.md`](reports/effort_curve.md)) compares three review modes:

- **manual review** (every frame);
- **a confidence-sorted list**;
- **DEPTH's cards.**

It reports the **break-even card time: 6.88 s** (EXP-003, verification split, 20 s per frame assumed
for manual review). DEPTH beats manual review at the recall promise whenever a card takes less than
this. At the assumed 8 s per card it does not: the DEPTH queue takes 65.5 analyst minutes for the 199
frames, manual review 56.3.

- **⏳ pending:** the timed, counterbalanced study built into the app (Study tab, [`user_study.md`](user_study.md)) will replace the assumed 8 s/card with a measurement.
- **⏳ pending:** the blinded false-alarm audit (Audit tab) will measure how many "false alarms" are unlabelled real objects. Catch trials check the auditors themselves.

### 4.5 Failure cases (measured, not assumed)

The failure gallery was measured on the earlier EXP-001 (its verification split, v1 test, 92 unique
frames) and has not been re-run for EXP-003. There, 19 of 138 pots (14%) never reached a human
([`failure_gallery.md`](reports/failure_gallery.md), with contact sheets):

- **32% of missed pots touch the frame edge**, against 11% of found pots;
- missed pots have lower contrast and sit farther in range;
- they are **not** smaller.

We tried seam inference across chunk boundaries (STUDY-11b): **no recall gain, not shipped**
([`seam_inference.md`](reports/seam_inference.md)). Most of these edges are augmentation crops, which dataset
v2b removes. We also tried a larger input with a flipped second view (STUDY-13,
[`scale_tta.md`](reports/scale_tta.md)): +0.10 AP on the calibration recording, but **−0.06 AP on unseen data**.
Not shipped.

**Real, out-of-domain water (STUDY-14, [`raw_recording.md`](reports/raw_recording.md)).** We ran the full agent on a
raw Humminbird recording with real per-ping GPS: the Colorado River at Horseshoe Bend, 150.6 s, with no
known crab pots.

- With EXP-003 it produced **2 review cards**, which is **48 cards per hour of sonar** at the detector floor.
- Every card is a false alarm or an unknown object, so this is the review load the model would put on an analyst there.
- All pins were placed from their own ping's GPS with a 7.9–8.0 m error radius.

### 4.6 Runtime and COOL

| run | host | total p50 | fps | survey-hour compute |
|---|---|--:|--:|--:|
| full product path (EXP-003) | laptop x86, stock OpenCV 5, 8 threads | 252 ms | 3.3 | 51 s (RTF 0.014) |
| OpenCV-only stages (decode, Stage 1, evidence, render) | laptop x86 | 25 ms | 40 | 4 s |
| full product path, 1 thread (EXP-001) | laptop x86 | 523 ms | 1.9 | 89 s |
| full product path, 4 threads (EC2, 2026-10-20) | c8i.xlarge x86, stock OpenCV 5 | 157 ms | 6.4 | 26.5 s · $0.0014 |
| full product path, 4 threads (EC2, 2026-10-20) | c8g.xlarge Graviton4, stock OpenCV 5 | 197 ms | 5.1 | 33.3 s · $0.0015 |
| real 1-h recording end to end (EC2) | c8i.xlarge x86 / c8g.xlarge Graviton4, stock | — | — | RTF 0.0135 / 0.0171 |
| load test, 4 clients + 1 survey, 120 s | c8g.large demo server (stock, rehearsal) | p95 2.0 s (mark 3 s) | 291/291 ok | **passed** |
| **⏳ pending:** Graviton **COOL** (c8g.xlarge + c8g.large) | Marketplace subscription waiting on AWS | — | — | — |

**Proposal targets vs delivered.** Stated plainly, because judges compare the two:

| proposal target | delivered | reading |
|---|---|---|
| mAP@0.5 ≥ 0.70 | **0.319** (both classes); **crab pots 0.482** (v2b test) | **not met.** Six clean training runs did not close it; the diagnosis points at the data. The recall promise below makes the shortfall safe to use: every pot the promise covers reaches a person. |
| < 300 ms per frame | 157 ms (x86) / 197 ms (Graviton stock) p50 on EC2, 4 threads; laptop 252 / 530 ms | **met on EC2** at p50 and p95. Graviton COOL pending. |
| ≥ 5 FPS | 6.4 FPS (x86) / 5.1 FPS (Graviton stock) on EC2; 3.3 on the laptop | met on EC2. It is also the wrong yardstick for side-scan: one frame per ~21 s of sonar arrives from the boat. What matters is survey-hours processed: **51 s of compute per hour of sonar (70× faster than real time)**. |
| (not in the proposal) | ≥ 79% of pots reach a person, verified on unseen data (81%) | added: a promise instead of an average |

The 3-way benchmark ([`infra/README.md`](../infra/README.md)) uses the same code, the same frames and
the same model (all sha256-pinned), with threads pinned per run. A run counts as COOL only if
`cv2.__file__` resolves under `/opt/cool`. It times the **product workload per stage** plus a
CV-only workload that isolates where COOL acts.

---

## 5. AWS deployment

The deployment is scripted, dry-run checked, and executed on the AWS day:

One CloudFormation stack (`infra/cfn/depth.yaml`):

- **EC2 c8g.large** (Graviton4) launched from the **COOL Marketplace AMI**, running `depth.service` under systemd
  with the COOL interpreter; web dependencies via `pip --target` — **the COOL venv is never modified**.
- **CloudFront** provides HTTPS and caches the static UI; the origin accepts only CloudFront's origin-facing
  prefix list and our secret `X-Origin-Verify` header.
- **S3** (versioned, TLS-only) holds survey results (30-day lifecycle), uploads (7-day) and benchmark results.
- **CloudWatch**: per-stage latency from the server's JSON frame log, CPU/memory, a dashboard, and alarms
  (recover, reboot, CPU, edge 5xx) that email through SNS.
- Access goes through **SSM** only (no SSH port, no key pair), IMDSv2, least-privilege **IAM** roles.
- An **EventBridge Scheduler** stops the instance after judging; the account **Budget** alerts at 25/50/100% + forecast.
- **Verified COOL:** `GET /api/cool` serves the AWS-signed instance identity (AMI, Marketplace product code,
  arm64) and the OpenCV libraries mapped into the server process; `python -m src.bench.verify_cool <url>` checks it.

`infra/deploy.sh` is a dry run by default (it prints the change set); `infra/teardown.sh` removes everything.

**Optional:** Amazon Bedrock (Claude) rewrites the mission brief. It writes, never decides: its text is
shown only if every number and ID traces back to the survey, otherwise the template is served
(`src/agentic/brief.py`).

**Scaling path (described, not built):** survey jobs → SQS → an Auto Scaling group of the same COOL AMI
on Spot → S3.

**⏳ pending:** the live endpoint URL and the 24-hour soak test.

---

## 6. Limitations (stated in the product, not hidden)

- **Scope.** One region's crab pots and one sonar family. On another sonar (`test_xsonar`) AP@0.5 is 0.34 and the val-tuned threshold does not transfer (recall 0.11): a new sonar needs re-calibration. The agent now detects that case and withdraws its promise (STUDY-19); the check is coarse — it catches another sonar or rendering, not subtle drift within one.
- **Detector.** Crab-pot AP@0.5 0.48 and F1 0.41 on the official split, against GhostVision's 0.71–0.73. EXP-003 raised the recall ceiling from 0.72 to 0.85 and the promise from ≥ 65% to ≥ 79%. EXP-004 / 005 (cleaner data) did not beat it ([`exp005_diagnosis.md`](reports/exp005_diagnosis.md)). EXP-002 failed as underfit (`optimizer=auto` built AdamW 0.00167; [`exp002_diagnosis.md`](reports/exp002_diagnosis.md)).
- **Wreck debris** is weak (AP 0.16): most training wrecks are off-domain fish-finder screenshots.
- **Calibration.** Calibration uses three recordings whose frames are correlated. That is why the promise is verified on a separate split.
- **Labels.** Labels are incomplete, so precision is a lower estimate (the audit will quantify this).
- **GPS.** The public crab-pot frames carry no GPS, so their demo track is synthetic and labelled as such everywhere, including the counterfactual's metres. The raw-recording path has **real** per-ping GPS and a measured range scale, but that recording is a river with no known pots: it measures false alarms, not recall.
- **Heights.** On crab-pot frames heights are relative (h/H). On raw recordings they are in metres from the sonar's measured depth. The range scale there is measured: the tracker alone made 5 false picks among the 11 first-estimate chunks, the agent's cross-check against the depth sounder resolves them (STUDY-16, [`sensor_check.md`](reports/sensor_check.md)), and the remaining uncertainty (± 10.2%) is carried into every error radius.
- **Effort.** Analyst seconds per card are assumed until the study measures them.

## 7. Responsible use

- **Human control.** Nothing is dispatched automatically. Every plan carries `human_approval_required`, re-survey passes are "PLANNED", and LOW-RISK finds are kept for audit, never deleted.
- **Protected sites.** Wrecks can be war graves or heritage sites. Public shares generalise their positions to about 1.1 km and drop passes that would point at them. The decision log is never public.
- **Honest geotags.** No GPS means no coordinates. Unknown orientation means a swath-wide error radius. Synthetic tracks are stamped on every export.
- **Provenance.** Every result carries the model and calibration hashes, the OpenCV build (COOL or stock), the code commit and the host.
- **Language models** may narrate but never decide, and their numbers are machine-checked.
- **Data.** CC-BY-SA sample frames are shipped with attribution. Datasets are not redistributed; their builders are in `DATASET/scripts/`.

Details: [`responsible_use.md`](responsible_use.md) · [`model_card.md`](model_card.md) ·
[`dataset_card.md`](dataset_card.md).

## 8. What did not work (published)

- **STUDY-01** — classical ROI gate: no gain; retired.
- **STUDY-03/04** — the acoustic shadow as a filter hurts recall; kept as evidence only.
- **STUDY-07** — re-look vs confidence: no gain on unseen data. The old test-tuned "CONFIRMED 0.737" is 0.58 on unseen data, and was retired.
- **STUDY-08** — range-gain detector input: no gain for EXP-001.
- **STUDY-15** — active vision (value-of-information tool choice): both registered validation checks failed, so it is off by default and labelled experimental ([`evidence_model_exp003.md`](reports/evidence_model_exp003.md)).
- **EXP-002, EXP-004, EXP-005** — training runs that did not beat EXP-003; each has a diagnosis.
- **STUDY-11b** — seam inference: no recall gain.
- **STUDY-13** — a larger input and a flipped second view: +0.10 AP on the calibration recording, but
  **−0.06 AP on unseen data**. Not shipped. It also changed the EXP-002 plan, which now trains 640 and
  1024 px and lets the held-out recordings and a speed gate choose.

Log: [`experiments.md`](experiments.md).

## 9. Reproduce

```bash
pip install -r requirements-dev.txt                   # runtime + test + lint dependencies
python -m src.detection.fetch_model                   # the detector (Release exp003-v1), SHA-256 checked
pytest -q                                             # the test suite (CI: .github/workflows/tests.yml)
python -m uvicorn src.dashboard.app:app --port 8000   # studio → "▶ 60-s demo"
python -m src.agentic.calibrate                       # fit + verify the guarantees
python -m src.detection.evaluate                      # deploy-faithful eval (+ mAP@0.5:0.95, confusion, PR)
python -m src.detection.study_scale_tta               # STUDY-13
python -m src.agentic.study_look_harder --verify      # STUDY-18: look harder (calibration -> test once)
python -m src.agentic.study_applicability --verify    # STUDY-19: does the promise apply here?
python -m src.bench.product_bench --label <host>      # benchmark this machine

# STUDY-12: the counterfactual trace on up to 80 v2b test frames
python -m src.agentic.study_causal \
    --frames DATASET/03_yolo_ready_dataset_v2b/test/images --limit 80

# STUDY-14: the raw recording with real GPS
python -m src.cv_pipeline.humminbird fetch
python -m src.cv_pipeline.humminbird report
```

Checked on a fresh clone (2026-10-12): `fetch_model` installed the published weights with a matching
hash, and the suite passed (202 passed, 3 skipped: the optional raw recording and the unpublished
EXP-001 weights). The datasets are git-ignored; their builders are in `DATASET/scripts/`.
