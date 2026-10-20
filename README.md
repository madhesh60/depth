# DEPTH — See · Prove · Decide · Act

[![tests](https://github.com/madhesh60/depth/actions/workflows/tests.yml/badge.svg)](https://github.com/madhesh60/depth/actions/workflows/tests.yml)

> **DEPTH** = **D**etect · **E**vidence · **P**rove · **T**riage · **H**azard-map — the pipeline
> stages spell the name.

An agentic system that finds **derelict fishing gear (ghost pots) and wreck debris in side-scan
sonar** and turns it into a human-approved cleanup plan. Every find **comes with evidence**; the
agent's tiers carry **calibrated statistical promises**; it spends compute, analyst minutes and boat
time only where they can change the outcome; every human decision becomes a training label. It runs
torch-free on **OpenCV 5**, CPU-only, built for **AWS Graviton + COOL**.

**Competition:** OpenCV AI Competition 2026 (AWS · OpenCV Foundation) · **Team:** Syndicate (solo —
Madhesh) · **Targets:** Agentic Vision · Best Use of COOL · **License:** AGPL-3.0

![DEPTH architecture: OpenCV 5 agent (Stage 1 → See → Prove → Decide → Act) on EC2 Graviton + COOL, with the human loop, exports and AWS services](docs/img/architecture.svg)

### At a glance

| | number | measured on |
|---|--:|---|
| **Recall promise** — pots that reach a person (95% confidence) | **≥ 79%**, held at **81.1%** | fit on held-out recordings, verified once on 199 unseen frames / 264 pots |
| Crab-pot detector, AP@0.5 · AP@0.5:0.95 | **0.482** (95% CI 0.43–0.55) · 0.165 | v2b test, 214 unique frames |
| OpenCV's seabed measurement moves the boat | **151 of 222** pins leave their own error circle without it | STUDY-12, v2b test |
| Stage-1 altitude, port vs starboard (no labels needed) | agree to **1.6 px** (random pairs 5.3 px) | STUDY-08 |
| Full product path on a laptop CPU | **252 ms/frame** p50 · one survey-hour in **51 s** | `benchmarks/`, x86, stock OpenCV 5 |
| Raw sonar recording, real GPS | range scale **2.03 cm/sample ± 10.2%**, measured | Colorado River, Humminbird |
| Tests | **225**, CI on every push; 3 need optional data and skip without it (the raw recording, the unpublished EXP-001 weights) | `pytest -q` |

**Not yet measured, and said so:** Graviton + COOL timings (scripted, AWS day), analyst minutes saved
(the timed study needs people), audited false-alarm precision (the blinded audit needs people). The
detector misses the proposal's 0.70 mAP target; §4 says by how much and why.

**For judges — five minutes from a clean clone:**

```bash
git clone https://github.com/madhesh60/depth.git
cd depth
pip install -r requirements.txt                       # torch-free runtime: OpenCV 5.0.0.93, FastAPI
python -m src.detection.fetch_model                   # the detector: 38 MB, SHA-256 checked
python -m src.cv_pipeline.humminbird fetch            # a raw sonar recording with real GPS: 11 MB
python -m uvicorn src.dashboard.app:app --port 8000   # open http://localhost:8000 → ▶ 60-s demo
```

### How the evidence maps to the judging criteria

| criterion | where DEPTH shows it |
|---|---|
| **Technical execution** — OpenCV 5 doing real work | Stage 1 (bottom track with `cv2.Sobel` / `cv2.blur`, slant→ground `cv2.remap`), YOLO11 through `cv2.dnn`, shadow + re-look evidence; deploy-faithful evaluation with CIs, mAP@0.5:0.95, confusion matrix, PR curves ([`eval_exp003_test.md`](docs/reports/eval_exp003_test.md)); 225 tests; deadlines + fallbacks on every tool |
| **Agentic Vision** — perception changes the next action | on by default and measured: OpenCV's seabed measurement decides where the pins go (without it 151 of 222 leave their own error circle and 4 of 40 re-survey passes regroup — [`causal_trace.md`](docs/reports/causal_trace.md)); the agent cross-checks that seabed against the depth sounder and re-tracks on a conflict ([`sensor_check.md`](docs/reports/sensor_check.md)); opposite-side passes ranked by information per boat-minute; a person's decline re-plans, an approval issues a work order (§2). Per-frame active vision is built and tested but failed its own validation, so it is off by default ([`evidence_model_exp003.md`](docs/reports/evidence_model_exp003.md)) |
| **Innovation** | statistical promises on an imperfect detector ([`calibration_exp003.md`](docs/calibration_exp003.md)); the agent cross-checks its own seabed track against the depth sounder ([`sensor_check.md`](docs/reports/sensor_check.md)); shadow-flip prediction for second looks |
| **Real-world impact** | raw Humminbird recordings with real GPS, routes, budgets, GPX/KML/GeoJSON, OGC API, webhooks, MCP; a severity hook for the review queue (suspended or tall finds weighted up; an operator policy, not yet exercised: no find in the demo data has a measured risk factor) |
| **UX** | zero-build studio: guided demo, evidence cards with the full agent trace, 3D twin, approvals inbox, failure drills (`?drills`) |
| **Responsible cloud delivery** | Graviton + COOL deploy, dry-run by default, budgets first, provenance on every result ([`infra/README.md`](infra/README.md)); approver PIN; protected wrecks generalised in public exports ([`responsible_use.md`](docs/responsible_use.md)) |

---

## 1. Why this is different

Detectors for sonar debris exist. An imperfect detector is not a tool an NGO can depend on — so DEPTH
wraps one in **promises you can check** and **measurements instead of assumptions**:

| | what DEPTH does | evidence |
|---|---|---|
| **Guaranteed tiers** | "≥ 79% of pots reach a human (95% confidence)" — fit on held-out recordings, verified once on unseen test (**held: 81.1%**; the test's own lower bound is 76.7%). It also says what it *cannot* promise (90% recall; any auto-confirm precision). | [`docs/calibration_exp003.md`](docs/calibration_exp003.md) |
| **Stage 1 measures the sonar** | bottom tracking gives the sonar's altitude per ping → relative object height, slant→ground geotags. Validated with no labels: port vs starboard on the same pings agree to **1.6 px** (random pairs: 5.3 px). | [`docs/reports/stage1_canonical.md`](docs/reports/stage1_canonical.md) |
| **Value-of-information agent** | spends compute only where it can change an outcome (with EXP-003, calibration found a second look cannot change a tier: re-looks did not beat plain confidence, STUDY-07, so it runs one inference per frame and logs each skipped look); orders the human queue by calibrated P(pot); **analyst budget** ("what do 10 minutes buy?") and **boat budget** | trace on every card |
| **Active vision** (experimental) | the agent picks its next OpenCV tool by whether the result could change its decision; evidence conflicts trigger another observation — §2. Validation failed, so it is **off by default** and labelled | [`docs/reports/evidence_model_exp003.md`](docs/reports/evidence_model_exp003.md) |
| **A physical second look** | plans re-survey passes on the **opposite side** at mid-swath, ranked by **information per boat-minute**; each is approved or rejected on its own, and a rejection makes the agent re-plan; each target predicts the bearing its **shadow must flip to** | map + GPX/GeoJSON |
| **Minutes saved — to be measured** | a timed, counterbalanced user study built into the app (not yet run with people). Under today's *assumed* timings the DEPTH queue costs **more** than manual review at the recall promise (65.5 vs 56.3 min); break-even card time **6.88 s**. We report that instead of an invented "4 h → 10 min" | Study tab · [`docs/reports/effort_curve.md`](docs/reports/effort_curve.md) |
| **Are false alarms false?** | a **blinded audit with catch trials** gives exact audited precision in a confidence band | Audit tab |
| **People decide, the agent re-plans** | a named person's ✓ / ✕ on a card (and a crew's *recovered* / *not found*) makes the agent re-plan the recovery route, queue, budget and re-survey passes; an **impact ledger** measures the reviewed precision (95% CI) | Survey tab · [`docs/agentic_vision.md`](docs/agentic_vision.md) |
| **Humans teach it** | ✓ / ✕ / **＋ missed pot** on any frame → fine-tune set + hard negatives | `python -m src.agentic.feedback export` |
| **The agent re-measures its own geometry** | on a raw recording it cross-checks OpenCV's seabed track against the **depth sounder** per chunk; on a conflict it **re-runs the OpenCV tracker** where the sounder points, trusts the image where the sounder lost lock, withholds what neither supports, then re-fits the range scale until it converges. Fresh 1-h recording: 280 of 280 chunks resolved, all 3 gross tracker failures corrected; one registered criterion failed and is reported | [`docs/reports/sensor_check.md`](docs/reports/sensor_check.md) |
| **Real sonar recordings, real GPS** | reads a **raw Humminbird recording** (`.DAT` + `.SON`/`.IDX`) directly — per-ping GPS, heading, speed and depth, checked against physics (GPS speed vs speed field r = 0.95, course vs heading 2.4° median) — and **measures the range scale** (sonar depth ÷ Stage-1 altitude: 2.19 cm/sample ± 14.5%), so pins and heights are in real metres | [`docs/reports/raw_recording.md`](docs/reports/raw_recording.md) |
| **OpenCV output drives actions — shown by counterfactual** | the same survey re-run with each OpenCV output withheld: without Stage 1, **151 of 222** pins fall outside their own error circle and 4 of 40 re-survey passes regroup (EXP-003, v2b test); without the shadow, nothing changes (evidence only). Live in the product: **"⊘ without Stage 1"** on the survey map draws the ghost pins | [`docs/reports/causal_trace.md`](docs/reports/causal_trace.md) |
| **Plugs into any console** | **OGC API – Features** (QGIS / ArcGIS read hazards, passes, routes and approved work orders), **signed webhooks** to mission-control / dispatch systems, and a drop-in `<depth-hazards>` web component — the Connect tab has live URLs and snippets | [`docs/integrations.md`](docs/integrations.md) |
| **Plugs into any agent (MCP)** | DEPTH is an MCP server — stdio for Claude Desktop / Code, Streamable HTTP at `/mcp` for remote agents (bearer token). 13 tools run the loop and read the evidence; the only write is **asking** a person, who approves in the studio | [`docs/mcp.md`](docs/mcp.md) |
| **An LLM that writes, never decides** | a one-page **mission brief** for the crew; optional Claude on Amazon Bedrock rewrites it, but its text is shown only if **every number and ID traces back to the survey** and the safety caveats are present — otherwise the deterministic template is served | Survey tab · `GET /api/brief` |
| **COOL, proven by provenance** | the benchmark times the product itself per stage, $/survey-hour, and labels a run COOL only if `cv2` loads from `/opt/cool` | [`infra/README.md`](infra/README.md) |

We also publish what did **not** work: a classical ROI gate (STUDY-01), the acoustic shadow as a
filter (STUDY-03/04), re-look vs plain confidence (STUDY-07), range-gain detector input (STUDY-08),
seam inference across chunk boundaries (STUDY-11b), and a larger input + flipped second view
(STUDY-13: +0.10 AP on the calibration recording, −0.06 on unseen data).

## 2. The agentic loop — OpenCV result → decision → next action → new evidence → changed plan

**On by default, with evidence:** the seabed cross-check (STUDY-16), the survey planner (Stage-1
geometry moves pins and passes, STUDY-12) and the human approvals (a decline re-plans, an approval
issues a work order; both in the test suite). **Per-frame active vision** (below the dashed line) is built and tested, but it failed its own
validation, so it is off by default and labelled experimental. In default mode the agent still asks of
every find whether another look could change the decision. With EXP-003 the answer is always no (a
re-look did not beat plain confidence, STUDY-07), so it runs one inference per frame.

```
Sonar frame ─► OpenCV 5: Stage 1 (bottom track, geometry) + YOLO11 via cv2.dnn
            ─► evidence: water column · shadow · height ─► calibrated tier + P(pot) ─► a person's review queue
Raw recording ─► OpenCV seabed track vs depth sounder ─► conflict? ─► OpenCV re-tracks ─► range scale re-fit ─► re-check
Survey ─► pins from the measured geometry ─► opposite-side passes ranked by information per boat-minute
Person ─► ✓ / ✕ a card or decline a pass ─► the agent re-plans route, queue, budget and passes · approve ─► work order
- - - - - - - - - - - - - - - - experimental, off by default - - - - - - - - - - - - - - - - -
Per find ─► next OpenCV tool = the cheapest one whose result could CHANGE the action (ACCEPT / REVIEW / WATCH), else stop
         ─► new evidence ─► belief + action updated ─► conflict? ─► another observation
```

| feature | what the agent does | where |
|---|---|---|
| **Sensor cross-check** | OpenCV seabed track vs depth sounder → conflict → OpenCV re-track with a new search window → accept / trust the image / withhold → re-fit the range scale → re-check. It changes where pins go and heights in metres (0.36 m → 0.11 m on a chunk where the sounder lost lock) | Survey → Agent · Seabed cross-check · STUDY-16 |
| **Smart survey planning** | passes ranked by expected **information per boat-minute**, not confidence: the most likely target is worth little and does not drive the plan | Agent tab · `resurvey_rank` |
| **Human control** | *Agent recommends: Perform opposite-side resurvey RS1 · Reason: uncertain visual evidence · [Approve] [Reject]* — reject and the pass is dropped for good, the boat time re-ranked, its targets moved to the front of the inspection queue | Agent tab · `/api/approvals/{id}/decide` |
| **Agent Trace** | on every evidence card: can another look change the decision? → evidence → P(pot) → tier → who acts next; with active vision also the tool it chose, the new evidence and any changed decision | Analyze |
| **Adaptive tool use** · experimental, off | not the same tools every time: a confident, consistent find is accepted after one ~1 ms check with **no re-look**; a borderline one gets the shadow check and, if that could still flip it, a zoom re-look; a find no result could move gets **no tool at all** — every skip is logged with why | `active.next_tool` |
| **Evidence conflict** · experimental, off | the detector says ACCEPT but the shadow says no (or the box sits in the water column) → *"Evidence conflict detected → another observation"*: the next informative tool, else an opposite-side pass | trace step `conflict` |
| **Active second look** · experimental, off | the re-look / shadow result changes the action (e.g. ACCEPT → REVIEW) and that changes the queue and the plan | `update_belief` (before → after) |

**Does OpenCV's measurement matter? The same survey, detector only vs + Stage 1 + OpenCV evidence** (both arms run the same `cv2.dnn` detections) — the 8 shipped frames, 27 hazards
(`python -m src.agentic.counterfactual` → [`docs/reports/opencv_counterfactual.md`](docs/reports/opencv_counterfactual.md)):

| downstream | Detector only | + Stage 1 + evidence — default | + Stage 1 + evidence — active vision (experimental) |
|---|---|---|---|
| decision (agent action) | detector only (8 accept) | unchanged | **3 accept; 7 of 27 hazards change action** |
| geolocation | slant range, frame-centre ping | **median pin moves 6.5 m; 17 of 27 leave their own error circle** | same |
| re-survey route | 6 passes, 15.5 boat-min | 6 passes, 15.4 boat-min (2 regrouped) | **1 pass, 3.3 boat-min, 0 identical** |
| human approval | 6 pass requests, 0 conflicts | same | **1 pass request, 7 conflicts flagged** |
| OpenCV tool calls | — | shadow on all 31 | shadow on 23 / 31, zoom re-look on **2** / 31 |

**Is it better? Measured, and the honest answer is "not proven".** Likelihood ratios fit on the
validation recordings; replayed on every candidate ([`docs/reports/evidence_model_exp003.md`](docs/reports/evidence_model_exp003.md)):

| split | ACCEPT precision (without → with) | real pots left in WATCH | log-loss |
|---|---|---|---|
| val (fit) | 64% → 70% | 27 → 10 | 0.459 → 0.449 |
| test (once) | 67% → 73% | 49 → 36 | **0.637 → 0.648 (worse)** |
| `test_xsonar` (fresh, another sonar) | **0.86 → 0.848** | **127 → 161** | — |

Both pre-registered gates failed (test calibration; the fresh split — run 1 had a replay bug that
guessed shadow orientation, fixed and re-run once, still failed; both runs are kept in the report).
So **active vision is off by default**; the Stage-1 effects above (geolocation, routes) are on and
validated (STUDY-12). Switch it on in the studio (**Active vision · experimental**) or with
`?active=1`; every result it produces is labelled experimental.

## 3. Try it (no sonar files needed)

```bash
pip install -r requirements.txt                       # torch-free runtime: OpenCV 5.0.0.93, FastAPI
python -m src.detection.fetch_model                   # the detector: 38 MB, SHA-256 checked
python -m src.cv_pipeline.humminbird fetch            # optional: a raw recording with real GPS: 11 MB
python -m uvicorn src.dashboard.app:app --port 8000   # open http://localhost:8000
```

Press **▶ 60-s demo** in the top bar for a guided walk through everything below, or explore the modes (plus a 3D twin view):
* **Analyze** — pick a sample frame → *Run See → Prove → Decide*: the tracked seabed, verdict-coloured
  boxes, an evidence card per find (tier promise, P(pot), shadow, relative height, full agent trace),
  the agent's-eye re-look view; label finds, or draw a **missed pot** (`M`).
* **Survey** — all samples (synthetic GPS) or the **raw recording (real GPS)** as a background job →
  hazards on a map with error radii, the inspection route, **re-survey passes**, analyst + boat budgets;
  ✓ a card and the **recovery route** re-plans; approvals inbox, mission brief, GeoJSON/GPX/KML/CSV/JSON.
* **3D twin** (Analyze and Survey toggles) — a physics-grounded digital twin: the seabed in true
  ground-range geometry, the sonar at its tracked altitude, finds with shadow-derived heights and the
  acoustic ray triangle behind each height; the survey laid out along the track with a replay of the
  boat sweeping its sonar fans. Orbit / fly / top cameras, wireframe, backscatter relief.
* **Study** — the timed user study (manual review vs DEPTH cards) + the live effort curve.
* **Audit** — the blinded false-alarm audit.
* **Connect** — MCP, OGC API – Features, signed webhooks and the drop-in panel, with live URLs and snippets.

The 8 sample frames ship with the app (CC-BY-SA, [`webui/samples/ATTRIBUTION.md`](webui/samples/ATTRIBUTION.md)).
They come from the v2b **test** split, so the default model never trained or tuned on them.
Their GPS is synthetic and labelled as such (the public frames carry none). The raw recording is real:
PINGMapper's sample data (MIT code, Zenodo 10.5281/zenodo.6604666), fetched hash-checked, not redistributed.

## 4. Results (honest — every number names its split)

**Detector — EXP-003, the deployed model** (YOLO11s, 640 px, dataset v2b, 2 classes), `cv2.dnn`
([`docs/reports/onboard_exp003.md`](docs/reports/onboard_exp003.md)):

| | value | split |
|---|--:|---|
| crab pots (ghost gear) on sonar: AP@0.5 (95% CI) · AP@0.5:0.95 | **0.482** (0.43–0.55) · 0.165 | v2b test, 285 pots |
| crab pots: P / R / F1 at the validation-tuned threshold | 0.50 / 0.47 / 0.49 | v2b test |
| wreck debris: AP@0.5 · AP@0.5:0.95 | 0.156 · 0.047 (weak) | v2b test, 70 boxes |
| **mAP@0.5 · mAP@0.5:0.95** (both classes) | **0.319** · 0.106 — the proposal's 0.70 is **not met** | v2b test |
| GhostVision head-to-head: F1 · AP@0.5 | 0.41 · 0.39 (GhostVision reports F1 0.71–0.73) | official 398-frame split |
| cross-sonar: ghost gear AP@0.5 | 0.34; the val threshold does not transfer (recall 0.11) | `test_xsonar` |

Full reports, each with a confusion matrix and a PR curve: [test](docs/reports/eval_exp003_test.md) ·
[official 398](docs/reports/eval_exp003_test_official398.md) · [cross-sonar](docs/reports/eval_exp003_test_xsonar.md).
Why the gap: most training "wrecks" are fish-finder screenshots and the wreck boxes are loose — a data
problem, not the recipe ([`docs/reports/exp003_diagnosis.md`](docs/reports/exp003_diagnosis.md)). The recall promise is
what makes this detector safe to use: every pot it covers reaches a person.

<img src="docs/img/pr_exp003_test.png" width="430" alt="EXP-003 precision-recall curves on v2b test, with the validation-tuned operating points">

Five training runs after it (EXP-003f, 004, 004g, 005, 005a) did not beat it on the held-out
recordings ([`docs/reports/exp005_diagnosis.md`](docs/reports/exp005_diagnosis.md)).

**Earlier baseline — EXP-001 (YOLO11s, 640 px, dataset v1, 4 classes), deploy-faithful `cv2.dnn` evaluation**
([`docs/reports/eval_exp001.md`](docs/reports/eval_exp001.md)):

| | value |
|---|--:|
| mAP@0.5, all classes (test) | 0.822 (0.827 re-measured via `cv2.dnn`) — **inflated** by an optical-only class |
| crab pots on **sonar** — AP@0.5 / recall | **0.473 / 0.599** — the number that matters |
| recall ceiling on unseen recordings (any proposal ≥ 0.05) | 0.72 (Rec19) – 0.86 (Rec3/4/6/10) |

**Agent** — recall promise **≥ 79%** with EXP-003 (fit on the validation recordings, held on test at
81.1%, lower bound 76.7%; EXP-001's was ≥ 65%); nothing auto-confirmed (no precision promise is
supportable); 1 inference/frame (re-look did not beat confidence on unseen data; the old
test-tuned "CONFIRMED 0.737" is 0.58 unseen — retired).

**Runtime (laptop x86, stock OpenCV 5, full product path, EXP-003):** 252 ms/frame p50 (network ≈ 92%;
the OpenCV-only stages run at 40 fps); one survey-hour of sonar (~169 frames) processed in **51 s**
(real-time factor 0.014). **On EC2 (2026-10-20, 4 threads, stock OpenCV 5):** 157 ms/frame on c8i.xlarge x86, 197 ms on c8g.xlarge Graviton4 — stock Graviton is *slower* at equal size, which is the bar COOL has to clear; the c8g.large demo server passed the load test (4 clients + 1 survey: 291/291, p95 2.0 s). Results are committed in [`benchmarks/`](benchmarks) · [`docs/reports/cool_benchmark.md`](docs/reports/cool_benchmark.md).
<!-- cool:readme -->
**Graviton + COOL:** measured automatically the moment the Marketplace subscription is active (`PHASE=cool ./infra/golive.sh`).
<!-- /cool:readme --> EC2 numbers come from
`infra/golive.sh` (x86 + Graviton stock now, Graviton COOL once the Marketplace subscription is active).

**Real recording** (raw Humminbird, Colorado River, 150.6 s, real per-ping GPS —
[`docs/reports/raw_recording.md`](docs/reports/raw_recording.md)): 3,453 pings per channel parsed with 0 malformed;
range scale 2.19 cm/sample first estimate → **2.03 cm/sample ± 10.2%** after the agent's sensor
cross-check (STUDY-16); the full agent in ~4 s (~40× faster than the recording); with EXP-003,
2 review cards = **48 cards per hour of sonar** on water with no known pots — the false-alarm load
this model would put on an analyst there.

**Pending:** the timed study and the blinded audit with people, and the Graviton **COOL** runs (the Marketplace subscription is with AWS Support). The
weights are published (GitHub Release `exp003-v1`, SHA-256 checked by `fetch_model`). EXP-003 is
deployed; EXP-002/004/005 were trained and did not beat it
([`docs/reports/exp005_diagnosis.md`](docs/reports/exp005_diagnosis.md)).

## 5. How it works

```
frame ─► Stage 1 canonicalise ─► See (YOLO11 · cv2.dnn) ─► Prove (evidence) ─► Decide (guaranteed tiers)
          └► active vision (experimental): value-of-information tool choice · conflict ─► another observation
survey ─► stitch · merge repeat sightings ─► routes · budgets · opposite-side re-survey ─► exports
human ─► labels · timed study · blinded audit          offline ─► v2b ─► train ─► onboard (one command)
```
Full design: [`docs/architecture.md`](docs/architecture.md). Agentic writeup: [`docs/agentic_vision.md`](docs/agentic_vision.md).
Every document, grouped by what it proves: [`docs/README.md`](docs/README.md).

## 6. Reproduce

```bash
pip install -r requirements-dev.txt                 # runtime + test + lint dependencies
pytest -q                                           # test suite
python -m ruff check .                              # lint (configuration: pyproject.toml)
python -m src.detection.evaluate                    # deploy-faithful detector evaluation
python -m src.agentic.calibrate                     # fit + verify the guaranteed tiers
python -m src.agentic.effort                        # analyst-effort curve
python -m src.agentic.counterfactual                # same survey: detector only vs + Stage 1 + evidence
python -m src.bench.product_bench --label my_host   # benchmark this machine (product workload)
python -m src.cv_pipeline.study_canonical           # STUDY-08: Stage 1 altitude, port vs starboard
python -m src.detection.study_scale_tta             # STUDY-13: input size + flip TTA
python -m src.cv_pipeline.humminbird report         # STUDY-14: raw recording, real GPS
python -m src.agentic.evidence_model                # STUDY-15: evidence LRs (val) → test → fresh split
python -m src.agentic.study_sensor                  # STUDY-16: seabed track vs depth sounder

# STUDY-12: the counterfactual trace on up to 80 v2b test frames
python -m src.agentic.study_causal \
    --frames DATASET/03_yolo_ready_dataset_v2b/test/images --limit 80

# A new Kaggle-trained model: diagnose on validation (never test), then plug it in (+ speed gate)
python -m src.detection.diagnose --zip EXP-003_complete.zip
python -m src.detection.onboard_model --zip EXP-003_complete.zip
```
Datasets are git-ignored (large); their builders are in `DATASET/scripts/`
([`docs/dataset_card.md`](docs/dataset_card.md)). Deploy: [`infra/README.md`](infra/README.md).

## 7. Repository

```
src/cv_pipeline/  Stage 1 (canonical.py) · raw Humminbird reader (humminbird.py) · retired ROI gate (STUDY-01)
src/detection/    cv2.dnn detector, calibration, evaluation, training, onboarding (+ speed gate), model fetch,
                  false-alarm audit, failure gallery, STUDY-11b / 13
src/agentic/      See→Prove→Decide→Act agent, sensor cross-check (STUDY-16), active vision (active,
                  evidence_model, counterfactual), guarantees, geo, routes, re-survey, person-confirmed loop,
                  approvals, mission brief, 3D twin, provenance, labels, study, effort, STUDY-12
src/bench/        COOL benchmark (product workload, provenance, comparison, COOL receipt)
src/dashboard/    FastAPI service (jobs, limits, rate limits, metrics) · MCP server · OGC API · webhooks
webui/            zero-build studio (+ three.js twin) · embed/depth-embed.js (drop-in panel) · 8 sample frames
infra/            Graviton + COOL deploy (dry-run by default) + 3-way benchmark
models/<MODEL>/   calibration.json + effort/audit data (weights: GitHub Release, fetch_model)
benchmarks/       measured benchmark results (JSON): laptop and EC2
tests/            the pytest suite, run by CI on every push (.github/workflows/tests.yml)
notebooks/        the Kaggle training notebook and its generator
DATASET/scripts/  dataset builders (the data itself is git-ignored)
docs/             technical report, architecture, cards, guides (MCP, integrations, training), project
                  record (proposal, progress, experiments); docs/reports/ holds the evaluation and study reports
```
Documentation map: [`docs/README.md`](docs/README.md). Logs: [`docs/experiments.md`](docs/experiments.md) (every study,
including negative results) · [`docs/progress.md`](docs/progress.md).

## 8. Responsible use & license

Human approval before any dispatch; low-risk finds kept for audit, never deleted; protected-wreck
coordinates and data retention are covered in [`docs/responsible_use.md`](docs/responsible_use.md).

**AGPL-3.0** ([`LICENSE`](LICENSE)) — builds on Ultralytics YOLO11 (AGPL-3.0). Data sources keep
their own licenses (crab-pot: the dataset card's metadata says CC-BY-SA-4.0, its text says GPL —
recorded in the attribution; UATD CC-BY-4.0; the PINGMapper recording is fetched, not redistributed) —
see the dataset card.
*Team Syndicate · OpenCV AI Competition 2026.*
