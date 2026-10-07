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

---

## 1. Why this is different

Detectors for sonar debris exist. An imperfect detector is not a tool an NGO can depend on — so DEPTH
wraps one in **promises you can check** and **measurements instead of assumptions**:

| | what DEPTH does | evidence |
|---|---|---|
| **Guaranteed tiers** | "≥ 65% of pots reach a human (95% confidence)" — fit on held-out recordings, verified once on unseen test (**held: 86.2%, LCB 80.5%**). It also says what it *cannot* promise (90% recall; any auto-confirm precision). | [`docs/calibration_exp001.md`](docs/calibration_exp001.md) |
| **Stage 1 measures the sonar** | bottom tracking gives the sonar's altitude per ping → relative object height, slant→ground geotags. Validated with no labels: port vs starboard on the same pings agree to **1.6 px** (random pairs: 5.3 px). | [`docs/stage1_canonical.md`](docs/stage1_canonical.md) |
| **Value-of-information agent** | spends compute only where it can change an outcome; orders the human queue by calibrated P(pot); **analyst budget** ("what do 10 minutes buy?") and **boat budget** | trace on every card |
| **Active vision** (experimental) | the agent picks its next OpenCV tool by whether the result could change its decision; evidence conflicts trigger another observation — §2. Validation failed, so it is **off by default** and labelled | [`docs/evidence_model_exp003.md`](docs/evidence_model_exp003.md) |
| **A physical second look** | plans re-survey passes on the **opposite side** at mid-swath, ranked by **information per boat-minute**; each is approved or rejected on its own, and a rejection makes the agent re-plan; each target predicts the bearing its **shadow must flip to** | map + GPX/GeoJSON |
| **Minutes saved — measured** | a timed, counterbalanced user study built into the app; the effort curve reports the **break-even card time** (7.95 s) instead of an invented "4 h → 10 min" | Study tab · [`docs/effort_curve.md`](docs/effort_curve.md) |
| **Are false alarms false?** | a **blinded audit with catch trials** gives exact audited precision in a confidence band | Audit tab |
| **People decide, the agent re-plans** | a named person's ✓ / ✕ on a card (and a crew's *recovered* / *not found*) makes the agent re-plan the recovery route, queue, budget and re-survey passes; an **impact ledger** measures the reviewed precision (95% CI) | Survey tab · [`docs/agentic_vision.md`](docs/agentic_vision.md) |
| **Humans teach it** | ✓ / ✕ / **＋ missed pot** on any frame → fine-tune set + hard negatives | `python -m src.agentic.feedback export` |
| **Real sonar recordings, real GPS** | reads a **raw Humminbird recording** (`.DAT` + `.SON`/`.IDX`) directly — per-ping GPS, heading, speed and depth, checked against physics (GPS speed vs speed field r = 0.95, course vs heading 2.4° median) — and **measures the range scale** (sonar depth ÷ Stage-1 altitude: 2.19 cm/sample ± 14.5%), so pins and heights are in real metres | [`docs/raw_recording.md`](docs/raw_recording.md) |
| **OpenCV output drives actions — shown by counterfactual** | the same survey re-run with each OpenCV output withheld: without Stage 1, **193 of 264** pins fall outside their own error circle and 13 of 63 re-survey passes regroup; without the shadow, nothing changes (evidence only). Live in the product: **"⊘ without Stage 1"** on the survey map draws the ghost pins | [`docs/causal_trace.md`](docs/causal_trace.md) |
| **Plugs into any console** | **OGC API – Features** (QGIS / ArcGIS read hazards, passes, routes and approved work orders), **signed webhooks** to mission-control / dispatch systems, and a drop-in `<depth-hazards>` web component — the Connect tab has live URLs and snippets | [`docs/integrations.md`](docs/integrations.md) |
| **Plugs into any agent (MCP)** | DEPTH is an MCP server — stdio for Claude Desktop / Code, Streamable HTTP at `/mcp` for remote agents (bearer token). 13 tools run the loop and read the evidence; the only write is **asking** a person, who approves in the studio | [`docs/mcp.md`](docs/mcp.md) |
| **An LLM that writes, never decides** | a one-page **mission brief** for the crew; optional Claude on Amazon Bedrock rewrites it, but its text is shown only if **every number and ID traces back to the survey** and the safety caveats are present — otherwise the deterministic template is served | Survey tab · `GET /api/brief` |
| **COOL, proven by provenance** | the benchmark times the product itself per stage, $/survey-hour, and labels a run COOL only if `cv2` loads from `/opt/cool` | [`infra/README.md`](infra/README.md) |

We also publish what did **not** work: a classical ROI gate (STUDY-01), the acoustic shadow as a
filter (STUDY-03/04), re-look vs plain confidence (STUDY-07), range-gain detector input (STUDY-08),
seam inference across chunk boundaries (STUDY-11b), and a larger input + flipped second view
(STUDY-13: +0.10 AP on the calibration recording, −0.06 on unseen data).

## 2. The agentic loop — OpenCV result → decision → next action → new evidence → changed plan

```
Sonar frame ─► OpenCV 5: Stage 1 (bottom track, geometry) + YOLO11 via cv2.dnn
           ─► evidence: confidence · geometry (water column, 0 ms) · shadow (~1 ms) · height · zoom re-look (~0.4 s)
           ─► agent belief P(pot) ─► action: ACCEPT / REVIEW / WATCH
           ─► next OpenCV tool = the cheapest one whose result could CHANGE that action (else stop)
           ─► new evidence ─► belief + action updated ─► conflict? ─► another observation
           ─► person approves ─► inspection route · opposite-side re-survey pass (rejected ─► the agent re-plans)
```

| feature | what the agent does | where |
|---|---|---|
| **Adaptive tool use** | not the same tools every time: a confident, consistent find is accepted after one ~1 ms check with **no re-look**; a borderline one gets the shadow check and, if that could still flip it, a zoom re-look; a find no result could move gets **no tool at all** — every skip is logged with why | `active.next_tool` |
| **Evidence conflict** | the detector says ACCEPT but the shadow says no (or the box sits in the water column) → *"Evidence conflict detected → another observation"*: the next informative tool, else an opposite-side pass | trace step `conflict` |
| **Active second look** | the re-look / shadow result changes the action (e.g. ACCEPT → REVIEW) and that changes the queue and the plan | `update_belief` (before → after) |
| **Smart survey planning** | passes ranked by expected **information per boat-minute**, not confidence: the most likely target is often worth ~0 bits and does not drive the plan | Agent tab · `resurvey_rank` |
| **Human control** | *Agent recommends: Perform opposite-side resurvey RS1 · Reason: evidence conflict on H001 · [Approve] [Reject]* — reject and the pass is dropped for good, the boat time re-ranked, its targets moved to the front of the inspection queue | Agent tab · `/api/approvals/{id}/decide` |
| **Agent Trace** | DETECT → evidence → decision → OpenCV tool → new evidence → decision changed → approval → survey plan, on every evidence card | Analyze |

**Does OpenCV matter? The same survey with vs without it** — the 8 shipped frames, 27 hazards
(`python -m src.agentic.counterfactual` → [`docs/opencv_counterfactual.md`](docs/opencv_counterfactual.md)):

| downstream | WITHOUT OpenCV | WITH OpenCV — default | WITH OpenCV — active vision (experimental) |
|---|---|---|---|
| decision (agent action) | detector only (8 accept) | unchanged | **3 accept; 7 of 27 hazards change action** |
| geolocation | slant range, frame-centre ping | **median pin moves 6.5 m; 17 of 27 leave their own error circle** | same |
| re-survey route | 6 passes, 15.5 boat-min | 6 passes, 15.4 boat-min (2 regrouped) | **3 passes, 6.9 boat-min, 0 identical** |
| human approval | 6 pass requests, 0 conflicts | same | **3 pass requests, 7 conflicts flagged** |
| OpenCV tool calls | — | shadow on all 31 | shadow on 23 / 31, zoom re-look on **2** / 31 |

**Is it better? Measured, and the honest answer is "not proven".** Likelihood ratios fit on the
validation recordings; replayed on every candidate ([`docs/evidence_model_exp003.md`](docs/evidence_model_exp003.md)):

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
pip install -r requirements.txt                    # torch-free runtime: OpenCV 5.0.0.93, FastAPI
python -m src.detection.fetch_model                # the trained detector (38 MB, SHA-256 checked)
python -m src.cv_pipeline.humminbird fetch         # optional: a raw sonar recording with real GPS (11 MB, SHA-256 pinned)
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
([`docs/onboard_exp003.md`](docs/onboard_exp003.md)):

| | value | split |
|---|--:|---|
| crab pots (ghost gear) on sonar: AP@0.5 (95% CI) | **0.482** (0.43–0.55) | v2b test, 285 pots |
| wreck debris: AP@0.5 | 0.156 (weak) | v2b test, 70 boxes |
| GhostVision head-to-head: F1 | 0.41 (GhostVision reports 0.71–0.73) | official 398-frame split |
| cross-sonar: ghost gear AP@0.5 | 0.34 | `test_xsonar` |

Five training runs after it (EXP-003f, 004, 004g, 005, 005a) did not beat it on the held-out
recordings ([`docs/exp005_diagnosis.md`](docs/exp005_diagnosis.md)).

**Earlier baseline — EXP-001 (YOLO11s, 640 px, dataset v1, 4 classes), deploy-faithful `cv2.dnn` evaluation**
([`docs/eval_exp001.md`](docs/eval_exp001.md)):

| | value |
|---|--:|
| mAP@0.5, all classes (test) | 0.822 (0.827 re-measured via `cv2.dnn`) — **inflated** by an optical-only class |
| crab pots on **sonar** — AP@0.5 / recall | **0.473 / 0.599** — the number that matters |
| recall ceiling on unseen recordings (any proposal ≥ 0.05) | 0.72 (Rec19) – 0.86 (Rec3/4/6/10) |

**Agent** — recall promise **≥ 79%** with EXP-003 (fit on the validation recordings, held on test at
81.1%, lower bound 76.7%; EXP-001's was ≥ 65%); nothing auto-confirmed (no precision promise is
supportable); 1 inference/frame (re-look did not beat confidence on unseen data; the old
test-tuned "CONFIRMED 0.737" is 0.58 unseen — retired).

**Runtime (laptop x86, stock OpenCV 5, full product path):** 238 ms/frame p50 (Stage 1 ≈ 5 ms,
network ≈ 86%); one survey-hour of sonar (~169 frames) processed in **43 s** (real-time factor
0.012). Graviton/COOL numbers: `infra/bench_cool.sh` on EC2 (pending).

**Real recording** (raw Humminbird, Colorado River, 150.6 s, real per-ping GPS —
[`docs/raw_recording.md`](docs/raw_recording.md)): 3,453 pings per channel parsed with 0 malformed;
range scale measured 2.19 cm/sample ± 14.5%; the full agent in 5.8 s (26× faster than the recording);
8 review cards = **191 cards per hour of sonar** on water with no known pots — the false-alarm load
this model would put on an analyst there.

**Pending, by design:** EXP-003, the recall lever. EXP-002 trained and **failed** (ghost AP 0.25 on the
held-out recordings: underfit by a silent optimizer switch, plus tile label poisoning; post-mortem in
[`docs/exp002_diagnosis.md`](docs/exp002_diagnosis.md)). The fixed kit is ready
([`docs/kaggle_training.md`](docs/kaggle_training.md)). Also pending: the timed study,
the audit, the EC2 benchmark, and publishing the weights as a GitHub Release.

## 5. How it works

```
frame ─► Stage 1 canonicalise ─► See (YOLO11 · cv2.dnn) ─► Prove (evidence) ─► Decide (guaranteed tiers)
          └► active vision (experimental): value-of-information tool choice · conflict ─► another observation
survey ─► stitch · merge repeat sightings ─► routes · budgets · opposite-side re-survey ─► exports
human ─► labels · timed study · blinded audit          offline ─► v2b ─► train ─► onboard (one command)
```
Full design: [`architecture.md`](architecture.md). Agentic writeup: [`docs/agentic_vision.md`](docs/agentic_vision.md).

## 6. Reproduce

```bash
pytest -q                                              # test suite
python -m src.agentic.calibrate                        # fit + verify the guaranteed tiers
python -m src.detection.evaluate                       # deploy-faithful detector evaluation
python -m src.cv_pipeline.study_canonical              # STUDY-08 (Stage 1)
python -m src.agentic.effort                           # analyst-effort curve
python -m src.bench.product_bench --label my_host      # benchmark this machine (product workload)
python -m src.detection.diagnose --zip EXP-003_complete.zip        # why a new model is (not) good: val + fit check
python -m src.detection.onboard_model --zip EXP-003_complete.zip   # plug in a new model (+ speed gate)
python -m src.agentic.study_causal --frames <v1>/test/images --limit 80   # STUDY-12 counterfactual
python -m src.agentic.evidence_model                  # STUDY-15: evidence LRs (val) -> test -> fresh split
python -m src.agentic.counterfactual                  # the same survey WITH vs WITHOUT OpenCV
python -m src.detection.study_scale_tta               # STUDY-13 input size + flip TTA
python -m src.cv_pipeline.humminbird report           # STUDY-14 raw recording, real GPS
```
Datasets are git-ignored (large); their builders are in `DATASET/scripts/`
([`docs/dataset_card.md`](docs/dataset_card.md)). Deploy: [`infra/README.md`](infra/README.md).

## 7. Repository

```
src/cv_pipeline/  Stage 1 (canonical.py) · raw Humminbird reader (humminbird.py) · retired ROI gate (STUDY-01)
src/detection/    cv2.dnn detector, calibration, evaluation, training, onboarding (+ speed gate), model fetch,
                  false-alarm audit, failure gallery, STUDY-11b / 13
src/agentic/      See→Prove→Decide→Act agent, active vision (active, evidence_model, counterfactual),
                  guarantees, geo, routes, re-survey, person-confirmed loop,
                  approvals, mission brief, 3D twin, provenance, labels, study, effort, STUDY-12
src/bench/        COOL benchmark (product workload, provenance, comparison)
src/dashboard/    FastAPI service (jobs, limits, rate limits, metrics) · MCP server · OGC API · webhooks
webui/            zero-build studio (+ three.js twin) · embed/depth-embed.js (drop-in panel)
infra/            Graviton + COOL deploy (dry-run by default) + 3-way benchmark
models/<MODEL>/   calibration.json + effort/audit data (weights: GitHub Release, fetch_model)
docs/             technical report, cards, calibration, evaluation, studies, MCP, integrations, …
.github/          CI: the test suite on every push
```
Logs: [`experiments.md`](experiments.md) (every study, including negative results) ·
[`progress.md`](progress.md) · [`TODO.md`](TODO.md).

## 8. Responsible use & license

Human approval before any dispatch; low-risk finds kept for audit, never deleted; protected-wreck
coordinates and data retention are covered in [`docs/responsible_use.md`](docs/responsible_use.md).

**AGPL-3.0** ([`LICENSE`](LICENSE)) — builds on Ultralytics YOLO11 (AGPL-3.0). Data sources keep
their own licenses (crab-pot: the dataset card's metadata says CC-BY-SA-4.0, its text says GPL —
recorded in the attribution; UATD CC-BY-4.0; the PINGMapper recording is fetched, not redistributed) —
see the dataset card.
*Team Syndicate · OpenCV AI Competition 2026.*
