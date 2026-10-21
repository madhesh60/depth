# DEPTH — See · Prove · Decide · Act

[![tests](https://github.com/madhesh60/depth/actions/workflows/tests.yml/badge.svg)](https://github.com/madhesh60/depth/actions/workflows/tests.yml)

> **DEPTH** = **D**etect · **E**vidence · **P**rove · **T**riage · **H**azard-map — the pipeline
> stages spell the name.

An agent that finds **derelict fishing gear (ghost pots) and wreck debris in side-scan sonar** and
turns it into a human-approved cleanup plan. The operator gives it a **goal** — the share of pots that
must reach a person, and the compute, analyst and boat time it may spend. The agent then **checks that
its promise applies to this recording**, **chooses how hard to look**, perceives every frame with OpenCV
5, **re-plans when its own measurements or budgets say so**, and asks a named person before anything is
dispatched. Every find comes with evidence; every promise is calibrated on held-out recordings and
verified once on unseen ones; every human decision becomes a training label. It runs torch-free on
**OpenCV 5**, CPU-only, built for **AWS Graviton + COOL**.

**Competition:** OpenCV AI Competition 2026 (AWS · OpenCV Foundation) · **Team:** Syndicate (solo —
Madhesh) · **Targets:** Agentic Vision · Best Use of COOL · **License:** AGPL-3.0

![DEPTH architecture: OpenCV 5 agent (Stage 1 → See → Prove → Decide → Act) on EC2 Graviton + COOL, with the human loop, exports and AWS services](docs/img/architecture.svg)

### At a glance

| | number | measured on |
|---|--:|---|
| **Recall promise** — pots that reach a person (95% confidence) | **≥ 84%** when the agent looks harder, held at **86.0%** · **≥ 78%** with one look, held at 81.1% | fit on held-out recordings, verified once on 199 unseen frames / 264 pots (the test frames that carry a recording ID: Rec8/9/14) — STUDY-18 |
| …and it survives correlated frames | block bootstrap: one look 78.2% → restated **79% → 78%**; looking harder 84.3% → stands. A pooled rate: per recording, one look held on **5 of 6** (Rec9: 74.7%) | STUDY-21, calibration split + per recording |
| **The agent knows when its promise does not apply** | another sonar: promise withdrawn in **22 of 22** surveys, where one look keeps only **23.9%** of pots · kept for all 3 test recordings | STUDY-19, verified once |
| Crab-pot detector, AP@0.5 · AP@0.5:0.95 | **0.482** (95% CI 0.43–0.55) · 0.165 | v2b test, all 214 unique crab-pot frames / 285 pots |
| OpenCV's seabed measurement moves the boat | **151 of 222** pins leave their own error circle without it | STUDY-12, v2b test |
| Stage-1 altitude, port vs starboard (no labels needed) | agree to **1.6 px** (random pairs 5.3 px) | STUDY-08 |
| Product frame, standard look | **157 ms** x86 · **197 ms** Graviton4 · looking harder ≈ **5×** that | EC2 4 vCPU, stock OpenCV 5 (COOL pending) · laptop, interleaved |
| OpenCV 5's new DNN engine (the one that loads) | **634 of 634** detections identical to the classic engine, 0 tier flips | STUDY-20, calibration split |
| Raw sonar recording, real GPS | range scale **2.03 cm/sample ± 10.2%**, measured | Colorado River, Humminbird |
| Tests | **281** passed, CI on every push (line coverage 60% of `src/`: **85% of the runtime modules**, 26% of the offline study scripts, whose outputs are the reports); a headless-Chrome UI smoke test drives every mode with 0 JS errors (`python tests/ui_smoke.py`); a few tests need optional data and skip without it | `pytest -q` · 2026-10-21 |

**Not yet measured, and said so:** Graviton + COOL timings (stock Graviton and x86 were measured on EC2 on
2026-10-20, §4; the COOL runs wait on the Marketplace subscription), the look-harder mode on EC2 and a load
test with it, analyst minutes saved (the timed study needs people), audited false-alarm precision (the blinded
audit needs people). The detector misses the proposal's 0.70 mAP target; §4 says by how much and why, and §5
lists every limit. **One-page judge guide (numbers, three clicks, glossary): [`docs/JUDGES.md`](docs/JUDGES.md).**

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
| **Technical execution** — OpenCV 5 doing real work | Stage 1 (bottom track with `cv2.Sobel` / `cv2.blur`, slant→ground `cv2.remap`), YOLO11 through `cv2.dnn` on OpenCV 5's new engine (recorded in every result's provenance; same answers as the classic engine — [`dnn_engine.md`](docs/reports/dnn_engine.md)), shadow + re-look evidence, a per-ping data-dropout mask on raw recordings; deploy-faithful evaluation with CIs, mAP@0.5:0.95, confusion matrix, PR curves ([`eval_exp003_test.md`](docs/reports/eval_exp003_test.md)); the promise checked for correlated frames ([`dependence.md`](docs/reports/dependence.md)); deadlines + fallbacks on every tool |
| **Agentic Vision** — perception changes the next action | **the core loop** ([`core.py`](src/agentic/core.py), §2) works toward the operator's goal: OpenCV statistics decide whether its promise applies (withdrawn on another sonar in 22 of 22 surveys, with a full-review request — [`applicability.md`](docs/reports/applicability.md)); it chooses how hard to look (the 2×2 tile look lifts the promise from 78% to 84%, held at 86.0% — [`look_harder.md`](docs/reports/look_harder.md)); it re-plans mid-run when its measured pace would overrun the compute budget or the closer look fails, and an analyze call prices its look against the server's queue. Also on by default and measured: OpenCV's seabed measurement decides where the pins go (without it 151 of 222 leave their own error circle and 4 of 40 re-survey passes regroup — [`causal_trace.md`](docs/reports/causal_trace.md)); the agent cross-checks that seabed against the depth sounder and re-tracks on a conflict ([`sensor_check.md`](docs/reports/sensor_check.md)); opposite-side passes ranked by information per boat-minute; a person's decline re-plans, an approval issues a work order (§2). Per-frame active vision is built and tested but failed its own validation, so it is off by default ([`evidence_model_exp003.md`](docs/reports/evidence_model_exp003.md)) |
| **Innovation** | statistical promises on an imperfect detector ([`calibration_exp003.md`](docs/calibration_exp003.md)) that are also checked for correlated frames and restated when they fall short ([`dependence.md`](docs/reports/dependence.md)), a goal-driven choice of how hard to look (two calibrated perception modes), and a conformal check that withdraws the promise where it does not apply; the agent cross-checks its own seabed track against the depth sounder ([`sensor_check.md`](docs/reports/sensor_check.md)); shadow-flip prediction for second looks |
| **Real-world impact** | raw Humminbird recordings with real GPS, routes, budgets, GPX/KML/GeoJSON, OGC API, webhooks, MCP; analyst effort measured honestly: at assumed timings DEPTH beats manual review only when fewer than 23% of frames hold a pot ([`effort_curve.md`](docs/reports/effort_curve.md)); a severity hook for the review queue (suspended or tall finds weighted up; an operator policy, not yet exercised: no find in the demo data has a measured risk factor) |
| **UX** | zero-build studio: guided demo, evidence cards with the full agent trace, 3D twin, approvals inbox, failure drills (`?drills`) |
| **Responsible cloud delivery** | Graviton + COOL deploy, dry-run by default, budgets first, provenance on every result ([`infra/README.md`](infra/README.md)); approver PIN with a wrong-PIN lockout; a public server refuses to start without its tokens; protected wrecks generalised in public exports ([`responsible_use.md`](docs/responsible_use.md)) |

---

## 1. Why this is different

Detectors for sonar debris exist. An imperfect detector is not a tool an NGO can depend on — so DEPTH
wraps one in **promises you can check** and **measurements instead of assumptions**:

| | what DEPTH does | evidence |
|---|---|---|
| **Calibrated promises, two ways to keep them** | "≥ 78% of pots reach a human (95% confidence)" with one look, **≥ 84%** when the agent looks harder — fit on held-out recordings, verified once on unseen test (**held: 81.1% and 86.0%**; lower bounds 76.7% and 82.0%). Clopper-Pearson alone gave 79% for one look; a block bootstrap over correlated frames supports 78.2%, so the promise was restated (STUDY-21). It also says what it *cannot* promise (90% recall; any auto-confirm precision; a per-survey guarantee). | [`docs/calibration_exp003.md`](docs/calibration_exp003.md) · [`look_harder.md`](docs/reports/look_harder.md) · [`dependence.md`](docs/reports/dependence.md) |
| **Knows when its promise does not apply** | before keeping a promise, the agent compares every frame with the recordings the promise was fit on (OpenCV statistics → conformal p-values). Another sonar: withdrawn in 22 of 22 surveys — there one look keeps only 23.9% of the pots — and a person is asked to approve a full review | [`docs/reports/applicability.md`](docs/reports/applicability.md) |
| **Stage 1 measures the sonar** | bottom tracking gives the sonar's altitude per ping → relative object height, slant→ground geotags. Validated with no labels: port vs starboard on the same pings agree to **1.6 px** (random pairs: 5.3 px). | [`docs/reports/stage1_canonical.md`](docs/reports/stage1_canonical.md) |
| **Goal-driven agent** | chooses how hard to look from the goal and the compute budget (the cheapest mode that keeps the goal, else the strongest the budget allows), prices each mode with its measured pace and re-plans mid-run if it would overrun; orders the human queue by calibrated P(pot); **analyst budget** ("what do 10 minutes buy?") and **boat budget**. Per find, a zoomed re-look cannot change a tier with EXP-003 (it did not beat plain confidence, STUDY-07), so it is skipped and the skip is logged | Agent tab · trace on every card |
| **Active vision** (experimental) | the agent picks its next OpenCV tool by whether the result could change its decision; evidence conflicts trigger another observation — §2. Validation failed, so it is **off by default** and labelled | [`docs/reports/evidence_model_exp003.md`](docs/reports/evidence_model_exp003.md) |
| **A physical second look** | plans re-survey passes on the **opposite side** at mid-swath, ranked by **information per boat-minute**; each is approved or rejected on its own, and a rejection makes the agent re-plan; each target predicts the bearing its **shadow must flip to** | map + GPX/GeoJSON |
| **Minutes saved — to be measured** | a timed, counterbalanced user study built into the app (not yet run with people). Under today's *assumed* timings (20 s per frame, 8 s per card) the DEPTH queue costs **more** than manual review at the recall promise on the pot-dense test (62.7 vs 55.7 min; break-even card time **7.11 s**); on mostly empty seabed it is faster only when fewer than **23%** of frames hold a pot (46 vs 56 min per survey-hour at none), because the detector still raises ~2 cards per empty frame. On 1 h of real river water (another sonar — the promise is withdrawn there) it raised **54 cards per hour**. We report that instead of an invented "4 h → 10 min" | Study tab · [`docs/reports/effort_curve.md`](docs/reports/effort_curve.md) |
| **Are false alarms false?** | a **blinded audit with catch trials** gives exact audited precision in a confidence band | Audit tab |
| **People decide, the agent re-plans** | a named person's ✓ / ✕ on a card (and a crew's *recovered* / *not found*) makes the agent re-plan the recovery route, queue, budget and re-survey passes; an **impact ledger** measures the reviewed precision (95% CI) | Survey tab · [`docs/agentic_vision.md`](docs/agentic_vision.md) |
| **Humans teach it** | ✓ / ✕ / **＋ missed pot** on any frame → fine-tune set + hard negatives | `python -m src.agentic.feedback export` |
| **The agent re-measures its own geometry** | on a raw recording it cross-checks OpenCV's seabed track against the **depth sounder** per chunk; on a conflict it **re-runs the OpenCV tracker** where the sounder points, trusts the image where the sounder lost lock, withholds what neither supports, then re-fits the range scale until it converges. Fresh 1-h recording: 280 of 280 chunks resolved, all 3 gross tracker failures corrected; one registered criterion failed and is reported | [`docs/reports/sensor_check.md`](docs/reports/sensor_check.md) |
| **Data dropouts flagged** | per ping, from the recording itself: a time gap (missing pings) or a dark ping (not imaged) → a `data_dropout` incident and a note on every find that touches one — the promise covers imaged seabed only. Real recordings: 0 of 6,906 and 2 of 140,032 pings flagged; checked on injected dropouts | [`docs/reports/raw_recording.md`](docs/reports/raw_recording.md) |
| **Real sonar recordings, real GPS** | reads a **raw Humminbird recording** (`.DAT` + `.SON`/`.IDX`) directly — per-ping GPS, heading, speed and depth, checked against physics (GPS speed vs speed field r = 0.95, course vs heading 2.4° median) — and **measures the range scale** (sonar depth ÷ Stage-1 altitude: first estimate 2.19 cm/sample ± 14.5%, **2.03 cm/sample ± 10.2%** after the sensor cross-check), so pins and heights are in real metres | [`docs/reports/raw_recording.md`](docs/reports/raw_recording.md) |
| **OpenCV output drives actions — shown by counterfactual** | the same survey re-run with each OpenCV output withheld: without Stage 1, **151 of 222** pins fall outside their own error circle and 4 of 40 re-survey passes regroup (EXP-003, v2b test); without the shadow, nothing changes (evidence only). Live in the product: **"⊘ without Stage 1"** on the survey map draws the ghost pins | [`docs/reports/causal_trace.md`](docs/reports/causal_trace.md) |
| **Plugs into any console** | **OGC API – Features** (QGIS / ArcGIS read hazards, passes, routes and approved work orders), **signed webhooks** to mission-control / dispatch systems, and a drop-in `<depth-hazards>` web component — the Connect tab has live URLs and snippets | [`docs/integrations.md`](docs/integrations.md) |
| **Plugs into any agent (MCP)** | DEPTH is an MCP server — stdio for Claude Desktop / Code, Streamable HTTP at `/mcp` for remote agents (bearer token). 13 tools run the loop and read the evidence; the only write is **asking** a person, who approves in the studio | [`docs/mcp.md`](docs/mcp.md) |
| **An LLM that writes, never decides** | a one-page **mission brief** for the crew; optional Claude on Amazon Bedrock rewrites it, but its text is shown only if **every number and ID traces back to the survey** and the safety caveats are present — otherwise the deterministic template is served. **Not deployed on the demo server** (it needs `anthropic[bedrock]` and a Bedrock IAM grant), so the demo serves the template | Survey tab · `GET /api/brief` |
| **COOL, proven by provenance** | the benchmark times the product itself per stage, $/survey-hour, and labels a run COOL only if `cv2` loads from `/opt/cool` | [`infra/README.md`](infra/README.md) |

We also publish what did **not** work: a classical ROI gate (STUDY-01), the acoustic shadow as a
filter (STUDY-03/04), re-look vs plain confidence (STUDY-07), range-gain detector input (STUDY-08),
seam inference across chunk boundaries (STUDY-11b), and a larger input + flipped second view
(STUDY-13: +0.10 AP on the calibration recording, −0.06 on unseen data).

## 2. The agent — a goal, OpenCV tools, and a loop that re-plans on what it sees

The core loop ([`src/agentic/core.py`](src/agentic/core.py)) runs every survey, on by default. The
operator states a **goal**; the agent decides how to meet it and logs every option it weighed (the
studio's **Agent** tab, `agent_run` in the API, `run_survey` over MCP):

```
Goal ─ recall to keep · compute budget · analyst minutes · boat minutes
 1 inspect      usable frames? (corrupt / blank ones skipped and reported)
    check       does the promise apply here? OpenCV statistics of every frame vs the calibration recordings
                 └ no ─► withdraw the promise · cheapest look · ask a person to approve a FULL REVIEW
 2 decide       how hard to look: standard (≥ 78%, 1 cv2.dnn pass/frame) or look harder (≥ 84%, + 2×2 tiles)
                 = the cheapest mode that keeps the goal, else the strongest the compute budget allows
 3 perceive     every frame: Stage 1 · YOLO11 via cv2.dnn · evidence — and watch the pace:
                 └ budget would overrun, or the closer look fails ─► RE-PLAN: standard look, re-tier
    sense       raw recording: OpenCV seabed track vs depth sounder ─► conflict ─► re-track ─► re-fit the scale
 4 plan         review queue · analyst budget · routes · opposite-side passes ranked by information per boat-minute
 5 hand off     one approval request per action ─► a named person decides ─► every decision RE-PLANS
- - - - - - - - - - - - - - - - experimental, off by default - - - - - - - - - - - - - - - - -
Per find ─► next OpenCV tool = the cheapest one whose result could CHANGE the action, else stop (STUDY-15)
```

**Autonomy levels — what the agent decides alone, what it asks, what only people do:**

| the agent decides alone | it asks a person | only a person does |
|---|---|---|
| how hard to look (goal + budgets), whether its promise applies, which seabed measurement to trust, where pins go, the queue order, which passes are worth boat time, when to stop | inspect · recover · re-survey · full review · public sharing — one request per action, with its rationale | approve or decline (name + approver PIN, with a lockout), confirm or reject a card, mark recovered / not found — each decision re-plans the route, queue, budget and passes |

It never dispatches; MCP and API clients can only ask. A recorded session on the real server shows the
whole chain — an external agent runs a survey over MCP, reads the evidence, asks; its own attempt to
approve is refused; a person approves; the agent reads the work order back:
[`mcp_session.md`](docs/reports/mcp_session.md) (`python -m src.dashboard.mcp_session`).

**Does it work? Every agent decision below was validated before it was switched on** (protocol committed
before the unseen data was scored):

| decision (on by default) | evidence |
|---|---|
| **look harder** to keep a stronger promise | promise 78% → **84%**; on unseen test 81.1% → **86.0%** (lower bound 82.0%), cards / frame 2.68 → 3.86, ≈ 5× the compute — [`look_harder.md`](docs/reports/look_harder.md) (STUDY-18; the 78% after STUDY-21) |
| **withdraw the promise** where it does not apply | another sonar: withdrawn in 22 of 22 surveys (one look keeps 23.9% of its pots there); kept for all 3 test recordings and the demo frames — [`applicability.md`](docs/reports/applicability.md) (STUDY-19) |
| **re-measure the seabed** when image and sounder disagree | fresh 1-h recording: 280 of 280 chunks resolved, 3 of 3 gross tracker failures corrected; one registered criterion failed and is reported — [`sensor_check.md`](docs/reports/sensor_check.md) (STUDY-16) |
| **put pins where the measured geometry says** | without Stage 1, 151 of 222 pins leave their own error circle — [`causal_trace.md`](docs/reports/causal_trace.md) (STUDY-12) |
| re-plan on a person's decision | in the test suite: a decline drops the pass and re-ranks boat time; an approval issues a work order |

The same goal produces different runs: with no recall asked for, the agent looks harder (≥ 84%); asked
for ≥ 78% it takes the cheaper single look; with a tight compute budget it prices both modes with its
measured pace and drops to one look, mid-run if needed; on the raw river recordings it withdraws the
promise and asks for a full review. A single analyze call has a 2-s answer budget: the agent first takes
off the compute already queued for the network, so an idle server looks harder and a busy one takes the
standard look (tested in the suite; the EC2 load test with it is pending). **Per-frame active vision**
(below the dashed line) is built and tested, but it failed its own validation, so it stays off by default
and labelled experimental.

| feature | what the agent does | where |
|---|---|---|
| **Sensor cross-check** | OpenCV seabed track vs depth sounder → conflict → OpenCV re-track with a new search window → accept / trust the image / withhold → re-fit the range scale → re-check. It changes where pins go and heights in metres (0.36 m → 0.11 m on a chunk where the sounder lost lock) | Survey → Agent · Seabed cross-check · STUDY-16 |
| **Smart survey planning** | passes ranked by expected **information per boat-minute**, not confidence: the most likely target is worth little and does not drive the plan | Agent tab · `resurvey_rank` |
| **Human control** | *Agent recommends: Perform opposite-side resurvey RS1 · Reason: uncertain visual evidence · [Approve] [Reject]* — reject and the pass is dropped for good, the boat time re-ranked, its targets moved to the front of the inspection queue | Agent tab · `/api/approvals/{id}/decide` |
| **Agent Trace** | on every evidence card: can another look change the decision? → evidence → P(pot) → tier → who acts next; with active vision also the tool it chose, the new evidence and any changed decision | Analyze |
| **Active vision** · experimental, off | per find, the next OpenCV tool is the cheapest one whose result could change the action (a confident, consistent find gets no re-look; a borderline one the shadow check, then a zoom re-look); an **evidence conflict** (the detector says accept, the shadow or the water column says no) triggers another observation or an opposite-side pass | `active.next_tool` · trace step `conflict` |

**Does OpenCV's measurement matter?** The same survey of the 8 shipped frames (27 hazards), detector only
vs + Stage 1 + OpenCV evidence, both arms on the same `cv2.dnn` detections: by default Stage 1 moves the median pin
**6.5 m** and **17 of 27** pins leave their own error circle; with active vision on, **7 of 27** hazards also
change action and the re-survey plan shrinks from 6 passes (15.5 boat-min) to 1 (3.3)
([`opencv_counterfactual.md`](docs/reports/opencv_counterfactual.md), `python -m src.agentic.counterfactual`).
**Is active vision better? Not proven:** both pre-registered gates failed — on test the log-loss got worse
(0.637 → 0.648), and on another sonar ACCEPT precision fell (0.86 → 0.848) with more real targets left in WATCH
(127 → 161) ([`evidence_model_exp003.md`](docs/reports/evidence_model_exp003.md); a replay bug in run 1 was fixed
and the gate re-run once — both runs are kept). So it stays **off by default**, labelled experimental in the studio
and behind `?active=1`; the Stage-1 effects (geolocation, routes) are on and validated (STUDY-12).

## 3. Try it (no sonar files needed)

```bash
pip install -r requirements.txt                       # torch-free runtime: OpenCV 5.0.0.93, FastAPI
python -m src.detection.fetch_model                   # the detector: 38 MB, SHA-256 checked
python -m src.cv_pipeline.humminbird fetch            # optional: a raw recording with real GPS: 11 MB
python -m uvicorn src.dashboard.app:app --port 8000   # open http://localhost:8000
```

Press **▶ 60-s demo** in the top bar for a guided walk through everything below, or explore the modes (plus a 3D twin view); **Terms** (top bar) explains the studio's words.
In **Survey**, the *Goal* selector sets the recall the agent must keep (default: the strongest it can) and
*Options → Compute budget* gives it a time limit; the **Agent** tab shows how it decided:
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

**Detector — EXP-003, the deployed model** (YOLO11s, 640 px, dataset v2b tiles, 2 classes), `cv2.dnn`
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

**Agent** — recall promise **≥ 84%** when it looks harder (2×2 tile pass, 5 inferences/frame; held on
test at 86.0%, lower bound 82.0%) and **≥ 78%** with one look (held at 81.1%, lower bound 76.7%;
EXP-001's is ≥ 63%, 65% before the same check), both fit on the validation recordings. Clopper-Pearson alone supported 79% for one
look; a block bootstrap over correlated frames supports 78.2%, so that promise was restated (STUDY-21). A
promise is a pooled rate: per recording, the one-look promise held on 5 of 6 (test Rec9: 74.7%); the promise is withdrawn on recordings
that do not look like them (another sonar: 22 of 22 surveys). Nothing auto-confirmed (no precision
promise is supportable); the per-candidate re-look did not beat confidence on unseen data (the old
test-tuned "CONFIRMED 0.737" is 0.58 unseen — retired).

**Runtime (laptop x86, stock OpenCV 5, standard look, EXP-003, 2026-10-12):** 252 ms/frame p50 (network
≈ 92%; the OpenCV-only stages run at 40 fps); one survey-hour of sonar (~169 frames) processed in **51 s**
(real-time factor 0.014). **On EC2 (2026-10-20, 4 threads, stock OpenCV 5):** 157 ms/frame on c8i.xlarge x86, 197 ms on c8g.xlarge Graviton4 — stock Graviton is *slower* at equal size, which is the bar COOL has to clear. **Looking harder** runs 5 forward passes: ≈ 5× the standard frame (laptop, the two modes interleaved, 2026-10-21: 406 → 2,110 ms p50, 4.7–5.9× per repeat — the laptop was slower that day than on 2026-10-12). The c8g.large demo server passed the load test (4 clients + 1 survey: 291/291, p95 2.0 s) **before** the look-harder mode existed; analyze now counts the server queue before it chooses, and the re-run is pending. The engine that loads is OpenCV 5's new one (STUDY-20: identical answers to the classic engine; its speed gain is within noise on the laptop and is measured on every EC2 machine by the fleet script). Results are committed in [`benchmarks/`](benchmarks) · [`docs/reports/cool_benchmark.md`](docs/reports/cool_benchmark.md) · [`docs/reports/dnn_engine.md`](docs/reports/dnn_engine.md).
<!-- cool:readme -->
**Graviton + COOL:** measured automatically the moment the Marketplace subscription is active (`PHASE=cool ./infra/golive.sh`).
<!-- /cool:readme --> EC2 numbers come from
`infra/golive.sh` (x86 + Graviton stock now, Graviton COOL once the Marketplace subscription is active).

**Real recording** (raw Humminbird, Colorado River, 150.6 s, real per-ping GPS —
[`docs/reports/raw_recording.md`](docs/reports/raw_recording.md)): 3,453 pings per channel parsed with 0 malformed;
range scale 2.19 cm/sample first estimate → **2.03 cm/sample ± 10.2%** after the agent's sensor
cross-check (STUDY-16); the full agent in ~4 s (~40× faster than the recording); with EXP-003,
2 review cards = **48 cards per hour of sonar** on water with no known pots — the false-alarm load
this model would put on an analyst there; the 1-h Solix recording (Pearl River) gives **54 cards per hour**
through the core loop, which withdraws its promise on both (another sonar) and asks for a full review.

**Pending:** the timed study and the blinded audit with people, and the Graviton **COOL** runs (the Marketplace subscription is with AWS Support). The
weights are published (GitHub Release `exp003-v1`, SHA-256 checked by `fetch_model`). EXP-003 is
deployed; EXP-002/004/005 were trained and did not beat it
([`docs/reports/exp005_diagnosis.md`](docs/reports/exp005_diagnosis.md)).

## 5. Limits — what DEPTH does not do (yet)

- **The detector is weak.** mAP@0.5 0.319 against the proposal's 0.70; crab-pot AP@0.5 0.482; wreck 0.156;
  on the official split F1 0.41 against GhostVision's reported 0.71–0.73; cross-sonar recall 0.11 at the
  validation threshold. DEPTH's promises make *this* detector usable; they do not make it accurate.
- **A promise is a pooled rate**, fit on 3 recordings and verified on 3 others — not a per-survey guarantee
  (one test recording reached 74.7% against the 78% promise). The applicability check catches another
  sonar or rendering, not subtle site or gain drift.
- **Vehicle motion.** Heave moves the seabed in the image and Stage 1 tracks the seabed per ping (within
  ±8 rows of the frame's bottom), so slant → ground uses each ping's own altitude. **Pitch and roll are not
  corrected** — the sonar's ping header carries no attitude. Data dropouts are **flagged, not repaired**: a
  pot inside a dropout cannot be found.
- **Boxes, not pixel masks.** The brief allows either; the proposal planned YOLO-seg, but the crab-pot
  labels are boxes.
- **Not validated in the field**: no recording has both real pots and real GPS; the river recordings have
  no known pots. **No human study and no blinded audit yet**: analyst minutes and audited precision are
  assumptions until people run them (Study and Audit tabs).
- **Not deployed yet**: COOL (Marketplace subscription pending), a public URL (CloudFront account
  verification pending), the optional Bedrock brief writer (off on the demo server). Active vision is built
  but off: it failed its own validation.

## 6. How it works

```
frame ─► Stage 1 canonicalise ─► See (YOLO11 · cv2.dnn) ─► Prove (evidence) ─► Decide (guaranteed tiers)
          └► active vision (experimental): value-of-information tool choice · conflict ─► another observation
survey ─► stitch · merge repeat sightings ─► routes · budgets · opposite-side re-survey ─► exports
human ─► labels · timed study · blinded audit          offline ─► v2b ─► train ─► onboard (one command)
```
Full design: [`docs/architecture.md`](docs/architecture.md). Agentic writeup: [`docs/agentic_vision.md`](docs/agentic_vision.md).
Every document, grouped by what it proves: [`docs/README.md`](docs/README.md).

## 7. Reproduce

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
python -m src.agentic.study_look_harder --verify    # STUDY-18: look harder (calibration -> test once)
python -m src.agentic.study_applicability --verify  # STUDY-19: does the promise apply? (fit -> verify once)
python -m src.bench.dnn_engine --label my_host      # STUDY-20: OpenCV 5's two DNN engines, speed + same answers
python -m src.agentic.study_dependence              # STUDY-21: does the promise survive correlated frames?
python -m src.agentic.effort --real-water           # analyst effort + a raw 1-h recording end to end

# STUDY-12: the counterfactual trace on up to 80 v2b test frames
python -m src.agentic.study_causal \
    --frames DATASET/03_yolo_ready_dataset_v2b/test/images --limit 80

# A new Kaggle-trained model: diagnose on validation (never test), then plug it in (+ speed gate)
python -m src.detection.diagnose --zip EXP-003_complete.zip
python -m src.detection.onboard_model --zip EXP-003_complete.zip
```
Datasets are git-ignored (large); their builders are in `DATASET/scripts/`
([`docs/dataset_card.md`](docs/dataset_card.md)). Deploy: [`infra/README.md`](infra/README.md).

## 8. Repository

```
src/cv_pipeline/  Stage 1 (canonical.py) · raw Humminbird reader (humminbird.py) · retired ROI gate (STUDY-01)
src/detection/    cv2.dnn detector, calibration, evaluation, training, onboarding (+ speed gate), model fetch,
                  false-alarm audit, failure gallery, STUDY-11b / 13
src/agentic/      the core loop (core.py: goal -> check the promise -> choose how hard to look -> perceive ->
                  plan -> ask), the frame agent, applicability (STUDY-19), look harder (STUDY-18), sensor
                  cross-check (STUDY-16), active vision (active, evidence_model, counterfactual), guarantees,
                  geo, routes, re-survey, approvals, mission brief, 3D twin, provenance, labels, study, effort
src/bench/        COOL benchmark (product workload, provenance, comparison, COOL receipt)
src/dashboard/    FastAPI service: app.py (wiring) · service.py (state) · routes/ (one router per area) ·
                  jobs, limits, rate limits, metrics · MCP server + a recorded session · OGC API · webhooks
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

## 9. Responsible use & license

Human approval before any dispatch; low-risk finds kept for audit, never deleted; protected-wreck
coordinates and data retention are covered in [`docs/responsible_use.md`](docs/responsible_use.md).

**AGPL-3.0** ([`LICENSE`](LICENSE)) — builds on Ultralytics YOLO11 (AGPL-3.0). Data sources keep
their own licenses (crab-pot: the dataset card's metadata says CC-BY-SA-4.0, its text says GPL —
recorded in the attribution; UATD CC-BY-4.0; the PINGMapper recording is fetched, not redistributed) —
see the dataset card.
*Team Syndicate · OpenCV AI Competition 2026.*
