# DEPTH — Judges' Guide

**DEPTH does not claim a better detector. It makes any detector dependable.** It turns side-scan sonar into a
cleanup plan that a named person approves. An agent promises how many pots reach a person, checks that the
promise applies to the recording, chooses how hard to look, perceives every frame with OpenCV 5, re-plans when
its own measurements or budgets say so, and asks before anything is dispatched.

| | |
|---|---|
| **Entry** | Team Syndicate (solo) · OpenCV AI Competition 2026 · **Agentic Vision** · **Best Use of COOL** |
| **Run it** | [Open in GitHub Codespaces](https://codespaces.new/madhesh60/depth?quickstart=1): one click, ready in about 3 minutes · or [five commands](#1-try-it-in-five-minutes) on a laptop, CPU only |
| **Live URL · video** | not live yet: CloudFront awaits AWS account verification (Codespaces runs the same studio) · not recorded yet |
| **Code health** | 322 tests · ruff · mypy · a headless-Chrome UI test, all in CI; CI also plays the guided demo inside the Codespaces image on 2 CPUs ([`codespaces.yml`](.github/workflows/codespaces.yml)) |

**On this page:** [1 Try it](#1-try-it-in-five-minutes) · [2 See the agent decide](#2-see-the-agent-decide) ·
[3 Key numbers](#3-key-numbers) · [4 Criteria map](#4-where-each-criterion-is-evidenced) ·
[5 Not done](#5-not-done--said-plainly) · [6 Glossary](#6-glossary)

---

## 1. Try It in Five Minutes

**In the browser:** the Codespaces link above. Its setup runs the same commands as below.

**On a laptop** (Python 3.10, CPU only):

```bash
git clone https://github.com/madhesh60/depth.git
cd depth
pip install -r requirements.txt                       # torch-free: OpenCV 5.0.0.93, FastAPI
python -m src.detection.fetch_model                   # the detector: 38 MB, SHA-256 checked
python -m src.cv_pipeline.humminbird fetch            # a raw sonar recording with real GPS: 11 MB
python -m uvicorn src.dashboard.app:app --port 8000   # open http://localhost:8000
```

Then three clicks:

1. **▶ Guided demo** (top bar): a 60-second walk through one frame and one survey.
2. **Analyze** → a sample → *Analyze frame*: the tracked seabed, one evidence card per object (overlapping boxes
   on one pot share a card), the agent's full trace, and "How the agent decided".
3. **Survey** → *GPS source: Raw recording · real per-ping GPS* → *Run survey* → **Agent** tab: the promise
   check (withdrawn on this river recording, with a full-review request), the seabed cross-check, the plan and
   the approval inbox.

On the deployed server, deciding needs the **approver PIN**; agents and API clients can only ask. Judges receive
the PIN with the submission. A local run sets no PIN.

---

## 2. See the Agent Decide

The same goal produces different runs, and each one is explained in plain words in the **Agent** tab.

| Do this | The agent | Where to look |
|---|---|---|
| **Survey**, Goal *Agent decides* (the default; 20-s compute budget) | keeps the strongest promise the budget allows. A fast machine looks harder (≥ 84%); a small one prices both looks by its measured pace and may switch mid-run (`replan`) | Agent → *How the agent decided* |
| Goal **≥ 78% · standard** | takes the cheaper single look: the goal is kept at about a fifth of the compute | same |
| GPS source **Raw recording · real per-ping GPS** | finds the recording unlike its calibration data, **withdraws the promise**, takes the cheapest look and asks a person to approve a full review | Agent → promise check · Approvals |
| `curl -s -X POST "http://localhost:8000/api/survey?use_samples=1&recall=0.95" \| grep -o "the goal cannot be met[^\"]*" \| head -1` | refuses the impossible goal in plain words and names the strongest promise it can keep (≥ 84%) | the API response (`agent_run`); about 30 s on a laptop |
| **Approvals** → approve or decline a request | re-plans the route, queue and passes, and logs who decided; a second, simultaneous decision gets a 409 | Survey → **Log** |
| Open `http://localhost:8000/?drills` → **Ready** (top right) → *Failure drills* | handles a failed tool, a timeout, a corrupt frame, a missing model or a failed brief writer, each with a stated fallback | the incident notice · Agent tab |

The full failure-handling table is in the [README](README.md#failure-handling). An external agent driving DEPTH
over MCP, whose own attempt to approve is refused, is recorded in
[`mcp_session.md`](docs/reports/mcp_session.md).

---

## 3. Key Numbers

Every number names its split. Thresholds are tuned on validation, and each test split was scored once.

| Claim | Number | Split · evidence |
|---|---|---|
| Recall promise, one look (95% confidence) | **≥ 78%** of pots reach a person; held at **81.1%** (lower bound 76.7%) | fit on val, verified once on test · [`calibration_exp003.md`](docs/calibration_exp003.md) |
| Recall promise, looking harder (2×2 tile pass) | **≥ 84%**; held at **86.0%** (lower bound 82.0%) | test · [`look_harder.md`](docs/reports/look_harder.md) (STUDY-18) |
| The promise survives correlated frames | block-bootstrap bound 78.2%, so the one-look promise was restated **79% → 78%**; looking harder stands (84.3%) | val · [`dependence.md`](docs/reports/dependence.md) (STUDY-21) |
| …per recording (a promise is a pooled rate) | one look ≥ 78% on 5 of 6 recordings (Rec9: 74.7%) | val + test · same |
| Works with any detector | the same code: EXP-001 **≥ 63%** (held at 86.2%), EXP-003 **≥ 78%** (held at 81.1%) | each on its own unseen test · [`calibration_exp001.md`](docs/calibration_exp001.md) |
| Knows when its promise does not apply | withdrawn in **22 of 22** surveys on another sonar, where one look keeps only 23.9% of pots | `test_xsonar` · [`applicability.md`](docs/reports/applicability.md) (STUDY-19) |
| One review card per object | **3.86 → 2.32** cards per frame looking harder (2.68 → 1.81 one look); the **same pots** reach a person; 1 of 462 cards holds two pots | test · [`card_grouping.md`](docs/reports/card_grouping.md) (STUDY-23) |
| OpenCV output moves the boat | without Stage 1, **95 of 142** pins leave their own error circle | test, 80 frames · [`causal_trace.md`](docs/reports/causal_trace.md) (STUDY-12) |
| Crab-pot detector, AP@0.5 | **0.482** (95% CI 0.43–0.55); mAP@0.5 0.319 **misses the 0.70 target** | v2b test · [`eval_exp003_test.md`](docs/reports/eval_exp003_test.md) |
| Against GhostVision (same split) | F1 **0.41** vs their reported 0.71–0.73: **DEPTH's detector is weaker** | official 398 · [`eval_exp003_test_official398.md`](docs/reports/eval_exp003_test_official398.md) |
| Speed per frame (stock OpenCV 5, 4 vCPU) | one look **138.5 ms** Graviton4 · 145.8 ms x86; looking harder 655 · 693 ms | EC2 c8g.xlarge · c8i.xlarge · [`cool_benchmark.md`](docs/reports/cool_benchmark.md) |
| OpenCV 5's two DNN engines | same answers (634 of 634 detections); on Graviton the classic engine is **1.41–1.47×** faster, so the server measures both at start-up and keeps the faster | [`dnn_engine.md`](docs/reports/dnn_engine.md) (STUDY-20) |
| The demo server under load | **383 / 383** ok, analyze p95 **2.44 s** (mark 3 s), 190 analyses a minute; the agent took the standard look 382 of 383 times | c8g.large · [`load_test__c8g_large_rehearsal.json`](benchmarks/load/load_test__c8g_large_rehearsal.json) |
| Analyst minutes (**assumed** 20 s/frame, 8 s/card) | one look **43.2 vs 55.7** manual · looking harder 56.5 vs 57.3 · empty seabed 34 / 47 vs 56 per survey-hour | test · [`effort_curve.md`](docs/reports/effort_curve.md) |

---

## 4. Where Each Criterion Is Evidenced

| Award · criterion | Where DEPTH shows it |
|---|---|
| **Overall · Technical execution** | OpenCV 5 does the perception and geometry on a CPU, torch-free; deploy-faithful evaluation with CIs; 322 tests in CI · [`technical_report.md`](docs/technical_report.md) |
| **Overall · Innovation** | a statistical recall promise on an imperfect detector, checked for correlated frames and withdrawn where it does not apply · [`dependence.md`](docs/reports/dependence.md), [`applicability.md`](docs/reports/applicability.md) |
| **Overall · Real-world impact** | raw Humminbird recordings with real GPS; GPX / KML / GeoJSON, OGC API, webhooks, MCP; analyst minutes with their assumptions stated · [`raw_recording.md`](docs/reports/raw_recording.md), [`integrations.md`](docs/integrations.md) |
| **Overall · UX · Documentation** | zero-build studio with a guided demo, evidence cards, approvals inbox, 3D twin and a *Terms* glossary; 23 registered studies, negatives included · [`experiments.md`](docs/experiments.md) |
| **Overall · Responsible cloud delivery** | one stack, dry run by default, 0 cfn-lint / cfn-guard findings, budget alerts, a scheduled stop, PIN lockout · [`infra/README.md`](infra/README.md), [`responsible_use.md`](docs/responsible_use.md) |
| **Agentic · OpenCV 5 + agent** | OpenCV output decides by default: the applicability check, Stage 1 pin placement, the sensor cross-check, the measured `cv2.dnn` engine · [`agentic_vision.md`](docs/agentic_vision.md) |
| **Agentic · Orchestration and autonomy** | goal → check the promise → choose the mode → perceive with a budget watch → plan → ask; re-plans on its own pace and on every decision · [`core.py`](src/agentic/core.py), Agent tab |
| **Agentic · Task effectiveness** | promises held on unseen test (81.1% · 86.0%); 2.32 cards per frame · [`look_harder.md`](docs/reports/look_harder.md), [`card_grouping.md`](docs/reports/card_grouping.md) |
| **Agentic · Failure handling and human control** | every failure an incident with a fallback; drills; ask-only agents; PIN with lockout; 409 on racing decisions · [README § Failure Handling](README.md#failure-handling), [`agentic_loop.md`](docs/agentic_loop.md) |
| **COOL · Verified COOL on Graviton** | a run counts as COOL only if `cv2` loads from `/opt/cool`; AWS-signed receipt + offline verifier (`verify_cool`); status in §5 · [`infra/README.md`](infra/README.md) |
| **COOL · Architecture, measured value, reproducibility** | one CloudFormation stack; a registered 3-way benchmark (STUDY-17): stock Graviton4 138.5 ms vs x86 145.8 ms at a 15% lower price; one command per phase (`infra/golive.sh`) · [`cool_benchmark.md`](docs/reports/cool_benchmark.md) |

---

## 5. Not Done — Said Plainly

- **COOL:** <!-- cool:status -->has not run yet. The AWS Marketplace subscription to the COOL AMI is blocked on payment: the Marketplace does not accept the account's card (it needs a Visa or Mastercard), and an AWS Support case is open. Until then the demo server runs stock OpenCV 5 in rehearsal mode and says so. The COOL benchmark, the AWS-signed receipt and its verifier are built and run in one command once the subscription is active (`PHASE=cool ./infra/golive.sh`).<!-- /cool:status -->
- **No public URL yet** (CloudFront account verification). Codespaces or the five commands run the same studio;
  the server is deployed in rehearsal mode.
- **The test split is not fresh any more.** STUDY-18, 19, 21 and 23 each scored the same 199 test frames once,
  and STUDY-23 was started because of a test-split result. Treat the test numbers as checked, not untouched.
- **The 8 demo frames all come from one test recording (Rec9)**, the one where the one-look promise fell short
  (74.7% against 78%). They show the product working, not a best case.
- **No real-LLM session recorded yet.** The recorded MCP session uses a scripted client;
  `python -m src.dashboard.llm_session` lets Claude drive DEPTH (tested with a stand-in model). Bedrock is
  blocked because this account cannot accept the Claude Marketplace agreement, so the run needs a Claude API key.
- **No human study and no blinded audit yet.** Analyst minutes and audited precision are not measured; the
  timings above are assumptions, labelled as such.
- **No field proof.** No recording has both real pots and real GPS: the pot frames carry synthetic GPS (labelled
  on screen), and the real-GPS river recordings hold no known pots.
- **Also:** boxes, not masks (the brief allows either); pitch and roll are not corrected (the sonar does not
  record them); the optional Bedrock brief writer is not deployed.

---

## 6. Glossary

The same terms are one click away in the studio (**Terms**, top bar).

| Term | Meaning |
|---|---|
| **recall promise** | "at least R% of pots reach a person (95% confidence)": fit on held-out recordings, verified once on unseen ones |
| **τ_review** | the detector-confidence threshold above which a find becomes a review card (0.06 standard, 0.055 looking harder) |
| **LCB** | lower confidence bound: the true rate is at least this, with 95% confidence |
| **CP** | Clopper-Pearson: the exact binomial bound behind the promise (it treats pots as independent; STUDY-21 checks that) |
| **LTT** | Learn-Then-Test: scanning thresholds in a fixed order, so trying many does not break the bound |
| **P(pot)** | a card's calibrated chance of being a real pot; it orders the review queue |
| **VOI** | value of information: "could another look change the decision?" If not, the agent does not spend it |
| **look harder** | a second pass over 2×2 tiles (5 forward passes per frame) that lifts the promise from 78% to 84% |
| **one card per object** | review boxes of one class that sit at least half inside each other share one card and one decision; no box is dropped (STUDY-23) |
| **applicability check** | OpenCV statistics of each frame vs the calibration recordings: is the promise valid here? |
| **Stage 1** | OpenCV sonar geometry: bottom track (altitude per ping), slant → ground range, water column |
| **STUDY-n** | a registered experiment in [`experiments.md`](docs/experiments.md): protocol first, then the data |
| **COOL** | the OpenCV-optimised AMI for AWS Graviton; a run counts as COOL only if `cv2` loads from `/opt/cool` |

**More:** [README](README.md) · [technical report](docs/technical_report.md) ·
[architecture](docs/architecture.md) · [agentic vision](docs/agentic_vision.md) ·
[experiment register](docs/experiments.md) · [documentation map](docs/README.md)
