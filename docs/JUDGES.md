# DEPTH for judges — one page

**DEPTH** (Detect · Evidence · Prove · Triage · Hazard-map) turns side-scan sonar into a cleanup plan a
person approves. **It does not claim a better detector — it makes any detector dependable:** an agent promises
how many pots reach a person, checks whether that promise applies to the recording, chooses how hard to look,
perceives every frame with OpenCV 5, re-plans when its own measurements or budgets say so, and asks a named
person before anything is dispatched. Solo entry, Team Syndicate, OpenCV AI Competition 2026 — targets:
**Agentic Vision** and **Best Use of COOL**.

## Try it

**In your browser, one click:** [Open in GitHub Codespaces](https://codespaces.new/madhesh60/depth?quickstart=1) —
the real studio, ready in about 3 minutes, free with a GitHub account (the setup runs the same commands as below;
CI runs that setup in the codespace's own image as its user on 2 CPUs: the studio is up seconds after the setup,
and the guided demo plays to its last step in about a minute with 0 JS errors —
[`codespaces.yml`](../.github/workflows/codespaces.yml)).

**On a laptop, 5 minutes, CPU only:**

```bash
git clone https://github.com/madhesh60/depth.git
cd depth
pip install -r requirements.txt                       # torch-free: OpenCV 5.0.0.93, FastAPI
python -m src.detection.fetch_model                   # the detector, 38 MB, SHA-256 checked
python -m src.cv_pipeline.humminbird fetch            # a raw sonar recording with real GPS, 11 MB
python -m uvicorn src.dashboard.app:app --port 8000   # open http://localhost:8000
```

Then three clicks:

1. **▶ Guided demo** (top bar) — a 60-second walk through one frame and one survey.
2. **Analyze** → a sample → *Analyze frame* — boxes, one evidence card per object (overlapping boxes on one pot
   share a card), the agent's full trace, and which perception mode it chose and why ("How the agent decided").
3. **Survey** → *Raw recording (real GPS)* → **Agent** tab — the core loop: the promise check (withdrawn on
   this river recording, with a full-review request), the seabed cross-check, the plan, the approval inbox.
   A survey gets 20 s of compute by default (*Options*): on two cores the agent prices both looks and takes
   the one that fits; clear the field to let it always look harder.

A public URL is **not live yet**: CloudFront waits on AWS account verification (see "Not done" below), so
Codespaces is the click-to-run path. On the deployed server, deciding needs the **approver PIN** (agents and API
clients can only ask); judges receive it with the submission, and the studio says so under the PIN field.

## The numbers (every one names its split)

| what | number | where |
|---|---|---|
| Recall promise, standard look (95% confidence) | **≥ 78%** of pots reach a person; held on unseen test at **81.1%** | [`calibration_exp003.md`](calibration_exp003.md) |
| Recall promise, looking harder (2×2 tile pass) | **≥ 84%**; held on unseen test at **86.0%** (lower bound 82.0%) | [`look_harder.md`](reports/look_harder.md) |
| The promise survives correlated frames? | standard: block-bootstrap bound 78.2% → restated **79% → 78%**; look harder 84.3% → stands | [`dependence.md`](reports/dependence.md) (STUDY-21) |
| …per recording (a promise is a pooled rate) | standard ≥ 78% on 5 of 6 recordings (Rec9: 74.7%) | same |
| The same calibration code, two detectors | EXP-001 (4-class, v1): ≥ 63%, held at 86.2% on its own unseen test · EXP-003: ≥ 78%, held at 81.1% — both after the same correlated-frames rule (it lowered each by 2 points) | [`calibration_exp001.md`](calibration_exp001.md) · [`calibration_exp003.md`](calibration_exp003.md) |
| The agent knows when its promise does not apply | another sonar: withdrawn in **22 of 22** surveys (one look keeps 23.9% of pots there) | [`applicability.md`](reports/applicability.md) (STUDY-19) |
| Crab-pot detector AP@0.5 | **0.482** (95% CI 0.43–0.55) — mAP@0.5 0.319 **misses the 0.70 target** | [`eval_exp003_test.md`](reports/eval_exp003_test.md) |
| Against GhostVision (same split) | F1 **0.41** vs their reported 0.71–0.73 — **DEPTH's detector is weaker** | [`eval_exp003_test_official398.md`](reports/eval_exp003_test_official398.md) |
| OpenCV 5's new DNN engine | loads by default; **634 of 634** detections identical to the classic engine, 0 tier flips; speed: within noise on the laptop | [`dnn_engine.md`](reports/dnn_engine.md) (STUDY-20) |
| Product frame, standard look (stock OpenCV 5, 4 vCPU) | **157 ms** x86 (c8i.xlarge) · **197 ms** Graviton4 (c8g.xlarge) | [`cool_benchmark.md`](reports/cool_benchmark.md) |
| Product frame, looking harder (the default when time allows) | **≈ 5×** the standard frame (laptop, interleaved: 406 → 2,110 ms) → about **0.8 s** on the same EC2 x86 (estimated; the EC2 run is pending) | `benchmarks/laptop_x86_modes*` |
| One review card per object | cards per frame **2.68 → 1.81** (one look), **3.86 → 2.32** (looking harder); the **same pots** reach a person; 1 of 462 cards holds two pots | [`card_grouping.md`](reports/card_grouping.md) (STUDY-23) |
| On one CPU core (on-board budget) | 599 ms/frame and 240 MB on one Graviton4 core → a survey-hour (280 frames) in ~168 s, **~21× faster than the sonar records** (no small board measured) | `benchmarks/graviton_stock_t1__product.json` × the real recording |
| Analyst minutes (ASSUMED 20 s/frame, 8 s/card; one card per object) | pot-dense test: one look **43.2 vs 55.7** manual, looking harder **56.5 vs 57.3**; empty seabed 34 / 47 vs 56 min per survey-hour (before one card per object: 62.7 vs 55.7, **slower**) | [`effort_curve.md`](reports/effort_curve.md) |
| Real 1-h recording (another sonar, promise withdrawn) | **48 review cards per hour** of sonar (54 before one card per object) | same |
| Seabed measurement moves the boat | without Stage 1, 95 of 142 pins leave their own error circle | [`causal_trace.md`](reports/causal_trace.md) (STUDY-12) |

## Not done — said plainly

- **COOL has not run yet.** The Marketplace subscription is still pending (an AWS Support case is open);
  the benchmark, receipt verifier and report are built and run the moment it is active
  (`PHASE=cool ./infra/golive.sh`).
- **No public URL yet** (CloudFront account verification). The server is deployed in rehearsal mode.
- **No human study and no blinded audit with people**: analyst minutes and audited precision are not
  measured; the timings above are assumptions, labelled as such (including that a card with several boxes on one
  object takes as long as a one-box card).
- **No field proof**: no recording has both real pots and real GPS. The pot frames carry synthetic GPS (labelled
  on screen); the real-GPS river recordings hold no known pots.
- The c8g.large **load test** (passed, p95 2.0 s) ran before the look-harder mode existed; the analyze
  agent now counts the server queue and takes the standard look when busy — the re-run is pending.
- **Boxes, not masks** (the brief allows either); **pitch and roll** are not corrected (not recorded by
  the sonar); the optional Bedrock brief writer is **not deployed** on the demo server.

## Glossary

The same terms are one click away in the studio (**Terms**, top bar).


| term | meaning |
|---|---|
| **recall promise** | "at least R% of pots reach a person (95% confidence)" — fit on held-out recordings, verified once on unseen ones |
| **τ_review** | the detector-confidence threshold above which a find becomes a review card (0.06 standard, 0.055 looking harder) |
| **LCB** | lower confidence bound: the value the true rate is at least, with 95% confidence |
| **CP** | Clopper-Pearson: the exact binomial bound behind the promise (it treats pots as independent — STUDY-21 checks that) |
| **LTT** | Learn-Then-Test: scanning thresholds in a fixed order so that trying many does not break the bound |
| **P(pot)** | a card's calibrated chance of being a real pot — orders the review queue |
| **VOI** | value of information: "could another look change the decision?" — if not, the agent does not spend it |
| **look harder** | a second pass over 2×2 tiles (5 forward passes per frame) that lifts the promise from 78% to 84% |
| **one card per object** | review boxes of one class that sit at least half inside each other share one card and one decision; no box is dropped (STUDY-23) |
| **applicability check** | OpenCV statistics of each frame vs the calibration recordings → is the promise valid here? |
| **Stage 1** | OpenCV sonar geometry: bottom track (altitude per ping), slant → ground range, water column |
| **STUDY-n** | a registered experiment in [`experiments.md`](experiments.md) — protocol first, then the data |
| **COOL** | the OpenCV-optimised AMI for AWS Graviton; a run counts as COOL only if `cv2` loads from `/opt/cool` |

More: [README](../README.md) · [technical report](technical_report.md) · [architecture](architecture.md) ·
[experiment log](experiments.md) · [documentation map](README.md).
