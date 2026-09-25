# DEPTH — See · Prove · Decide · Act

> **DEPTH** = **D**etect · **E**vidence · **P**rove · **T**riage · **H**azard-map — the pipeline
> stages spell the name.

An agentic system that finds **derelict fishing gear (ghost pots) and wreck debris in side-scan
sonar** and turns it into a human-approved cleanup plan. Every find **comes with evidence**; the
agent's tiers carry **calibrated statistical promises**; it spends compute, analyst minutes and boat
time only where they can change the outcome; every human decision becomes a training label. It runs
torch-free on **OpenCV 5**, CPU-only, built for **AWS Graviton + COOL**.

**Competition:** OpenCV AI Competition 2026 (AWS · OpenCV Foundation) · **Team:** Syndicate (solo —
Madhesh) · **Targets:** Agentic Vision · Best Use of COOL · **License:** AGPL-3.0

---

## 1. Why this is different

Detectors for sonar debris exist. An imperfect detector is not a tool an NGO can depend on — so DEPTH
wraps one in **promises you can check** and **measurements instead of assumptions**:

| | what DEPTH does | evidence |
|---|---|---|
| **Guaranteed tiers** | "≥ 65% of pots reach a human (95% confidence)" — fit on held-out recordings, verified once on unseen test (**held: 86.2%, LCB 80.5%**). It also says what it *cannot* promise (90% recall; any auto-confirm precision). | [`docs/calibration_exp001.md`](docs/calibration_exp001.md) |
| **Stage 1 measures the sonar** | bottom tracking gives the sonar's altitude per ping → relative object height, slant→ground geotags. Validated with no labels: port vs starboard on the same pings agree to **1.6 px** (random pairs: 5.3 px). | [`docs/stage1_canonical.md`](docs/stage1_canonical.md) |
| **Value-of-information agent** | re-looks only where a tier could change; orders the human queue by calibrated P(pot); **analyst budget** ("what do 10 minutes buy?") and **boat budget** | trace on every card |
| **A physical second look** | plans re-survey passes on the **opposite side** at mid-swath, ranked by uncertainty per metre; each target predicts the bearing its **shadow must flip to** — a test speckle can't pass | map + GPX/GeoJSON |
| **Minutes saved — measured** | a timed, counterbalanced user study built into the app; the effort curve reports the **break-even card time** (7.95 s) instead of an invented "4 h → 10 min" | Study tab · [`docs/effort_curve.md`](docs/effort_curve.md) |
| **Are false alarms false?** | a **blinded audit with catch trials** gives exact audited precision in a confidence band | Audit tab |
| **Humans teach it** | ✓ / ✕ / **＋ missed pot** on any frame → fine-tune set + hard negatives | `python -m src.agentic.feedback export` |
| **COOL, proven by provenance** | the benchmark times the product itself per stage, $/survey-hour, and labels a run COOL only if `cv2` loads from `/opt/cool` | [`infra/README.md`](infra/README.md) |

We also publish what did **not** work: a classical ROI gate (STUDY-01), the acoustic shadow as a
filter (STUDY-03/04), re-look vs plain confidence (STUDY-07), range-gain detector input (STUDY-08).

## 2. Try it (no sonar files needed)

```bash
pip install -r requirements.txt                    # torch-free runtime: OpenCV 5.0.0.93, FastAPI
python -m uvicorn src.dashboard.app:app --port 8000   # open http://localhost:8000
```

Press **▶ 60-s demo** in the top bar for a guided walk through everything below, or explore the four modes (plus a 3D twin view):
* **Analyze** — pick a sample frame → *Run See → Prove → Decide*: the tracked seabed, verdict-coloured
  boxes, an evidence card per find (tier promise, P(pot), shadow, relative height, full agent trace),
  the agent's-eye re-look view; label finds, or draw a **missed pot** (`M`).
* **Survey** — all samples as a background job → hazards on a map with error radii, recovery and
  inspection routes, **re-survey passes**, analyst + boat budgets, GeoJSON/GPX/KML/CSV/JSON.
* **3D twin** (Analyze and Survey toggles) — a physics-grounded digital twin: the seabed in true
  ground-range geometry, the sonar at its tracked altitude, finds with shadow-derived heights and the
  acoustic ray triangle behind each height; the survey laid out along the track with a replay of the
  boat sweeping its sonar fans. Orbit / fly / top cameras, wireframe, backscatter relief.
* **Study** — the timed user study (manual review vs DEPTH cards) + the live effort curve.
* **Audit** — the blinded false-alarm audit.

The 8 sample frames ship with the app (CC-BY-SA, [`webui/samples/ATTRIBUTION.md`](webui/samples/ATTRIBUTION.md)).
The demo GPS is synthetic and labelled as such (the public frames carry none).

## 3. Results (honest — every number names its split)

**Detector — EXP-001 (YOLO11s, 640 px, dataset v1), deploy-faithful `cv2.dnn` evaluation**
([`docs/eval_exp001.md`](docs/eval_exp001.md)):

| | value |
|---|--:|
| mAP@0.5, all classes (test) | 0.822 (0.827 re-measured via `cv2.dnn`) — **inflated** by an optical-only class |
| crab pots on **sonar** — AP@0.5 / recall | **0.473 / 0.599** — the number that matters |
| recall ceiling on unseen recordings (any proposal ≥ 0.05) | 0.72 (Rec19) – 0.86 (Rec3/4/6/10) |

**Agent** — recall promise ≥ 65% held on test; nothing auto-confirmed (no precision promise is
supportable); 1 inference/frame (re-look did not beat confidence on unseen data; the old
test-tuned "CONFIRMED 0.737" is 0.58 unseen — retired).

**Runtime (laptop x86, stock OpenCV 5, full product path):** 238 ms/frame p50 (Stage 1 ≈ 5 ms,
network ≈ 86%); one survey-hour of sonar (~169 frames) processed in **43 s** (real-time factor
0.012). Graviton/COOL numbers: `infra/bench_cool.sh` on EC2 (pending).

**Pending, by design:** EXP-002 (the recall lever — kit ready, [`docs/exp002_kaggle.md`](docs/exp002_kaggle.md)),
the timed study, the audit, the EC2 benchmark.

## 4. How it works

```
frame ─► Stage 1 canonicalise ─► See (YOLO11 · cv2.dnn) ─► Prove (evidence) ─► Decide (guaranteed tiers)
survey ─► stitch · merge repeat sightings ─► routes · budgets · opposite-side re-survey ─► exports
human ─► labels · timed study · blinded audit          offline ─► v2b ─► train ─► onboard (one command)
```
Full design: [`architecture.md`](architecture.md). Agentic writeup: [`docs/agentic_vision.md`](docs/agentic_vision.md).

## 5. Reproduce

```bash
pytest -q                                              # test suite
python -m src.agentic.calibrate                        # fit + verify the guaranteed tiers
python -m src.detection.evaluate                       # deploy-faithful detector evaluation
python -m src.cv_pipeline.study_canonical              # STUDY-08 (Stage 1)
python -m src.agentic.effort                           # analyst-effort curve
python -m src.bench.product_bench --label my_host      # benchmark this machine (product workload)
python -m src.detection.onboard_model --zip EXP-002_complete.zip   # plug in a new model
```
Datasets are git-ignored (large); their builders are in `DATASET/scripts/`
([`docs/dataset_card.md`](docs/dataset_card.md)). Deploy: [`infra/README.md`](infra/README.md).

## 6. Repository

```
src/cv_pipeline/  Stage 1 (canonical.py) + the retired ROI gate (STUDY-01 record)
src/detection/    cv2.dnn detector, calibration, evaluation, training, onboarding, audit
src/agentic/      See→Prove→Decide→Act agent, guarantees, geo, routes, re-survey, labels, study, effort
src/bench/        COOL benchmark (product workload, provenance, comparison)
src/dashboard/    FastAPI service (jobs, limits, metrics)          webui/  zero-build studio
infra/            Graviton + COOL deploy (dry-run by default) + 3-way benchmark
models/<MODEL>/   calibration.json + effort/audit data (weights git-ignored)
docs/             model/dataset cards, calibration, evaluation, Stage-1 study, effort, user study, …
```
Logs: [`experiments.md`](experiments.md) (every study, including negative results) ·
[`progress.md`](progress.md) · [`TODO.md`](TODO.md).

## 7. Responsible use & license

Human approval before any dispatch; low-risk finds kept for audit, never deleted; protected-wreck
coordinates and data retention are covered in [`docs/responsible_use.md`](docs/responsible_use.md).

**AGPL-3.0** ([`LICENSE`](LICENSE)) — builds on Ultralytics YOLO11 (AGPL-3.0). Data sources keep
their own licenses (crab-pot CC-BY-SA-4.0, UATD CC-BY-4.0, …) — see the dataset card.
*Team Syndicate · OpenCV AI Competition 2026.*
