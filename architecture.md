# DEPTH — architecture (as built)

**Related:** [`README.md`](README.md) · [`experiments.md`](experiments.md) · [`progress.md`](progress.md) ·
[`infra/README.md`](infra/README.md) · [`docs/agentic_vision.md`](docs/agentic_vision.md) · proposal of record
[`AGENT.md`](AGENT.md) (the as-built design below supersedes its Lambda/DynamoDB/Amplify/SageMaker plan)

DEPTH turns side-scan sonar frames into a **human-approved cleanup plan** for derelict fishing gear
(ghost pots) and wreck debris. Every find comes with evidence; the agent's tiers carry **calibrated
statistical promises**; every human decision is kept as a training label; and the whole thing runs
torch-free on **OpenCV 5**, CPU-only, on **AWS Graviton with COOL**.

## 1. Principles

1. **Promises, not guesses.** Thresholds are fit on held-out recordings with finite-sample bounds
   (Clopper–Pearson, Learn-Then-Test) and verified once on unseen test frames. If a promise is not
   achievable, the product says so.
2. **Measured, not assumed.** Every headline number names its data split; assumptions (analyst
   timings, ping rate, prices) are labelled and replaced by measurements as soon as they exist
   (the Study tab, the COOL benchmark, the audit).
3. **Human-gated.** Nothing is dispatched automatically; LOW-RISK items are kept for audit, never
   deleted.
4. **Publish what didn't work.** STUDY-01 (classical ROI gate), STUDY-03/04 (shadow as a filter),
   STUDY-07 (re-look vs confidence), STUDY-08 (range-gain input) are negative results kept in the
   record.
5. **One product, one deploy target.** The same FastAPI app runs locally and on the COOL server.

## 2. System at a glance

```
                              ┌──────────── OFFLINE ─────────────────────────────────────────────────┐
 7 raw archives ─► v1 (4 cls) ─► v2b: sonar-only, 2 cls, 1 copy per Roboflow frame, val = held-out
                  recordings, unique-frame test, official-398 + cross-sonar tests          (DATASET/scripts)
                  └► build_tiles.py (full + 2×2 tiles, sonar-aware copy-paste) ─► train.py (Kaggle T4)
                     └► <name>_complete.zip ─► onboard_model.py: verify cv2.dnn ─► models/<name>/
                        ─► calibrate (val recordings) ─► evaluate (test · official 398 · cross-sonar)
                              └──────────────────────────────────────────────────────────────────────┘
 RUNTIME (per frame, CPU, OpenCV 5)                                                  models/<MODEL>/calibration.json
  frame ─► STAGE 1  canonicalise: palette→luminance · orientation by source rule · bottom track
         │          (altitude px / ping) · slant→ground · water-column mask          src/cv_pipeline/canonical.py
         ─► SEE     YOLO11 ONNX via cv2.dnn · letterbox · class-aware NMS             src/detection/infer.py
         ─► PROVE   water_column_check · thin-line shadow · relative height (tracked altitude)
         │          · agent's-eye re-look view (display)                              src/agentic/{shadow,perception,tools}.py
         ─► DECIDE  guaranteed tiers (CONFIRMED / REVIEW / LOW-RISK) · value-of-information tool use
         │          · calibrated P(pot) · full trace                                  src/agentic/{agent,policy,guarantees}.py
         ─► ACT     geotag (ground range + own ping) ─┐                               src/agentic/geo.py
 SURVEY  frames ─► chunk stitching ─► repeat-sighting merge ─► recovery route · inspection route
                   · analyst budget · boat budget ─► opposite-side re-survey passes ─► GeoJSON/GPX/KML/CSV/JSON
                                                            src/agentic/{pipeline,stitch,resurvey,mission}.py
 HUMAN LOOP  labels (✓ / ✕ / ＋missed) ─► fine-tune set     src/agentic/feedback.py
             timed study (manual vs cards) ─► effort curve  src/agentic/{study,effort}.py
             blinded false-alarm audit ─► audited precision src/detection/fp_audit.py
 SERVE   FastAPI (jobs, limits, metrics) + zero-build studio (Analyze · Survey · Study · Audit)   src/dashboard/, webui/
 CLOUD   CloudFront ─► EC2 c8g (Graviton4) · COOL AMI · systemd ─► S3 · CloudWatch · Budgets · SSM   infra/
```

## 3. Stage 1 — sonar canonicalisation (`src/cv_pipeline/canonical.py`)

The classical ROI gate was retired (STUDY-01: it fires more on empty seabed than on debris). Stage 1
now **measures geometry** the later stages need, ~5 ms per 640² frame:

| step | output | used by |
|---|---|---|
| palette → luminance (HSV value for colour palettes) | `gray`, `palette` | shadow, bottom track |
| orientation by **source rule** (PINGMapper `*_ss_*` ⇒ nadir top), user override, else *unknown* — never guessed (the old auto-guess was right 26% of the time) | `orientation` | everything geometric |
| **bottom tracking** — first dip-then-rise after the transducer ring-down, per ping, median+mean along track | `altitude_px`, per-ping line | relative height, ground range, UI overlay |
| slant → ground: `sqrt(slant² − alt²)` (`cv2.remap` view) | `ground_range_px` | geotag |
| range-gain normalisation (detector-input option) | `gain` image | measured: no gain for EXP-001 → detector stays on `raw` |
| water-column mask | `in_water_column` | evidence card |

Validation without labels on 107 un-augmented originals (STUDY-08): port vs starboard on the same
pings agree to a median **1.6 px** (null 5.3 px); consecutive chunks 1.2 px; 97% tracked.

## 4. See — the detector (`src/detection/infer.py`, `calibration.py`)

YOLO11s exported to ONNX and run by **`cv2.dnn`** (OpenCV 5 new engine with fallback), letterbox →
decode → class-aware `NMSBoxesBatched`. Everything model-specific comes from
`models/<MODEL>/calibration.json` (`$DEPTH_MODEL`, default EXP-001): class names, input size,
per-class floors, the guaranteed class, detector input, tiers. Weights resolve from `$DEPTH_ONNX` →
`models/<MODEL>/best.onnx` → `runs/<MODEL>/weights/best.onnx`. Threads: `$DEPTH_THREADS`.

## 5. Prove — evidence per candidate

* **water_column_check** — is the box above the tracked seabed?
* **thin-line acoustic shadow** — PINGMapper shadows are 2–4 px lines; contrast is measured against
  same-size flank windows (selection-unbiased); CLEAR / WEAK / NONE. Evidence for the card, **not a
  gate** (AUC ≈ 0.6 true vs false detections — STUDY-03/06).
* **relative height** h/H = Ls/(R+Ls) against the **tracked** altitude, slant→ground corrected.
  Metres only with a measured altitude in metres (these frames carry no range scale).
* **agent's-eye view** — the exact re-look crop (zoom + CLAHE), display-only.

## 6. Decide — guaranteed tiers + value of information (`src/agentic/`)

`calibrate.py` fits, on the calibration recordings only:
* **τ_review** — the highest threshold whose Clopper–Pearson upper bound on the miss rate stays ≤ α
  (fixed-sequence LTT) → "≥ R% of pots reach a human (95%)"; never-proposed pots count as misses;
* **τ_confirm** — the lowest threshold whose lower bound on precision stays ≥ target → "≥ P% of
  CONFIRMED are real (95%)", or none;
* **P(pot)** bins for queue order and budget forecasts; seven scoring policies compared at the same
  promise, a re-look policy is adopted only with a significant paired-bootstrap gain.

EXP-001: **≥ 65% of pots reach a human, held on test (86.2%, LCB 80.5%)**; 90% not achievable
(recall ceiling 0.72 on the calibration recording); no precision promise → nothing auto-confirmed;
re-look did not beat confidence (1 inference/frame). The agent re-looks only where it could change a
tier (value of information); every call — including the ones it chose not to make — is in the trace.

## 7. Act — survey level (`src/agentic/pipeline.py`, `resurvey.py`, `mission.py`, `geo.py`)

* **geotag** — no GPS ⇒ no coordinates (honest). With a track: each object gets its **own ping**
  (along-track offset from the frame-centre fix) and its **ground** range; error radius shown.
* **chunk stitching** — an object cut by a chunk boundary is one hazard.
* **repeat-sighting merge** — different-frame, same-class detections whose error circles overlap
  (tight gate; closest first; never two detections of one frame) → one hazard with `sightings`.
* **routes** — nearest-neighbour recovery route (CONFIRMED) and inspection route (budgeted REVIEW).
* **analyst budget** — which cards fit N minutes and the expected real pots (Σ P(pot)).
* **opposite-side re-survey** — for each uncertain target, a pass on the far side with the target at
  mid-swath; aligned targets share a pass; passes ranked by Σ p(1−p) per metre; boat-time budget;
  each target predicts the bearing its shadow must flip to. Status always PLANNED.
* **exports** — GeoJSON / GPX / KML / CSV / JSON, synthetic GPS always labelled; every format but CSV
  carries the provenance stamp; `trace` = the agent decision log (JSONL, never public); `?public=1`
  generalises protected-site (wreck) positions.
* **mission brief** (`brief.py`) — a one-page hand-over written from a compact facts JSON *after* every
  decision. Deterministic template always; optional Claude on Amazon Bedrock (`DEPTH_BRIEF_LLM=bedrock`)
  whose text is accepted only if every number / ID is a survey fact and the mandatory caveats are
  present — else the template is served. The LLM never sees coordinates and never changes a decision.

## 8. Human in the loop

| tool | what it measures / produces | where |
|---|---|---|
| labels | ✓ real / ✕ not a pot / **＋ missed pot** → append-only log → YOLO fine-tune set + hard negatives; reviewer agreement vs GT | `feedback.py`, Analyze + Survey |
| timed study | counterbalanced 2×2 Latin square, manual review vs DEPTH cards, scored vs labels → s/frame, s/card, recall both ways | `study.py`, Study tab |
| effort curve | recall vs analyst minutes: manual / detector list / DEPTH queue; promise point; **break-even card time** (7.95 s at 20 s/frame); forecast vs actual (84.5 vs 90) | `effort.py`, Study tab |
| false-alarm audit | blinded, with catch trials; exact audited precision in the conf ≥ 0.176 band; taxonomy; κ | `fp_audit.py`, Audit tab |

## 9. Serving (`src/dashboard/`, `webui/`)

| endpoint | purpose |
|---|---|
| `GET /api/health` | OpenCV version + `cv2.__file__` (COOL provenance), model state, calibration summary, limits, jobs |
| `GET /api/metrics` | rolling per-stage p50/p95 on this host + arch / EC2 type / COOL |
| `POST /api/analyze` | one frame → Stage 1 + evidence cards + traces (+ `frame_ref` for labels) |
| `POST /api/jobs/survey`, `GET /api/jobs/{id}` | background survey jobs (no proxy timeouts) with progress |
| `GET /api/brief?survey_id=` | the mission brief + `writer` (template / llm), `grounding` report, `fallback_reason`, the facts it was written from |
| `GET /api/report/{fmt}` | mission exports (persisted) — **brief** (markdown) · geojson · gpx · kml · csv · json · **trace** (agent decision log, JSONL); `?public=1` generalises protected-site locations; every format but CSV carries the **provenance stamp** |
| `POST /api/feedback`, `GET /api/feedback/stats` | human labels |
| `/api/study/*`, `GET /api/effort` | timed study + effort curves |
| `/api/audit/*` | blinded audit |

Hardening: plain-`def` endpoints (thread pool) with one `cv2.dnn` lock held per frame; images only,
≤ 20 MB, ≤ 50 frames; CORS off unless configured; model warmed at startup; Leaflet vendored
(SRI-checked); keyless basemaps with an offline grid fallback. UI: a zero-build panelled studio with
four modes (Analyze · Survey · Study · Audit).

## 10. Cloud + COOL (`infra/`)

EC2 **c8g.xlarge** from the **COOL AMI** (OpenCV 5 under `/opt/cool`), `systemd` unit running the
app with the COOL interpreter (web deps via `pip --target`; the COOL venv is never modified), behind
**CloudFront** (HTTPS; security group open only to CloudFront's origin-facing prefix list; no SSH —
SSM), **S3** (models, uploads with 7-day expiry, results, benchmarks), **CloudWatch** (status-check
auto-recover + email; JSON frame logs), **Budgets** alarm first. `deploy_aws.sh` is dry-run by
default. The **3-way benchmark** (x86 stock · Graviton stock · Graviton COOL) times the product
workload per stage with pinned threads and reports $/1k frames and $ + real-time factor per
survey-hour; a run is labelled COOL only if `cv2` loads from `/opt/cool`. Local x86 reference:
238 ms/frame p50; one survey-hour in 43 s (RTF 0.012).

Scaling (described, not built): survey jobs → SQS → an Auto Scaling group of the same COOL AMI
workers (Spot) → S3.

## 11. Data + models

| dataset | role |
|---|---|
| v1 (4 classes, 29k images) | trained EXP-001 (the deployed model); its unseen crab-pot sonograms (v1 val Rec19, v1 test) host EXP-001's calibration/verification |
| **v2b** (2 classes, sonar only, deduped) | EXP-002: val = held-out recordings Rec10/12/16; test = 214 unique crab-pot frames; `test_official398` (GhostVision head-to-head); `test_xsonar` (orange Contact crops) |

EXP-002 (1024 px, full + tiles, optional sonar-aware copy-paste) is built and round-trip tested; the
GPU run is on Kaggle ([`docs/exp002_kaggle.md`](docs/exp002_kaggle.md)).

## 12. Code map

```
src/cv_pipeline/  canonical.py (Stage 1) · orientation.py · study_canonical.py (STUDY-08)
                  pipeline.py (retired ROI gate, STUDY-01 record)
src/detection/    infer.py · calibration.py · evaluate.py · export_onnx.py · train.py ·
                  onboard_model.py · fp_audit.py · frames.py · error_analysis.py · tiled_infer.py
src/agentic/      agent.py · perception.py · shadow.py · tools.py · evidence.py · policy.py ·
                  guarantees.py · calibrate.py · geo.py · stitch.py · resurvey.py · mission.py ·
                  pipeline.py · feedback.py · study.py · effort.py · twin.py (3D twin) ·
                  provenance.py · brief.py (mission brief) · types.py
src/bench/        product_bench.py · fingerprint.py · compare.py (COOL benchmark)
src/dashboard/    app.py · jobs.py · metrics.py · samples.py
webui/            index.html · app.js · twin3d.js (three.js twin) · styles.css · samples/ (8 CC-BY-SA
                  frames) · audit/ · vendor/leaflet · vendor/three
infra/            deploy_aws.sh · setup_cool_instance.sh · depth.service · bench_cool.sh
models/<MODEL>/   calibration.json · effort_curve.json · fp_audit_val.json (weights git-ignored)
DATASET/scripts/  audit_dataset · build_dataset_v1/v2/v2b · build_tiles · visualize_labels
```

## 13. Known limits (stated in the product)

* Recall ceiling of EXP-001 (0.72–0.86 by recording) caps the promise at 65% → EXP-002.
* No precision promise yet → every find goes to a human.
* Synthetic GPS in the demo (the HF frames carry none); metres of height need a range scale.
* Minutes saved and label noise are **pending human measurements** (Study, Audit).
* COOL numbers pending the EC2 runs; the local reference is x86.
