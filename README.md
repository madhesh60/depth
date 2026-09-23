# GhostGear Sonar — See · Prove · Decide · Act

> Agentic detection of derelict fishing gear (ghost pots/nets) and wreck debris in **side-scan
> sonar**. Every find is **proven** (acoustic-shadow physics + a zoom-in re-look), **triaged** by a
> human-gated agent, and turned into a **cleanup route** with geotagged reports — through a live web
> dashboard.

**Competition:** OpenCV AI Competition 2026 (Sponsor: AWS · Administrator: OpenCV Foundation)
**Team:** Syndicate (solo — Madhesh) · **Awards targeted:** Agentic Vision · Best Use of COOL
**Deadline:** 2026-10-26, 23:59 PT · **License:** AGPL-3.0 (see [§7](#7-license))

---

## 1. What this is

Manual interpretation of side-scan sonar (SSS) is the bottleneck in marine-debris surveys: a trained
analyst needs 4–8 h per mission, debris blends into seafloor clutter, and calls are subjective. This
is an **end-to-end pipeline** that ingests sonar frames, detects man-made debris, **proves each
detection with physical evidence**, and emits a human-approved cleanup plan + geotagged
GeoJSON/GPX/KML/CSV.

The differentiator is not "YOLO on sonar" (published work already does that) — it is the
**See → Prove → Decide → Act** loop, our Agentic-Vision entry. Full writeup:
[`docs/agentic_vision.md`](docs/agentic_vision.md).

| Stage | Engine | What it does |
|---|---|---|
| **See** | YOLO11 via OpenCV 5 `cv2.dnn` (no torch at inference) | full-frame detection; `fishing_gear` runs hot (0.10) to catch small pots |
| **Prove** | OpenCV 5 (pure) | per candidate: **re-look persistence** (zoom in, re-detect) + acoustic **shadow** where it exists + height estimate |
| **Decide** | deterministic, human-gated agent | an **adaptive controller** selects tools by evidence, triages CONFIRMED/REVIEW/REJECTED, logs every step |
| **Act** | reporting + map | confirmed hazards → nearest-neighbour recovery route + honest geotagged reports |

> **Honest architecture note (a deliberate strength).** We first planned a classical-CV Stage-1 ROI
> *gate* to cut Stage-2 false positives ≥60%. **STUDY-01 disproved it** on this data (classical CV
> fires *more* on empty seafloor than on debris; GT coverage caps ~72% at 141 ROIs/frame), so the
> ROI-gate and the FP-reduction target are **retired**. Likewise **STUDY-03/04** show the acoustic
> shadow is *non-discriminative* here (CLEAR-rate 14.5% on true vs 14.6% on false detections), so it
> is **shown as evidence, never used as a silent gate**. We publish the cues that *didn't* work — the
> discriminators that *do* (re-look persistence + high detector confidence) are calibrated on the real
> test split. OpenCV 5 is used substantively end-to-end: preprocessing → `cv2.dnn` inference → shadow
> physics → overlay rendering.

---

## 2. Live demo (30 seconds, no sonar files needed)

```bash
pip install -r requirements.txt            # torch-free runtime: opencv 5.0.0.93, fastapi, ...
uvicorn src.dashboard.app:app --port 8000  # open http://localhost:8000
```

Then: click a **sample frame** → **Run See → Prove → Decide** (watch the stepper, the evidence cards,
and the agent's tool-call trace) → open the **Run a survey** tab for the map, recovery route, and
GeoJSON/GPX/KML/CSV downloads. The footer shows the live OpenCV version; synthetic-GPS demos are
clearly labelled.

---

## 3. Detection classes & the honest-naming fix

The **deployed baseline model (EXP-001)** is 4-class. During the dataset audit we found the class
*names* over-claimed what the data is, so **dataset v2 renames to match reality** and drops
off-domain data (`build_dataset_v2.py`, trains EXP-002):

| EXP-001 (4-class, deployed) | What the data actually is | v2 (honest, sonar-only) |
|---|---|---|
| `fishing_gear` | crab-pots in side-scan sonar (+ a few optical net photos) | **`ghost_gear`** |
| `structural_fragment` | shipwreck + UATD placed test objects | **`wreck_debris`** |
| `pipe_cylinder` | UATD cylinders — *forward-looking* sonar test objects | held out (different sensor) |
| `natural_formation` | **ICRA19 "bio": fish/plants in optical camera photos** (a control class — *not* rock clusters) | dropped (off-domain) |

v2 also **fixes three label bugs** (955 real pipeline/platform frames mislabelled "empty seabed"
recovered/dropped; 1,547 empty crab-pot frames restored as negatives) and **locks the official
crab-pot 398-frame test split** for a head-to-head vs the published GhostVision baseline. See
[`docs/dataset_card.md`](docs/dataset_card.md).

---

## 4. Results (honest — read per class, not just the aggregate)

| Metric (EXP-001, test) | Target | Value |
|---|---|---|
| mAP@0.5 (aggregate) | ≥ 0.70 | **0.822** ✅ *(inflated — see below)* |
| Precision / Recall | 0.80 / 0.70 | 0.808 / 0.800 |
| Latency / throughput (T4) | <300 ms / ≥5 FPS | ~11.5 ms / ~87 FPS |

> The aggregate is **inflated by domain segregation** (`natural_formation` is optical-only, R 0.99).
> The real target — `fishing_gear` (ghost gear) on **sonar** — is recall ~0.47 @conf 0.25, lifted to
> ~0.69 by per-class thresholds; EXP-002 (higher resolution) targets the rest.

**Agent value (STUDY-04, real crab-pot test split, `python -m src.agentic.calibrate`):** the raw hot
detector is precision 0.60; the agent's **CONFIRMED** tier (re-look persistence ⋃ high detector
confidence) is **precision 0.737 at 30% recall-share** — better than either signal alone — while
REJECTED stays recall-safe (91% of true pots retained, never deleted).

---

## 5. Quickstart (train / infer / evaluate)

```bash
# Train Stage 2 (GPU). EXP-001 done; EXP-002 trains on the honest v2 split (docs/exp002_kaggle.md).
python src/detection/train.py --model yolo11s.pt --imgsz 1024 --batch 8 --epochs 30 --name EXP-002

# Export the trained weights to ONNX + verify the cv2.dnn load path (the deploy path)
python src/detection/export_onnx.py runs/EXP-002/weights/best.pt

# Detect (cv2.dnn — no torch), run the agent, run a full survey
python -m src.detection.infer <image_or_dir>
python -m src.agentic.agent <image_or_dir> --out runs/agent
python -m src.agentic.pipeline --survey <dir> --gps synthetic --out runs/survey

# Calibrate the agent on the test split (STUDY-03/04 numbers) and run the tests
python -m src.agentic.calibrate --frames 160 --out runs/prove
pytest -q
```

---

## 6. Repository layout

```
src/
  detection/      # YOLO train / infer (cv2.dnn) / export_onnx / error_analysis / tiled_infer
  cv_pipeline/    # Stage-1 classical CV — the CPU workload benchmarked for the COOL award
  agentic/        # See→Prove→Decide→Act: perception, shadow, evidence, policy, agent, tools,
                  #   geo, mission, pipeline, calibrate  (the Agentic-Vision entry)
  dashboard/      # FastAPI backend exposing the loop
webui/            # zero-build static dashboard (served by FastAPI)
infra/            # AWS: lambda_handler.py (cv2.dnn, honest geotag), benchmark_graviton.sh (COOL)
tests/            # 30 tests (pytest)
docs/             # agentic_vision, dataset_card, model_card, responsible_use, dataset_report, ...
DATASET/          # git-ignored (data lives in S3); scripts/ tracked for reproducibility
runs/             # git-ignored — weights, benchmark JSON, agent overlays
```

Key docs: [`architecture.md`](architecture.md) · [`experiments.md`](experiments.md) ·
[`progress.md`](progress.md) · [`docs/agentic_vision.md`](docs/agentic_vision.md) ·
[`docs/model_card.md`](docs/model_card.md) · [`docs/responsible_use.md`](docs/responsible_use.md) ·
[`infra/README.md`](infra/README.md) (AWS + the 3-way COOL benchmark).

---

## 7. License

**AGPL-3.0** — see [`LICENSE`](LICENSE). This project builds on Ultralytics YOLO11 (AGPL-3.0); the
network-use clause applies, so the full source is public. © 2026 Team Syndicate.

Data sources carry their own licenses (CC-BY-SA-4.0 crab-pot, CC-BY-4.0 UATD, JAMSTEC-derived
TrashCan/ICRA19, etc.) — see [`docs/dataset_card.md`](docs/dataset_card.md). Responsible-use notes
(human approval before dispatch, protected-wreck coordinates, data retention) are in
[`docs/responsible_use.md`](docs/responsible_use.md).

*Team Syndicate · OpenCV AI Competition 2026.*
