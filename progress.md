# Progress Log — Marine Debris Detection System

**Team Syndicate · OpenCV AI Competition 2026**
This is the living status of the project. Update it at the end of every working session:
what moved, what's blocked, what's next. Newest entries at the top of §4.

**Related:** [`README.md`](README.md) · [`architecture.md`](architecture.md) ·
[`experiments.md`](experiments.md) · [`TODO.md`](TODO.md)

---

## 1. Timeline & hard dates

| Date | Milestone |
|---|---|
| 2026-08-13 | Proposal submission window opened |
| 2026-08-26 | Build phase begins |
| **2026-09-18** | **Today — foundation / planning** |
| 2026-09-21 → 10-02 | **Grant check-in** (30-min Zoom, required to unlock 2nd 50% of grant) |
| 2026-10-26 23:59 PT | **Final submission deadline** |
| 2026-10-27 → 11-09 | Judging |
| 2026-11-10 | Winners announced (OpenCV Live!) |

> **~5.5 weeks of build time remain.** The grant check-in is imminent — a concise,
> credible progress story is needed for it (see §5).

---

## 2. Milestone tracker

| Milestone | Status | Notes |
|---|---|---|
| Grant awarded ($150) | ✅ Done | Secured. 2nd half unlocked at check-in. |
| Proposal (AGENT.md) | ✅ Done | Source of record. |
| Dataset built (v0, 32,981 imgs) | ✅ Done | 7 sources remapped to 5 classes; split 30444/1263/1274. |
| Dataset audit | ✅ Done | Issues catalogued in `docs/dataset_report.md`. |
| Project docs (this set) | ✅ Done | README / architecture / progress / experiments / TODO. |
| Dataset fixes (v1) | ✅ Done | Clean split built: 0 leakage, full-frame boxes removed, 1,099 background negatives, 4-class taxonomy locked. `03_yolo_ready_dataset_v1/`. |
| **Dataset v2 (honest, sonar-only)** | ✅ Done | `build_dataset_v2.py`: 2-class (`ghost_gear`+`wreck_debris`), fixes 3 WINNING_REPORT data bugs (mpulse full-frame mislabel dropped, 1,547 empty crab-pot frames recovered, optical dropped, UATD held out). train 6,291 / val 626 / test 469, leakage 0/0/0, official crab-pot 398-test LOCKED. **EXP-002 trains on this.** |
| OpenCV 5 (rule 1) | ✅ Done | Env on **5.0.0.93** (was 4.12); requirements split + pinned exactly; 27/27 tests pass; `/api/health` reports it. |
| Stage 1 classical CV | ✅ Done (reframed twice) | **2026-09-24: in the product path as sonar canonicalisation** (`canonical.py`, STUDY-08: bottom-tracked altitude validated port-vs-starboard 1.6 px; ground range → geotag; ~5 ms/frame). Earlier: **Reframed** (STUDY-01): not a Stage-2 ROI gate (no discriminative power on this sonar) — now the CPU **sonar-preprocessing workload** for the COOL benchmark. `src/cv_pipeline/` + 3-way harness. Local x86: 29 ms/frame, 30.5 FPS. Needs Graviton+COOL run. |
| Stage 1→2 wiring (cv2.dnn) | ✅ Done | `infer.py` (ONNX via `cv2.dnn`, full_frame + roi_guided) + `ablation_fp.py` + `tune_coverage.py`. ONNX verified loading/forward. |
| Stage 2 baseline train | ✅ Done | EXP-001 (yolo11s detect, v1, Kaggle T4): test mAP@0.5 **0.822**, P 0.808, R 0.800 — all targets met. |
| Evaluation + ablation | ✅ Done (EXP-001) | `evaluate.py` (deploy-faithful cv2.dnn): val-tuned thresholds/test-once, per-sensor & per-source tables, bootstrap CIs → `docs/eval_exp001.md` (STUDY-05). Aggregate 0.827 reproduces 0.822; honest sonar `fishing_gear` AP 0.473. GhostVision head-to-head pending EXP-002. |
| Reporting engine | ✅ Done | `src/agentic/{geo,mission}.py`: honest geotag + GeoJSON/GPX/KML/CSV/JSON exports. |
| Dashboard | ✅ Done | Zero-build static web app (`webui/`), served by FastAPI; hardened (job queue, per-frame model lock, upload limits, shipped samples, vendored Leaflet). See→Prove→Decide→Act stepper, evidence cards (conf→relook), agent trace, Leaflet hazard map + recovery route, downloads, honesty banners. Verified under uvicorn. |
| AWS deployment | 🟡 Scripted, not run | `infra/deploy_aws.sh` (dry-run default) + `setup_cool_instance.sh` + `depth.service`; EC2 c8g COOL AMI + CloudFront + S3 + alarms. Scheduled for the AWS day. |
| **COOL benchmark (Arm vs x86)** | 🟡 Tooling done, EC2 runs pending | Product-workload benchmark (`src/bench/`, `infra/bench_cool.sh`): per stage, threads pinned, $/survey-hour, COOL provenance. Local x86 reference: 238 ms/frame p50, RTF 0.012. |
| Agentic loop (See→Prove→Decide→Act) | ✅ Done | **Primary Agentic-Vision entry.** All 4 stages + an **adaptive escalation controller** (STUDY-04) + live dashboard. Two calibrated CONFIRMED paths (re-look ⋃ high-confidence → precision **0.737 @ 30% recall**), cross-pass corroboration, 6-tool toolbox, full audit trace, human-gated. `src/agentic/` + `webui/`; 30/30 tests. Remaining is *deployment* (AWS/COOL), not the brain. |
| Submission package (report + video) | ⬜ Not started | Due 2026-10-26. |

Legend: ✅ done · 🟡 in progress · ⬜ not started · ⛔ blocked

---

## 3. Current state (2026-09-18)

**What exists**
- Grant secured; proposal finalised (`AGENT.md`).
- Dataset v0 built and audited: `DATASET/03_yolo_ready_dataset/data.yaml` is the training
  entry point. Build scripts frozen in `DATASET/archive_scripts/` (idempotent).
- Pinned dependencies (`requirements.txt`).
- Full project documentation set created (this commit).

**What does not exist yet**
- No `src/`, `infra/`, or `tests/` code — greenfield.
- Dataset v0 has **known blockers** (see §6) that must be fixed before the first real train.
- AWS CLI not installed; no cloud resources provisioned; COOL not yet exercised.

**Environment**
- Global Python 3.10 (no venv). `torch`, `torchvision`, `opencv-python`, `numpy`,
  `PyYAML`, `Pillow` present. `pip install -r requirements.txt` adds the rest.

---

## 4. Session log

### 2026-09-26 (sweep 23) — Technical report draft (required submission item)
- `docs/technical_report.md`: the rules' sections (problem, users, architecture, OpenCV 5
  implementation, AWS deployment, evaluation, limitations, responsible use) + what did not work +
  reproduce. Every number links the script-generated report it comes from; unmeasured items are
  marked ⏳ pending and left empty (COOL on EC2, EXP-002, study, audit, live URL). The OpenCV-ops
  table was checked against the code (`grep cv2.` per module). TODO.md updated.

### 2026-09-26 (sweep 22) — Architecture diagram (required submission item)
- `docs/img/architecture.svg`: OpenCV 5 stages on EC2 Graviton + COOL, CloudFront, S3, CloudWatch,
  Budgets/SSM/IAM, optional Bedrock, the COOL benchmark, the offline data → training → calibration
  path, the human loop and what leaves the machine. AWS parts drawn dashed and labelled "scripted,
  deployed on the AWS day" (honest status). Embedded in README + `architecture.md` §2; every label
  checked to fit its box in a browser render (101 text elements, 0 overflows).

### 2026-09-26 (sweep 21) — STUDY-12 made visible: "⊘ without Stage 1" on the survey map
- `geo.stage1_counterfactual()` re-geotags every hazard of the current survey with Stage-1 geometry
  removed (slant range + frame-centre ping) — no re-inference; `/api/survey` + survey jobs return
  `stage1_counterfactual` (sample survey: **31 of 46** pins leave their own error circle, median
  6.6 m — identical to STUDY-12's offline arm).
- UI: a toggle beside Map | 3D twin draws ghost pins linked to the measured ones (pink = outside
  its circle) with the summary line; the 60-s demo shows it for five seconds with the live numbers.
- Tests +1 unit, API check; suite 113 passed. Browser-verified (92 overlay layers, clean toggle).

### 2026-09-26 (sweep 20) — STUDY-12: the counterfactual trace (OpenCV output → the agent's actions)
- **`src/agentic/study_causal.py`** → `docs/causal_trace.md`: the rules ask for "a trace showing
  OpenCV 5 output changing a later decision/action"; a counterfactual shows *cause*. The same survey
  runs as full / Stage-1 geometry withheld / shadow withheld and every downstream decision is diffed.
- **Result (v1 test, 80 unique frames, 264 hazards):** without Stage 1 — 0 tier flips (never a gate)
  but the median pin moves 8.6 m, **193 of 264 pins land outside their own stated error circle**, 13
  of 63 re-survey passes regroup, 2 inspection stops move. Without the shadow — **no decision
  changes** (evidence only, by design). Plus one hazard's end-to-end chain (bottom track → cv2.dnn →
  tier → P(pot) → queue rank → budget → inspection stop → re-survey pass + predicted shadow flip).
- Caught before reading numbers: the first run's single synthetic line made different recordings
  share fixes (11 spurious merges); the study now lays recordings on parallel lines.
- STUDY-11b (seam inference, negative) logged in `experiments.md` (was only in its report).
- Tests +3 (`test_causal.py`).

### 2026-09-26 (sweep 19) — Mission brief: an LLM that writes, never decides
- **`src/agentic/brief.py`**: a one-page hand-over for the crew / manager (bottom line, the promise,
  first hour of review, routes, re-survey passes with their shadow-flip prediction, caveats). The
  **template** writer is deterministic and ships in every survey's report set (`mission.brief.md`,
  public variant too). The optional **Claude on Amazon Bedrock** writer (`DEPTH_BRIEF_LLM=bedrock`,
  `AnthropicBedrockMantle`, default `anthropic.claude-opus-5`) sees only a compact facts JSON (no
  coordinates) built **after** every decision is made.
- **Grounding check** gates the LLM text: every number must be a survey fact (as given, rounded, or a
  0–1 share as a percentage; number words count too), every hazard / pass ID must exist, the mandatory
  caveats (human approval · synthetic GPS · assumed card time) must be present, ≤ 350 words. Any
  failure — SDK missing, no credentials, API error, refusal, truncation, an ungrounded number — serves
  the template, and the response says which writer ran and why. Limitation stated in the module: the
  check proves each number came from the survey, not that it sits next to the right noun.
- The template passes its own check (sample survey: 33 numbers traced). `GET /api/brief`,
  `/api/report/brief`; UI **Mission brief** panel (writer tag, "✓ N numbers traced", copy) +
  agent-log `write_brief` line; the 60-s demo ends on it.
- Fix: the survey map could open at world zoom — `fitBounds` ran before the container had a size
  (mode just switched / pane hidden) and Leaflet cached the 0-px size. Now the fit reads the element
  and a `ResizeObserver` re-fits when the view is lost (a user's own zoom is kept). Browser-verified.
- Tests +7 (`test_brief.py`, fake Bedrock client — no network) + API brief checks; suite 109 passed.

### 2026-09-26 (sweep 18) — "▶ 60-s demo": the judge path, guided
- One click in the top bar drives the whole loop with a narration bar (7 steps, ~60 s): analyze a
  frame (Stage-1 seabed, evidence, calibrated tier) → 3D twin with the acoustic triangle of a find
  with a measured shadow → survey job (map, budgets, re-survey passes) → 3D replay → measured effort →
  provenance + decision log + human gate. Any key or click stops it. Browser-verified end to end
  (7/7 steps, no console errors).

### 2026-09-25 (sweep 16) — Failure gallery with measured failure modes (STUDY-11a)
- `src/detection/failure_gallery.py` → `docs/failure_gallery.md` + two contact sheets (12 misses
  across sizes, 12 most confident false alarms). **Finding:** 32% of missed pots touch the frame edge
  (11% of found) — chunk-boundary cuts are the largest failure mode; misses are lower-contrast and
  farther in range, and NOT smaller. The report's reading is generated from the numbers.

### 2026-09-25 (sweep 15) — Provenance stamps, agent decision log, protected-site redaction
- **`src/agentic/provenance.py`**: every analyze result and every report carries a stamp — model +
  ONNX sha256, calibration sha256 (the promises in force), OpenCV version + `cv2` path (`/opt/cool`
  ⇒ COOL), code commit, host / EC2 type, UTC time. GeoJSON properties, GPX/KML descriptions, JSON;
  CSV stays a clean table. UI: provenance line in the dock.
- **Agent decision log** (`/api/report/trace`, `mission.trace.jsonl`): provenance → survey → every
  frame's Stage-1 record → every tool call (rationale, latency, conf before/after, detail) → every
  verdict → mission decisions (budget, routes, re-survey plan, human gate). Sample survey: 8 frames,
  155 tool calls, 48 verdicts.
- **Responsible use:** `?public=1` exports generalise protected-site (wreck) positions to ~1.1 km and
  drop re-survey passes that would point at them; ghost-gear targets keep full precision; the decision
  log is **never** public (403). UI "public share" toggle on the Reports panel.
- Pipeline dock now shows five stages: **Canonicalise** → See → Prove → Decide → Act.
- Tests +3 (`test_provenance.py`). Browser-verified.

### 2026-09-25 (sweep 14) — Physics-grounded 3D digital twin (three.js, DepthWizard-class viewer)
- **`src/agentic/twin.py`** builds two scenes from MEASURED geometry only: the **frame twin** (seabed =
  Stage-1 slant→ground remap, sonar path = tracked altitude per ping, finds at their ping / ground
  range, heights = shadow h/H × tracked altitude — only for CLEAR/WEAK shadows) and the **survey
  twin** (each frame's swath ribbon on its side of the track, hazards at their geotags with error
  rings, recovery / inspection routes, opposite-side re-survey passes, the track). Frames without
  tracked geometry are not drawn (reason shown).
- **`webui/twin3d.js`** (three.js r160 vendored, MIT, import map — still zero-build): orbit · fly
  (WASD/QE) · top cameras, textured / wireframe, **backscatter relief** (labelled "not bathymetry";
  markers ride the relief), vertical exaggeration, PNG snapshot. Clicking a find draws its
  **acoustic ray triangle** (sonar → object top → end of shadow) with h, H and the shadow length;
  cards ⇄ 3D selection is linked. **Replay**: the boat sweeps its port/starboard sonar fans along the
  track and finds appear as it passes them.
- UI: Analyze toolbar **2D | 3D twin**, Survey **Map | 3D twin**; the 2D gaze tour is suppressed in
  3D; card selection now scrolls only the inspector (scrollIntoView scrolled the page).
- Tests +4 (`test_twin.py`) incl. the invariant **every hazard's geotag lies on its own frame's
  swath ribbon** and "height ⇔ real shadow". Browser-verified at 1440×900.

### 2026-09-25 (review sweep 13) — Documentation truth pass (I-10 / X-5)
- **`architecture.md` rewritten as built** (was: ROI gate, Lambda/DynamoDB/Amplify/SageMaker): Stage 1
  canonicalisation → See → Prove → Decide (guaranteed tiers) → Act (geotag, stitching, merge, routes,
  re-survey) → human loop (labels, study, audit) → serving → Graviton/COOL → data/models → code map →
  stated limits.
- **README** (judge-facing): what is different + where the evidence is, the four studio modes,
  honest results with splits named, what is pending by design, reproduce commands.
- **CLAUDE.md** brought to the as-built state (it said "nothing under src/", "AWS CLI not
  installed", the Lambda plan); git-workflow section kept verbatim. **TODO.md** is now the real
  backlog: what needs a person (grant check-in by 2 Oct, Kaggle EXP-002, AWS day, study, audit),
  what follows, the submission package.
- `docs/model_card.md`: retired "CONFIRMED 0.737" and the GPU latency → the unseen-data promise and
  the measured CPU path; `AGENT.md` banner points at the as-built design;
  `docs/agentic_vision.md` gains §6 (Stage-1 measurement, physical second look, labels, measured
  effort, audit) and an updated rubric map.
- Full suite: **95 tests pass**.

### 2026-09-25 (review sweep 12) — Blinded false-alarm audit with catch trials (M-4, STUDY-10 set up)
- **`src/detection/fp_audit.py`**: are EXP-001's "false alarms" really false? **Exact band, not a
  sample** — every false alarm with conf ≥ 0.176 on the calibration recordings (60 of them; 66 true
  pots in the band; raw precision 0.52) plus **15 catch trials** (known labelled pots), shuffled,
  drawn identically (**blind**: no confidence, no label shown). Tags: real object · clutter · noise
  · unsure. Summary: FP taxonomy (majority across auditors), share of real objects with a Wilson CI,
  **audited precision** (exact once the band is fully tagged), catch accuracy per auditor (a result
  only counts if the auditor recognised the known pots), Cohen's κ between auditors.
- Crops ship in `webui/audit/` (CC-BY-SA); the answer key lives in `models/EXP-001/fp_audit_val.json`
  — outside `webui/`, verified not served (404).
- **Audit** tab: keyboard 1–4, back, resume; live result panel. API `/api/audit/items|tag|summary`.
- QA note (not a result): many of the most confident "false alarms" look like bright returns with a
  shadow tail — exactly the hypothesis the audit tests; the tagging is for humans, not the AI.
- Tests +4 (`test_fp_audit.py`). Browser-verified; test tag deleted.

### 2026-09-25 (review sweep 11) — Physical second look: opposite-side re-survey + repeat-sighting merge (A-1, X-3)
- **`src/agentic/resurvey.py` → `plan_resurvey`**: for every uncertain (REVIEW) target the agent
  plans the pass that can settle it — a straight line on the **opposite side** with the target at
  **mid-swath** — groups targets whose lines align into one pass, and ranks passes by **uncertainty
  resolved per metre of boat travel** (Σ p(1−p), p = calibrated P(pot)). A **boat-time budget**
  keeps the densest passes (greedy knapsack) — the boat-side twin of the analyst budget. Each target
  carries a **falsifiable prediction**: its shadow pointed θ, on the new pass it must point θ+180°
  (a standing object's shadow flips; speckle doesn't). Sample survey, 8-min budget: 3 passes,
  6.3 min, 4.5 of 9.6 units of uncertainty.
- **`merge_repeat_sightings`**: different-frame detections of one class merge when their geotag
  error circles overlap (≤ max(3 m, ½√(e₁²+e₂²)) — tight: pot strings are 10–30 m apart), closest
  pairs first; **found + fixed** a transitive-fusion bug (two distinct detections in ONE frame could
  merge through a third sighting) — clusters never hold two detections of the same frame.
  `sightings` per hazard; tiers/scores untouched. Straight synthetic track ⇒ 0 merges (honest).
- Exports: GeoJSON LineStrings (`kind: resurvey_plan`, status PLANNED), GPX routes, KML lines.
  UI: violet dashed passes with heading arrows + popups (targets, uncertainty, shadow predictions),
  banner + agent-log lines, "re-survey boat time" input, `×N` sightings in the hazard table.
- Tests +4 (`test_resurvey.py`: opposite side at 16 m, predicted shadow bearing flip, grouping,
  budget, no-GPS). Browser-verified.

### 2026-09-25 (review sweep 10) — Timed user study mode + analyst-effort curve (X-2, STUDY-09)
- **Found by computing first:** with the ASSUMED timings (20 s/frame manual, 8 s/card) DEPTH only
  *matches* a perfect human at the recall promise (19.3 vs 19.5 min on the 92 verification frames)
  — the minutes-saved claim rests entirely on two stopwatch numbers nobody had measured. So the
  headline is now a **break-even card time (7.95 s)**: faster card review than that and DEPTH wins.
- **`src/agentic/effort.py`**: recall-vs-minutes on the verification split for manual review,
  a detector list (survey order) and the DEPTH queue (P(pot) order) + the promise point, the
  break-even, per-survey-hour minutes, and **forecast vs actual** (the agent forecast 84.5 real pots
  at the promise point; 90 were real). Writes `models/<MODEL>/effort_curve.json` (UI replays it),
  `docs/effort_curve.md` + chart.
- **`src/agentic/study.py` + Study mode:** counterbalanced **2×2 Latin square** (frame halves ×
  task order; nobody sees a frame twice), manual arm (click every pot, timed per frame) and cards
  arm (the agent's queue, Y/N, timed per card), server-side scoring vs labels (clicks within a box
  +8 px; cards IoU ≥ 0.3; DEPTH recall counts pots in frames that produced no card — fixed a bug
  that silently excluded them). Pooled summary → the effort panel switches from "assumed" to
  "measured"; live what-if sliders; `$DEPTH_STUDY_DIR` for a bigger frame set.
- API: `/api/study/plan|frame|result|summary`, `/api/effort`. Protocol: `docs/user_study.md`.
- Tests +4 (`test_study.py`). Browser-verified the full session flow; test session deleted.

### 2026-09-25 (review sweep 9) — Every human decision becomes a training label (A-2)
- **`src/agentic/feedback.py`**: append-only `runs/feedback/decisions.jsonl` (confirm / reject /
  **missed**), latest decision wins per object (same frame, IoU ≥ 0.7); uploaded frames persisted
  only when labelled (sha256); `export` → a YOLO fine-tune set (positives = confirmed + missed) +
  `hard_negatives.json` (explicit rejections), frames flagged `partial` (only reviewed objects are
  labelled); `agreement()` scores reviewers against ground truth where it exists (the shipped
  samples) — reviewer precision is the ceiling on what the human loop can deliver.
- **API:** `POST /api/feedback`, `GET /api/feedback/stats`; analyze/survey payloads carry a
  `frame_ref` (sample id or upload sha) so any label can find its image again.
- **UI:** "✓ real pot / ✕ not a pot" on every evidence card; survey hazard-table ✓/✕ now save real
  labels; **"＋ missed pot" (M)** — drag a box around a pot the detector never proposed → positive
  label (fixes *recall*, which is EXP-001's binding limit); live "human labels" counter.
- `python -m src.agentic.feedback stats | export --out DATASET/feedback_v1`. Tests +4
  (`test_feedback.py`). Browser-verified (card label, missed-pot box, counter); test labels deleted.

### 2026-09-25 (review sweep 8) — EXP-002 kit: train on Kaggle, plug in with ONE command
Review I-3 / C-5 (the recall ceiling is the binding constraint).
- **Three silent failures found and fixed before they cost a GPU run:** v2b `data.yaml` was cp1252
  (em-dash byte 0x97; ultralytics reads UTF-8) and had `path: .`, which ultralytics resolves against
  the *working directory* (verified: it looked for `MARINE_DEBRIS/val/images`) — builder fixed,
  file rewritten, `check_det_dataset` now resolves; the review's `copy_paste 0.3` is a **no-op** for
  box-only labels (ultralytics `CopyPaste` returns early without masks); and `evaluate.py`'s
  GhostVision head-to-head matched **0 of 398** official frames (`Path.stem` ate the Roboflow hash)
  — now 398/398.
- **`build_tiles.py`**: full frames + 2×2 overlapping tiles (boxes clipped, kept if ≥ 60% visible);
  val/test stay full-frame → v2b 1,773 frames + 3,973 tiles. **`--paste N` sonar-aware copy-paste**:
  real pots cut out *with their shadow tails*, pasted at the **same slant range** below the Stage-1
  tracked seabed on empty same-sonar frames, Poisson-blended (`cv2.seamlessClone`), with a
  visibility gate (bank 1,018 pots · 230 backgrounds). Visually QA'd.
- **`train.py`** (EXP-002 recipe): v2b(+tiles), 1024 px, 30 ep, patience 8, cosine LR, sonar-aware
  aug; after training → ONNX → **`cv2.dnn` forward verified** → `model_meta.json` → one zip.
- **`onboard_model.py`**: zip → sha256 + `cv2.dnn` check → `models/<name>/` registry → calibrate the
  guaranteed tiers on v2b val recordings (verified once on test) → evaluate on v2b test, official
  398 (GhostVision) and cross-sonar → `docs/onboard_<name>.md` vs EXP-001, with a **do-not-switch
  gate** (calibration failed / recall promise < 0.3 / below EXP-001).
- **Round-trip verified on CPU:** a 1-epoch smoke model went train → export → verify → zip → onboard
  → agent running it (2 classes, 320 px); the gate correctly refused it. Found + fixed a crash in
  `calibrate.auc_gain_ci` when no resample has both TPs and FPs.
- **Runtime is model-agnostic:** class names, guaranteed class, input size and ONNX path come from
  `models/<MODEL>/` (`DEPTH_MODEL`); the COOL server kit takes `DEPTH_MODEL` too.
- `docs/exp002_kaggle.md` rewritten as copy-paste notebook cells. Tests: +5 (`test_training_kit.py`).

### 2026-09-24 (review sweep 7) — COOL benchmark v3 (product workload) + Graviton/COOL deploy kit
Review M-5 / I-1 prep / X-6 / §3.7 (not yet executed on AWS — scheduled for the AWS day).
- **`src/bench/product_bench.py`** times what `/api/analyze` runs, per stage (decode → Stage 1 →
  `cv2.dnn` → evidence/tiers → render) + a `cv` workload (OpenCV image ops only, where COOL/KleidiCV
  acts); threads pinned; **$/1k frames** and **per survey-hour** compute, real-time factor and $.
  `fingerprint.py` records arch / vCPUs / EC2 type / AMI / `cv2.__file__` (`/opt/cool` ⇒ COOL) /
  model + calibration sha256 / git commit. `compare.py` → `docs/cool_benchmark.md` + chart with the
  (B→C) COOL-on-identical-Graviton and (A→C) x86→Graviton+COOL rows.
- **Local reference (laptop x86, stock OpenCV 5):** product **p50 238 ms/frame** (target < 300),
  one thread 522 ms; the network is ~86% of the time; OpenCV-only share 22 ms; **one survey-hour of
  sonar (≈169 frames) → 43 s compute, real-time factor 0.012**.
- **`infra/bench_cool.sh`** replaces `benchmark_graviton.sh`: product workload; stock runs in a private
  venv (Ubuntu 24.04 / PEP 668); **nothing is ever installed into `/opt/cool/venvs`**; refuses to
  label a run COOL unless cv2 loads from `/opt/cool`; checks the build has `dnn`.
- **Deploy kit:** `setup_cool_instance.sh` (repo + sha256-checked model + web deps via `pip --target`
  + systemd; waits for `model_loaded` and `is_cool_path`), `depth.service`, `deploy_aws.sh`
  (**dry-run by default**: budget alarm → private S3 w/ 7-day uploads → scoped IAM → SG open to
  CloudFront origin IPs only, no SSH → EC2 c8g COOL AMI, IMDSv2 → CloudFront HTTPS → status-check
  auto-recover + email). The pinned web stack was verified in a clean venv holding only numpy +
  OpenCV (model warm 0.3 s, analyze OK).
- **Retired** `infra/lambda_handler.py` (second, detection-only product with duplicated thresholds;
  Lambda can't use COOL) and the old Stage-1 `benchmark_graviton.sh`.
- **Live runtime metrics:** `/api/metrics` (rolling per-stage p50/p95 on this host + arch / EC2 type /
  COOL provenance); JSON frame log lines for CloudWatch (`DEPTH_LOG_JSON=1`); UI model registry shows
  host + live p50.
- Tests: +4 (`tests/test_bench.py`) + metrics API test.

### 2026-09-24 (review sweep 6) — Stage 1 in the product path: sonar canonicalisation (STUDY-08)
Review I-4 / X-1 / part of X-3.
- **`src/cv_pipeline/canonical.py`** — palette → luminance, orientation by rule, **bottom tracking**
  (sonar altitude in px, per ping), slant → ground `cv2.remap`, range-gain normalisation,
  water-column mask; every op timed for the COOL benchmark. ~5 ms/frame.
- **Validated without labels** on 107 un-augmented originals: port vs starboard on the same pings
  agree to a median **1.6 px** (null 5.3 px); consecutive chunks 1.2 px; 97% tracked.
- **Measured, not assumed:** range-gain input does not help EXP-001 (Δ AP −0.006, CI spans 0) →
  detector stays on `raw` (`calibration.json detector.input`); no pot/candidate in the water column.
- **Agent:** Stage 1 runs first in `ReLookAgent.run_frame`; shadow relative height uses the tracked
  altitude at the object's ping; new traced tool `water_column_check`; `FrameResult.stage1` record.
- **Act:** geotag uses the Stage-1 **ground range** and each object's **own ping** (along-track
  offset from the frame-centre fix) — objects in one frame no longer share a boat position; the
  synthetic track now tiles chunks (`frame_len_m` = 640 px × 0.05 m).
- **UI:** tracked seabed drawn on the sonar viewer, Stage-1 line first in the agent log, card fact
  "on seabed / water column · ground px"; model pill re-polls until warm (was stuck at "lazy") and
  shows "· COOL" when `cv2` loads from `/opt/cool`.
- Tests: +8 (`tests/test_canonical.py`, synthetic sonograms with known altitude).

### 2026-09-24 (review sweep 5) — Backend hardening: job queue, limits, shipped samples, offline-safe map
Review C-7 / I-9 + failure scenarios 10–13.
- **Non-blocking server:** endpoints are plain `def` (thread pool); one `threading.Lock` guards the
  shared `cv2.dnn` net and is held **per frame** (`run_survey(frame_lock=…)`), not per survey.
  Measured on the laptop while an 8-frame survey job ran: `/api/health` p50 **18 ms** (46 polls);
  `/api/analyze` **2.6 s** vs 2.9 s standalone (before: it waited for the whole survey).
- **Survey jobs** (`src/dashboard/jobs.py`): `POST /api/jobs/survey` → job id, `GET /api/jobs/{id}`
  → progress + result. Bounded (50 records, finished evicted first), reports persisted to
  `runs/jobs/<survey_id>/` (downloads survive restarts; optional S3 mirror via `$DEPTH_S3_BUCKET`).
  The UI polls and streams `frame_done k/N` into the agent log. Sync `/api/survey` kept for ≤ 12 frames.
- **Input limits:** images only (415), ≤ 20 MB/file (413), ≤ 50 frames (413), undecodable → 400.
  CORS off unless `$DEPTH_CORS_ORIGINS`. Model loads + warms up at startup (`/api/health` shows it,
  plus `cv2_file` / `is_cool_path` for COOL provenance).
- **Samples ship with the app:** `webui/samples/` — 8 Rec6 crab-pot sonograms unseen by EXP-001
  (one un-rotated copy each, consecutive chunks so stitching shows), labels, CC-BY-SA attribution.
- **Map:** Leaflet 1.9.4 **vendored** (`webui/vendor/leaflet`, SRI-verified = upstream hashes);
  CARTO dark (now key-gated) → keyless Esri satellite + OSM layer switch; tiles failing → grid
  fallback (overlays stay exact); **geotag error radii** drawn; scale bar. Synthetic demo track
  moved from marsh to open Chesapeake water (37.80 N, 76.15 W).
- Tests: +4 job-queue tests (`tests/test_jobs.py`) + 6 API tests (limits, 415/400/413, job path
  end-to-end with the real model). Browser-verified: job progress, map, downloads, no console errors.

### 2026-09-24 (review sweep 4) — Guaranteed tiers, value-of-information agent, budget mode (STUDY-07)
Review C-3 / I-5 / I-6 / M-1 / M-2 / X-4.
- **`src/agentic/guarantees.py`** — exact Clopper–Pearson bounds (pure Python, matches SciPy to 5 d.p.)
  + Learn-Then-Test threshold fitting for a **recall promise** and a **precision promise**.
- **`src/agentic/calibrate.py` rewritten** — collect on unique frames (cached), fit on validation,
  verify once on test, compare 7 policies at the same promise, write `calibration.json` + report.
  Result: **≥ 65% of pots reach a human, held on test (86.2%)**; no ≥85% precision promise possible
  → nothing auto-confirmed; confidence alone beats every re-look variant on unseen data; the old
  0.737 claim is 0.58 on unseen data (retired).
- **Agent** — calibrated mode: value-of-information tool use (`needs_relook`), batched re-look
  (`Perceptor.relook_batch`: 4 crops per inference, ~3× cheaper), P(pot) per card, tier promise in
  every trace; legacy ladder kept as fallback/tests. REJECTED → **LOW-RISK** (alias kept).
- **Survey** — REVIEW queue by P(pot), **budget mode** (`plan_budget`: cards that fit N minutes +
  expected real pots; sec/card ASSUMED 8 s until the user study), **inspection route**.
- **UI** — Guarantees panel (promises + "held on test"), tier promise + P(pot) on cards, "no re-look
  (VoI 0)" wording, budget input, Analyst-effort panel (expected pots vs minutes). **Fixed a
  pre-existing bug:** `display:flex` overrode `hidden`, so the placeholder covered the sonar image and
  the "working…" toast never hid. Static UI now served with `Cache-Control: no-cache`.
- Found: CARTO dark tiles now return "API KEY REQUIRED" → fixed in the next (backend) component.
- Tests: 47 (+4 calibrated-agent tests); API tests exercise the calibrated mode with the real model.

### 2026-09-24 (review sweep 3) — Dataset v2b: dedupe, recording-level val, unique-frame test
Review I-2 / C-4. `DATASET/scripts/build_dataset_v2b.py` → `03_yolo_ready_dataset_v2b/`.
- **Roboflow copies deduped:** train 5,275 copies → **1,615 unique frames** (3,241 rotated copies
  dropped; least-changed copy kept). Official test 398 → **214 unique frames**.
- **val = held-out training recordings** Rec10/12/16 (163 frames, 186 pots, same Humminbird sonar as
  test) + v2 wreck/seabed val. The orange Contact_sslo "valid" became `test_xsonar/` (cross-sonar).
- `test_official398/` kept for the GhostVision head-to-head only; `groups.json` for group bootstrap.
- **Pixel leakage probe:** 0 test frames with a train twin (max corr 0.92, different recordings).
- **Finding for EXP-001:** its only unseen crab-pot sonograms are v1 val (66 unique frames, Rec19)
  and v1 test (92 unique) — every EXP-001 guarantee must be fit/verified there (next component).
- Audit: 0 leakage, 0 malformed; `DATASET/exports/audit_summary_03_yolo_ready_dataset_v2b.json`.

### 2026-09-24 (review sweep 2) — Honesty fixes: orientation by rule, thin-line shadow, no cross-pass
Review items C-6 / I-8 (physics + geo part). See `experiments.md` STUDY-06.
- **`src/cv_pipeline/orientation.py`** — nadir from a *source rule* (PINGMapper `*_ss_port/star*` →
  top), explicit user override, else **unknown**. The auto-guess (26% right) is now a labelled
  diagnostic only. `FrameResult.orientation` carries the provenance to the UI.
- **Shadow** (`shadow.py`): thin 3-px darkest line + flank-referenced contrast (seabed no longer
  reads as "weak shadow" 30% of the time); relative height `h/H`; metres only with a measured
  altitude; slant→ground correction hook for Stage-1 bottom tracking; unknown orientation ⇒ not measured.
- **Removed `match_other_pass`** and its +0.10 ranking boost / "overlapping pass" note (physically
  wrong). **Added `stitch.py`** — chunk-boundary stitching; survey counts a split object once and keeps
  the stronger sighting (`TrackedObject.also_in`).
- Geotag: unknown orientation ⇒ boat fix + swath-wide error (no fake range).
- UI/exports: "rel. height" (% of altitude) instead of metres; orientation source in the caption.
- Tests: 39 fast (+9: orientation, stitching, relative height, thin line, unknown-orientation) + 4 API.

### 2026-09-24 (review sweep 1) — Detector core: one threshold source, per-class NMS, warm-up
Start of the NEEDTOIMPROVE review sweep (M-3 + part of I-5/I-6).
- **`src/detection/calibration.py` + `models/EXP-001/calibration.json`** — the ONE place every
  runtime threshold comes from (per-class conf, NMS IoU, class-aware NMS, re-look conf, decision
  tiers). Detector, agent, dashboard (and next: Lambda/benchmark) read it; a missing/partial file
  falls back to the shipped defaults, so nothing can change silently. Tracked in git (tiny JSON).
- **Class-aware NMS** (`cv2.dnn.NMSBoxesBatched`) — a crab pot next to a wreck box is no longer
  suppressed by it. On 60 v1-val sonar frames: 221 detections either way (single-class frames), so
  no regression; unit test proves the cross-class case.
- **Runtime:** `configure_runtime()` pins `cv2.setNumThreads` from `$DEPTH_THREADS`;
  `$DEPTH_DNN_ENGINE` selects auto|new|classic (default AUTO = OpenCV 5 new engine w/ fallback;
  laptop timings too noisy to prefer one); `warmup()`; the re-look detector now **shares** the
  loaded network (`with_conf`) instead of loading a second copy.
- Tests: +4 (`tests/test_calibration.py`) → 30 fast tests + API smoke.

### 2026-09-24 (cont.3) — Frontend rebuilt as a DepthWizard-class agentic studio
- **Studied the user's reference** `C:\Users\RAJ\Desktop\ExP\DepthWizard` (React+TS+Tailwind+Zustand
  Tauri studio) and rebuilt DEPTH's web app to that bar — **kept zero-build vanilla** (the live judge
  demo must stay up without a toolchain; Tauri/React would jeopardise it) while adopting the
  instrument aesthetic + panelled layout.
- **Studio shell** (`webui/index.html` + `styles.css` full rewrite): full-viewport, no page scroll;
  topbar (brand + **Analyze/Survey mode switch** + status pills); 3-column **workspace** — left rail
  (**Sources** + **Model Registry**) · center **canvas** (toolbar + interactive viewer) · right
  **inspector** (Evidence / Survey reports+hazards); **bottom pipeline dock** = See→Prove→Decide→Act
  status-node stepper + a **live streaming agent log** (tool calls appear line-by-line with timings as
  the loop runs — the "agentic running" feel).
- **Class cut/toggle:** per-class legend chips in the toolbar — click to **cut/show a detected class**
  (dims its boxes + evidence cards); combined with the detector-gate slider.
- **Smoother zoom:** wheel now eases (.10s) instead of stepping; drag stays instant; cinematic
  gaze-tour + agent's-eye PiP retained.
- **Cleaner aesthetic:** DepthWizard palette (near-black surfaces #0a0e13, teal accent #1fb6d5, thin
  5px scrollbars, tiny uppercase panel titles, mono tabular metrics, status-dot nodes) — dropped the
  marketing hero/emoji-steps.
- **Preserved every DOM id** the viewer/agent logic uses (`app.js` rewritten around the new shell but
  same contract) → **backend untouched, 30/30 tests still valid**.
- **Verified:** `node --check`, all `#id` refs resolve, uvicorn serves the studio (title=DEPTH,
  app.js/styles 200, studio elements present) + `/api/analyze` returns candidates + relook_view.
  *Pixel-level polish is best judged in a browser — run `uvicorn src.dashboard.app:app --port 8000`.*

### 2026-09-24 (cont.2) — "See what the agent sees": agent's-eye OpenCV view + cinematic synchronized zoom
- **OpenCV, upgraded.** `zoom_relook` built the exact upscaled+CLAHE image the re-look detector
  sees, then discarded it. Added **`Perceptor.relook_view()`** (pure cv2, **no extra inference** — the
  calibrated agent path + 30/30 tests are untouched): reproduces that crop as **display images** — a
  **LANCZOS4** zoom + the **CLAHE** "try-harder" pass (BGR) + the object box in crop px. Evidence-card
  crops now upscale with **INTER_CUBIC** (was blocky INTER_NEAREST) at 300px. `/api/analyze` attaches
  `relook_view{zoom_png, enhanced_png, scale, obj_box}` per candidate.
- **Cinematic synchronized zoom (the ask).** After a run the viewer **auto-tours** every find —
  flying/zooming the user's viewport to each candidate at the agent's *actual* re-look scale (smoother
  .95s ease), popping an **"agent's eye" picture-in-picture** of the CLAHE re-look, drawing the shadow
  strip/echo, and toasting the re-fire. **Any scroll/drag/zoom hands control back to the user**
  (opt-out for that frame); `▶ agent tour` / `⏸ stop` toggles it manually.
- **Dynamic interactiveness.** Evidence cards gain a **raw ⇄ enhanced (CLAHE)** before/after toggle
  using the agent's-eye images; plus the box↔card linking, keyboard (←/→ · G · ±/0) and detector-gate
  slider from the prior drop.
- **Verified:** `node --check`, DOM ids resolve, python parse, uvicorn `/api/analyze` returns
  `relook_view` (zoom 26 KB / enhanced 36 KB, scale 3.3×), **30/30 tests pass**.

### 2026-09-24 (cont.) — Rebrand to **DEPTH** + interactive sonar viewer (peak, robust frontend)
- **Renamed the product to DEPTH** across all surfaces (README, `webui/` title + brand, dataset &
  model cards) — reverted the inherited "GhostGear Sonar". README adds the backronym
  **DEPTH = Detect · Evidence · Prove · Triage · Hazard-map** (the loop spells the name). GitHub repo
  stays `depth` (rename deferred by user).
- **Built an interactive sonar viewer** (`webui/`, still zero-build): cursor-anchored **wheel zoom**,
  **drag-pan**, fit/±, and **live SVG vector detection boxes** over the raw frame — verdict-coloured,
  constant on-screen stroke at any zoom (`--inv` scale var), dashed for REJECTED. Boxes are clickable
  objects **linked to evidence cards** (hover a card → its box highlights; click a box → scroll+focus
  its card).
- **"Replay agent gaze"** per candidate — animates the viewer to pan+zoom into the candidate at the
  agent's *actual* re-look scale, draws the measured **shadow strip + echo** overlay, and toasts the
  outcome ("agent zoomed 2.1× → re-fired at 0.54"). Makes *"the picture result changed the agent's
  next step"* literally visible; the user can also zoom/pan the frame himself.
- **Detector-gate slider** (dim boxes+cards below a live confidence gate) + **keyboard nav**
  (←/→ cycle candidates, +/−/0 zoom, **G** replay gaze) — the user explores recall/precision himself.
- **Verified:** `node --check` clean; all referenced DOM ids resolve; uvicorn end-to-end — health
  OpenCV 5.0.0, `/` (title=DEPTH), `/app.js`, `/styles.css` all 200, `/api/analyze` returns the
  `bbox` + `relook.scale` the viewer needs. **Backend unchanged** (30/30 tests still valid).
- **Deferred (user):** AWS/MCP setup → a dedicated "peak architecture" day (state saved in memory).
- **Next:** more novel agentic/frontend (before/after re-look compare slider, persisted human
  corrections → retrain seed, client re-triage explorer); EXP-002 training support.

### 2026-09-24 — `evaluate.py`: deploy-faithful, honest per-domain metrics (#10/#11/#13) + AWS CLI in
- **Built `src/detection/evaluate.py`** — the missing evaluator the WINNING_REPORT leaned on four
  times. It scores the **exact shipped ONNX** through the torch-free **`cv2.dnn`** path (same as the
  Lambda/dashboard), collects detections once at a low floor and caches them, then computes everything
  from the cache: VOC all-points AP@0.5, P/R/F1, **val-tuned per-class thresholds (test scored once)**,
  **per-sensor & per-source tables**, and **bootstrap 95% CIs**. 30/30 tests still green.
- **Ran it full (val 1,204 + test 1,276 imgs, 1,000 bootstrap resamples).** Results (`docs/eval_exp001.md`):
  - **Aggregate mAP@0.5 0.827** — *independently reproduces* the ultralytics 0.822 (cross-validates
    both the train run and this evaluator).
  - **Honest per-sensor truth:** `fishing_gear` (crab pots) on **SONAR** = AP **0.473 / R 0.599**;
    `natural_formation` is **optical** (AP 0.990 optical vs 0.000 on 2 sonar boxes); **optical debris =
    total miss** (fishing 0.000/19, struct 0.000/25). The 0.827 is domain-inflated — now provable.
  - **Val-tuned thresholds** (never touched test): fishing 0.15 / pipe 0.40 / struct 0.25 / natural 0.50.
  - **GhostVision head-to-head SKIPPED honestly** (EXP-001 leaked v1 on the official crab-pot split);
    gated behind `--leakage-free` for EXP-002. Logged as **STUDY-05**.
  - **Closes NEEDTOFIX #10 (val-tuned/test-once), #11 (per-domain tables), #13 (bootstrap CIs);
    scaffolds #24.**
- **AWS setup started (user is retraining EXP-002 in parallel):** installed **AWS CLI v2.37.1**
  (user-scoped, no admin) + persisted PATH; `uv` already present; created profile **`hackathon`**,
  region **us-east-1** (standard account → advanced rule-set). Verified `aws login` +
  `aws configure agent-toolkit` are valid subcommands. **Handed off the 2 interactive steps**
  (browser `aws login`; the `configure agent-toolkit` wizard that installs the `aws-mcp` server +
  skills). On return: verify identity/skills, add `AWS_MCP_PROXY_PROFILES=hackathon` to the `aws-mcp`
  entry in `~/.claude.json`, append AWS advanced rules to `CLAUDE.md` (idempotent markers), set a
  **budget alarm** before standing up any resource. Services to be chosen together (COOL Graviton EC2
  first — the primary award).
- **Next:** finish AWS MCP setup on the user's return; when EXP-002 (v2) lands, export ONNX → re-run
  `evaluate.py --leakage-free` for the GhostVision head-to-head; then Graviton+COOL benchmark.

### 2026-09-23 (cont.7) — NEEDTOFIX/​WINNING_REPORT sweep: honesty + compliance across infra & docs
- **Cleared the remaining `NEEDTOFIX.md` items that were code/docs (not AWS-run or GPU-train):**
- **#4 fake geotag → honest** (`infra/lambda_handler.py`): stop stamping one lat/lon on every
  object; `geotag()` now offsets each detection by its across-track ground range from the boat fix
  (side + heading), mirroring `src/agentic/geo.py`; **no GPS → `gps_available=False` + null coords**.
  Verified: 2 detections → 2 distinct coords; null coords without GPS.
- **#5 Lambda "runs Stage-1" claim → corrected**: it's a detection-only endpoint (STUDY-01 retired
  ROI-gating); the docstring no longer claims Stage-1, which is the *separate* COOL benchmark workload.
- **#6/#7/#19 COOL proof** (`infra/benchmark_graviton.sh`): pinned `opencv-python-headless==5.0.0.93`
  (was unpinned → silently 5.0.0); removed the false "install the COOL wheel" (COOL is an **AMI** —
  use `/opt/cool` venv); now **proves COOL by provenance** (`cv2.__file__` under `/opt/cool` + version
  + build info + AMI/instance via IMDSv2), **refusing to mislabel** a run as cool unless cv2 truly
  loads from `/opt/cool`. "KleidiCV detected" demoted to informational (stock Arm bundles it). `bash -n` clean.
- **#12/#25 README → judge-facing + honest**: removed "Why this wins" and "AWS CLI not installed";
  fixed the false **"natural_formation = rock clusters"** claim (it's ICRA19 *optical fish/plants*);
  reflect the built dashboard + agentic loop; added a live-demo quickstart, an honest per-class
  results table (aggregate is domain-inflated), the STUDY-04 agent numbers, and a License section.
- **#9 LICENSE**: added canonical **AGPL-3.0** (Ultralytics YOLO is AGPL; repo is public).
- **#26 AGENT.md encoding**: fixed the UTF-8 mojibake (`â€˜ghost netsâ€™â€"` → `'ghost nets'—`, 3→0).
- **#22 export_onnx.py**: created `src/detection/export_onnx.py` (the script `infer.py` tells users to
  run) — exports `.pt`→ONNX and **verifies the `cv2.dnn` load + forward** (output `(1,8,8400)` on the
  existing weights). `#28` docs: added `docs/{dataset_card,model_card,responsible_use}.md`.
- **Left for the AWS/GPU phase (cannot do from here):** #10/#17 EXP-002 retrain on v2 (Kaggle GPU);
  #12 val-tuned thresholds (needs a val sweep = `evaluate.py`); #8/#18 stand up the c8g COOL EC2.
  These are *run/train/deploy* actions, not code defects — flagged in TODO.
- **Full suite still 30/30.** Next: EXP-002 + `evaluate.py` (val-tuned thresholds, GhostVision
  head-to-head, bootstrap CIs), then the Graviton+COOL deploy.

### 2026-09-23 (cont.6) — Decide becomes an adaptive controller; a 2nd calibrated CONFIRMED path (STUDY-04)
- **Made the agent genuinely agentic, not a fixed script** — the substance the Agentic-Vision award
  weights most (orchestration & autonomy 25% + task success 20%). `_decide_candidate` is now an
  **adaptive escalation controller**: re-look first → shadow + `estimate_height` → **escalate to an
  enhanced CLAHE re-look only when still uncertain** (skipped once confident — no wasted inference) →
  triage. Candidates take different tool paths and stop at different depths; the trace records the
  route **and which CONFIRMED path won**.
- **De-risked the headline claim with data BEFORE coding it (STUDY-04).** Mined 160 test frames /
  461 detections. My corroboration hypothesis (high-conf **and** CLEAR shadow) was **overturned**:
  requiring a CLEAR shadow *lowers* precision (0.40 vs 0.69 conf-only) — and the committed calibrate
  shows **CLEAR-rate 14.5% (TP) vs 14.6% (FP)**, i.e. non-discriminative (reconfirms STUDY-03 with a
  cleaner statistic). Shadow stays evidence-shown + height, never a gate.
- **What the data *did* support → shipped:** plain **high detector confidence** (`conf ≥ 0.60`) is
  precision **0.826**. New policy = `relook ≥ 0.40` **OR** `conf ≥ 0.60`: CONFIRMED precision **0.737
  at 30% recall-share**, beating the old re-look-only tier (**0.713 / 26%**) on *both* axes. By path:
  re-look 0.71, high-confidence-only 0.92, both 0.70. REJECT stays recall-safe (91% of true pots
  retained, never deleted).
- **Two tools added** (toolbox now 6: detect · zoom_relook · enhance_relook · shadow_check ·
  estimate_height · match_other_pass): `estimate_height` (named, traced) and **`match_other_pass`** —
  survey-level **cross-pass corroboration** (same recording+channel, adjacent ping, same across-track
  position → trace step + evidence note + review-ranking boost). Deliberately **non-gating**, so the
  CONFIRMED tier stays exactly the calibrated set (honest).
- **Live-demo impact:** the bundled 5-sample survey went **0 confirmed / 3 review → 1 confirmed / 2
  review** — a conf-0.68 pot with a clear h~0.7 m shadow, previously stuck in REVIEW, is now
  auto-confirmed via the high-confidence path → a real recovery route on the map.
- **Tests 30/30** (added: high-confidence path + escalation-skip, the two new tools, and a survey
  cross-pass integration test asserting the verdict is never changed). Docstrings across `agentic/`
  updated to the STUDY-04 numbers; `calibrate.py` now prints the per-path precision + the shadow
  non-discrimination statistic.
- **Next:** stand up the Graviton + COOL EC2 with the COOL AMI and put the live link behind it
  (the last big Agentic-Vision + COOL deliverable); then EXP-002 (imgsz retrain) to raise the
  underlying recall the agent triages.

### 2026-09-23 (cont.5) — Live dashboard: the See→Prove→Decide→Act loop becomes a judge-usable demo
- **Built the web dashboard** (`webui/index.html` + `styles.css` + `app.js`) — the piece that had been
  "Next" for four sessions and the make-or-break for the Agentic-Vision **user-experience** score and
  the live demo link. Deliberately a **zero-build static app** (no npm/Vite toolchain to break on the
  Graviton demo server that must stay live 27 Oct–9 Nov) served directly by FastAPI. Deep-ocean sonar
  aesthetic; Leaflet via CDN with graceful degradation.
- **What a judge sees:** a **See→Prove→Decide→Act stepper** that lights up as the loop runs; one-click
  **sample frames** (real sonar thumbnails via a new `/api/sample_thumb/{id}` endpoint) + drag-drop
  upload; an **Analyze** view with a verdict overlay/raw toggle, per-candidate **evidence cards**
  (zoomed crop with the shadow overlay, a **detector-conf → re-look-conf arrow** that visualises "the
  picture result changed the agent's next step", shadow quality + est. height + echo, evidence notes)
  and an expandable **agent trace** (every tool call, timed); a **Survey** view with a Leaflet
  **hazard map** (verdict-coloured markers + the nearest-neighbour recovery route), a **mission banner**
  ("human approval required — nothing auto-dispatched"), one-click **downloads** (GeoJSON/GPX/KML/CSV/
  JSON), frame thumbnails, and a **hazards table with per-row approve/reject** (human-in-the-loop).
- **Honesty surfaced in the UI** (avoids the "misrepresents capabilities" rejection risk): OpenCV
  version + model status in the footer/pills; a bright **"⚠ SYNTHETIC DEMO GPS — not real coordinates"**
  banner on the map/exports; "no GPS → table only" when a file has none; "REJECTED retained for audit,
  not deleted" in the footer.
- **Backend:** added `/api/sample_thumb/{id}` (downscaled real-sonar gallery cards) and made `app.py`
  serve the no-build `webui/` directly (prefers `webui/dist` if a real build ever lands).
- **Verified** end-to-end through both the Starlette TestClient **and real uvicorn**: `/`, `/app.js`,
  `/styles.css`, `/api/health`, `/api/samples`, `/api/sample_thumb`, `/api/analyze` (0.7 s/frame),
  `/api/survey` (5 frames, 1.5 s), and all five `/api/report/{fmt}` downloads → 200. `node --check`
  clean; all 27 JS id-references resolve in the HTML.
- **Honest demo gap noted → motivates next session:** on the 5 bundled samples the loop currently
  returns **0 CONFIRMED / 3 REVIEW** (re-look alone confirmed none), so the recovery route is empty.
  Yet a conf-0.68 candidate with a **clear** acoustic shadow at a plausible crab-pot height (h~0.7 m)
  is sitting in REVIEW — strong corroborated evidence the single-signal policy ignores. → **STUDY-04**.
- **Next:** upgrade the Decide agent from a fixed 3-call sequence to an **adaptive controller** +
  **cross-pass corroboration** + a **multi-signal CONFIRMED path** (recall-safe, must preserve the
  STUDY-03 CONFIRMED precision), calibrated honestly on the test split (STUDY-04). Then AWS/COOL.

### 2026-09-23 (cont.4) — WINNING_REPORT critical fixes: OpenCV 5 pinned; dataset v2 (honest, sonar-only)
- **Acted on `docs/WINNING_REPORT.md`** (the full strategic audit). This session cleared the two
  rule-/model-critical blockers it flagged, before spending any more Kaggle hours on the buggy v1.
- **OpenCV 5 for real (rule 1).** Local env moved off OpenCV 4.12 → **5.0.0.93** (removed the stale
  dual `opencv-python`/`-headless` 4.x install; `cv2.dnn` present; Intel IPP here, KleidiCV is the
  Arm/COOL equivalent). Split & **pinned exactly**: `requirements.txt` (lean torch-free runtime:
  opencv-python-headless 5.0.0.93 / numpy 2.2.6 [OpenCV 5 needs numpy≥2] / fastapi / uvicorn /
  python-multipart / boto3), `requirements-train.txt` (Kaggle: ultralytics/torch/onnxruntime),
  `requirements-dev.txt` (pytest). **Full suite 27/27 passes on OpenCV 5 + numpy 2.2.6.** The
  dashboard `/api/health` footer now honestly reports `cv2 5.0.0`.
- **Dataset v2 — `DATASET/scripts/build_dataset_v2.py` (2-class, sonar-only, honest).** Fixes the 3
  data bugs: (1) Marine PULSE pipeline/platform — **all 1,243 mpulse images were full-frame-boxed**
  in v0, so v1 dropped them to "background" (955 real objects taught as empty) → **DROPPED**
  (unlocalised); only mpulse `seabed_surface` kept, as background. (2) **1,547 empty crab-pot frames
  RECOVERED** from the raw archive as natural-seabed negatives (the cure for the 79% empty-seabed→
  fishing-gear FP). (3) Optical (ICRA19/TrashCan) **dropped**; **UATD** (forward-looking, placed
  objects) **excluded from train/val/test** and exported as a separate held-out generalisation eval
  (`_holdout_uatd_fls/`, 3,927 imgs). Classes renamed to match reality: **`ghost_gear`** (side-scan
  crab pots) + **`wreck_debris`** (KLSG barges + AI4Shipwrecks sonar).
- **Built & audited v2:** train **6,291** (1,550 bg) · val 626 (104 bg) · test 469 (65 bg); boxes
  ghost_gear **9,281** / wreck_debris 615; **leakage 0/0/0**, **0 full-frame boxes**; COCO→YOLO
  conversion spot-checked exact. The **official crab-pot 398-frame test split is respected + LOCKED**
  (`official_crabpot_test.txt`) → enables the head-to-head vs GhostVision. Honest residual:
  `wreck_debris` is low-volume (val 20 / test 70 boxes) → high-variance per-class AP; report it, and
  it's a secondary class (crab pots are the target).
- **Next:** point EXP-002 at `03_yolo_ready_dataset_v2/data.yaml` (≈25–30 ep, patience 8 — v1 curve
  peaked ~ep16); write `evaluate.py` (val-tuned thresholds, per-source tables, GhostVision head-to-head,
  bootstrap CIs); add `docs/dataset_card.md`; then resume the React dashboard.

### 2026-09-23 (cont.3) — FastAPI backend: agentic loop exposed as an API
- **Built the backend** (`src/dashboard/app.py`, `samples.py`): `/api/health` (honest OpenCV version
  + model status for the footer), `/api/samples` (one-click demo frames), `/api/analyze` (sample or
  upload → annotated frame + per-candidate evidence cards + full tool-call trace, base64), `/api/survey`
  (many frames → hazards + mission plan + thumbnails), `/api/report/{fmt}` (geojson/gpx/kml/csv/json
  download). Model loads lazily; serves `webui/dist` when built.
- **Tests: 28/28 pass** (added `test_api.py` — health/samples/validation always; analyze+survey+report
  path when model+samples present). Verified end-to-end via TestClient (analyze ~1.7s/frame).
- **Next:** the React dashboard (DepthWizard-style See→Prove→Decide→Act stepper, evidence cards, map).

### 2026-09-23 (cont.2) — ACT: honest geotag + cleanup route + orchestrator (loop complete)
- **Built the Act stage + orchestrator** (`src/agentic/geo.py`, `mission.py`, `pipeline.py`):
  `AgenticPipeline.run_survey()` runs See→Prove→Decide per frame, geotags candidates, promotes
  CONFIRMED/REVIEW to survey-level `TrackedObject`s, and builds a human-approved `MissionPlan`
  (nearest-neighbour recovery route + resurvey list + GPX/GeoJSON/KML/CSV exports).
- **Honest geotagging:** real per-ping GPS → detection lat/lon via across-track range + heading±90°
  (with a coarse error estimate); **no GPS → no coordinates + a `gps_available=False` flag** (never
  the "one fake lat/lon on every object" bug from WINNING_REPORT). A clearly-labelled **synthetic**
  demo track (`SYNTHETIC DEMO GPS - not real coordinates`) makes the map/route demonstrable.
- **End-to-end run** (8 demo frames): 23 hazards (5 confirmed / 18 review) → 5-stop 22.8 m recovery
  route; exports validated; `human_approval_required=True` throughout (nothing auto-dispatched).
- **Tests: 23/23 pass** (added `test_geo.py` 5 + `test_mission.py` 4 — geodesy, side parsing, NN
  route, no-GPS honesty, all exporters).
- **The agentic brain (See→Prove→Decide→Act) is complete.** Next: FastAPI backend → React dashboard
  (DepthWizard-style stepper) to make it a judge-usable live demo.

### 2026-09-23 (cont.) — DECIDE agent: triage + tool-call trace (Agentic-Vision substance)
- **Built the Decide stage** (`src/agentic/agent.py` + `tools.py`): `ReLookAgent.run_frame()` runs
  See→Prove→Decide per frame and, for each candidate, calls tools in sequence
  (`shadow_check` → `zoom_relook` → an **enhanced CLAHE re-look** when still uncertain) then triages
  via `policy.py` into CONFIRMED / REVIEW / REJECTED. Every tool call is logged as an `AgentStep`, so
  each decision carries a full audit trail; `render()` draws verdict-coloured overlays.
- **Real trace captured** (the award evidence — a perception result changing the next step): a
  conf-0.18 candidate did *not* re-fire on the first zoom, but the agent's CLAHE "try harder" re-look
  re-fired at **0.54 → auto-CONFIRMED**; its weak acoustic shadow (contrast 0.36, h~0.2 m) shown as
  supporting evidence. Nadir auto-calibrated to `bottom` on that starboard channel (orientation-robust).
- **Refactor:** evidence fusion/notes extracted to module functions in `evidence.py` so the agent and
  `EvidenceGatherer` share one source of truth. `perception.zoom_relook(enhance=)` added (CLAHE pass).
- **Tests: 14/14 pass** (`tests/test_agent.py` added — trace shape, enhanced-relook upgrade, all 3
  tiers, serialisation, render). Deterministic (perceptor stubbed).
- **Next:** ACT (`mission.py` cleanup route + `geo.py` honest geotag + `pipeline.py` orchestrator) → FastAPI + dashboard.

### 2026-09-23 — Agentic loop begins: See→Prove→Decide foundation (Prove is multi-evidence, honest)
- **Started the See→Prove→Decide→Act agentic layer** (`src/agentic/`) — the Agentic-Vision entry.
  This session delivered the **Prove foundation** + its honest calibration (STUDY-03).
- **Key finding (STUDY-03), reached by empirical de-risking BEFORE building the headline claim:** the
  acoustic **shadow does NOT gate** detections on this data — YOLO's TP and FP have identical echo
  brightness and neither shows a systematic dark shadow (shadow-gating *lowers* precision 0.59→0.31).
  A clear shadow exists only on a minority of larger objects. **Re-look persistence** (zoom 2.5×,
  re-detect) is the real discriminator: raw hot precision **0.60 → 0.77** at the auto-CONFIRMED tier.
- **Built & tested:** `types.py` (serialisable Verdict/ShadowProof/Evidence/Candidate/FrameResult),
  `perception.py` (Perceptor: detect + `zoom_relook` [validated params] + CLAHE), `shadow.py`
  (ShadowProver: nadir auto-calibration + shadow measure *where present* + height + overlay),
  `evidence.py` (EvidenceGatherer: re-look⊕shadow⊕echo → evidence_score), `policy.py` (pre-registered
  triage rules), `calibrate.py` (the STUDY-03 table on the real test split). 9/9 unit tests pass
  (`tests/test_shadow.py`, `tests/test_evidence.py`), incl. shadow geometry for all 4 nadir edges.
- **Design decision (user-confirmed):** Prove = multi-evidence; Decide = **recall-safe** 3-tier triage
  (REJECTED is retained-for-audit + deprioritised, never deleted). Shadow kept as *evidence shown where
  it exists* + height, never a silent gate — honest, and avoids the "misrepresenting capabilities"
  rejection risk. Dashboard target: DepthWizard-style web app (FastAPI + React).
- **Next:** DECIDE agent (`agent.py`, tool-call trace) → ACT (`mission.py`/`geo.py`) → FastAPI + dashboard.

### 2026-09-22 (cont.2) — tiled inference (modest); EXP-002 ready; AWS infra scaffolded
- **Tiled inference** (`src/detection/tiled_infer.py`, SAHI-style slice→detect→NMS-merge):
  sonar sample, fishing_gear recall **0.565→0.645** (+8pts, small crab-pots) but precision
  drops (structural 0.96→0.69 — big objects fragment across tiles). **Secondary lever, not
  default;** EXP-002 retrain remains the real fix. Kept as a tool + a legit COOL workload.
- **EXP-002 ready:** `docs/exp002_kaggle.md` — exact "Save & Run All" steps, `--imgsz 1024
  --batch 8 --epochs 40` (fits Kaggle's 12h cap; survives low internet). `train.py` aug left
  untouched (proven) so the unattended run can't fail on a new arg.
- **AWS infra scaffolded** (`infra/`, deploy-ready, not deployed — credits pending):
  `lambda_handler.py` (cv2.dnn ONNX → geotagged JSON, no torch, arm64-ready, locally
  smoke-tested), `benchmark_graviton.sh` (3-way COOL runner + input-manifest sha256 + S3
  upload), `README.md` (architecture, instance types, deploy checklist, COOL scorecard).
- **Next:** EXP-002 on Kaggle tomorrow → send zip → I re-run `error_analysis.py` per-domain.

### 2026-09-22 (cont.) — deep error analysis of EXP-001; per-class conf lever banked
- Built `src/detection/error_analysis.py` and mined every EXP-001 artifact (`results.csv`,
  args, P/R/F1 curves, confusion matrix, val batches). Findings:
  - **Run was 40 epochs, not 100** (docs corrected). Val mAP50 **peaked ~epoch 16** (0.704)
    then declined to 0.663 by ep40 while train loss fell → **overfit after ~ep20**; `best.pt` ≈ ep25.
  - **Aggregate mAP 0.822 is inflated by domain segregation:** `natural_formation` (R 0.99)
    is optical-only and trivially separable. `fishing_gear`/`pipe`/`structural` are ~all sonar.
  - **Optical debris = total miss** (fishing_gear 0/19, structural 0/25); model maps optical→
    natural_formation. Product is **sonar** — report sonar metrics separately.
  - **fishing_gear is crab-pot sonar** (R 0.47); `uatd` pipe/struct 0.95/0.97; `shipwreck`
    struct weak 0.25; misses are small (91% <10% frame).
  - **Free lever banked:** per-class confidence thresholds in `infer.py` (`PER_CLASS_CONF`,
    fishing_gear 0.10) → fishing_gear recall **0.47→0.69**, no retrain, others unchanged.
- **Next:** EXP-002 (`--imgsz 1280`) on Kaggle for the small-object ceiling; EXP-003 sonar-only.

### 2026-09-22 — Stage 1→2 wired via cv2.dnn; ROI-gating is a NEGATIVE result → Stage 1 reframed
- **Wired Stage 2 for the Lambda path:** extracted EXP-001 artifacts locally, verified
  `best.onnx` loads + runs through `cv2.dnn.readNetFromONNX` (output `(1,8,8400)`; 4 box + 4 cls).
  Built `src/detection/infer.py` (`YoloOnnxDetector`, full_frame + roi_guided modes, letterbox
  + NMS, CLI) and `src/detection/ablation_fp.py` (full vs ROI-gated TP/FP/recall + Stage-1 GT
  coverage), plus `src/cv_pipeline/tune_coverage.py` (geometry-filter sweep).
- **The two-stage FP premise fails on this data (STUDY-01).** On a 300-frame strided test
  sample: Stage-1 **GT coverage 24.7%**; ROI-gating drops recall **0.708→0.147** for a fake
  52% "FP reduction" (it just discards true detections). Sweeping geometry filters, coverage
  **caps at 72.4% with ALL filters off — at 141 ROIs/frame** (~28% of debris never segments).
  Killer probe: on **empty-seafloor** frames Stage 1 emits **60 ROIs/frame** vs **8** on frames
  with objects — it fires *more* on clutter than on debris. It thresholds brightness/geometry
  and ignores the real sonar cue (highlight + acoustic shadow).
- **Decision (locked):** retire the ROI-gating / ≥60% FP-reduction target; **YOLO is the
  detector**; **reframe Stage 1 as the CPU sonar-preprocessing workload for the COOL benchmark**
  (its resize/threshold/contour ops are still the COOL sweet-spot — it just no longer gates
  Stage 2). Negative result + evidence kept as a submission asset. Shadow-aware Stage 1 deferred.
- **`fishing_gear` diagnosis done (`fn_gallery.py`):** the 0.371 recall is a **small-object**
  problem, not confusion. Confusion matrix: misses → *background* 0.63 (not other classes) and
  background → fishing_gear 0.79 (top FP sink). FN gallery: 54% of boxes missed, **91% of misses
  <10% of frame width** (median 6%) — small low-contrast returns in speckle clutter.
- **Next (Kaggle GPU):** EXP-002 = higher resolution, the clear lever:
  `python src/detection/train.py --model yolo11s.pt --imgsz 1280 --batch 8 --epochs 100 --name EXP-002`.
  Then EXP-003 tiled train/infer (also fills Stage-1's reframed tiling role), EXP-004 oversample.

### 2026-09-21 — EXP-001 baseline trained (Kaggle T4) — all aggregate targets met
- **First real training run is in.** YOLO11s (detect, not seg), dataset v1, 640, sonar-aware
  aug, 40 ep / batch 16 on a Kaggle T4. Logged as **EXP-001** in `experiments.md` with full
  per-class results.
- **Test-split result (1,276 imgs / 1,787 inst):** mAP@0.5 **0.822**, mAP@0.5:0.95 **0.514**,
  precision **0.808**, recall **0.800** — **every aggregate target cleared** (≥0.70 / ≥0.45 /
  ≥0.80 / ≥0.70). Inference ~11.5 ms/frame on T4.
- **The one problem: `fishing_gear`.** Recall **0.371** (AP@0.5 0.45) — we miss ~63% of the
  single most dangerous class (ghost nets / rope). Aggregate metrics pass only because the other
  three classes are strong (pipe_cylinder AP 0.94 on 65 boxes — high-variance; struct 0.923;
  natural 0.973). Counter to blocker B4, imbalance hurt the *majority* merged class, not the rare one.
- **Next:** pull `confusion_matrix.png` + a fishing_gear FN gallery to classify the misses;
  run **EXP-002** (originals-only vs baked-aug ablation), then attack fishing_gear recall
  (oversample / focal, EXP-004). Export `best.onnx` for the Stage-1→Stage-2 wiring.

### 2026-09-21 — Stage 1 CV pipeline + COOL benchmark harness; verified COOL scorecard
- **Verified what COOL actually is** (was an assumption): Cloud-Optimized OpenCV, a
  KleidiCV-accelerated Arm/Graviton build that speeds up **resize, adaptive-gaussian
  threshold, contour detection** (~1.5× avg). Proposal's expansion was correct.
- **Got the "Best Use of COOL" scorecard:** verified COOL integration on Graviton **30%** +
  measured performance vs baseline **20%** + architecture **25%** + innovation **15%** +
  reproducibility **10%**. ⇒ **half the award is a reproducible Graviton-vs-x86 benchmark of
  the core workload**, not a fancier model.
- **Built Stage 1** (`src/cv_pipeline/`): `config.py` (all tunables, COOL-friendly defaults),
  `pipeline.py` (`Stage1Pipeline.process()`, per-op timed, + `draw_candidates()`),
  `benchmark.py` (3-way harness), `README.md`. Deliberately composed of COOL-accelerated ops.
- **Tested on real sonar frames** (v1 test set): local x86 stock OpenCV 4.12, 400 frames →
  **29.2 ms/frame, 30.5 FPS**, avg 10 ROIs/frame. Per-op: **threshold+contours+resize = ~76%
  of compute** — i.e. the compute concentrates in exactly the ops COOL accelerates, so the
  Graviton+COOL speedup will be attributable. Well under 300 ms / above 5 FPS.
- **Key design calls:** `fastNlMeansDenoising` OFF by default (slow + NOT KleidiCV-accelerated,
  would dilute COOL's measured gain) — median denoise default, NLMeans kept in `QUALITY_PRESET`
  for the ablation. Benchmark is **three-way** (x86-stock / Graviton-stock / Graviton-COOL) so
  the Graviton-stock→Graviton-COOL delta isolates COOL on identical hardware.
- **Architecture guidance:** AWS footprint for the COOL award is small — EC2 Graviton (c7g/c8g)
  + COOL running Stage 1, EC2 x86 baseline, S3 for frames/results, CloudWatch/CSV for metrics.
  Lambda/DynamoDB/Amplify/SageMaker are overall-award/demo polish, built only after COOL locked.
- **Next:** run the benchmark on a Graviton instance with COOL vs stock (needs AWS CLI + EC2);
  wire Stage 1 → Stage 2 (ROI crop → `cv2.dnn.readNetFromONNX`) once EXP-001 weights land.

### 2026-09-18 — CRITICAL FIX: v1 split had dead classes; rebuilt & stratified
- **Bug found via audit:** the on-disk v1 dataset did **not** match its manifest — a stale/partial
  build had **zero `pipe_cylinder` and zero `structural_fragment` boxes in val AND test** (2 of 4
  classes unmeasurable), and `pipe_cylinder` was nearly gone overall (105 boxes). Any mAP would
  have silently excluded two classes. v0 source verified intact (all 4 classes healthy).
- **Fix:** rewrote the split to stratify by **(source, dominant-class)** at the recording/clip
  level — guarantees every source and every class is represented ~80/10/10 in val/test while
  staying leakage-free. Rebuilt from source. New counts: **train=27,380 / val=1,220 / test=957**,
  leakage **0/0/0**, **all 4 classes now present in every split** (val pipe=100/struct=479,
  test pipe=105/struct=551).
- **Residual (honest):** optical-sensor share still skews val 13% / test 30% (only a few optical
  video clips; whole clips can't be subdivided). Options for later: make val/test sonar-only, or
  accept + report the mix. Not a blocker for EXP-001 but note it when reading metrics.
- **Next:** eyeball QA renders → `pip install -r requirements.txt` → EXP-001 baseline.

### 2026-09-18 — Split refinement + Stage 2 trainer
- **Dataset v1 split hardened.** Replaced the base-frame split key with a source-aware
  `seq_group` (video-clip id for icra/vid, recording+channel for crabpot, wreck/object for
  shipwreck, survey site for mpulse, per-frame otherwise). Whole groups are placed greedily
  to keep per-class box share + image count near 80/10/10. Also drop degenerate **sliver**
  boxes (w or h < 0.01). Re-audit: **leakage 0/0/0**, 5,618 groups; counts refreshed
  (train=26,389 / val=1,172 / test=1,734; sensor split ~80/20 sonar/optical).
- **Stage 2 trainer added** (`src/detection/train.py`): reads v1 `data.yaml`, sonar-aware
  augmentation (hue/sat off, no rotation/vflip, along-track hflip only), prints inverse-freq
  class weights for the imbalance, and reports test-split mAP/P/R at the end. `--seg` switches
  to instance segmentation. Verified `--help` runs without the training deps installed.
- **Next:** eyeball QA renders → `pip install -r requirements.txt` → run **EXP-001** baseline.

### 2026-09-18 — Foundation, docs cleanup & dataset audit
- Read full context (AGENT.md proposal, competition rules, dataset report, CC playbook).
- Established peak architecture and winning plan (`architecture.md`).
- Created the formal documentation set: `README.md`, `architecture.md`, `progress.md`,
  `experiments.md`, `TODO.md`; defined naming conventions.
- Moved reference docs into `docs/` and rewrote them formally: `competition_rules.md`,
  `claude_code_playbook.md`, `dataset_report.md` (removed clumsy `comptetion.md`,
  `HOWTOWORK.md`, root `dataset_report.md`).
- Built `DATASET/scripts/audit_dataset.py` and ran a full ground-truth audit. **Key
  findings:** dataset is **4-class** (`fishing_gear`, not 5 w/ `rope_line`); **all boxes,
  zero polygons** (already converted); **cross-split leakage** from `crab_pot`; 1,258
  full-frame + 1,713 tiny boxes. Old report was stale → rewritten (`docs/dataset_report.md`).
- Built **dataset v1** (`03_yolo_ready_dataset_v1/`) via `build_dataset_v1.py`: source-frame
  grouped split (**leakage 0/0/0**, re-audited), full-frame boxes dropped (1,096), corrupt
  scan (0), 1,099 background negatives, 4-class taxonomy locked (`rope_line` stays merged).
  Added `visualize_labels.py` (QA renders in `DATASET/exports/qa_*`).
- **Next:** eyeball QA renders → Stage 1 CV (`src/cv_pipeline/`) → baseline train EXP-001 on v1.

---

## 5. Grant check-in talking points (Sep 21 – Oct 2)

Keep it concrete and honest:
1. **Problem & impact** — ghost-net detection from side-scan sonar; 4–8h manual review → minutes.
2. **Dataset** — 7 sources → cleaned, leakage-free **v1: ~29k images** (26,533 train incl.
   1,120 background / 1,204 val / 1,276 test), 4-class, audited with a regenerable report.
3. **Architecture** — OpenCV 5 pipeline; Stage 1 classical CV is the **COOL benchmark workload**
   on Graviton (vs x86); YOLO11 does full-frame detection (EXP-001 mAP@0.5 0.822). Classical
   ROI-gating was tested and retired as a documented negative result (STUDY-01).
4. **Plan to deadline** — phased execution in `TODO.md`; primary path is Best Use of COOL.
5. **Evidence of active development** — this documentation set + dataset audit + committed repo.

---

## 6. Known blockers (must clear before first real training run)

Re-derived from the verified audit ([`docs/dataset_report.md`](docs/dataset_report.md),
`python DATASET/scripts/audit_dataset.py`). The earlier polygon/5-class blockers are
**resolved** — the dataset was reprocessed (polygons→boxes, `rope_line` merged into
`fishing_gear`, collapsed to 4 classes).

| # | Blocker | Status |
|---|---|---|
| B1 | Cross-split leakage | ✅ Resolved — v1 split by source frame; re-audit 0/0/0 |
| B2 | 1,258 full-frame boxes | ✅ Resolved — dropped in v1 (audit: 0 remain) |
| B3 | Tiny boxes | ✅ Handled — degenerate dropped; small-plausible kept + flagged for QA |
| B4 | Class imbalance (`fishing_gear` ≈ 11.7× `pipe_cylinder`) | 🟡 Open — training-time (weights/focal/aug), not a data defect |
| B5 | Taxonomy decision | ✅ Resolved — **4-class**, `rope_line` merged into `fishing_gear` |
| B6 | AWS CLI not installed | 🟡 Open — blocks cloud + COOL work |

Details and fixes tracked in [`TODO.md`](TODO.md) Phase 1 & Phase 4.

---

## 7. Metrics dashboard (fill as results arrive)

| Metric | Target | Current | Δ |
|---|---|---|---|
| mAP@0.5 | ≥ 0.70 | **0.822** (EXP-001) | +0.122 ✅ |
| mAP@0.5:0.95 | ≥ 0.45 | **0.514** (EXP-001) | +0.064 ✅ |
| Precision | ≥ 0.80 | **0.808** (EXP-001) | +0.008 ✅ |
| Recall | ≥ 0.70 | **0.800** (EXP-001) | +0.100 ✅ |
| `fishing_gear` recall (watch) | ≥ 0.70 | sonar **0.599** @val-tuned 0.15 (STUDY-05, test-once); 0.47@0.25→0.69@0.10 | 🟡 |
| `fishing_gear` AP@0.5 (sonar) | — | **0.473** [CI 0.42–0.51] (STUDY-05) — the honest product number | 🟡 |
| Agent CONFIRMED-tier precision | > raw | **0.737** @ 30% recall-share vs 0.60 raw (STUDY-04) | +0.14 ✅ |
| ~~FP reduction (Stage 1)~~ | ~~≥ 60%~~ | **retired** (STUDY-01: gating costs recall) | — |
| Latency / frame (Stage 2, T4) | < 300 ms | ~11.5 ms (EXP-001) | ✅ |
| Throughput (Stage 2, T4) | ≥ 5 FPS | ~87 FPS (EXP-001) | ✅ |
| COOL: Graviton vs x86 latency | measured | — | — |

_Link each filled row to the `EXP-NNN` that produced it. Note: aggregate targets met, but
`fishing_gear` (mission-critical) recall is the open gap._
