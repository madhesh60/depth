# Experiment Register — Marine Debris Detection System

Every training / evaluation run gets one entry. This is the research loop, not a scoreboard:
each experiment states a **hypothesis**, records the exact config, reports metrics, and ends
with **error analysis** + the **next** experiment. No result is trusted without knowing the
dataset version, model, augmentation, and hyperparameters that produced it.

**Related:** [`architecture.md`](architecture.md) · [`progress.md`](progress.md) ·
[`TODO.md`](TODO.md)

---

## Conventions

- **ID:** `EXP-NNN` (zero-padded, monotonic). Run artifacts: `runs/EXP-NNN/` (git-ignored).
- **Dataset version:** `vN` — bump whenever labels/splits change. Never train on an unversioned dataset.
- **No test-set leakage:** the `test` split is touched **only** for final reporting, never for tuning.
- **One variable at a time** where possible, so an improvement is attributable.
- **Special attention to false negatives** on dangerous debris (`fishing_gear` — ghost nets/rope).
- Record per-class metrics + confusion matrix, not just aggregate mAP.
- Copy the template below for each new run; keep newest at the top of the log.

---

## Metric definitions

| Metric | Meaning | Target |
|---|---|---|
| Precision | TP / (TP+FP) — how trustworthy a detection is | ≥ 0.80 |
| Recall | TP / (TP+FN) — how much debris we catch | ≥ 0.70 |
| mAP@0.5 | mean AP at IoU 0.5 | ≥ 0.70 |
| mAP@0.5:0.95 | mean AP over IoU 0.5→0.95 | ≥ 0.45 |
| ~~FP reduction~~ | ~~FP(full-frame) → FP(ROI-guided)~~ — **retired**, see STUDY-01 (Stage-1 gating costs recall) | ~~≥ 60%~~ |

---

## Experiment template (copy for each run)

```
### EXP-NNN — <short title>
- Date:
- Status:            planned | running | done | discarded
- Hypothesis:        what we expect to change and why
- Baseline compared: EXP-MMM (or "none")

Config
- Model:            yolo11s-seg | yolov8s-seg | ...
- Task:             segment | detect
- Dataset version:  vN   (note fixes applied vs previous version)
- Input size:       640 | 768 | 896
- Epochs / Batch:   /
- Optimizer / LR:
- Augmentation:     flip, brightness, contrast, blur, mosaic, underwater color shift, ...
- Class weighting:  none | inverse-freq | focal
- Hardware:         local CPU/GPU | SageMaker <instance>
- Command / config: path or one-liner to reproduce

Results
- Precision:
- Recall:
- mAP@0.5:
- mAP@0.5:0.95:
- Per-class (P/R/AP):   # live 4-class taxonomy
    fishing_gear:
    pipe_cylinder:
    structural_fragment:
    natural_formation:
- Confusion matrix:  (link to runs/EXP-NNN/confusion_matrix.png)
- Inference latency:  full_frame __ ms | roi_guided __ ms
- FP reduction (Stage 1): __%

Error analysis
- False negatives grouped by cause (low-vis / occlusion / small object / blur / domain / annotation):
- False positives (natural formation misclassified, shadows, rock clusters):
- Worst class + likely reason:

Decision & next
- Keep / revert / iterate:
- Next experiment (EXP-___): the single change to try next, and why
```

---

## Planned experiment ladder

**Dataset v1 is ready** (`DATASET/03_yolo_ready_dataset_v1/data.yaml`, leakage-free, 4-class).
These are queued, not run:

Reprioritised 2026-09-22 after the deep error analysis (EXP-001 log): fishing_gear is a
small-object sonar problem; optical debris is a total miss but off-domain; resolution + sonar
focus first, aug-ablation demoted.

| ID | Hypothesis | Key change |
|---|---|---|
| ✅ EXP-001 | Baseline (**done** — 40 ep) | YOLO11s detect, v1, 640, sonar aug → mAP@0.5 0.822 (aggregate), sonar fishing_gear R 0.47 |
| **EXP-002** (next) | **Higher resolution recovers small fishing_gear** (91% of misses <10% frame; ceiling ~0.78) | `--imgsz 1280 --batch 8` (else = EXP-001). Cheapest, highest-EV lever. |
| EXP-003 | **Sonar-only** training raises the target-domain numbers + removes the optical→natural_formation shortcut (optical debris is 0% and off-product) | filter to crabpot/uatd/mpulse/seabed/shipwreck |
| EXP-004 | **Tiled train+infer** (SAHI-style slicing) beats one large frame for tiny targets — doubles as Stage-1's reframed tiling role | slice → detect per tile → merge |
| EXP-005 | Oversampling fishing_gear + small-object aug (copy_paste>0, scale-up mosaic) lifts recall | minority oversample + aug |
| EXP-006 | Aug ablation: are the ~62% baked-in v0 augs helping or just doubling online aug? | originals-only vs baked-aug (demoted) |

**Already banked (no retrain):** per-class confidence thresholds in `infer.py` — fishing_gear
recall 0.47→0.69 at conf 0.10. Report metrics **split sonar vs optical**, not just aggregate.

Promote each into the log below with full results as it runs.

---

## Experiment log

### STUDY-12 — Counterfactual: does OpenCV 5 output change what the agent *does*?

- Tool: `python -m src.agentic.study_causal [--frames …/v1/test/images --limit 80]` →
  `docs/causal_trace.md`. Same frames, synthetic track (one recording = one line), model and
  calibration in every arm; analyst budget 2 min. Arms: **full** · **no_stage1** (Stage-1 geometry
  withheld from Prove/Act; detections and tiers identical) · **no_shadow**. No re-look arm: EXP-001's
  policy spends no re-looks (STUDY-07).

| downstream decision (v1 test, 80 unique frames, 264 hazards) | no_stage1 | no_shadow |
|---|---|---|
| tier flips · queue order (Kendall τ) · budget picks changed | 0 · 1.0 · 0 | 0 · 1.0 · 0 |
| geotag shift, median / max | **8.6 / 15.6 m** | 0 |
| pins pushed outside their own stated error circle | **193 of 264** | 0 |
| re-survey passes with identical targets | **50 of 63** | 63 of 63 |
| inspection stops moved (route length) | 2 (5883.6 → 5919.4 m) | 0 |

  Shipped samples (8 frames, 46 hazards): median shift 6.6 m, 31 of 46 pins outside their circle,
  10 inspection stops moved, 5 of 6 passes identical.
- **Reading.** Stage 1 never changes a tier (it is not a gate) but it decides **where the boat goes**:
  without it most pins would sit outside their own error circle, and a fifth of the re-survey passes
  would regroup. The shadow changes **no** decision — by design it is evidence for the human card.
- A first run laid every recording on one synthetic line: same-numbered chunks of different
  recordings shared a fix and 11 spurious repeat-sighting merges appeared without Stage 1. Fixed in
  the study (parallel lines 500 m apart) before reading any number; the product's demo track is
  single-recording.

### STUDY-11b — Seam inference across chunk boundaries (NEGATIVE — not shipped)

- Tool: `python -m src.detection.study_seam` → `docs/seam_inference.md`. Windows straddling each chunk
  boundary (`seam.py`), mapped back and merged with class-aware NMS; unique frames, IoU ≥ 0.3, paired
  frame-bootstrap 95% CI.
- Recall ceiling unchanged on both unseen splits (0.724 → 0.724 calibration; 0.862 → 0.862
  verification); AP@0.3 slightly **lower** (−0.018, −0.012 [−0.024, −0.001]); +0.8–1.3 inferences/frame.
- Premise checked: chunk k's right edge continues into chunk k+1's left edge (seam discontinuity 1.56
  vs 2.52 reversed), so `stitch.py` holds. The edge cuts in these Roboflow copies are mostly
  augmentation crops → re-test on v2b's un-augmented test frames with EXP-002.

### STUDY-11a — Failure gallery: why EXP-001 misses pots (measured, not assumed)

- Tool: `python -m src.detection.failure_gallery` → `docs/failure_gallery.md` (verification split,
  92 unique frames, 138 pots, floor 0.05, match IoU ≥ 0.3).
- **19 pots (14%) never reach a human.** Compared with the 119 found pots (medians / shares):

| | missed | found |
|---|--:|--:|
| **touches the frame edge** (cut by the chunk boundary) | **32%** | 11% |
| local contrast (box ÷ ring) | 1.17 | 1.34 |
| slant range (row px) | 379 | 322.5 |
| size (px) | 52 | 36 |

- **Reading.** The largest single failure mode is **objects cut by the chunk boundary**; misses are
  also lower-contrast and farther in range. They are *not* smaller on this split — the "small
  object" explanation (EXP-001 error analysis on v1) does not carry over. Next: seam inference across
  chunk boundaries (STUDY-11b) and range-gain-normalised training for EXP-002.

### STUDY-10 — Label-noise audit: how many "false alarms" are unlabelled real objects? — SET UP

- Tool: `python -m src.detection.fp_audit build` + the **Audit** tab (blind, catch trials).
- Band: all 60 EXP-001 false alarms with conf ≥ 0.176 on v1 val (Rec19); 66 TPs in the band →
  raw precision 0.52. 15 catch trials (known pots).
- Output: audited precision (exact in the band), FP taxonomy, catch accuracy, κ.
- Result: _pending — needs 2+ human auditors (~15 min each). Report it whatever it shows._

### STUDY-09 — Analyst effort: does DEPTH save review minutes? (curve + timed study)

**Why.** Review X-2: the impact claim ("minutes per survey-hour") was never measured.
**Setup.** `python -m src.agentic.effort` on the EXP-001 verification split (92 unique frames,
138 pots — every frame has ≥ 1 pot); `docs/effort_curve.md`. Timings ASSUMED until the Study-mode
sessions exist (`docs/user_study.md`).

| at the recall promise (≥ 65%) | manual (every frame) | DEPTH queue |
|---|--:|--:|
| minutes for 92 frames (20 s/frame, 8 s/card) | 19.3 | 19.5 (146 cards) |
| per survey-hour (~169 frames) | 35.5 | 35.7 |
| **break-even card time** | — | **7.95 s** |
| agent forecast vs actual real pots | — | 84.5 vs 90 |

**Reading.** With these assumptions DEPTH merely ties a *perfect* manual reviewer; the win requires
card review under ~8 s and/or imperfect manual recall — both are exactly what the study measures.
DEPTH's order equals confidence order for EXP-001 (STUDY-07); its measurable extras are the promise
(when to stop) and a forecast that held (slightly conservative). Status: **study pending** (needs
3+ people). A higher recall ceiling (EXP-002) moves the promise and the card count.

### EXP-002 — YOLO11s @1024 on v2b + tiles (+ sonar-aware copy-paste variant EXP-002p) — READY TO RUN

- Status: **kit built and round-trip tested; GPU run pending (Kaggle T4)** — `docs/exp002_kaggle.md`.
- Why: EXP-001's recall ceiling (0.72 on its calibration recording) caps the recall promise at 65%;
  triage cannot recover unproposed pots (STUDY-07).
- Recipe: v2b (2 classes, sonar only, deduped; `best.pt` selected on held-out recordings Rec10/12/16),
  full frames + 2×2 tiles, 1024 px, 30 epochs, patience 8, cosine LR, sonar-aware augmentation.
  EXP-002p adds 600 sonar-aware copy-paste frames. Choose between them **on validation**.
- Evaluation (automatic via `onboard_model`): guarantees fit on val, verified once on v2b test;
  deploy-faithful AP on v2b test (unique frames), official 398 split (GhostVision F1 0.71–0.73), and
  cross-sonar `test_xsonar`.
- Targets: ceiling ≥ 0.85 · crab-pot AP@0.5 ≥ 0.60 · F1 within 0.05 of GhostVision · promise ≥ 90%.
- Result: _pending — record every number here, including misses._


### STUDY-08 — Stage-1 sonar canonicalisation: measured geometry, and does it help the detector?

**Why.** Review I-4/X-1: after STUDY-01, Stage 1 existed only to be timed for COOL. Give it a real
job in the product path — turn a frame into **measured sonar geometry** — and test, not assume,
whether its image normalisation helps the deployed detector.

**Setup.** `src/cv_pipeline/canonical.py` (pure CPU OpenCV/NumPy): palette → luminance; canonical
orientation by source rule; **bottom tracking** (first dip-then-rise after the transducer ring-down
in a robust row profile, then per ping in a window, median + mean along track); slant → ground
range (`cv2.remap`); range-gain normalisation; water-column mask. `python -m
src.cv_pipeline.study_canonical` → `docs/stage1_canonical.md`.
Part A (no labels): **un-augmented originals only** — frame keys with exactly one copy in the raw
HF archive (107 of 1,214; Roboflow crop/zoom copies change the pixel scale, so an altitude in px is
only meaningful on originals — found when the first run on "unique" v1 frames gave port/starboard
gaps of ~10 px driven by augmented train copies). Part B: EXP-001 on its 158 unseen unique sonograms.

**Result.**
| check | value |
|---|--:|
| bottom tracked | 104 / 107 originals (97%); 3 with no water-column step → *not measured* |
| altitude | median 29.6 px of 640 (5–95%: 11.5–33.7) |
| **port vs starboard, same pings (20 pairs)** | **median 1.6 px apart, 70% within 2 px** (null, different recordings: 5.3 px / 24%) |
| consecutive chunks, one channel (34) | median 1.2 px change |
| synthetic sonograms, known altitude (unit tests) | within 1.5 px; follows a sloping seabed |
| Stage-1 cost | ~1.3 ms luminance + ~4 ms bottom track per 640² frame (laptop, under load) |
| detector input `gain` vs `raw` (AP@0.3) | 0.514 vs 0.521, Δ −0.006 (95% CI −0.033..+0.019) — **no gain** |
| detector input `gain` vs `raw` (recall ceiling) | 0.790 vs 0.794, Δ −0.003 (CI −0.026..+0.020) |
| water column (150 tracked frames) | 0 of 259 pots, 0 of 534 candidates lie in it |

**Decision.** Stage 1 is now **in the product path as a measurement stage**: the tracked altitude
feeds the shadow's relative height (h/H against a *measured* altitude instead of the slant-range
proxy), ground range feeds the geotag, the per-ping position gives each object its own along-track
fix, and a `water_column_check` step is on every card. The detector keeps **`raw`** input
(`calibration.json` `detector.input`) — range-gain normalisation neither helps nor hurts a model
trained on raw frames; it stays available (and benchmarked) for EXP-002. The water-column check
never fired on this data (the column is only ~10–35 px); it is a safety net for deeper/other
sonars, not a filter. Metres of height still need a range scale (px → m) that these frames don't
carry — relative height only.


### STUDY-07 — Guaranteed tiers (conformal / LTT) + agent vs "sort by confidence" on unseen data

**Why.** Review C-3/I-5/M-1: the agent's thresholds (0.40/0.60/0.15/0.12) were tuned *and* reported
on the same test frames, and it was never compared with the simple baseline at equal recall/effort.

**Setup.** `python -m src.agentic.calibrate`. EXP-001 via `cv2.dnn`, detector floor 0.05, one-to-one
matching at IoU ≥ 0.3, **unique frames only**. Calibration = v1 val crab-pot sonograms (66 frames,
Rec19, 134 pots); verification = v1 test sonograms (92 frames, Rec3/4/6/10, 138 pots) — the only
sonograms EXP-001 never trained on. Recall promise: Clopper–Pearson UCB on the miss rate ≤ α (LTT,
fixed sequence); precision promise: CP LCB ≥ 0.85 (n ≥ 15), δ = 0.05. Seven scoring policies
(confidence; single/mosaic re-look × lift/CLAHE/demote). Full tables: `docs/calibration_exp001.md`.

**Result.**
| | calibration | verification |
|---|--:|--:|
| recall ceiling (any proposal ≥ 0.05) | 0.724 | 0.862 |
| **recall promise (95%)** | **≥ 65%** (90% not achievable) | **held** — 0.862 (LCB 0.805) |
| precision promise ≥ 85% | not achievable by any policy (best LCB 0.62–0.66) | — |
| AUC TP vs FP — confidence | 0.742 | **0.764** |
| AUC — best re-look variant | 0.756 (single demote, +0.014, CI −0.013..+0.042) | 0.710 |
| old test-tuned rules, CONFIRMED precision | 0.588 | 0.578 (claimed 0.737) |
| thin-line shadow AUC TP vs FP | — | 0.599 |

**Decision.** Ship `baseline_conf` tiers: τ_review = 0.05, no τ_confirm (nothing auto-confirmed),
P(pot) bins for queue order; the value-of-information agent spends **no re-look inference** on
tiering (1 inference/frame). A first tie-break (AUC at 2 d.p.) picked the demote variant, which then
lost on verification — the rule now requires a significant paired-bootstrap gain, and the report
discloses that the rule change followed that test result. **The recall ceiling, not triage, is the
binding constraint → EXP-002 (1024 px, tiles, v2b) is the lever.**

### STUDY-06 — Orientation by source rule + thin-line shadow re-measured (honesty fixes)

**Why.** Review §3.4: the auto-nadir guess picked the correct edge on only 26% of real PINGMapper
sonograms (whose nadir is always the top edge), so the shadow search and the geotag range pointed the
wrong way on ~3 of 4 frames; heights used an assumed 10 m altitude; `match_other_pass` "corroborated"
detections across adjacent chunks that image different seabed.

**Setup.** Viewed real frames from every crab-pot source (`runs/_look/`): `Rec*_wcp_ss_{port,star}`
sonograms → water column at the top, shadows are **2–4 px vertical lines** below each pot; `Contact_*_sslo`
(orange) → horizontal range, sonar side not recoverable from the crop; `BC_POST`/`baycove`/`TI` →
rotated mosaics. Shadow statistic measured on **952 labelled pots vs 952 same-size random seabed boxes**
(v1 val wcp frames, nadir forced top).

| Shadow measure | AUC pot vs random seabed | CLEAR rate pots / seabed |
|---|--:|--:|
| old wide-column mean (for reference, review pilot) | ~0.62 (their thin-line) | — |
| thin darkest line, naive contrast | 0.556 | WEAK on **30%** of empty seabed |
| **thin darkest line, flank-referenced (shipped)** | **0.571** | ~0.18 / ~0.13 |

**Result.**
1. Orientation now comes from `src/cv_pipeline/orientation.py`: PINGMapper sonograms → `top` by rule
   (100% correct by construction on that source); everything else **unknown** → shadow *not measured*,
   geotag falls back to the boat fix with a swath-wide error. `auto` survives only as a labelled diagnostic.
2. The shadow separates real pots from seabed only weakly (AUC 0.57) — it remains **evidence shown on
   the card, never a filter** (consistent with STUDY-03/04). The unbiased reference stops speckle from
   being reported as a shadow.
3. Height is **relative** (`h/H`, % of sonar altitude); metres only when the altitude is measured.
4. `match_other_pass` (+0.10 score boost, "also seen in an overlapping pass") **removed**; replaced by
   chunk-boundary **stitching** (same range, adjacent chunks, touching edges → one hazard; keeps the
   stronger sighting; never changes a verdict).

**Decision.** Ship. Tagline changes from "proven by physics" to "every find comes with evidence".

### STUDY-05 — Deploy-faithful evaluation: val-tuned thresholds, per-domain tables, bootstrap CIs
- Date: 2026-09-24 · Status: done · Code: `src/detection/evaluate.py`; reproduce:
  `python -m src.detection.evaluate --bootstrap 1000`. Artifacts: `docs/eval_exp001.md`,
  `runs/EXP-001/eval/metrics.json`.
- Hypothesis / purpose: the 0.822 headline is **domain-inflated** and its per-class thresholds were
  picked by looking at the **test** set (WINNING_REPORT #10/#11/#13). Re-evaluate the **exact shipped
  ONNX** through the **`cv2.dnn`** path (not a torch re-run) with (i) thresholds tuned on **val**, test
  scored **once**; (ii) **per-sensor & per-source** tables; (iii) **bootstrap 95% CIs**. AP is an
  independent Pascal-VOC all-points implementation, so agreement with ultralytics is a cross-check.

Results — **the aggregate is confirmed AND shown to be misleading:**
- **Aggregate mAP@0.5 = 0.827** — independently reproduces the ultralytics 0.822 (validates both the
  training run and this evaluator; small delta = 101-pt interp vs VOC all-points).
- **Per-class (val-tuned t, test scored once):** `fishing_gear` AP **0.463** [CI 0.42–0.51], R 0.587;
  `pipe_cylinder` 0.945 [0.87–1.00] (only 65 boxes → wide CI, high-variance); `structural_fragment`
  0.919 [0.89–0.94]; `natural_formation` 0.981 [0.97–0.99].
- **Per-sensor proves the inflation:** `fishing_gear` on **SONAR** AP **0.473 / R 0.599** (the honest
  product figure); `natural_formation` is **optical** (267 optical boxes AP 0.990 vs its 2 sonar boxes
  0.000); **optical debris = total miss** (`fishing_gear` 0.000/19, `structural_fragment` 0.000/25) —
  the model maps optical→natural_formation, exactly as EXP-001 error analysis found.
- **Per-source:** crabpot `fishing_gear` 0.473; `shipwreck` struct weak **0.347** (R 0.25); `uatd`
  pipe/struct 0.950/0.987; `vid` optical debris 0.000.
- **Val-tuned thresholds** (never touched test): fishing **0.15** / pipe 0.40 / struct 0.25 /
  natural 0.50 — vs the shipped `PER_CLASS_CONF` fishing 0.10.
- **GhostVision head-to-head SKIPPED (honest):** EXP-001 trained on v1, which mixed the official
  crab-pot test frames into training → a leaked number. Gated behind `--leakage-free`; re-run on
  EXP-002 (v2 respects the split): `--official-split .../official_crabpot_test.txt --leakage-free`.
- **Closes** WINNING_REPORT/NEEDTOFIX **#10** (val-tuned, test-once), **#11** (per-domain tables),
  **#13** (bootstrap CIs); **scaffolds #24** (GhostVision, ready for EXP-002).
- Next: EXP-002 on v2 → re-run evaluate.py with `--leakage-free` for the head-to-head.

### STUDY-04 — Adaptive triage controller + a second CONFIRMED path (high detector confidence)
- Date: 2026-09-23 · Status: done · Code: `src/agentic/` (`policy.py`, `agent.py`, `tools.py`,
  `pipeline.py`, `calibrate.py`); reproduce: `python -m src.agentic.calibrate --frames 160 --out runs/prove`
- Hypothesis: STUDY-03 auto-confirms via **one** signal (re-look ≥ 0.40). Two ideas could safely
  confirm *more* true pots (→ a non-empty recovery route in the demo, higher task success): (a) a
  **corroboration** rule "high detector conf **and** a CLEAR acoustic shadow at a plausible height",
  and (b) plain **high detector confidence**. Mine the real test split to see which — if either —
  holds precision, then bake only the honest one into the policy.

Results — **(a) is rejected, (b) is a clean win; the controller is made genuinely adaptive:**
- **Shadow-corroboration fails (hypothesis overturned).** On 160 crab-pot test frames / 461
  fishing_gear detections (raw precision 0.599), requiring a CLEAR shadow *lowers* precision:
  `conf≥0.55 & CLEAR` → **0.40** vs `conf≥0.55` alone → **0.69**. The committed run confirms the
  mechanism: **CLEAR-shadow rate is 14.5% on TP vs 14.6% on FP** — statistically identical, i.e.
  non-discriminative (reconfirms STUDY-03 with a cleaner statistic). Shadow stays **evidence-shown +
  height**, never a gate.
- **High detector confidence is a strong, independent CONFIRMED path.** `conf ≥ 0.60` → precision
  **0.826** (n=23). The **union** used by the new policy — `relook ≥ 0.40` **OR** `conf ≥ 0.60` —
  gives CONFIRMED precision **0.737 at 30% recall-share**, beating the STUDY-03 re-look-only tier
  (**0.713 / 26%**) on *both* axes. By path (committed calibrate): re-look **0.71** (n=91),
  high-confidence-only **0.92** (n=13), both **0.70** (n=10). REJECT stays recall-safe: **91% of true
  pots retained** in CONFIRMED+REVIEW after deprioritising (never deleting) REJECTED.
- **Decide is now an adaptive escalation controller** (`agent.py`), not a fixed 3-call sequence:
  re-look first → shadow + `estimate_height` (evidence for the card) → **escalate to an enhanced
  CLAHE re-look only when still uncertain** (skipped once confident — no wasted second inference) →
  triage. Different candidates take different tool paths and stop at different depths; the trace
  records the whole route **and which CONFIRMED path won** (`confirm_path` ∈ relook / high_confidence
  / both). Two tools added: `estimate_height` (named, traced) and `match_other_pass`.
- **Cross-pass corroboration** (`match_other_pass`, survey-level, `pipeline._corroborate`): a
  candidate that re-appears in an overlapping pass (same recording+channel, adjacent ping, same
  across-track position) gets a trace step + evidence note + a review-ranking boost. Deliberately
  **non-gating** — it never changes a verdict, so the CONFIRMED tier stays exactly the calibrated set.
- **Effect on the live demo:** the bundled 5-sample survey went **0 confirmed / 3 review → 1 confirmed
  / 2 review** (a conf-0.68 pot with a clear h~0.7 m shadow, previously stuck in REVIEW, now
  auto-confirmed via the high-confidence path), producing a real recovery route.

Error analysis
- The two confirm paths are complementary: re-look rescues *low-confidence* true pots that persist on
  zoom; the confidence path trusts the detector when it is already decisive. Their overlap ("both",
  n=10) is small, so the union genuinely adds recall.
- `match_other_pass` is a filename/geometry heuristic (no per-frame slant model here), hence kept
  non-gating and off the calibrated metric; it aids the human review queue and demonstrates the agent
  *considering* a second pass. A physically-rigorous version needs the slant-range geometry from 5.5.

Decision & next
- **Ship the union policy** (`TriageConfig.confirm_conf=0.60`) + the adaptive controller + the two
  tools + non-gating cross-pass. 12/12 agent tests, full suite green.
- Next: deploy on Graviton + COOL (the live link); EXP-002 (imgsz retrain) to raise the underlying
  recall the agent triages; a physically-grounded `match_other_pass` once slant-range Stage 1 lands.

### STUDY-03 — See→Prove→Decide: shadow is NOT a gate; re-look persistence IS (agentic triage)
- Date: 2026-09-23 · Status: done · Code: `src/agentic/` (`shadow.py`, `perception.py`,
  `evidence.py`, `policy.py`, `calibrate.py`); reproduce: `python -m src.agentic.calibrate --frames 110`
- Hypothesis: the acoustic **shadow** (bright echo + dark far-range shadow — the side-scan debris
  cue STUDY-01 said Stage 1 ignores) proves a detection is a real object, so shadow-gating the hot
  detector (fishing_gear conf 0.10) kills its false positives while keeping recall.

Results — **the shadow premise fails on this data; re-look persistence is the real discriminator:**
- **Shadow does NOT separate the detector's TP from FP.** On 120 crab-pot test frames, YOLO's true
  and false fishing_gear detections have **identical echo brightness** (TP echo 2.08× bg vs FP 2.19×)
  and the region below the echo is on average **no darker than the flanks for either** (median shadow
  contrast ≈ 0, negative for both). A shadow gate *lowers* precision (0.59 → 0.31–0.50). A clear,
  measurable shadow exists only on a **minority of larger / higher-relief objects** (e.g. the pipe-like
  return in `runs/prove/`); most small low-relief crab-pots in heavy speckle cast none. Same class of
  finding as STUDY-01: the intuitive physical cue has no per-object discriminative power *here*.
- **Re-look persistence works.** Zooming 2.5× around a candidate and re-detecting, then keeping only
  detections that re-fire strongly, lifts precision **0.59 → 0.77**; raising raw confidence barely
  moves it (0.59 → 0.63). The agent's "squint and look again" is the discriminator, not the shadow.
- **Calibration on 110 frames / 309 detections** (`calibrate.py`, triage in `policy.py`):
  raw hot precision **0.602** → **CONFIRMED** tier (`relook ≥ 0.40`) precision **0.774** (22% of true
  pots), **REVIEW** 0.570 (68%, the ranked human queue), **REJECTED** 0.545 (10%, deprioritised).
  Deprioritising REJECTED retains **90% of true pots** and removes 15 false alarms.

Error analysis
- FPs are bright speckle patches indistinguishable from small pots by brightness → echo-based cues
  (incl. the detector's own confidence) can't separate them. Multi-scale *consistency* (re-look) can:
  a physical object re-fires at higher effective resolution, a speckle artefact often does not.
- Shadow is real and visible on a minority; kept as **evidence shown where it exists** (+ a height
  estimate `h = altitude·Ls/(range+Ls)`, pixel scale cancels) — compelling for the card/video — but
  **never a silent gate**.

Decision & next
- **Prove = multi-evidence** (re-look persistence [primary] + acoustic shadow where present + echo +
  height), fused into an `evidence_score` for ranking. **Decide = 3-tier triage**
  CONFIRMED / REVIEW / REJECTED, **recall-safe** (REJECTED is retained-for-audit + deprioritised,
  never deleted — no potential hazard is hidden from the human). Deterministic, logged, human-gated —
  the Agentic-Vision substance. Thresholds pre-registered in `policy.TriageConfig`, tuned on this
  split once (not per-frame).
- Next: **DECIDE** agent (`agent.py`, tool-call trace incl. a re-look that flips a decision) → **ACT**
  (`mission.py` cleanup route + `geo.py` honest geotag) → FastAPI + dashboard.

### STUDY-02 — Tiled (SAHI-style) inference for small objects (MODEST, no retrain)
- Date: 2026-09-22 · Status: done · Tool: `src/detection/tiled_infer.py`
- Hypothesis: slicing 640² frames into an overlapping 2×2 grid (each tile upscaled to 640 by
  the detector) makes small crab-pots ~2× larger to `best.onnx`, lifting recall without retrain.
- Result (60 sonar frames, conf 0.25, IoU 0.5): fishing_gear recall **0.565 → 0.645** (+8pts);
  but precision falls (fishing 0.52→0.48, pipe 1.0→0.8, **structural 0.96→0.69** — large frags
  split across tile seams and double-count). Recall for the larger classes unchanged.
- Decision: **keep as a secondary tool, not the default.** Net gain is real but small and costs
  precision on big objects; the retrain (EXP-002 imgsz 1024) is the better fix. Tiling is also a
  legitimate COOL workload (slice/resize on Graviton). Revisit selective/large-object-aware tiling.

### STUDY-01 — Stage-1 classical-CV ROI-gating (NEGATIVE RESULT)
- Date:              2026-09-22
- Status:            done
- Hypothesis:        Stage-1 classical CV (denoise → adaptive threshold → contours → geometry
                     filter) proposes ROIs that (a) cover real debris and (b) suppress background,
                     so gating the YOLO detector to Stage-1 ROIs should cut false positives ≥60%
                     while keeping recall.
- Baseline compared: EXP-001 full-frame detector
- Method:           `src/detection/ablation_fp.py` (full_frame vs roi_guided, gate IoU 0.10,
                    match IoU 0.5) + `src/cv_pipeline/tune_coverage.py` (geometry-filter sweep),
                    300-frame strided test sample; background-ROI probe on 80 empty-label frames.

Results — **the premise fails on this data:**
- Stage-1 **GT coverage 24.7%** with default config (104/421 GT boxes fall in any ROI) →
  hard recall ceiling for roi_guided.
- roi_guided vs full_frame: recall **0.708 → 0.147** (only 20.8% of TPs kept) for FP 124 → 59
  (52.4% "reduction"). The FP drop is an artefact of discarding true detections.
- Geometry-filter sweep: even with **all filters off** ("ceiling" config) coverage tops out at
  **72.4%** — at **141 ROIs/frame**. ~28% of debris does not segment at all (low-contrast,
  blends into seafloor). `min_area_frac` was the dominant lever (0.30 → 0.63 coverage).
- **Discriminative-power probe:** on background (empty-seafloor) frames Stage 1 emits **60.2
  ROIs/frame** vs **8.0** on frames with objects; only **1%** of background frames emit zero.
  Stage 1 fires *more* on clutter than on debris.

Error analysis
- Stage 1 thresholds on **brightness + blob geometry**; the real side-scan-sonar debris cue is
  the **highlight + acoustic-shadow pair**, which this pipeline ignores. Rippled seafloor / sand
  waves / speckle generate abundant bright blobs → clutter dominates.
- Classical contour CV therefore works neither as a per-box recall gate (72% ceiling, huge ROI
  counts) nor as a frame-level triage (louder on empty background).

Decision & next
- **Retire the ROI-gating / ≥60% FP-reduction target.** YOLO (EXP-001) is the detector.
- **Reframe Stage 1** as the CPU **sonar-preprocessing workload** benchmarked for the COOL award
  (Graviton vs x86) — its ops (resize, adaptive threshold, contours) are still the COOL sweet-spot;
  it just no longer feeds ROIs to Stage 2. This negative result + evidence is a submission asset.
- Shadow-aware Stage 1 (highlight+shadow detection) is a possible future upgrade, deferred.

### EXP-001 — Baseline YOLO11s (detect) on dataset v1
- Date:              2026-09-21
- Status:            done
- Hypothesis:        Establish an honest first baseline on the clean v1 split — where do we
                     stand against the targets (mAP@0.5 ≥ 0.70, P ≥ 0.80, R ≥ 0.70) with a
                     stock YOLO11s and no per-class tricks, and which class is the weak link?
- Baseline compared: none (this is the reference point for EXP-002+)

Config
- Model:            yolo11s.pt (detect — **not** seg; 9.41 M params, 21.4 GFLOPs, fused)
- Task:             detect
- Dataset version:  v1  (`03_yolo_ready_dataset_v1`, leakage-free, 4-class, split by clip)
- Input size:       640
- Epochs / Batch:   **40** (`args.yaml` — not 100; patience 20 not triggered, LR schedule
                    completed at 40) / 16
- Optimizer / LR:   auto optimizer, lr0 0.01 → lrf 0.01 (linear), momentum 0.937, wd 5e-4
- Extra aug (args): copy_paste 0.0, mixup 0.0, erasing 0.4, auto_augment randaugment
- Augmentation:     sonar-aware — hsv_h=0, hsv_s=0, hsv_v=0.2; degrees=0, flipud=0,
                    fliplr=0.5; translate=0.1, scale=0.5; mosaic=1.0, close_mosaic=10
- Class weighting:  none (inverse-freq weights printed for reference only, not applied)
- Hardware:         Kaggle T4 (16 GB), GPU
- Command / config: `python src/detection/train.py --model yolo11s.pt --epochs 100 --batch 16 --name EXP-001`

Results  (test split — 1,276 images / 1,787 instances; all 4 targets met in aggregate ✅)
- Precision:        0.808   (≥ 0.80 ✅)
- Recall:           0.800   (≥ 0.70 ✅)
- mAP@0.5:          0.822   (≥ 0.70 ✅)
- mAP@0.5:0.95:     0.514   (≥ 0.45 ✅)
- Per-class (imgs / inst — P / R / AP@0.5 / AP@0.5:0.95):
    fishing_gear:         537 / 952 — 0.535 / 0.371 / 0.450 / 0.194   ⬅ worst; mission-critical
    pipe_cylinder:         65 /  65 — 0.909 / 0.954 / 0.940 / 0.523   (high-variance, 65 boxes)
    structural_fragment:  402 / 501 — 0.951 / 0.897 / 0.923 / 0.554
    natural_formation:    214 / 269 — 0.835 / 0.978 / 0.973 / 0.784
- Confusion matrix:  runs/EXP-001/confusion_matrix.png (in MyDrive/EXP-001_results)
- Inference latency:  T4 — 1.1 ms preprocess + 9.8 ms inference + 0.6 ms postprocess ≈ 11.5 ms/frame (~87 FPS)
- FP reduction (Stage 1): n/a — full-frame detector only; Stage-1→Stage-2 ROI study is EXP-later

Error analysis
- **`fishing_gear` is the failure mode, not the minority class.** Recall 0.371 means we miss
  ~63% of ghost-net/rope instances — the single most dangerous class (per experiments.md §Conventions).
  Its AP@0.5 (0.45) is the only class below the 0.70 target; aggregate metrics pass *only because*
  the other three classes are strong. This is the headline problem to fix.
- **Imbalance did NOT hurt the rare class.** Counter to blocker B4's worry, `pipe_cylinder`
  (rarest, 4% of boxes) scored AP@0.5 0.94 — but on just 65 test boxes, so treat as high-variance.
  The imbalance lever should target `fishing_gear` *recall*, not `pipe_cylinder`.
- **Why fishing_gear is hard — CONFIRMED (2026-09-22, `src/detection/fn_gallery.py`):** it's a
  **small-object detectability** problem, not confusion or the merged taxonomy.
    - Normalised confusion matrix: true fishing_gear → predicted **background 0.63** (missed
      outright), ~0.00 into any other class. And true **background → fishing_gear 0.79** — it is
      also the dominant false-positive sink (matches P 0.535). Hard both ways.
    - FN gallery (test split, conf 0.25): **517/952 (54%)** GT boxes missed; **91% of the misses
      are < 0.10 of frame width** (median long side **0.059**, 31% < 0.05). Eyeballed frames:
      small low-contrast bright returns embedded in heavy speckle/sand-wave clutter.
    - Note: some crabpot samples carry **baked-in rotation borders** from v0 augmentation
      (black triangular corners) — a possible extra confound; flag for a data spot-check.
- **Implication for next experiments:** the lever is **effective resolution** (higher `imgsz`
  and/or tiling so small targets are larger to the detector) + small-object augmentation, NOT
  the aug-on/off ablation. Reprioritised the ladder accordingly.

- **DEEP ANALYSIS (2026-09-22, `src/detection/error_analysis.py`, full 1,276-img test set).**
  Four findings the aggregate mAP 0.822 hides:
  1. **Aggregate is inflated by domain segregation.** `natural_formation` (R 0.993) lives *only*
     in optical (267 optical vs 2 sonar GT boxes) — trivially separable, it pads the mean.
     `fishing_gear`/`pipe`/`structural` are ~all sonar. Honest per-class @conf 0.25:
     fishing_gear P0.50/R0.46, pipe P0.91/R0.95, structural P0.96/R0.89, natural P0.84/R0.99.
  2. **Optical debris is a total miss** (not the product domain): fishing_gear **0/19**,
     structural **0/25** on optical frames — the model maps "optical → natural_formation".
     Product is side-scan **sonar**; sonar-only recall is the number that matters.
  3. **Per-source:** fishing_gear in test = essentially all **crab-pot sonar** (R 0.466);
     `uatd` pipe/structural excellent (0.95/0.97); `shipwreck` structural weak (0.25 — large,
     ambiguous fragments); optical `vid`/JAMSTEC rope = 0.0.
  4. **Free recall lever — lower the fishing_gear threshold.** Recall vs conf (sonar):
     **@0.25 0.466 → @0.10 0.689 → @0.05 0.781**; other classes flat. Implemented as
     **per-class confidence thresholds** in `infer.py` (`PER_CLASS_CONF`: fishing_gear 0.10,
     rest 0.25) — nearly doubles fishing_gear recall at inference, no retrain. Precision falls
     (acceptable for human-review hazard triage). The ~0.78 ceiling still needs EXP-002 resolution.
  5. **Training dynamics (`results.csv`):** val mAP50 peaked **epoch 16 (0.704)** then declined to
     0.663 by ep40 while train loss kept falling — **overfitting after ~ep20**, and precision was
     traded up for recall down late. So `best.pt` ≈ ep25; extra epochs won't help without more
     regularisation/data. (Also: val is optical-heavy → val 0.70 < test 0.82; not a real gain.)

Decision & next
- **Keep** as the reference baseline. It clears every aggregate target, so v1 + the sonar-aware
  aug recipe is sound; the open problem is `fishing_gear` recall, not overall capacity.
- Next experiment (**EXP-002**): before touching the class problem, isolate whether the ~62%
  baked-in augmentation is helping or just doubling the online aug — originals-only train vs. this
  run (per `docs/colab_baseline.md` §7). Then attack `fishing_gear` recall via oversampling /
  focal (EXP-004) once the FN cause is confirmed from the confusion matrix.

<!-- template retained below for the next run:
### EXP-NNN — <short title>  (copy from the template section above) -->
