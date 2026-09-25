# Agentic Vision entry — See · Prove · Decide · Act

**Marine-debris (ghost-gear) detection in side-scan sonar · Team Syndicate · DEPTH**

> A chatbot that *explains* results does not qualify. DEPTH qualifies because **the picture result
> changes the agent's next step**, under a **stated objective**: meet two calibrated promises — how
> many pots reach a human, and how many auto-confirmed finds are real — while spending the least
> compute and the fewest human minutes. Every decision is logged, every tier is calibrated on held-out
> data, and a human approves every action.

Code: [`src/agentic/`](../src/agentic) · Live demo: the web app (`webui/`, served by
`src/dashboard/app.py`) · Evidence: [`docs/calibration_exp001.md`](calibration_exp001.md)
(regenerate with `python -m src.agentic.calibrate`).

---

## 1. The loop

```mermaid
flowchart LR
  SEE["SEE<br/>YOLO11 · OpenCV 5 cv2.dnn<br/>floor 0.05 · class-aware NMS"] --> VOI{"value of information:<br/>can more looking change<br/>this tier or queue slot?"}
  VOI -->|"conf < τ_review"| LOW["LOW-RISK<br/>no compute · kept for audit"]
  VOI -->|"no"| TIER
  VOI -->|"yes"| RL["re-look<br/>(single or 4-crop mosaic)<br/>+ optional CLAHE pass"]
  RL --> TIER{"guaranteed tiers<br/>(calibration.json)"}
  TIER -->|"score ≥ τ_confirm"| CONF["CONFIRMED<br/>precision promise"]
  TIER -->|"else"| REV["REVIEW card<br/>ordered by calibrated P(pot)"]
  CONF --> ACT
  REV --> BUD["budget mode<br/>analyst minutes → which cards"]
  BUD --> ACT["ACT<br/>stitch chunk boundaries · recovery + inspection routes<br/>GeoJSON/GPX/KML/CSV · human approval"]
```

- **SEE** — `perception.Perceptor` runs YOLO11 through OpenCV 5 `cv2.dnn` (no torch), with
  class-aware NMS. Every threshold comes from one file, `models/<MODEL>/calibration.json`.
- **PROVE** — per candidate, *only when it can matter*: a zoom-in **re-look** (single crop, or 4 crops
  packed into one inference), a CLAHE "try harder" pass, the thin-line **acoustic shadow** and a
  **relative height** (orientation from a source rule — never guessed). Evidence is always shown on
  the card; the shadow is never a filter.
- **DECIDE** — **guaranteed tiers** (§4): CONFIRMED / REVIEW / LOW-RISK, with thresholds fit by
  Clopper–Pearson / Learn-Then-Test on held-out frames. The agent spends re-look compute only where it
  can change a tier or a REVIEW-queue position, and records the calls it chose *not* to make.
- **ACT** — chunk-boundary **stitching** (one object cut by a chunk edge = one hazard), a
  nearest-neighbour **recovery route** (CONFIRMED) and **inspection route** (the budgeted REVIEW
  cards), honest geotags, exports. Nothing is dispatched without a human.

---

## 2. The toolbox (`tools.py`)

Each tool returns `(payload, AgentStep)` — the result the agent reasons over, plus a timed,
human-readable trace entry.

| Tool | What it does | Drives the next step by… |
|---|---|---|
| `detect` | full-frame YOLO11 (`cv2.dnn`, class-aware NMS) | producing the candidate set |
| `relook_band` / `zoom_relook` / `mosaic_relook` | crop + upscale + re-detect (1 or 4 crops per inference) | lifting or vetoing the calibrated score, where it can change the outcome |
| `enhance_relook` | CLAHE contrast boost, then re-look | a "try harder" pass, only if calibration showed it helps |
| `shadow_check` | darkest 3-px shadow line vs flank-referenced background | physical evidence for the card (never a gate) |
| `estimate_height` | `h/H = Ls/(R+Ls)` — relative to sonar altitude | plausibility evidence; metres only with a *measured* altitude |
| `stitch_boundary` | same range, adjacent chunks, touching edges | counts a split object once; never changes a verdict |

The rule core (`policy.py`) is the **sole decision authority** — deterministic, reproducible,
human-gated. An LLM may only *narrate*: `brief.py` lets Claude on Amazon Bedrock write the mission
brief from a facts JSON built after every decision, and serves its text only if every number and ID
traces back to the survey (§5).

---

## 3. Autonomy — a stated objective and value-of-information tool use

`agent.ReLookAgent._decide_calibrated` asks, for every candidate, *"can more looking change this
outcome?"* (`policy.GuaranteedTiers.needs_relook`):

```
conf < τ_review                → LOW-RISK, no compute ("a re-look cannot lift it into review")
conf ≥ τ_confirm (lift mode)   → CONFIRMED, no compute ("already above τ_confirm")
score depends on the re-look   → re-look (4 crops / inference in mosaic mode) → maybe CLAHE pass
otherwise                      → REVIEW card, no compute; P(pot) from the calibration
then: shadow + relative height for the card (cheap, no inference) → decide → trace
survey: stitch boundaries → REVIEW queue by P(pot) → budget plan → routes → human gate
```

**Budget mode.** Given "the analyst has N minutes", the agent orders the REVIEW queue by calibrated
P(pot), reports how many cards fit and the expected number of real pots they contain
(`mission.plan_budget`), and routes an **inspection route** through them. Seconds-per-card is
*assumed* (8 s) until the timed user study measures it.

**What the measurements decided (EXP-001).** The calibration compared seven policies — plain detector
confidence and six re-look variants — at the same recall promise. **None beat plain confidence
significantly on the calibration split**, and on the verification split confidence ranked best (AUC
0.764 vs 0.65–0.71 for every re-look variant). So, for this model, the agent's value-of-information
rule spends **no re-look inference on tiering: 1 inference per frame** (the old ladder spent 4–7).
The re-look stays as a display-only "agent's eye" view on each card. The same machinery re-decides
automatically for EXP-002.

---

## 4. Guaranteed tiers — calibrated on validation, verified once on test

Full, regenerable report: [`calibration_exp001.md`](calibration_exp001.md). EXP-001's only unseen
crab-pot sonograms are v1 val (**66 unique frames**, calibration) and v1 test (**92 unique frames**,
verification); Roboflow copies are removed.

| Promise (95% confidence) | Requested | What EXP-001 can support | Verified on test |
|---|---|---|---|
| **Recall** — pots that reach a human (CONFIRMED+REVIEW) | ≥ 90% | **≥ 65%** — 90% is impossible: the detector never proposes 28% of calibration pots (ceiling 0.72) | **held** — 86.2% (lower bound 80.5%) |
| **Precision** — CONFIRMED finds that are real | ≥ 85% | **none** — the best any threshold supports is ~62–66% | — (nothing auto-confirmed) |

So with EXP-001 **every find goes to a human**, in P(pot) order. That is the honest consequence of the
guarantee machinery, and it is the quantitative case for EXP-002: raising the proposal ceiling is the
only lever that can raise the recall promise, and a better-separated score is the only way to earn an
auto-confirm tier.

**The old headline, re-scored on unseen data.** The previous rules (re-look ≥ 0.40 OR conf ≥ 0.60,
tuned on the test frames) claimed CONFIRMED precision 0.737. On the unseen splits they give
**0.588 (calibration) and 0.578 (verification)** — they are retired.

**Why the shadow is not a gate.** Thin-line shadow AUC, true vs false detection: **0.60** on the
verification split (random seabed vs real pots: 0.57, STUDY-06). The detector's false positives are
mostly real 3-D returns too, so a shadow proves "something stands up", not "it's a crab pot". It is
shown as evidence with a relative height, never used to accept or reject.

---

## 5. Failure handling & human control

- **Nothing is ever deleted.** LOW-RISK = "below τ_review — retained for audit". Its size is bounded
  by the recall promise.
- **REVIEW is a budgeted human queue** ordered by calibrated P(pot), with per-row approve/dismiss.
- **No auto-dispatch.** Every `MissionPlan` carries `human_approval_required = True`; the inspection
  route is labelled "pending human approval".
- **Honest geometry.** Orientation comes from a source rule (PINGMapper sonograms: nadir at the top);
  unknown sources are *not measured* rather than guessed. Geotags need real GPS; demo tracks are
  stamped `SYNTHETIC DEMO GPS`; unknown orientation ⇒ swath-wide error radius.
- **Language models write, never decide.** The mission brief (`brief.py`) is written from a facts
  JSON built after every tier, route and pass is fixed. If Claude on Amazon Bedrock writes it, the text
  is machine-checked (every number and ID must be a survey fact; human-approval / synthetic-GPS /
  assumed-card-time caveats must be present) and replaced by the deterministic template on any failure.
- **Known limits (stated, not hidden):** one bay's crab pots and one sonar brand; calibration is a
  single recording (hence the separate verification split); labels are incomplete, so precision is a
  lower estimate; seconds-per-card is assumed until the user study.

---

## 6. Closing the loop — measure the sonar, act physically, learn from people

**The agent measures before it judges (Stage 1, STUDY-08).** Before detection it tracks the seabed
in every frame: the sonar's altitude per ping (validated with no labels — port and starboard, which
see the same pings, agree to a median 1.6 px). That measured altitude drives the relative height on
every card, the slant→ground range of every geotag, and a traced `water_column_check` tool step.
Orientation is decided by a source rule and never guessed; "unknown" switches the physics off.

**A physical second look (`resurvey.py`).** For what stays uncertain, the agent plans the boat pass
that can settle it: a straight line on the **opposite side** of the target with the target at
mid-swath. A real object standing off the seabed casts its shadow *away* from the sonar, so seen
from the other side its shadow must **flip** — the plan states the bearing it must point to, a
falsifiable prediction speckle cannot satisfy. Targets whose lines align share one pass; passes are
ranked by uncertainty resolved per metre of travel (Σ p(1−p)); a boat-time budget keeps the densest
ones. Repeat sightings of one object across passes merge into one hazard (tight error-circle gate;
never two detections of the same frame). Plans are PLANNED — a human approves them.

**Learning from people (`feedback.py`).** Every approve / dismiss is saved, and a reviewer can draw
a **missed pot** the detector never proposed — a positive label exactly where the model is blind
(its binding limit is recall). `python -m src.agentic.feedback export` turns them into a fine-tune
set + hard negatives; where labels exist, reviewer agreement is scored.

**Measuring the impact instead of claiming it (`study.py`, `effort.py`).** The Study tab runs a
counterbalanced 2×2 Latin-square user study (manual frame review vs the agent's cards, timed, scored
against labels). The effort curve then reports minutes to the recall promise for manual review, a
raw detector list and the DEPTH queue, plus the **break-even card time** (7.95 s at 20 s/frame) —
with assumed timings DEPTH only ties a perfect human, and the study decides the real answer. The
agent's own forecast of "pots these minutes will buy" held on unseen data (84.5 forecast vs 90 real).

**Auditing the labels (`fp_audit.py`).** A blinded audit with catch trials asks whether the
detector's most confident "false alarms" are really unlabelled objects — giving an exact audited
precision in that confidence band, with the auditors' own reliability measured.

---

## 7. Mapping to the Agentic-Vision rubric

**Required evidence — "a trace showing OpenCV 5 output changing a later decision/action":**
[`docs/causal_trace.md`](causal_trace.md) (STUDY-12) runs the same survey with each OpenCV output
withheld and diffs every downstream decision. Without Stage 1's bottom track, 193 of 264 pins land
outside their own error circle and 13 of 63 re-survey passes regroup (tiers unchanged — it is not a
gate); without the shadow nothing changes (evidence only). It also prints one hazard's full chain,
and every survey exports its decision log (`mission.trace.jsonl`).

| Criterion | Weight | Where it's met |
|---|---:|---|
| OpenCV 5 + agent doing real work | 30% | Stage-1 bottom tracking / `cv2.remap` / luminance, `cv2.dnn` detect, re-look + CLAHE, thin-line shadow, `seamlessClone` copy-paste for training; the agent's tools *are* CV ops |
| Orchestration & autonomy | 25% | stated objective; value-of-information tool use; analyst **and boat** budgets; stitching; repeat-sighting merge; routes; opposite-side re-survey planning (§3, §6) |
| Task success | 20% | a recall promise that **held on unseen test** (86%); honest "no auto-confirm"; forecast held; effort measured, not assumed (§4, §6) |
| Failure handling & human control | 15% | bounded LOW-RISK tier, budgeted REVIEW queue, human-approval gate, missed-pot labels, blinded audit (§5, §6) |
| User experience | 10% | studio with four modes (Analyze · Survey · Study · Audit): guarantee badges, evidence cards, seabed overlay, map + routes + re-survey passes, live effort curve, downloads |

**Reproduce:** `python -m src.agentic.calibrate` (fit + verify + report + `calibration.json`) ·
`python -m src.agentic.pipeline --survey <dir>` · `python -m src.agentic.effort` · `pytest -q`.
