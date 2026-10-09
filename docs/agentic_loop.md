# The agentic loop: SEE → PROVE → DECIDE → ACT, with a person in charge

DEPTH is not *image → YOLO → box*. After OpenCV 5 sees, an agent keeps asking one question —
**can another observation change this decision?** — spends evidence tools only where the answer is
yes, records every call it makes *and every call it skips*, updates its belief, decides a tier,
checks the analyst and boat budgets, plans routes, and then **asks a named person** before anything
moves. Every failure is caught, contained and told to that person. This page is the map; the code is
`src/agentic/` and the studio shows all of it.

## 0 · Per chunk of a raw recording: the agent re-measures its own geometry (on by default)

Every metre DEPTH reports hangs on the sonar's altitude, and two independent sensors measure it:
OpenCV's bottom tracker (Stage 1) and the depth sounder in each ping header. Neither is always right,
so `sensor_check.py` runs a loop per 500-ping chunk, each step an `AgentStep`:

| step | what the agent does |
|---|---|
| `bottom_track` | OpenCV tracker on the native sonogram |
| `sounder_check` | is the sounder steady (windowed spread ≤ 50%) or did it lose lock? |
| `crosscheck` | ≥ 60% of pings within 15% → **agree**; else **conflict** |
| `retrack_guided` | conflict + steady sounder → **re-run the OpenCV tracker** in a ±30% window the sounder sets; a coherent edge → **corrected** / **recovered**; speckle is never forced |
| `image_check` | unsteady sounder → a coherent image edge wins → **image trusted** (sounder flagged) |
| `retrack_cross_channel` | still nothing → re-track inside the other channel's accepted line (same pings) |
| `resolve` | the verdict and what it means downstream; nothing coherent → **not measured** (slant range, error radius + altitude bound, no metres) |
| `estimate_scale` | re-fit metres per sample from sounder-backed chunks; moved > 2% → re-check every chunk |

The survey's plan opens with `sensor_crosscheck` → `geometry_update` / `geometry_withheld` (which
hazards were placed with corrected geometry). Studio: Survey → Agent → **Seabed cross-check**.
Validation: [`sensor_check.md`](sensor_check.md) (STUDY-16; fresh 1-h recording, criteria committed
before the run; one of three failed and is reported).

## 1 · Per candidate (Analyze → evidence card → "Agent decision")

Each find's trace (`agent.py`, `AgentStep.status` = `done | skipped | failed`) is one chain:

| stage | step | what it records |
|---|---|---|
| Ask | `voi_check` | "can another observation change the decision?" — **yes** (uncertain band) or **no** + why (above τ_confirm, below τ_review, class outside the guarantee, calibration found re-looks do not move the score) |
| Evidence | `zoom_relook` / `mosaic_relook` / `enhance_relook` | done, or **skipped** (VoI 0, or the per-frame compute budget is spent), or **failed** (fallback stated) |
| Evidence | `water_column_check`, `shadow_check`, `estimate_height` | Stage-1 geometry + physics; skipped when not measurable (never guessed) |
| Update | `update_belief` | calibrated P(pot) from the detector alone → after the evidence |
| Decide | `decide` | the tier and the promise it carries |
| Next | `handoff` | what happens next and who must approve it |

**Stop rule / compute budget:** at most `$DEPTH_MAX_RELOOKS` (16) re-looks per frame, spent on the
likeliest band candidates; the rest are recorded as skipped and go to a person on their detector score.

Honest note for the deployed EXP-003: its calibration found that a re-look does **not** move the
calibrated score (`relook_mode: null`), so the agent answers "no" and skips it for every find. The
"another observation" that *can* change a card is physical: an opposite-side sonar pass (§2).

## 2 · Per survey (Survey → Agent tab)

`mission.planner_log` — deterministic, reproducible:

`triage` → `voi_check` (open uncertainty Σp(1−p); options: a person's card or an opposite-side pass) →
`budget_check` (analyst minutes ÷ s/card → cards that fit, expected pots) → `stop_rule` (where it stops
and what it defers) → `human_override` (a person's order is kept) → `route_update` (inspection route) →
`recovery_route` → `resurvey_plan` (passes chosen **and skipped** against the boat budget) →
`dispatch_gate` → `request_approval` (the agent files approval requests; it never executes).

After **every person's decision** the agent re-plans and records what changed
(`MissionPlan.replans`: queue, inspection / recovery stops and metres, passes, cards in budget). If a
re-plan makes a pending request stale, the agent **withdraws it and asks again**
(`approvals.withdraw`; a decided request is never touched).

## 3 · Human in the loop

| where | a person can | it becomes |
|---|---|---|
| Analyze evidence card | **Approve / Reject / Override** the tier (Confirmed / Review / Low risk) | a training label (`feedback.py`; override → confirmed = positive, → low risk = hard negative, → review = no label) |
| Analyze, Evidence header | **Mark missed** — draw the box the detector missed | a positive label |
| Survey Hazards | ✓ / ✕, recovered / not found, **↑ prioritise / ↓ deprioritise / reset** (override the agent's order) | re-plan + decision log; ✓ / ✕ also a training label (`source: survey`) |
| Survey Approvals | approve / decline the agent's requests (name required; `DEPTH_APPROVER_PIN` optional) | the only way anything is dispatched (webhook `approval.decided`) |

The agent's own verdict is never overwritten; the person's decision sits beside it.
**The LLM** (`brief.py`, optional Claude on Bedrock) writes the mission brief only — it cannot decide,
approve or dispatch, and an ungrounded or failed draft is replaced by the template.

## 4 · Deliberate failure handling (`incidents.py`)

Every failure is an incident `{code, severity, stage, title, message, fallback, human_action}`, shown as
a notice, listed in the Agent tab / evidence panel, and written to the audit log.

| code | fallback | drill |
|---|---|---|
| `no_detection` / `low_confidence_only` | frame kept, nothing promoted; "mark missed" offered | ✓ |
| `corrupt_frame` / `invalid_frame` | frame skipped (analyze: 422 with the incident); the survey continues | ✓ |
| `missing_gps` | positions, routes, passes withheld; frame-order queue | ✓ |
| `invalid_geometry` | position withheld for the affected hazard (non-finite / out of range / > 2 km uncertainty) | ✓ |
| `orientation_unknown` | no ground range / shadow / height; finds still reach a person | |
| `model_unavailable` | 503 with the reason; nothing decided | ✓ |
| `tool_failed` | detector score used; the find goes to a person, **never auto-confirmed**; one bad frame never sinks a survey | ✓ |
| `storage_failed` (S3) | results kept on local disk, downloads still work | ✓ |
| `llm_failed` | template brief served | ✓ |
| `view_failed` | 3D twin / counterfactual hidden; decisions unaffected | |

**Drills:** open the studio with `?drills` (e.g. `http://localhost:8000/?drills`), then Ready (top right) → *Failure drills*; or `?simulate=<code>` on `/api/analyze`,
`/api/survey`, `/api/jobs/survey`, `/api/brief`. Drill incidents are tagged `simulated`.

## 5 · The audit log

`GET /api/survey/{id}/log` (studio: Survey → **Log**) merges, oldest first: agent plan steps, incidents,
approval requests / decisions / withdrawals, people's decisions, re-plans and who wrote the brief —
each with its actor (**agent · person · system · llm**). `GET /api/report/trace?survey_id=` is the
replayable JSONL flight recorder (now with plan, incident and re-plan lines).

Tests: `tests/test_agentic_loop.py`.

## 6 · Active vision (experimental) — and a person's "no" to a pass

With the **Active vision · experimental** switch on (`?active=1`), each find's card opens with an
**Agent trace**: DETECT → evidence collected → agent decision → OpenCV tool selected → new evidence →
(evidence conflict) → decision changed → human approval → new survey plan, built only from the
logged steps. Tools are chosen by value of information (`active.py`); see
[`evidence_model_exp003.md`](evidence_model_exp003.md) for why it is not on by default.

Every opposite-side pass is its own approval request — the Agent tab shows *Agent recommends:
Perform opposite-side resurvey RSn · Reason: … · [Approve] [Reject]*. Reject → the agent drops that
pass for good, re-ranks the boat time over the rest, and moves its targets to the front of the
inspection queue (`human_declined` in the plan). The Agent tab also shows **Does OpenCV change the
plan?** — the same survey re-planned without the OpenCV evidence ([`opencv_counterfactual.md`](opencv_counterfactual.md)).
