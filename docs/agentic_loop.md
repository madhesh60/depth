# The agentic loop: SEE → PROVE → DECIDE → ACT, with a person in charge

DEPTH is not *image → YOLO → box*. After OpenCV 5 sees, an agent keeps asking one question —
**can another observation change this decision?** — spends evidence tools only where the answer is
yes, records every call it makes *and every call it skips*, updates its belief, decides a tier,
checks the analyst and boat budgets, plans routes, and then **asks a named person** before anything
moves. Every failure is caught, contained and told to that person. This page is the map; the code is
`src/agentic/` and the studio shows all of it.

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

**Drills:** System (top right) → *Failure drills*, or `?simulate=<code>` on `/api/analyze`,
`/api/survey`, `/api/jobs/survey`, `/api/brief`. Drill incidents are tagged `simulated`.

## 5 · The audit log

`GET /api/survey/{id}/log` (studio: Survey → **Log**) merges, oldest first: agent plan steps, incidents,
approval requests / decisions / withdrawals, people's decisions, re-plans and who wrote the brief —
each with its actor (**agent · person · system · llm**). `GET /api/report/trace?survey_id=` is the
replayable JSONL flight recorder (now with plan, incident and re-plan lines).

Tests: `tests/test_agentic_loop.py`.
