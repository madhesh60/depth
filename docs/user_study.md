# Timed user study — protocol (≈ 10 minutes per person)

**Goal.** Replace the two assumed numbers behind every "minutes saved" claim — seconds per frame of
manual review and seconds per DEPTH card — with measured ones, plus how many pots people actually
find each way. Built into the app: **Study** tab.

## Run it

1. Start the app (`python -m uvicorn src.dashboard.app:app --port 8000`) or use the live link.
2. Open **Study**, type the participant's initials, **Start session**.
3. The participant does both tasks (the app times everything):
   - **Manual review** — a raw sonar frame; click every crab pot, then *Next* (Enter). Backspace undoes.
   - **DEPTH cards** — the agent's review queue; *Y* real pot / *N* not a pot.
4. Hand the laptop to the next person → **New session**. Aim for **3+ people** (the more, the tighter).

## Design (why the numbers are fair)

- **Counterbalanced 2×2 Latin square:** the labelled frames are split into two halves; each person
  reviews one half manually and the other half as cards, so nobody sees a frame twice. Which half
  goes to which task, and which task comes first, rotate with the session number — learning and
  fatigue effects cancel across sessions.
- **Scored against the labels, server-side:** a manual click counts if it lands in a labelled pot
  (+8 px); one click per pot; extra clicks are false marks. A card is real if it overlaps a labelled
  pot at IoU ≥ 0.3.
- **DEPTH-card recall counts every labelled pot in the card frames** — including pots the detector
  never proposed. It is bounded by the model, not the person.
- Frames: the 8 shipped labelled samples (4 per task per person). For a larger study point
  `$DEPTH_STUDY_DIR` at a folder with `images/` + `labels/` (e.g. unseen v2b test frames).
- Stored: `runs/study/<session>.json` (initials only — no other personal data).

## What it feeds

- **Study results** panel: median s/frame and s/card, recall both ways, minutes per survey-hour,
  and the per-frame speed-up.
- `src/agentic/effort.py` / **Analyst effort** panel: once a session exists, the effort curve uses
  the measured timings instead of the assumed ones (`python -m src.agentic.effort` regenerates
  `docs/effort_curve.md`). The **break-even card time** says how fast card review must be for DEPTH
  to beat manual review at the recall promise.

Report whatever the study shows — including if manual review is faster.
