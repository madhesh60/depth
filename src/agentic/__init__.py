"""
src.agentic — the DEPTH agent: a goal, OpenCV tools, and a loop that re-plans on what it observes.

* **The core loop** (``core.SurveyAgent``) runs every survey toward the operator's goal: inspect the
  frames → check that the calibrated promise applies to this recording (``applicability``, STUDY-19) →
  choose how hard to look (``standard`` or ``look_harder``, STUDY-18) → perceive every frame, re-planning
  if the compute budget would overrun or the closer look fails → plan for people and the boat →
  one approval request per action (``approvals``).
* **The frame agent** (``agent.TriageAgent``) runs See → Prove → Decide on one frame:
  Stage-1 geometry (``src.cv_pipeline.canonical``), the detector over ``cv2.dnn``
  (``perception.Perceptor``), physical evidence (water column, acoustic shadow and height —
  ``shadow.ShadowProver``), then the calibrated tiers (``policy.GuaranteedTiers``): a recall promise fit
  on held-out recordings, verified once on unseen ones (``calibrate``, ``guarantees``), and checked for
  correlated frames (``study_dependence``, STUDY-21). Every candidate carries a full tool-call trace.
* **Act** — geotags, chunk stitching, repeat-sighting merge, routes and opposite-side re-survey passes
  (``geo``, ``stitch``, ``resurvey``, ``mission``); nothing is dispatched without a named person.

Honesty first: on this sonar the acoustic shadow barely separates true from false finds (AUC 0.563) and a
zoomed re-look did not beat plain confidence (STUDY-07), so both are evidence shown to a person, never a
silent gate. EXP-003 supports no precision promise, so nothing is auto-confirmed; no candidate is
discarded without a person being able to see it.
"""
