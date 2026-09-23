"""
src.agentic — the See → Prove → Decide → Act loop for marine-debris sonar.

A trustworthy agentic layer on top of the EXP-001 YOLO detector:

* **See**    — detect candidates full-frame (``perception.Perceptor`` over ``cv2.dnn``).
* **Prove**  — gather *physical* evidence per candidate (``evidence.EvidenceGatherer``):
               re-look persistence (zoom in and re-detect — the workhorse discriminator),
               the acoustic **shadow** where it genuinely exists (``shadow.ShadowProver``),
               echo strength, and a shadow-derived height estimate.
* **Decide** — a deterministic, logged, human-gated triage agent (``agent.ReLookAgent``)
               sorts each candidate into CONFIRMED / REVIEW / REJECTED and records a full
               tool-call trace.
* **Act**    — turn CONFIRMED objects into a human-approved cleanup route + geotagged reports
               (``mission`` / ``geo``).

Design note (honesty first): on this side-scan crab-pot data the acoustic shadow is only
measurable on a minority of larger/high-relief objects and does **not** separate the detector's
true positives from its false positives (STUDY-03/04: its CLEAR-rate is ~14.5% on both TP and FP).
The discriminators that *do* work are re-look persistence and high detector confidence — the two
independently-calibrated CONFIRMED paths whose union lifts precision from ~0.60 (raw) to ~0.74 at
~30% recall. The shadow is therefore used as **evidence shown where present** (plus a height
estimate), never as a silent gate — and no candidate is ever discarded without a human seeing it.
"""
