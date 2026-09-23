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
true positives from its false positives (see ``experiments.md`` STUDY-03). The discriminator that
*does* work is re-look persistence (raw precision 0.61 → 0.79 at the CONFIRMED tier). The shadow is
therefore used as **evidence shown where present**, never as a silent gate — and no candidate is
ever discarded without a human seeing it.
"""
