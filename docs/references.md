# References & design justification

Published work that backs specific design choices in this project. Cite these in the technical
report. **Be precise about modality** — the strongest reference below is from optical satellite
imagery, not sonar; we borrow the *methodology*, not the data or model.

---

## [R1] Rußwurm, Venkatesa & Tuia (2023) — data-centric marine-debris detection

- **Title:** *Large-scale Detection of Marine Debris in Coastal Areas with Sentinel-2.*
- **Authors:** Marc Rußwurm, Sushen Jilla Venkatesa, Devis Tuia (EPFL).
- **arXiv:** 2307.02465 · **Code:** github.com/marccoru/marinedebrisdetector (MIT).
- **Modality:** Sentinel-2 **optical** satellite (12-band). **Target:** *floating* surface
  plastic. **Model:** U-Net++ segmentation.

**Their core finding (verified quote):**
> "this performance is due to our particular dataset design with extensive sampling of negative
> examples and label refinements rather than depending on the particular deep learning model."

**Why we cite it (honest scope):** their *conclusion generalises across modalities* — data
design beats model choice — even though their **data/weights/model are optical and not reusable
for side-scan sonar** (their hosted data/weights are also now private). Two of our choices are
the same principle applied to sonar:

1. **Extensive negative sampling** → dataset v1 ships **1,120 background (empty-label) frames,
   99% sonar** (the product domain) for false-positive control.
2. **Label refinement** → the documented audit + v1 rebuild (dropped 1,096 full-frame boxes,
   sliver/degenerate boxes, corrupt scan, leakage-free split by clip, taxonomy lock). See
   `docs/dataset_report.md` and `DATASET/scripts/{audit_dataset,build_dataset_v1}.py`.

Our own STUDY-01 (a model-side trick — classical ROI-gating — gave no gain) independently
echoes the "data > model" lesson.

**Honest gap this reference exposes (drives v2, see `experiments.md`):** our negatives are
concentrated in one source — `mpulse` (1,086 of 1,120) — while **`crabpot`, the source of the
`fishing_gear` false positives (background→fishing_gear 0.79, precision 0.50), has only 21**.
The paper prescribes *more, domain-matched* negatives; v2 should mine empty/clutter crab-pot
sonar frames. Do **not** claim `natural_formation` suppresses sonar FPs — our data shows it is
optical-only (267 optical vs 2 sonar test boxes), so it does no hard-negative work on sonar.
