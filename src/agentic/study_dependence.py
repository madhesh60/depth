"""
study_dependence.py — STUDY-21: does the recall promise survive correlated frames?

The Clopper-Pearson bound behind "≥ 79% of pots reach a person (95% confidence)" treats every pot as an
independent trial. Pots are not independent: port and starboard frames of the same pings show the same
seabed, consecutive frames of a recording share a site, a gain setting and sometimes an object, and a
recording shares one sonar set-up. Correlation makes the real uncertainty larger than the bound says.
(``guarantees.py`` claimed that ``calibrate.py`` reports a frame-group bootstrap; it did not — this study
is that check.)

**Protocol (fixed in this file and committed before any number below was computed)**

1. Data: the cached per-frame detections ``calibrate.py`` and STUDY-18 already scored
   (``runs/calib/<MODEL>_val.json``, ``_test.json`` and their ``_tiles`` twins) — no new inference. The
   test split was scored once (STUDY-18); here its cached predictions are only broken down by recording.
   No threshold is fit or changed by this study. Modes: ``standard`` (τ_review from ``calibration.json``)
   and ``look_harder`` (``modes.look_harder``, the STUDY-18 merge of the tile pass).
2. Unit of dependence: a **block** = the frames of one recording whose ping index (``_port_00012`` /
   ``_star_00012``) falls in the same run of ``b`` consecutive indices — port and starboard of the same
   pings always share a block. ``b`` ∈ {1, 4, 8}.
3. **A1 — block bootstrap (calibration split, the gate).** At each mode's shipped τ: resample blocks with
   replacement within each recording (each recording keeps its number of blocks), B = 20,000, seed 0;
   recall = reached pots ÷ all pots of the resample; the 5th percentile is the dependence-aware 95% lower
   bound. Also reported: the CP lower bound at the same τ and the design effect (bootstrap variance ÷
   binomial variance) → the effective number of independent pots.
4. **A2 — recordings (reported, no gate).** Recall at the shipped τ on each of the 3 calibration and 3
   test recordings with its own CP 95% lower bound, and leave-one-recording-out on the calibration
   split: the promise refit on two recordings (``best_recall_guarantee``, as ``calibrate.fit`` does) and
   checked on the third. Also reported: a two-stage bootstrap (recordings, then blocks within them) —
   with 3 recordings this bound is itself very uncertain, so it is a sensitivity check, not a gate.
5. **Decision rule R1 (fixed now).** For each mode: if the A1 lower bound with the largest block
   (``b`` = 8) is ≥ the promise, the promise stands as stated. If it is lower, the promise is restated
   to that bound rounded down to 0.01 — ``calibration.json`` keeps the Clopper-Pearson value as
   ``recall_promise_cp`` and ``recall_promise`` becomes the restated one, so the studio, the API, MCP
   and the mission plan all show it; τ is unchanged, and the test verification is re-compared against
   the restated promise (no test number is recomputed).
6. Wording rule (whatever R1 says): the headline says **promise**, not guarantee, and names its
   assumption — the pots of a new survey are exchangeable with the calibration pots; A2's per-recording
   spread is reported next to it, because the promise is a pooled rate over recordings like the
   calibration ones, not a per-survey guarantee.

    python -m src.agentic.study_dependence            # A1 + A2 → docs/reports/dependence.md
    python -m src.agentic.study_dependence --write    # + apply R1 to calibration.json (only if R1 restates)
"""
