# Responsible use

This system is decision-support for marine-debris surveys, not an autonomous actuator. The
following constraints are built into the design, not just promised.

## Human control

- **Nothing is auto-dispatched.** Every `MissionPlan` carries `human_approval_required = True`; the
  recovery route and re-survey list are *proposals* a human must approve.
- **No candidate is ever deleted.** The agent's REJECTED tier is *deprioritised but retained for
  audit* — a reviewer can still see everything (recall-safe: ~91% of true pots stay in CONFIRMED +
  REVIEW). Confident-but-wrong suppression is avoided by design.
- **REVIEW is a human queue** with per-item approve/reject in the dashboard; corrections can become
  new training labels (a supervised feedback loop, not silent auto-learning).

## Honest geotagging

- Coordinates are attached **only when real per-ping GPS exists**. With no GPS, the output is flagged
  `gps_available = False` with null coordinates — we never stamp one fabricated position onto every
  object.
- Demonstration tracks are clearly labelled **"SYNTHETIC DEMO GPS — not real coordinates"** in the
  UI and every export.
- Reported geo error is a coarse, honest estimate that grows with across-track range. On a **real
  recording** it is built from stated parts:
  - GPS, 3 m (assumed consumer grade);
  - the measured range-scale uncertainty;
  - the measured heading error.
- Metres of height appear only where the sonar's own depth sounding provides a measured altitude.

## Sensitive locations

- **Do not publish exact coordinates of protected shipwrecks** (war graves, heritage sites). Treat
  wreck detections as sensitive; share only with authorised parties and, for public artefacts,
  reduce coordinate precision.
- The tool is for hazard reduction (ghost gear), not salvage targeting.

## Data & privacy

- No personal data is processed — inputs are seabed sonar frames.
- Raw recordings are untrusted binary input. The reader bounds-checks every length and offset, caps
  file and sample sizes, and skips corrupt pings with a recorded issue. Nothing in a recording is
  executed. The PINGMapper sample is fetched with every file SHA-256 pinned and is never redistributed.
- Uploaded frames should be auto-deleted after a short retention window (e.g. 7 days via an S3
  lifecycle rule) on any hosted deployment; the demo holds results in memory only.

## Honesty about capability (anti-overclaim)

- We report metrics **per class and per sensor**, never a single inflated aggregate.
- We publish the cues that did **not** work (STUDY-01 ROI-gating; STUDY-03/04 acoustic shadow is
  non-discriminative) alongside the ones that do.
- CPU latency is measured on the real server, not quoted from a GPU.
- Known limits (one bay, one sonar brand, small-object recall, optical out-of-domain) are stated in
  the [model card](model_card.md).

## Licensing

AGPL-3.0 (Ultralytics YOLO) — full source is public. Dataset sources retain their own licenses; see
the [dataset card](dataset_card.md). Respect each source's attribution and share-alike terms.


## Implemented safeguards (2026-09-25, extended 2026-09-28)

- **Public share mode** (`/api/report/<fmt>?public=1`, "public share" toggle): wreck / structural
  hazards (possible war graves, heritage sites) are generalised to ~1.1 km (0.01°) with an honest
  error radius, and re-survey passes that would reveal them are dropped. Ghost-gear cleanup targets keep
  full precision. Full precision stays with the survey owner.
- **The agent decision log is never public** — it carries exact positions (HTTP 403).
- **Provenance on every output** — model / calibration hashes, OpenCV build (COOL or stock), code
  commit and host, so any published hazard can be traced to what produced it.
- **Human approval** before any dispatch; LOW-RISK items are kept for audit, never deleted.
- **Decisions are signed and logged:** a named person confirms / rejects / marks recovered; the agent's
  own verdict is kept alongside; every decision is in the decision log (2026-09-28).
- **Public-demo protection (2026-09-28):** per-client rate limits on the expensive endpoints (429 +
  Retry-After; `src/dashboard/ratelimit.py`), a cap on queued survey jobs (503), a bearer token on
  `/mcp`, an admin token for webhook management (both generated root-only by the setup script), and
  webhook targets checked against private / link-local addresses. A production deployment also sets
  `DEPTH_APPROVER_PIN` so only people holding the PIN can decide; the public judge demo leaves it unset
  so judges can try the human loop.
