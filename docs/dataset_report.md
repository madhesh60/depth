# Dataset Report — Marine Debris (YOLO-Ready v0)

**Generated:** 2026-09-18 · **Source of truth:** `DATASET/03_yolo_ready_dataset/`
**Audit script:** [`DATASET/scripts/audit_dataset.py`](../DATASET/scripts/audit_dataset.py)
(re-run to regenerate) · **Machine summary:** `DATASET/exports/audit_summary.json`

> This report was produced by scanning every label file. **§0 is the current cleaned dataset
> (v1) used for training.** §1–§7 document the original v0 split and the audit that motivated
> the cleaning (retained for provenance).

---

## 0. v1 — cleaned & training-ready ✅ (current)

Built by [`build_dataset_v1.py`](../DATASET/scripts/build_dataset_v1.py) from v0; independently
re-audited. Location: `DATASET/03_yolo_ready_dataset_v1/` (`data.yaml`, `manifest.json`).

| Split | Images | Background | Boxes (per class 0/1/2/3) |
|---|---:|---:|---|
| train | 26,533 | 1,120 | 19,527 / 1,636 / 10,422 / 5,389 |
| val | 1,204 | 15 | 1,008 / 120 / 507 / 260 |
| test | 1,276 | 58 | 952 / 65 / 501 / 269 |

**Classes (`nc=4`):** `0 fishing_gear · 1 pipe_cylinder · 2 structural_fragment · 3 natural_formation`.

**Split is jointly stratified so val AND test mirror train** on both class mix and sensor domain
(greedy fill-equalisation on original frames). Resulting composition is near-identical across all
three splits — the earlier test-set distribution inversion is gone:

| Measure | train | val | test |
|---|---:|---:|---:|
| `fishing_gear` share | 52.8% | 53.2% | 53.3% |
| `structural_fragment` share | 28.2% | 26.8% | 28.0% |
| `natural_formation` share | 14.6% | 13.7% | 15.1% |
| `pipe_cylinder` share | 4.4% | 6.3% | 3.6% |
| sonar / optical | 80 / 20 | 79 / 21 | 80 / 20 |

| Problem (v0) | Fix in v1 | Verified result |
|---|---|---|
| Cross-split leakage (838 shared frames) | Split whole **recording/clip** groups (5,618) | **0 / 0 / 0** shared frames |
| Test-set class + sensor mix diverged from train | **Joint stratification** on class box-mix + sensor (originals) | val ≈ test ≈ train (table above) |
| Baked augmentation scattered across splits | Train keeps augs; val/test = originals only | honest eval set |
| Full-frame boxes | Dropped on write (`w&h>0.95`) | 1,166 removed → **0 remain** |
| Degenerate / sliver boxes | Dropped (`area<1e-5`, side`<0.01`) | 75 slivers removed |
| No background negatives | Emptied-label images kept as negatives | **1,193** background images |
| Corrupt images unknown | Every image opened/verified | **0 corrupt** |
| Polygons / malformed | — | **0 / 0** |

**Deliberately retained (not bugs):** plausibly-small boxes (train 1,469 / val 10 / test 60) kept
to preserve small-debris recall — flagged for visual QA via `visualize_labels.py`.

**Remaining training-time concerns (not dataset defects):**
- **Class imbalance** — `fishing_gear` (21,487) ≈ 11.8× `pipe_cylinder` (1,821). Handle with
  minority oversampling / focal loss / augmentation; always report per-class metrics.
- **Sparse minority in eval** — `pipe_cylinder` has only 65 (test) / 120 (val) boxes (few clips
  carry it), so its per-class AP is high-variance — caveat it in results.
- **Mixed sensors** — ~80% sonar / 20% optical, now balanced across all splits (tagged in
  `manifest.json`). Consider a sonar-only fine-tune for the mission domain (EXP-006).

---

## 1. Headline numbers  *(v0 — as-built, superseded)*

| | Images | Annotations |
|---|---:|---:|
| **train** | 30,444 | 44,133 |
| **val** | 1,263 | 1,865 |
| **test** | 1,274 | 1,883 |
| **Total** | **32,981** | **47,881** |

- **Annotation format:** 100% YOLO **bounding boxes** (5 tokens). **Zero polygon lines.**
- **Background (empty-label) images:** 0
- **Malformed / out-of-range lines:** 0
- **Predominant resolution:** 640×640 (Roboflow-standardised) with a minority tail of
  mixed native sizes (480×360, 617×768, 1728×1818, …).

---

## 2. ⚠️ Critical finding — the taxonomy changed since the proposal

The live `data.yaml` declares **4 classes**, not the 5 described in the proposal. The
maximum class id present in any label file is **3** — there is **no class 4**.

| Live dataset (`data.yaml`, `nc: 4`) | Proposal taxonomy (`AGENT.md`, 5 classes) |
|---|---|
| 0 `fishing_gear` | 0 `ghost_net` + 3 `rope_line` (merged) |
| 1 `pipe_cylinder` | 1 `pipe_cylinder` |
| 2 `structural_fragment` | 2 `structural_fragment` |
| 3 `natural_formation` | 4 `natural_formation` |

**What happened (reconstructed from the counts):** the dataset was reprocessed since the
previous report. All polygon annotations were converted to boxes, `rope_line` was folded
into `fishing_gear`, and the taxonomy collapsed 5→4 classes. Evidence: the previous report's
`ghost_net` (24,418) + `rope_line` (307) = **24,725** = today's class 0 exactly; every other
class total is unchanged; total annotations (47,881) is identical.

**Consequences of the previous report being stale:** its claims that "`rope_line` has 0
usable boxes", "1,461 polygon lines need conversion", and "5 classes" are **no longer true**.
Those blockers are resolved by the reprocessing.

**Open decision (for the owner):** ship the cleaner 4-class `fishing_gear` taxonomy, or
re-introduce `rope_line` as a distinct 5th class to match the proposal. See
[`TODO.md`](../TODO.md) Phase 1 and [`architecture.md`](../architecture.md) §12.

---

## 3. Class distribution

| ID | Class | Train | Val | Test | **Total** | Share |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `fishing_gear` | 22,755 | 995 | 975 | **24,725** | 51.6% |
| 1 | `pipe_cylinder` | 2,922 | 121 | 123 | **3,166** | 6.6% |
| 2 | `structural_fragment` | 12,042 | 492 | 497 | **13,031** | 27.2% |
| 3 | `natural_formation` | 6,414 | 257 | 288 | **6,959** | 14.5% |
| | **Total** | **44,133** | **1,865** | **1,883** | **47,881** | 100% |

**Imbalance:** `fishing_gear` outnumbers `pipe_cylinder` ~7.8:1. Mitigate with class
weighting / focal loss / targeted augmentation, and **always report per-class metrics** —
aggregate mAP will hide weak minority-class recall.

---

## 4. Provenance — source datasets

Images carry a source prefix, so provenance is recoverable. Image counts:

| Prefix | Source dataset | Sensor | Train | Val | Test | **Total** |
|---|---|---|---:|---:|---:|---:|
| `crabpot_` | Crab Pot Dataset | Side-scan sonar | 12,297 | 512 | 516 | **13,325** |
| `uatd_` | UATD (Underwater Acoustic Target Detection) | Forward-look sonar | 9,459 | 381 | 393 | **10,233** |
| `icra_` | Trash-ICRA19 | Optical (RGB) | 4,815 | 197 | 195 | **5,207** |
| `vid_` | TrashCan | Optical (RGB) | 1,242 | 48 | 55 | **1,345** |
| `mpulse_` | Marine PULSE | Side-scan sonar | 1,125 | 58 | 60 | **1,243** |
| `seabed_` | SeabedObjects-KLSG | Side-scan sonar | 1,122 | 46 | 43 | **1,211** |
| `shipwreck_` | AI4Shipwrecks | Side-scan sonar | 384 | 21 | 12 | **417** |
| | **Total** | | **30,444** | **1,263** | **1,274** | **32,981** |

**Sensor mix:** ~80% sonar (crabpot, uatd, mpulse, seabed, shipwreck ≈ 26,429 images),
~20% optical (icra, trashcan ≈ 6,552). Domain shift between sonar and optical is a modelling
risk — track sonar-only vs mixed metrics (see [`experiments.md`](../experiments.md) EXP-006).

---

## 5. Data-quality issues (verified this audit)

| # | Issue | Count | Severity | Recommended fix |
|---|---|---:|:---:|---|
| **D1** | **Cross-split leakage** — same source frame in multiple splits (base key ignoring aug/hash) | train↔val **361**, train↔test **359**, val↔test **118** | 🔴 High | Re-split by **grouped** source frame so all augmentations of a frame stay in one split. **100% of leaked keys are `crabpot_`** sequential frames. |
| **D2** | **Degenerate full-frame boxes** (w>0.95 & h>0.95) | **1,258** (train 1,140 / val 58 / test 60) | 🟡 Med | Drop or relabel — mostly Marine PULSE (classification-origin). Harms localisation. |
| **D3** | **Tiny boxes** (area < 0.0005 of frame) | **1,713** (train 1,545 / val 93 / test 75) | 🟡 Med | Review; may be valid small debris or annotation noise. Set a min-box filter and sample-inspect. |
| **D4** | **Class imbalance** | 7.8:1 (class 0 vs 1) | 🟡 Med | Class weights / focal loss / minority augmentation. |
| **D5** | **Mixed sensors** (sonar + optical) | ~80/20 | 🟢 Low | Track domain-split metrics; consider sonar-only fine-tune. |

> **D1 is the most important pre-training fix.** Leakage inflates validation/test scores and
> produces a model that looks better than it is — fatal for credible evaluation evidence.
> The leak is concentrated entirely in the `crabpot_` source (adjacent sonar frames +
> augmentation variants landing across splits).

---

## 6. What was checked ✅

- Per-split image/label file parity — matched (30,444 / 1,263 / 1,274; no orphans).
- Every annotation line parsed for token count, class id, and 0–1 range.
- Box vs polygon format classification.
- Full-frame and tiny-box detection.
- Source-prefix provenance tally.
- Cross-split leakage via normalised base key (strips `_augN` and `.rf.<hash>`).

## 7. Recommended path to dataset v1

1. **Fix leakage (D1)** — regroup and re-split by source frame; re-run this audit to confirm 0 shared keys.
2. **Handle full-frame boxes (D2)** and review tiny boxes (D3).
3. **Decide the taxonomy** (§2) — 4-class `fishing_gear` vs re-introduce `rope_line`.
4. **Tag `dataset v1`**, record the exact transform script in `DATASET/scripts/`, push to S3.
5. Only then run the first real training experiment (EXP-001).

_Regenerate this report after any change: `python DATASET/scripts/audit_dataset.py`._
