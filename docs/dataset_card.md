# Dataset card — DEPTH

Honest provenance for the training data. The data itself is git-ignored (S3); the build scripts
(`DATASET/scripts/`) are tracked. Regenerate the audit: `python DATASET/scripts/audit_dataset.py <root>`.

## Sources & sensors (7 archives)

| Source | Sensor | Content | License |
|---|---|---|---|
| PINGEcosystem crab-pot | **side-scan sonar** | derelict crab-pots (the real target) | CC-BY-SA-4.0 (confirm GPL note) |
| UATD | **forward-looking sonar** | placed test objects (cylinder, tyre, cage…) | CC-BY-4.0 |
| SeabedObjects-KLSG | side-scan sonar | shipwrecks | research use |
| AI4Shipwrecks | side-scan sonar | shipwrecks | check page before publishing |
| Marine PULSE | side-scan sonar | seabed surface, pipelines, platforms | check page |
| ICRA19 ("bio") | **optical camera** (JAMSTEC) | fish / plants / sea life | JAMSTEC-derived, research use |
| TrashCan | **optical camera** (JAMSTEC) | debris in video | JAMSTEC-derived, research use |

> Three sensor modalities (side-scan, forward-looking, optical) look nothing alike. Mixing them
> lets a model learn shortcuts (e.g. "optical photo → natural_formation"), which is exactly why v2
> is **sonar-only**.

## Class taxonomy — honest naming

The v1 class *names* over-claimed. What the data actually is, and the v2 rename:

| v1 (4-class, EXP-001) | Reality | v2 (2-class, sonar-only) |
|---|---|---|
| `fishing_gear` | crab-pots (side-scan) + a few optical net photos | **`ghost_gear`** |
| `structural_fragment` | KLSG/AI4Shipwrecks wrecks + UATD placed objects | **`wreck_debris`** |
| `pipe_cylinder` | UATD cylinders (forward-looking test objects) | held out (`_holdout_uatd_fls/`) |
| `natural_formation` | **ICRA19 "bio": fish/plants in optical photos** — *not* rock clusters | dropped (off-domain) |

## Versions

- **v0** — 7 sources → 5 classes, 32,981 imgs. Superseded (leakage, full-frame boxes).
- **v1** (`03_yolo_ready_dataset_v1/`, EXP-001 trained here) — 4-class, leakage-free (split by
  recording/clip, 0 shared frames), full-frame + sliver boxes removed, 1,120 background negatives.
  train 26,533 / val 1,204 / test 1,276.
- **v2** (`03_yolo_ready_dataset_v2/`, EXP-002 trains here) — 2-class, **sonar-only, honest**
  (`build_dataset_v2.py`). Fixes three v1 label bugs (below). train 6,291 (1,550 bg) / val 626 /
  test 469; leakage 0/0/0; **official crab-pot 398-frame test split locked** for the GhostVision
  head-to-head. Held-out UATD forward-looking eval kept separate.

## Bugs found & fixed (v1 → v2)

1. **955 real pipeline/platform frames were labelled "empty seabed"** — v0 gave each Marine PULSE
   image one full-image box; v1 then dropped full-frame boxes, leaving them empty. → dropped as
   unlocalised (only true seabed kept as background).
2. **1,547 empty crab-pot frames were discarded** — the best natural-seabed negatives for the top
   false alarm (empty seabed → fishing_gear). → recovered as background.
3. **Class names renamed** to match the sensor/content (above).

## Known limitations

- Class imbalance (`fishing_gear` ≫ `pipe_cylinder`); `wreck_debris`/`pipe` are low-volume →
  high-variance per-class AP (always report per class, not just aggregate).
- Crab-pot frames carry **no GPS** — geotagging is demonstrated on synthetic/PINGMapper tracks and
  clearly labelled as such (see [`responsible_use.md`](responsible_use.md)).
- Trained on one bay's crab-pots and one sonar brand; generalisation is unproven — stated, not hidden.
