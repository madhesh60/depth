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
- **v2b** (`03_yolo_ready_dataset_v2b/`, **EXP-002 trains here**) — v2 with its three *evaluation
  traps* removed (`build_dataset_v2b.py`, below). train **1,773** (1,615 unique Roboflow frames +
  158 wreck/seabed; 462 bg) / val **234** (recordings Rec10/12/16 held out whole: 163 frames, 186
  pots + 71 wreck/seabed) / test **285** (**214 unique** crab-pot frames + 71 wreck) — plus
  `test_official398/` (the exact official 398, GhostVision comparison only) and `test_xsonar/`
  (555 orange Contact_sslo crops, cross-sonar). `groups.json` maps every image to its frame and
  recording group for group bootstrap. Leakage 0/0/0 by frame key and by held-out recording.

## Evaluation traps found & fixed (v2 → v2b)

1. **Baked-in Roboflow augmentation** — the HF crab-pot archive was exported *with* augmentation:
   1,615 train frames appeared as 5,275 copies (2-12 each), 3,241 of them **rotated** (black
   corners; rotation breaks sonar range/shadow geometry). → one least-changed copy per frame;
   `train.py` augments at train time.
2. **Duplicate test frames** — official test = 398 images but **214 unique frames**, so metrics
   double-counted and bootstrap CIs were too narrow. → `test/` = unique frames (primary);
   the 398 are kept only for the like-for-like GhostVision number.
3. **Validation was a different sonar** — the official `valid` is 555 orange `Contact_*_sslo`
   crops (horizontal range) vs greyscale Humminbird test sonograms. → val = held-out *training
   recordings* of the test sonar; Contact_sslo became a cross-sonar test.
4. Pixel check: no unique test frame has a train twin (max thumbnail correlation 0.92, and that
   pair is from different recordings). `Rec14_Sensor_Depth` (train) ≠ `Rec14_wcp` (test).

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
  `test_xsonar/` (Contact_sslo) and the UATD hold-out now measure it.
- After dedupe the data is **small** (1,015 unique crab-pot sonogram frames in total; test 214):
  expect wide confidence intervals and report them.
- **EXP-001's own clean data is tiny:** it trained on v1, whose train split contains the official
  crab-pot test recordings. Its only unseen crab-pot sonograms are v1 val (**66 unique frames**,
  Rec19; 439 images incl. copies) and v1 test (**92 unique frames**). All EXP-001 agent/guarantee
  numbers are fit on the former and checked on the latter.
