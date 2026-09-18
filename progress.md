# Progress Log — Marine Debris Detection System

**Team Syndicate · OpenCV AI Competition 2026**
This is the living status of the project. Update it at the end of every working session:
what moved, what's blocked, what's next. Newest entries at the top of §4.

**Related:** [`README.md`](README.md) · [`architecture.md`](architecture.md) ·
[`experiments.md`](experiments.md) · [`TODO.md`](TODO.md)

---

## 1. Timeline & hard dates

| Date | Milestone |
|---|---|
| 2026-08-13 | Proposal submission window opened |
| 2026-08-26 | Build phase begins |
| **2026-09-18** | **Today — foundation / planning** |
| 2026-09-21 → 10-02 | **Grant check-in** (30-min Zoom, required to unlock 2nd 50% of grant) |
| 2026-10-26 23:59 PT | **Final submission deadline** |
| 2026-10-27 → 11-09 | Judging |
| 2026-11-10 | Winners announced (OpenCV Live!) |

> **~5.5 weeks of build time remain.** The grant check-in is imminent — a concise,
> credible progress story is needed for it (see §5).

---

## 2. Milestone tracker

| Milestone | Status | Notes |
|---|---|---|
| Grant awarded ($150) | ✅ Done | Secured. 2nd half unlocked at check-in. |
| Proposal (AGENT.md) | ✅ Done | Source of record. |
| Dataset built (v0, 32,981 imgs) | ✅ Done | 7 sources remapped to 5 classes; split 30444/1263/1274. |
| Dataset audit | ✅ Done | Issues catalogued in `docs/dataset_report.md`. |
| Project docs (this set) | ✅ Done | README / architecture / progress / experiments / TODO. |
| Dataset fixes (v1) | ✅ Done | Clean split built: 0 leakage, full-frame boxes removed, 1,099 background negatives, 4-class taxonomy locked. `03_yolo_ready_dataset_v1/`. |
| Stage 1 classical CV | ⬜ Not started | `src/cv_pipeline/`. |
| Stage 2 baseline train | ⬜ Not started | Needs v1 dataset. |
| Evaluation + ablation | ⬜ Not started | mAP/P/R, Stage-1 FP-reduction study. |
| Reporting engine | ⬜ Not started | JSON/CSV/GeoJSON + geotagging. |
| Dashboard | ⬜ Not started | FastAPI + map + transparency view. |
| AWS deployment | ⬜ Not started | AWS CLI not yet installed. |
| **COOL benchmark (Arm vs x86)** | ⬜ Not started | **Bonus-prize deliverable (Best Use of COOL).** |
| Agentic loop (stretch) | ⬜ Not started | Only if ahead of schedule. |
| Submission package (report + video) | ⬜ Not started | Due 2026-10-26. |

Legend: ✅ done · 🟡 in progress · ⬜ not started · ⛔ blocked

---

## 3. Current state (2026-09-18)

**What exists**
- Grant secured; proposal finalised (`AGENT.md`).
- Dataset v0 built and audited: `DATASET/03_yolo_ready_dataset/data.yaml` is the training
  entry point. Build scripts frozen in `DATASET/archive_scripts/` (idempotent).
- Pinned dependencies (`requirements.txt`).
- Full project documentation set created (this commit).

**What does not exist yet**
- No `src/`, `infra/`, or `tests/` code — greenfield.
- Dataset v0 has **known blockers** (see §6) that must be fixed before the first real train.
- AWS CLI not installed; no cloud resources provisioned; COOL not yet exercised.

**Environment**
- Global Python 3.10 (no venv). `torch`, `torchvision`, `opencv-python`, `numpy`,
  `PyYAML`, `Pillow` present. `pip install -r requirements.txt` adds the rest.

---

## 4. Session log

### 2026-09-18 — Foundation, docs cleanup & dataset audit
- Read full context (AGENT.md proposal, competition rules, dataset report, CC playbook).
- Established peak architecture and winning plan (`architecture.md`).
- Created the formal documentation set: `README.md`, `architecture.md`, `progress.md`,
  `experiments.md`, `TODO.md`; defined naming conventions.
- Moved reference docs into `docs/` and rewrote them formally: `competition_rules.md`,
  `claude_code_playbook.md`, `dataset_report.md` (removed clumsy `comptetion.md`,
  `HOWTOWORK.md`, root `dataset_report.md`).
- Built `DATASET/scripts/audit_dataset.py` and ran a full ground-truth audit. **Key
  findings:** dataset is **4-class** (`fishing_gear`, not 5 w/ `rope_line`); **all boxes,
  zero polygons** (already converted); **cross-split leakage** from `crab_pot`; 1,258
  full-frame + 1,713 tiny boxes. Old report was stale → rewritten (`docs/dataset_report.md`).
- Built **dataset v1** (`03_yolo_ready_dataset_v1/`) via `build_dataset_v1.py`: source-frame
  grouped split (**leakage 0/0/0**, re-audited), full-frame boxes dropped (1,096), corrupt
  scan (0), 1,099 background negatives, 4-class taxonomy locked (`rope_line` stays merged).
  Added `visualize_labels.py` (QA renders in `DATASET/exports/qa_*`).
- **Next:** eyeball QA renders → Stage 1 CV (`src/cv_pipeline/`) → baseline train EXP-001 on v1.

---

## 5. Grant check-in talking points (Sep 21 – Oct 2)

Keep it concrete and honest:
1. **Problem & impact** — ghost-net detection from side-scan sonar; 4–8h manual review → minutes.
2. **Dataset** — 32,981 images / 47,881 boxes from 7 sources, unified to a 4-class taxonomy;
   audited with a documented, regenerable quality report.
3. **Architecture** — two-stage OpenCV 5 pipeline; Stage 1 classical CV is the **COOL core
   workload** on Graviton, benchmarked vs x86.
4. **Plan to deadline** — phased execution in `TODO.md`; primary path is Best Use of COOL.
5. **Evidence of active development** — this documentation set + dataset audit + committed repo.

---

## 6. Known blockers (must clear before first real training run)

Re-derived from the verified audit ([`docs/dataset_report.md`](docs/dataset_report.md),
`python DATASET/scripts/audit_dataset.py`). The earlier polygon/5-class blockers are
**resolved** — the dataset was reprocessed (polygons→boxes, `rope_line` merged into
`fishing_gear`, collapsed to 4 classes).

| # | Blocker | Status |
|---|---|---|
| B1 | Cross-split leakage | ✅ Resolved — v1 split by source frame; re-audit 0/0/0 |
| B2 | 1,258 full-frame boxes | ✅ Resolved — dropped in v1 (audit: 0 remain) |
| B3 | Tiny boxes | ✅ Handled — degenerate dropped; small-plausible kept + flagged for QA |
| B4 | Class imbalance (`fishing_gear` ≈ 11.7× `pipe_cylinder`) | 🟡 Open — training-time (weights/focal/aug), not a data defect |
| B5 | Taxonomy decision | ✅ Resolved — **4-class**, `rope_line` merged into `fishing_gear` |
| B6 | AWS CLI not installed | 🟡 Open — blocks cloud + COOL work |

Details and fixes tracked in [`TODO.md`](TODO.md) Phase 1 & Phase 4.

---

## 7. Metrics dashboard (fill as results arrive)

| Metric | Target | Current | Δ |
|---|---|---|---|
| mAP@0.5 | ≥ 0.70 | — | — |
| mAP@0.5:0.95 | ≥ 0.45 | — | — |
| Precision | ≥ 0.80 | — | — |
| Recall | ≥ 0.70 | — | — |
| FP reduction (Stage 1) | ≥ 60% | — | — |
| Latency / frame | < 300 ms | — | — |
| Throughput | ≥ 5 FPS | — | — |
| COOL: Graviton vs x86 latency | measured | — | — |

_Link each filled row to the `EXP-NNN` that produced it._
