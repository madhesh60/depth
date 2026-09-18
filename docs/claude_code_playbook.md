# Claude Code Playbook — Working Method for This Project

> How to drive this build with Claude Code: **you make the ML and product decisions;
> Claude executes, tests, documents, and moves the project forward.** Reference material,
> not a spec. See [`CLAUDE.md`](../CLAUDE.md) for the enforced session instructions.

---

## 1. Principle

Use Claude Code across **more than coding** — architecture, ML research, data engineering,
error analysis, backend, AWS, and skeptical self-review. The goal is a repeatable loop:

```
YOU define the goal → Claude researches → Claude proposes options → YOU choose
   → Claude implements → Claude tests → YOU inspect results → Claude improves
```

Not: *"Claude → build everything → hope it works."* You should end the month a better
engineer, not dependent on the tool.

---

## 2. The five+ roles to use Claude for

| Role | Example ask |
|---|---|
| **Software architect** | "Study the repo and design the architecture. Don't modify anything. Identify components, data flow, APIs, ML pipeline, AWS services, risks. Write to `architecture.md`." |
| **ML research assistant** | "Compare YOLO, RT-DETR, segmentation and hybrid OpenCV+DL for sonar debris. Focus on recall, speed, hackathon feasibility. Recommend with evidence. Then design 3 experiments — don't train yet." |
| **Data engineer** | "Inspect the dataset: corrupt/duplicate images, missing/incorrect annotations, extreme resolutions, class imbalance, train/val leakage. Produce a report first; then build scripts to detect these automatically." |
| **Error analyst** | "Analyse false negatives. Group by cause (low-vis, occlusion, small object, blur, domain shift, annotation error). Find patterns and recommend the next dataset/augmentation change." |
| **Backend/AWS engineer** | "Implement the upload→inference→report API. Inspect the repo first, design the contract, implement, add tests + Docker + docs." / "Design the cheapest practical AWS architecture for the demo; audit for cost, security, failure points." |
| **Skeptical judge** | "Act as a skeptical international hackathon judge. Find every weakness in novelty, ML validity, OpenCV/AWS usage, UX, reliability, demoability. Do not praise." |

---

## 3. The experiment loop

Maintain [`experiments.md`](../experiments.md). Every run records model, dataset version,
input size, augmentation, hyperparameters, and metrics (P / R / mAP / per-class / confusion),
plus error analysis and the next hypothesis. This turns random training into a research loop:

```
Hypothesis → Experiment → Training → Evaluation → Error analysis → New hypothesis
```

Ask: **"Why is my model failing?"** — not just **"What model should I use?"**

---

## 4. Give real acceptance criteria

Weak: *"Make detection better."*
Strong: *"Improve ghost-net recall without dropping precision below the EXP-001 baseline. Run
the eval set, compare, and revert if it doesn't help."*

Explicit success criteria + verification beats vague instructions every time.

---

## 5. Git as a safety net

`Claude changes code → tests → you inspect → commit → next task.` Small, typed commits
(`feat:`, `fix:`, `data:`, `infra:`, `docs:`, `exp:`). Never let the project become one giant
uncommitted directory.

---

## 6. Documents Claude keeps current

| File | Update cadence |
|---|---|
| [`progress.md`](../progress.md) | End of every session |
| [`experiments.md`](../experiments.md) | After every training run |
| [`TODO.md`](../TODO.md) | As tasks open/close |
| [`architecture.md`](../architecture.md) | When a structural decision changes |

---

## 7. Priority of Claude-Code time for this project

| Task | Priority |
|---|---|
| Dataset inspection / cleaning | Very high |
| ML experiment automation | Very high |
| Error analysis | Very high |
| Architecture | Very high |
| Backend / API | Very high |
| AWS deployment (+ COOL benchmark) | Very high |
| Testing · code review | Very high |
| Research | High |
| Frontend | High |
| Documentation · presentation | Medium |
| Random feature development | Low |

> **Do not burn the month on a fancy frontend while the model and pipeline are weak.**

---

## 8. Month plan (mirrors [`TODO.md`](../TODO.md))

- **Week 1 — Foundation:** architecture, dataset audit, research, model selection, AWS design, baseline.
- **Week 2 — ML:** data quality → augmentation → training → evaluation → error analysis → repeat (controlled experiments, not 20 random models).
- **Week 3 — Product:** model → inference API → backend → AWS → dashboard. Judges see a product, not a notebook.
- **Week 4 — Winning prep:** reliability → performance → UI polish → AWS deploy → demo rehearsal → cleanup → full pre-submission audit.

---

## 9. Standing rules for Claude (see `CLAUDE.md`)

1. Inspect the repo before making claims; don't guess about files, APIs, datasets, or library behaviour.
2. Explain the approach and risks before major implementation.
3. Prefer simple, maintainable design; no unnecessary abstractions.
4. Run tests/validation after significant changes; never hide errors.
5. Never modify the test set or introduce data leakage.
6. Track decisions in `architecture.md`, progress in `progress.md`, experiments in `experiments.md`, open work in `TODO.md`.
7. Analyse false positives/negatives, not just aggregate accuracy.
8. For AWS: prioritise reliability, security, reasonable cost.
9. Every feature must support the core problem — no impressive-but-pointless additions.
10. Large task → break into smaller verifiable stages: **PLAN → IMPLEMENT → TEST → REVIEW → DOCUMENT → COMMIT.**
