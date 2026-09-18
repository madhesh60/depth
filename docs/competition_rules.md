# Competition Rules & Requirements — OpenCV AI Competition 2026

> Reference digest of the official rules, curated for Team Syndicate. Authoritative source
> is the organisers' site; this file is a working summary. See [`AGENT.md`](../AGENT.md) for
> our submitted proposal and [`architecture.md`](../architecture.md) for how we satisfy each
> requirement.

**Sponsor:** Amazon Web Services · **Administrator:** OpenCV Foundation

---

## 1. Key dates

| Date | Event |
|---|---|
| 2026-08-12 | Competition launch & registration |
| 2026-08-13 | Proposal submissions open |
| 2026-08-26 | **Build phase begins** (two months) |
| 2026-09-21 → 10-02 | Grant recipients' required 30-min Zoom check-in |
| **2026-10-26, 23:59 PT** | **Final submission deadline** |
| 2026-10-27 → 11-09 | Judging |
| 2026-11-10 | Winners announced (OpenCV Live! webinar) |

---

## 2. Core eligibility requirements

Every entry **must**:

1. Use **OpenCV 5** for *substantive* image or video analysis.
2. Run a **meaningful component on AWS**.

Projects may use any language or supporting hardware provided these two requirements are met.
The competition emphasises **Physical AI and Generative AI** where visual understanding drives
useful decisions, actions, predictions, or interactions.

---

## 3. Grant terms

- 50 teams received a **$150 AWS Cloud Compute Grant**. *(Team Syndicate — awarded.)*
- Grant recipients must complete **one 30-min Zoom check-in** (Sep 21 – Oct 2) and show active
  development to unlock the remaining 50% of the grant.
- New AWS customers may also access **up to $200 in Free Tier credits** (subject to AWS terms).
- All teams that submitted a proposal remain eligible to continue the build phase and win.

---

## 4. Featured award paths (each a separate $1,000 prize)

Entries may pursue either path, both, or neither, and remain eligible for prizes.

### 4.1 Best Use of COOL — *Team Syndicate's focus*
Use the **Cloud-Optimized OpenCV Library (COOL)** (AWS Marketplace, optimised for Graviton)
to accelerate vision operations and scale from prototype to cloud deployment.

**To qualify:** COOL must execute the **claimed core workload** on AWS Graviton (Arm), or the
Arm component of a documented hybrid architecture.

**Strong submissions:**
- Show COOL executing the core image/video workload on the Arm path.
- Report reproducible measurements (latency, throughput, utilisation, cost, or developer
  productivity) against an appropriate **baseline**.
- Explain any x86/Arm coexistence, container, server, or serverless architecture.

**Required evidence:** COOL version · AWS instance/deployment config · reproducible method,
inputs, baselines, results · evidence COOL runs the claimed core workload.

### 4.2 Agentic Vision — *Team Syndicate's stretch*
Build a workflow where an agent uses OpenCV 5 tools in a multi-step
**perception → decision → action** loop. Image/video results **must influence a subsequent
plan, tool call, action, or human-approval request**. A chatbot that only explains a fixed
vision result does **not** qualify.

**Required evidence:** agent workflow diagram (perception/decision/action) · a trace showing
OpenCV 5 output changing a later decision/action · evaluation of task success, failure
handling, observability, and human control.

> Note: merely using an AI coding assistant to *write* the entry does **not** count as an
> agentic workflow.

---

## 5. Final submission requirements

- [ ] **Technical report** — problem, users, architecture, OpenCV 5 implementation, AWS
      deployment, evaluation, limitations, responsible-use considerations.
- [ ] **Judge-accessible code repository or archive** (need not be open source).
- [ ] **Pinned dependencies** + clear build, deployment, and test instructions.
- [ ] **Architecture diagram** showing OpenCV 5 + AWS components (and COOL / agent where relevant).
- [ ] **Working web endpoint** or an arranged **live screen-share** demo.
- [ ] **Video ≤ 5 minutes** (public or unlisted) showing team, app working, architecture, results.
- [ ] **Evaluation evidence** appropriate to the project, including failure cases / limitations.

---

## 6. Suggested project areas (organisers especially welcome)

Active perception for autonomous inspection with agentic orchestration/MCP · physics-informed
video prediction · multi-agent visual SLAM · real-time spatial digital twins · **COOL-based
server/serverless/container/hybrid x86-Arm vision pipelines** · developer agents integrating
COOL with tools such as Claude Code / Codex / Kiro / MCP · healthcare, safety, accessibility,
agriculture, **environmental monitoring**, smart cities, education, retail, sports analytics.

> Marine-debris detection maps directly onto **environmental monitoring** + **COOL-based
> hybrid x86/Arm vision pipelines** — our two strongest alignment points.
