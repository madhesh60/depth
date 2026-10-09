# DEPTH — video script (≤ 5:00)

**One sentence the judges should remember:** *DEPTH isn't a better detector. It turns an imperfect one
into something a cleanup crew can depend on: a promise, a proof and a plan, with a person deciding
every action.*

Required by the rules: team, the app working, the architecture, results. Record at 1920×1080 with the
studio at 100% zoom and the dock set to "Agent log ▴" (slim). Every number said aloud is on screen,
and on screen from a regenerated report.

| time | on screen | narration (spoken) |
|---|---|---|
| **0:00–0:20** | Title card: DEPTH · Team Syndicate · Madhesh (solo). Then a sonogram with a ghost pot. | "Lost crab pots keep fishing for years. Side-scan sonar can see them — but someone has to find them in hours of sonar, and a detector that misses things silently is not a tool a crew can plan a boat day on." |
| **0:20–0:45** | Architecture diagram (`docs/img/architecture.svg`); highlight the five OpenCV stages, then EC2 Graviton + COOL. | "DEPTH is an OpenCV 5 agent on AWS Graviton with COOL. It measures the sonar, sees, proves, decides with calibrated promises, and plans — and a named person approves every action." |
| **0:45–1:25** | **Analyze**: pick a sample → seabed line appears → evidence card (tier, P(pot), shadow, relative height). Toggle **3D twin**: the acoustic triangle. | "First, the agent measures the sonar itself — the seabed line is its altitude, ping by ping, checked without labels: port and starboard agree to 1.6 pixels. Every find gets evidence and a tier that carries a promise." |
| **1:25–1:55** | Guarantees panel: "≥ 79% of pots reach a human · held on unseen test: 81%". Then `docs/calibration_exp003.md` table. | "The promise is fit on held-out recordings and verified once on unseen ones: at least 79% of pots reach a person — it held at 81%. And DEPTH says what it cannot promise: with this model, no auto-confirm." |
| **1:55–2:30** | **Survey**: run → KPI tiles, map, inspection route, re-survey passes (violet). Click a pass: "shadow must flip 110° → 290°". Toggle **⊘ without Stage 1**: ghost pins. | "A survey becomes a plan: a budgeted review queue, an inspection route, and second-look passes from the opposite side — where a real object's shadow must flip. Remove the seabed measurement and 193 of 264 pins fall outside their own error circle: OpenCV decides where the boat goes." |
| **2:30–2:40** | Survey → GPS source "Raw recording · real per-ping GPS": the track on the river in satellite imagery; the note "range scale measured 2.19 cm/sample ± 14%". | "And it reads real sonar recordings straight off the unit: every pin from its own ping's GPS, every metre measured — the sonar's depth against the seabed line." |
| **2:40–3:15** | Hazard table: **✓ confirm** three cards → the numbered **recovery route** appears; mark one **recovered**; impact ledger + reviewed precision. | "Nothing moves without a person. Confirm a card and the agent re-plans the recovery route. The crew reports back, and DEPTH keeps an impact ledger — including how precise its queue really was." |
| **3:15–3:45** | **Connect** tab: MCP, OGC API, webhooks; QGIS showing the `hazards` / `work_orders` layers; the drop-in panel. Claude calling `run_survey` and `request_human_approval`; the request appearing in **Approvals**. | "It plugs into what ocean agencies already run: QGIS and ArcGIS read it as a standard OGC layer, operations systems get signed webhooks, and any AI agent can drive it over MCP — agents can ask, only people approve." |
| **3:45–4:25** | COOL benchmark chart (x86 stock · Graviton stock · Graviton COOL) + provenance line "COOL"; the accuracy–speed–cost table. | "On Graviton with COOL, the whole product workload — not a toy kernel — runs at <X> ms per frame, <Y>× … versus x86, at <$Z> per survey-hour. Provenance proves which OpenCV build produced every result." *(fill from `docs/cool_benchmark.md` after the EC2 runs)* |
| **4:25–4:50** | `experiments.md` headings: STUDY-01, 07, 11b, 13 (negative results); failure gallery. | "We publish what didn't work — the ROI gate, re-looks, seam inference, test-time upscaling — and where it fails: pots cut at frame edges, low contrast, far range. That's what EXP-002 trains against." |
| **4:50–5:00** | Closing card: repo URL, live URL, "a promise · a proof · a plan". | "DEPTH: a promise, a proof and a plan — so the boat goes where the pots are." |

## Recording checklist

- Live URL working (CloudFront) and the model provenance line reading **COOL** before recording the
  Graviton segment; otherwise record locally and say "recorded locally; the live link runs on Graviton".
- Clear the approvals and feedback logs first (fresh `runs/approvals`, `runs/feedback`) so the counts start at 0.
- Use the **60-s demo** button as a rehearsal of the Analyze → Survey path.
- Captions on (judges watch muted); keep the pointer still between clicks.
- Upload unlisted; put the link in the README and the submission form.
