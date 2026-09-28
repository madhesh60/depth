# Outreach — a real-world quote and real GPS (drafts for the owner to send)

Two asks close the biggest real-world-impact gaps:

- **A quote from an organisation.** Two or three sentences from someone who plans ghost-gear removal
  or runs sonar surveys, saying whether this would help.
- **Real GPS.** One side-scan recording (Humminbird `.DAT` / `.SON` + `.IDX`, or PINGMapper output with
  per-ping positions) so the map shows real coordinates instead of the labelled demo track.

**Who to write to** (find current contacts on each organisation's site; start with the first):

1. **The crab-pot sonar dataset's authors** (PINGEcosystem / PINGMapper, the source of
   `sss-crab-pot-detection-ds`). Their raw recordings carry GPS, and they know the use case best.
2. **A Chesapeake Bay derelict-crab-pot removal programme** — the data come from the bay.
3. **National Institute of Ocean Technology (NIOT), Chennai** — the ocean-observation and
   marine-technology agency the integration layer (MCP / OGC / webhooks) was built for.
4. **Global Ghost Gear Initiative** (Ocean Conservancy) — the international ghost-gear network.

Send before **Oct 10** to leave time for replies. Quote the reply verbatim in the report and the video
only with the sender's permission, and name the organisation only if they agree.

---

## Email A — dataset authors / survey teams (quote + a recording with GPS)

**Subject:** Your crab-pot side-scan dataset → a human-approved cleanup plan (OpenCV AI Competition)

Hello <name>,

I'm building **DEPTH** for the OpenCV AI Competition 2026 (AWS / OpenCV Foundation), using your
side-scan crab-pot dataset (CC BY-SA, credited in the app and the report). It turns sonar frames into a
cleanup plan that a person approves:

- it tracks the seabed to measure the sonar's altitude;
- it promises a share of pots that reach a reviewer — at least 65%, which held at 86% on unseen recordings;
- it plans opposite-side re-survey passes, and the recovery route only includes pots a person confirmed.

Code and report: https://github.com/madhesh60/depth

Two small questions:

1. **Would this be useful to a team planning derelict-pot removal?** Two or three sentences of honest
   feedback (critical is fine) would mean a lot. I'd quote it only with your permission.
2. **Is there a recording with GPS I could use**, e.g. one of the raw Humminbird recordings behind the
   dataset, or a PINGMapper export with per-ping positions? Today the map uses a clearly labelled
   synthetic track because the published frames carry no coordinates.

3. **A licence question:** the dataset card's metadata says CC BY-SA 4.0, but its text says GPL. Which
   applies? I redistribute 8 sample frames with attribution and want to get it right.

Thank you for publishing the data.

<your name> — Team Syndicate

---

## Email B — agencies / cleanup organisations (quote + fit with their systems)

**Subject:** Ghost-gear detection from side-scan sonar that plugs into your GIS and operations systems

Hello <name>,

I'm building **DEPTH**, an open-source system for the OpenCV AI Competition 2026 that finds derelict
fishing gear and wreck debris in side-scan sonar and turns it into a plan that a person approves. It runs
on AWS and speaks the standards your systems already use:

- **OGC API – Features** — hazards, routes and approved work orders open directly in QGIS or ArcGIS;
- **signed webhooks** — survey results and approvals go to operations systems;
- **MCP** — AI agents can use it, but they can only *ask* for approval.

Positions of possible war graves or heritage wrecks are generalised in public shares.

Would this fit how your teams plan survey or recovery work? A short reply — even "this part is useless,
we'd need X" — would help me make it genuinely usable. I would quote it only with your permission.

Code and report: https://github.com/madhesh60/depth

<your name> — Team Syndicate
