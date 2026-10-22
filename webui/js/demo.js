/* DEPTH studio — the guided 60-second demo (the judge path).
   Zero-build: index.html loads webui/js/*.js in order as classic scripts sharing one global scope;
   the boot runs on DOMContentLoaded (core.js), after every file has loaded. API: src/dashboard/routes/. */
"use strict";

/* ------------------------------------------------------------------ GUIDED 60-SECOND DEMO (the judge path) */
const Demo = {
  on: false, token: 0,
  steps: 7,                                      // the judge path: the agentic loop end to end
  async run() {
    this.on = true; const tok = ++this.token;
    $("#demoBtn").classList.add("is-on"); $("#demoBtn").textContent = "Stop demo"; $("#demoBar").hidden = false;
    const stopper = e => { if (e.target.closest && (e.target.closest("#demoBar") || e.target.closest("#demoBtn"))) return; this.stop(); };
    setTimeout(() => { window.addEventListener("keydown", this._k = () => this.stop(), { once: true }); window.addEventListener("pointerdown", this._p = stopper); }, 400);
    const alive = () => this.on && tok === this.token;
    const wait = ms => new Promise(r => setTimeout(r, ms));
    const say = async (i, txt, ms) => {
      if (!alive()) throw 0;
      $("#demoStep").textContent = `${i}/${this.steps}`; $("#demoTxt").innerHTML = txt;
      $("#demoDots").innerHTML = Array.from({ length: this.steps }, (_, k) => `<i class="${k < i ? "on" : ""}"></i>`).join("");
      if (ms) await wait(ms);
      if (!alive()) throw 0;
    };
    const mode = m => $(`.mode-btn[data-mode="${m}"]`).click();
    const wasActive = localStorage.getItem("depth.active");
    const setActive = on => { localStorage.setItem("depth.active", on ? "1" : "0"); $$(".activeToggle").forEach(x => x.checked = on); };
    const g = (state.calibration && state.calibration.guarantees) || {};
    try {
      mode("analyze"); if (Twin.dim === "3d") $('.dim-btn[data-dim="2d"]').click();
      await say(1, "<b>DEPTH</b> turns side-scan sonar into a human-approved cleanup plan. One frame first: <b>Stage 1</b> (OpenCV) tracks the seabed — the dashed line — the sonar's altitude, measured per ping.", 600);
      const card = $('.sample-card[data-id="sample-01"]') || $(".sample-card"); if (card) card.click();
      setActive(false); await runAnalyze(); await wait(3000);
      await say(2, `Every find gets <b>evidence</b> and a <b>calibrated tier</b>: ${g.recall_promise ? `≥ ${Math.round(g.recall_promise * 100)}%` : "a calibrated share"} of pots reach a person — a promise fit on held-out recordings and <b>held on unseen test</b>${(g.verified_on_test || {}).recall ? ` (${Math.round(g.verified_on_test.recall * 1000) / 10}%)` : ""}.`, 6000);
      cancelTour(); toastHide();
      setActive(true); await runAnalyze(); state._tourOptOut = true; cancelTour(); await wait(400);
      const cs = state._analyze.candidates || [];
      const changed = c => c.card_of == null && c.evidence && c.evidence.action && c.evidence.action !== c.evidence.action0;
      let ci = cs.findIndex(c => changed(c) && c.evidence.conflict); if (ci < 0) ci = cs.findIndex(changed);
      if (ci >= 0) { selectCandidate(ci); await wait(500); const at = $(`.ev-card[data-id="${ci}"] .atrace`); if (at) at.scrollIntoView({ block: "center" }); }
      const ev = ci >= 0 ? cs[ci].evidence : null;
      await say(3, "<b>Active vision</b> (experimental, labelled): per find the agent asks which OpenCV tool could change its decision, runs only that one, and updates its belief. "
        + (ev ? `Here the OpenCV evidence moves P(pot) ${Math.round((cs[ci].p_pot ?? ev.p_pot ?? 0) * 100)}% → ${Math.round(ev.p_evidence * 100)}%: <b>${ACT[ev.action0] || ev.action0} → ${ACT[ev.action] || ev.action}</b>${ev.request_resurvey ? ", and the agent asks for an opposite-side pass" : ""}. ` : "")
        + "It failed its own validation, so it stays off by default.", 9000);
      setActive(wasActive === "1"); cancelTour(); toastHide();
      await say(4, "Failures are handled on purpose. A drill: the evidence tool crashes → the agent falls back to the detector score, sends the find to a person and <b>never auto-confirms</b>.", 0);
      await runAnalyze("tool_failed"); await wait(4200);
      $$(".notice .nt-x").forEach(b => b.click());
      mode("survey");
      const rec = $("#gpsRecording") && !$("#gpsRecording").hidden;
      if (rec) { $("#gpsMode").value = "recording"; }
      await say(5, rec ? "A <b>raw sonar recording</b> with real GPS. Every metre on the map hangs on the sonar's altitude, so the agent <b>cross-checks OpenCV's seabed track against the depth sounder</b>, re-runs the tracker where they conflict and trusts the image where the sounder lost lock." : "Now a whole survey: hazards with error radii, analyst + boat budgets, and opposite-side re-survey passes.", 0);
      await runSurvey(); await wait(1500);
      const agentTab = [...$$(".tabs .tab")].find(x => x.textContent.trim() === "Agent"); if (agentTab) agentTab.click();
      await wait(600);
      const chip = $('#agentPanel .geo-chip.geo-corrected') || $("#agentPanel .geo-chip.geo-image_trusted");
      if (chip) { chip.click(); await wait(300); const gd = $("#geoDetail"); if (gd) gd.scrollIntoView({ block: "start", behavior: "smooth" }); }
      if (rec) await say(5, "Red: the tracker alone. Blue: the sounder. Green: what the agent accepted after re-measuring. Checked on a <b>1-hour recording it never saw</b> — registered criteria, one failure reported (STUDY-16).", 8000);
      const disp = (state.survey && state.survey.dispatch) || {};
      const k = Object.keys(disp).find(x => x.startsWith("resurvey:") && disp[x].status === "pending");
      if (k) {
        const reco = $("#agentPanel .recos"); if (reco) reco.scrollIntoView({ block: "start", behavior: "smooth" });
        await say(6, "The agent <b>recommends</b> a re-survey pass — it can only ask. A person rejects it…", 3600);
        if (!($("#apprName").value || "").trim()) $("#apprName").value = "Guided demo";
        await Appr.decide(disp[k].id, "declined"); await wait(800);
        const t2 = [...$$(".tabs .tab")].find(x => x.textContent.trim() === "Agent"); if (t2) t2.click();
        await wait(300);
        const dn = [...$$("#agentPanel .ch-name")].find(x => x.textContent.includes("Pass declined"));
        if (dn) dn.closest("li").scrollIntoView({ block: "center", behavior: "smooth" });
        await say(6, "…and the agent <b>re-plans</b>: the pass is dropped for good, the boat time is re-ranked, and its target moves to the front of the inspection queue. Every step is in the decision log.", 7000);
      }
      const bt = [...$$(".tabs .tab")].find(x => x.textContent.trim() === "Brief"); if (bt) bt.click();
      await say(7, "The agent hands over a one-page <b>mission brief</b>; every number in it is machine-checked against the survey. Every export carries <b>provenance</b> (model, calibration, the OpenCV build) and a decision log. Nothing is dispatched without a named person. ✓", 7000);
    } catch (e) { /* stopped */ }
    if (tok === this.token) this.stop();
  },
  stop() {
    this.on = false; this.token++;
    $("#demoBtn").classList.remove("is-on"); $("#demoBtn").textContent = "Guided demo"; $("#demoBar").hidden = true;
    if (this._k) window.removeEventListener("keydown", this._k); if (this._p) window.removeEventListener("pointerdown", this._p);
    if (Twin.survey) Twin.survey.play = false;
  },
};
