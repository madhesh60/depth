"""
brief.py — the **mission brief**: a one-page, plain-English hand-over for the boat crew / manager.

Every decision is already made (tiers, budget, routes, re-survey passes) before the brief exists — the
brief only *says* it. Two writers, one contract:

* :func:`template_brief` — deterministic markdown from :func:`facts`. Always available, reproducible,
  part of every survey's report set (``/api/report/brief``).
* :func:`llm_brief` — optional: Claude on **Amazon Bedrock** rewrites the same facts for readability
  (``DEPTH_BRIEF_LLM=bedrock``; AWS credentials from the instance role). The model **writes, never
  decides**: it sees only the compact facts JSON, and its text is accepted only if :func:`check`
  passes —

  - every number in it is a fact (as given, rounded, or a 0–1 share written as a percentage);
  - every hazard / pass ID it names exists;
  - the mandatory caveats are present (human approval; synthetic GPS; assumed card time);
  - it stays under the word cap.

  Any failure — SDK missing, no credentials, API error, refusal, truncation, an ungrounded number — falls
  back to the template, and the response says which writer produced the text and why.

Limitation (stated, not hidden): the check proves every number *came from the survey*; it cannot prove
the number is attached to the right noun. The facts are kept small to make such swaps unlikely.

    python -m src.agentic.brief runs/jobs/<survey_id>/mission.json [--llm]
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from .mission import SENSITIVE_CLASSES

log = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic.claude-opus-5"
MAX_WORDS = 350
_FIRST_CARDS = 5


# ---- facts: the only thing a writer may use --------------------------------------------------------
def facts(survey: dict, prov: Optional[dict] = None, public: bool = False) -> dict:
    """Compact, decision-complete facts of a survey (``SurveyResult.to_dict()`` or the ``json`` report).
    No coordinates — the brief names hazards by ID; positions live in the GIS exports."""
    m = survey.get("mission") or {}
    tracked = survey.get("tracked") or []
    prov = prov or survey.get("provenance") or {}
    g = m.get("guarantees") or {}
    ver = g.get("verified_on_test") or {}
    b = m.get("budget") or {}
    rp = m.get("resurvey_plan") or {}
    idx = {t["oid"]: t for t in tracked}
    by_cls: dict[str, int] = {}
    for t in tracked:
        by_cls[t["cls_name"]] = by_cls.get(t["cls_name"], 0) + 1
    counts = m.get("counts") or {}
    gps = "none" if not m.get("gps_available") else ("synthetic" if m.get("gps_synthetic") else "real")
    first = [idx[i] for i in (b.get("review_ids") or m.get("review_queue") or [])[:_FIRST_CARDS] if i in idx]
    lines = rp.get("lines") or []
    F: dict[str, Any] = {
        "survey_id": survey.get("survey_id"),
        "frames": len(survey["frames"]) if survey.get("frames") is not None else None,
        "model": prov.get("model"),
        "opencv": (prov.get("opencv") or {}).get("version"),
        "cool": bool((prov.get("opencv") or {}).get("is_cool_path")),
        "commit": prov.get("code_commit"),
        "gps": gps,
        "hazards": {"total": len(tracked), "confirmed": counts.get("confirmed", 0),
                    "review": counts.get("review", 0), "low_risk": counts.get("low_risk", 0)},
        "by_class": by_cls,
        "repeat_sightings_merged": m.get("repeat_merges", 0),
        "promise": {"recall": g.get("recall_promise"), "verified_recall_on_test": ver.get("recall"),
                    "held_on_test": ver.get("recall_promise_held"), "precision": g.get("precision_promise")},
        "auto_confirm": g.get("tau_confirm") is not None,
        "budget": ({"minutes": b.get("minutes"), "sec_per_card": b.get("sec_per_card"),
                    "sec_per_card_measured": not str(b.get("sec_per_card_source", "")).upper().startswith("ASSUMED"),
                    "cards_affordable": b.get("cards_affordable"), "cards_total": b.get("cards_total"),
                    "expected_pots_in_budget": b.get("expected_pots_in_budget"),
                    "expected_pots_in_queue": b.get("expected_pots_in_queue"),
                    "share_of_expected_pots": b.get("share_of_expected_pots"),
                    "minutes_for_whole_queue": b.get("minutes_for_whole_queue")} if b else None),
        "first_cards": [{"id": t["oid"], "class": t["cls_name"],
                         "p_pot": None if t.get("p_pot") is None else round(t["p_pot"], 2),
                         "shadow": t.get("shadow_quality"), "sightings": t.get("sightings", 1)} for t in first],
        "inspection_route": {"stops": len(m.get("inspection_route") or []), "length_m": m.get("inspection_length_m")},
        "recovery_route": {"stops": len(m.get("recovery_route") or []), "length_m": m.get("route_length_m")},
        "resurvey": ({"passes": len(lines), "boat_minutes": rp.get("boat_minutes_planned"),
                      "targets_covered": rp.get("targets_covered"), "targets_total": rp.get("targets_total"),
                      "first": ({"id": lines[0]["id"], "targets": len(lines[0]["targets"]),
                                 "side": lines[0]["targets_on"], "heading_deg": lines[0]["heading_deg"],
                                 "boat_min": lines[0]["boat_min"],
                                 "example": lines[0]["predictions"][0]["id"],
                                 "shadow_was_deg": lines[0]["predictions"][0]["shadow_was"],
                                 "shadow_must_point_deg": lines[0]["predictions"][0]["shadow_must_point"]}
                                if lines and lines[0].get("predictions") else None)} if rp else None),
        "protected_sites": sum(1 for t in tracked if t["cls_name"] in SENSITIVE_CLASSES),
        "public": public,
    }
    return F


def _pct(x: Optional[float]) -> str:
    return "?" if x is None else f"{100 * x:.0f}%"


def _n(x: Any) -> str:
    if x is None:
        return "?"
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)


# ---- writer 1: the deterministic template ----------------------------------------------------------
def template_brief(F: dict) -> str:
    h, p, b, rs = F["hazards"], F["promise"], F.get("budget"), F.get("resurvey")
    frames = f" in {F['frames']} frames" if F.get("frames") else ""
    L = [f"# Mission brief — {F.get('survey_id') or 'survey'}", "",
         f"_DEPTH · model {F.get('model')} · OpenCV {F.get('opencv')} ({'COOL build' if F.get('cool') else 'stock build'}) · "
         f"commit {F.get('commit')} · GPS: {F['gps']}{' (SYNTHETIC DEMO - not real positions)' if F['gps'] == 'synthetic' else ''}_", "",
         "## Bottom line", ""]
    if h["total"] == 0:
        L.append(f"No possible hazards were found{frames}.")
    else:
        conf = (f"{h['confirmed']} confirmed" if h["confirmed"] else "none confirmed automatically")
        L.append(f"**{h['total']} possible hazards**{frames}: {conf}, **{h['review']} for a human to review**, "
                 f"{h['low_risk']} low-risk (kept for audit).")
        cls = " · ".join(f"{c} {n}" for c, n in sorted(F["by_class"].items(), key=lambda kv: -kv[1]))
        L.append(f"By class: {cls}.")
        if F.get("repeat_sightings_merged"):
            L.append(f"{F['repeat_sightings_merged']} repeat sightings were merged into existing hazards.")
    L.append("")
    if p.get("recall") is not None:
        held = (f" — verified on unseen test data: {_pct(p.get('verified_recall_on_test'))}"
                if p.get("verified_recall_on_test") is not None else "")
        L.append(f"**The promise:** at least {_pct(p['recall'])} of real pots reach a human{held}.")
    if p.get("precision") is None:
        L.append("There is no precision promise, so nothing is confirmed automatically — every find is a card for a human.")
    L += ["", "## First hour", ""]
    if b:
        L.append(f"With **{_n(b['minutes'])} analyst-minute{'' if b['minutes'] == 1 else 's'}** at {_n(b['sec_per_card'])} s per card"
                 f"{'' if b['sec_per_card_measured'] else ' (assumed until the timed study measures it)'}: "
                 f"review **{b['cards_affordable']} of {b['cards_total']} cards** — expected "
                 f"{_n(b['expected_pots_in_budget'])} of {_n(b['expected_pots_in_queue'])} real pots in the queue "
                 f"({_pct(b['share_of_expected_pots'])}). The whole queue takes {_n(b['minutes_for_whole_queue'])} min.")
    elif h["review"]:
        L.append("No analyst budget was set; review the queue in order (most-likely-real first).")
    if F["first_cards"]:
        L.append("")
        L.append("Start with: " + " · ".join(
            f"**{c['id']}** ({c['class']}, P(pot) {_n(c['p_pot'])}, shadow {c['shadow']})" for c in F["first_cards"]) + ".")
    L += ["", "## On the water", ""]
    ir, rr = F["inspection_route"], F["recovery_route"]
    if F["gps"] == "none":
        L.append("No GPS in this survey: the plan is frame-ordered and there are no routes.")
    else:
        if ir["stops"]:
            L.append(f"- **Inspection route:** {ir['stops']} stops, {_n(ir['length_m'])} m (the cards above, nearest-first).")
        L.append(f"- **Recovery route:** {rr['stops']} stops, {_n(rr['length_m'])} m." if rr["stops"]
                 else "- **Recovery route:** none yet — nothing is confirmed until a human approves it.")
    if rs and rs.get("passes"):
        L.append(f"- **Re-survey:** {rs['passes']} pass{'' if rs['passes'] == 1 else 'es'}, {_n(rs['boat_minutes'])} boat-minutes, covering "
                 f"{rs['targets_covered']} of {rs['targets_total']} uncertain targets from the other side.")
        f1 = rs.get("first")
        if f1:
            L.append(f"- **Pass {f1['id']}** ({f1['targets']} targets on the {f1['side']} side, heading "
                     f"{_n(f1['heading_deg'])}°, {_n(f1['boat_min'])} min): a real object's shadow must flip — "
                     f"e.g. {f1['example']} from {_n(f1['shadow_was_deg'])}° to {_n(f1['shadow_must_point_deg'])}°. "
                     f"If it does not, the find was an artefact.")
    L += ["", "## Caveats", ""]
    if F["gps"] == "synthetic":
        L.append("- GPS is a **synthetic demo track**: positions are illustrative, not real.")
    if b and not b["sec_per_card_measured"]:
        L.append("- The per-card review time is **assumed**; the timed study replaces it with a measurement.")
    if F.get("protected_sites"):
        L.append(f"- {F['protected_sites']} finds are wreck-type (possible protected sites)"
                 + (": their positions are generalised in this public share." if F.get("public")
                    else ": generalise their positions before sharing publicly."))
    L.append("- Expected pot counts are calibrated expectations, not guarantees.")
    L.append("- **Nothing is dispatched without human approval.**")
    return "\n".join(L) + "\n"


# ---- the grounding check ---------------------------------------------------------------------------
_ID_LIKE = re.compile(r"\b[A-Za-z][A-Za-z_]*[-_]?\d[\w.\-]*")          # H041, EXP-001, Rec6_…, c8g.xlarge
_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)*")
_LIST = re.compile(r"^\s*\d+[.)]\s", re.M)
_REF = re.compile(r"\b(?:H\d{2,4}|RS\d{1,3})\b")
_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
          "eleven": 11, "twelve": 12, "dozen": 12, "twenty": 20, "hundred": 100, "thousand": 1000}


def _numbers(text: str) -> list[str]:
    t = _LIST.sub(" ", text)
    t = _ID_LIKE.sub(" ", t)
    out = []
    for tok in _NUM.findall(t):
        tok = tok.rstrip(",")
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", tok):
            tok = tok.replace(",", "")
        out += [x for x in tok.split(",") if x]
    return out


def _as_float(tok: str) -> Optional[float]:
    try:
        return float(tok)
    except ValueError:
        return None                                   # versions like 5.0.0 → string match only


def _fact_numbers(F: dict) -> tuple[set[str], list[float]]:
    blob = json.dumps(F)
    toks = set(_numbers(blob))
    vals = [v for v in (_as_float(t) for t in toks) if v is not None]
    return toks, vals


def _supported(tok: str, toks: set[str], vals: list[float]) -> bool:
    if tok in toks:
        return True
    x = _as_float(tok)
    if x is None:
        return False
    d = len(tok.split(".")[1]) if "." in tok else 0
    tol = 0.5 * 10 ** -d + 1e-9
    return any(abs(x - f) <= tol or abs(x - 100 * f) <= tol for f in vals)


def check(text: str, F: dict) -> dict:
    """Grounding report for a written brief against its facts."""
    toks, vals = _fact_numbers(F)
    nums = _numbers(text)
    bad = [n for n in nums if not _supported(n, toks, vals)]
    words = [w for w in re.findall(r"[A-Za-z]+", text.lower()) if w in _WORDS]
    bad += [w for w in words if not _supported(str(_WORDS[w]), toks, vals)]
    ids = set(json.dumps(F).split('"')) | {c["id"] for c in F.get("first_cards", [])}
    bad_ids = sorted({r for r in _REF.findall(text) if r not in ids})
    low = text.lower()
    need = {"human approval": "human" in low and "approv" in low}
    if F.get("gps") == "synthetic":
        need["synthetic GPS"] = "synthetic" in low
    if (F.get("budget") or {}).get("sec_per_card_measured") is False:
        need["assumed card time"] = "assum" in low
    missing = [k for k, ok in need.items() if not ok]
    n_words = len(re.findall(r"\S+", text))
    ok = not bad and not bad_ids and not missing and n_words <= MAX_WORDS
    return {"ok": ok, "numbers_checked": len(nums) + len(words), "unsupported_numbers": bad,
            "unknown_ids": bad_ids, "missing_caveats": missing, "words": n_words, "max_words": MAX_WORDS}


# ---- writer 2 (optional): Claude on Amazon Bedrock --------------------------------------------------
SYSTEM = f"""You write the mission brief for a marine-debris survey crew and their manager.
The survey is finished and every decision is already made; you only explain it clearly.

Rules — the brief is machine-checked and discarded if any rule is broken:
- Use ONLY the facts in the JSON the user sends. Add no outside knowledge.
- Write every quantity in digits exactly as it appears in the facts. A 0–1 share may be written as a
  percentage (0.248 -> 25%). Do not add, subtract, multiply or convert numbers, and do not spell numbers as words.
- Name hazards and re-survey passes only by the IDs in the facts (for example H041, RS1).
- You must say that nothing is dispatched without human approval.
- If "gps" is "synthetic", say the positions are from a synthetic demo track.
- If budget.sec_per_card_measured is false, say the per-card review time is assumed.
- Markdown with these sections: Bottom line, First hour, On the water, Caveats. No numbered lists.
- At most {MAX_WORDS - 50} words. Plain, direct language for a boat crew."""


def _client():
    import anthropic                                   # optional dependency: pip install "anthropic[bedrock]"
    cls = getattr(anthropic, "AnthropicBedrockMantle", None)
    if cls is None:
        raise RuntimeError("anthropic SDK has no AnthropicBedrockMantle (upgrade: pip install -U 'anthropic[bedrock]')")
    return cls(aws_region=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")


def llm_brief(F: dict, client=None, model: Optional[str] = None) -> str:
    """Claude rewrites the facts. Raises on any failure (the caller falls back to the template)."""
    client = client or _client()
    msg = client.messages.create(
        model=model or os.environ.get("DEPTH_BRIEF_MODEL", DEFAULT_MODEL),
        max_tokens=16000, system=SYSTEM,
        messages=[{"role": "user", "content": "Facts (JSON):\n" + json.dumps(F, indent=1)}])
    if msg.stop_reason == "refusal":
        raise RuntimeError("the model declined to write the brief")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("the brief was truncated")
    text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text").strip()
    if not text:
        raise RuntimeError("empty brief")
    return text + "\n"


def llm_enabled() -> bool:
    return os.environ.get("DEPTH_BRIEF_LLM", "").lower() == "bedrock"


def write(survey: dict, prov: Optional[dict] = None, public: bool = False, writer: str = "auto",
          client=None) -> dict:
    """The brief + how it was written. ``writer``: ``template`` | ``llm`` | ``auto`` (LLM when enabled)."""
    F = facts(survey, prov, public)
    tpl = template_brief(F)
    out = {"text": tpl, "writer": "template", "model": None, "fallback_reason": None,
           "grounding": check(tpl, F), "facts": F}
    want_llm = writer == "llm" or (writer == "auto" and (llm_enabled() or client is not None))
    if not want_llm:
        return out
    model = os.environ.get("DEPTH_BRIEF_MODEL", DEFAULT_MODEL)
    try:
        text = llm_brief(F, client=client, model=model)
    except Exception as e:                                # SDK missing, no creds, API error, refusal …
        log.warning("LLM brief failed: %s", e)
        out["fallback_reason"] = f"{type(e).__name__}: {e}"[:300]
        return out
    g = check(text, F)
    if not g["ok"]:
        out["fallback_reason"] = "LLM brief failed the grounding check"
        out["rejected"] = {"text": text, "grounding": g}
        return out
    out.update(text=text, writer="llm", model=model, grounding=g)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="mission brief from a survey JSON report")
    ap.add_argument("survey_json")
    ap.add_argument("--llm", action="store_true", help="try Claude on Bedrock (falls back to the template)")
    a = ap.parse_args()
    with open(a.survey_json, encoding="utf-8") as f:
        d = json.load(f)
    r = write(d, writer="llm" if a.llm else "template")
    print(r["text"])
    print(json.dumps({k: r[k] for k in ("writer", "model", "fallback_reason", "grounding")}, indent=1))


if __name__ == "__main__":
    main()
