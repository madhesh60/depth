"""
routes/people.py — the people loop: training labels, the timed user study, the effort curve and the blinded
audit.
"""
from __future__ import annotations

import json
from typing import Optional

import cv2
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import Response

from src.agentic import effort as effort_mod, study as study_mod
from src.detection import fp_audit
from src.detection.calibration import SHIPPED_MODEL, load_calibration
from .. import service as S

router = APIRouter()


@router.post("/api/feedback")
def feedback(rec: dict = Body(...)):
    """One human decision (confirm / reject / missed) on a box → an append-only training label. The frame must
    be a sample in the catalogue or a recently analysed upload."""
    ref = rec.get("frame_ref") or S._frame_ref(str(rec.get("frame_id", "")))
    if not isinstance(ref, dict):
        raise HTTPException(400, "frame_ref must be an object: {\"sample\": <id>} or {\"upload\": <sha>}")
    rec["frame_ref"] = ref
    img = S._UPLOADS.get(ref.get("upload")) if ref.get("upload") else None
    if ref.get("upload") and img is None and S._FEEDBACK.image_path(ref) is None:
        raise HTTPException(410, "that uploaded frame is no longer cached - re-run analyze, then label it")
    try:
        saved = S._FEEDBACK.add(rec, img)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"saved": saved, "stats": S._FEEDBACK.stats()}


@router.get("/api/feedback/stats")
def feedback_stats():
    return {**S._FEEDBACK.stats(), "agreement_with_gt": S._FEEDBACK.agreement()}


# ---- timed user study + analyst-effort curve (src/agentic/study.py, effort.py) ------------------
@router.get("/api/study/plan")
def study_plan(participant: str = Query("anon", max_length=40)):
    """Counterbalanced session: half the frames manual, half as the agent's cards (in queue order)."""
    try:
        plan = study_mod.plan(participant)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    pipe = S.get_pipeline()
    g = load_calibration().guaranteed_class
    cards = []
    for fr in plan["card_frames"]:
        fp = study_mod.frame_path(fr["frame_id"])
        img = cv2.imread(str(fp)) if fp else None
        if img is None:
            continue
        with S._INFER_LOCK:
            res = pipe.run_frame(img, frame_id=fr["frame_id"])
        for i, c in enumerate(res.candidates):
            if c.verdict is None or c.verdict.value not in ("confirmed", "review") or c.cls_name != g or c.card_of is not None:
                continue
            cards.append({"card_id": f"{fr['frame_id'][-12:]}-{i}", "frame_id": fr["frame_id"], "bbox": list(c.bbox),
                          "boxes": [list(c.bbox)] + [list(res.candidates[k].bbox) for k in c.card_members],
                          "p_pot": c.evidence.p_pot if c.evidence else None, "conf": round(c.conf, 3),
                          "verdict": c.verdict.value,
                          "crop_png": S._evidence_crop(img, c, 280, [res.candidates[k].bbox for k in c.card_members])})
    cards.sort(key=lambda c: (-(c["p_pot"] or 0), -c["conf"]))           # the DEPTH queue order
    S._STUDY[plan["session_id"]] = {"plan": plan, "cards": {c["card_id"]: c for c in cards}}
    while len(S._STUDY) > 32:
        S._STUDY.popitem(last=False)
    return {**plan, "cards": cards}


@router.get("/api/study/frame/{frame_id}")
def study_frame(frame_id: str):
    fp = study_mod.frame_path(frame_id)
    if not fp or not fp.exists():
        raise HTTPException(404, "not a study frame")
    return Response(fp.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/api/study/result")
def study_result(body: dict = Body(...)):
    sess = S._STUDY.get(str(body.get("session_id", "")))
    if sess is None:
        raise HTTPException(404, "unknown or expired study session - start a new one")
    rec = study_mod.score(sess["plan"], body, sess["cards"])
    study_mod.save(rec)
    S._STUDY.pop(sess["plan"]["session_id"], None)
    return {"session": {k: v for k, v in rec.items() if k != "raw"}, "summary": study_mod.summary()}


@router.get("/api/study/summary")
def study_summary():
    params, prov = effort_mod.measured_params()
    return {"summary": study_mod.summary(), "effort_params": params, "effort_provenance": prov}


@router.get("/api/effort")
def effort(sec_per_frame: Optional[float] = Query(None, gt=0, le=600),
           sec_per_card: Optional[float] = Query(None, gt=0, le=120),
           manual_recall: Optional[float] = Query(None, gt=0, le=1),
           card_accuracy: Optional[float] = Query(None, gt=0, le=1),
           mode: str = Query("standard", pattern="^(standard|look_harder)$",
                             description="perception mode: its own queue and promise")):
    """Recall-vs-minutes curves for manual review / detector list / DEPTH queue with any timings."""
    ep = S.REPO / "models" / load_calibration().data.get("model", SHIPPED_MODEL) / "effort_curve.json"
    if not ep.exists():
        raise HTTPException(404, "no effort data for this model - run python -m src.agentic.effort")
    E = json.loads(ep.read_text(encoding="utf-8"))
    modes = ["standard", *sorted((E.get("modes") or {}).keys())]
    E = effort_mod.for_mode(E, mode)
    params, prov = effort_mod.measured_params()
    for k, v in (("sec_per_frame", sec_per_frame), ("sec_per_card", sec_per_card),
                 ("manual_recall", manual_recall), ("card_accuracy", card_accuracy)):
        if v is not None:
            params[k], prov = v, "set in the UI"
    C = effort_mod.curves(E, **params)
    return {"curves": C, "provenance": prov, "frames": E["frames"], "pots": E["pots"],
            "frames_per_survey_hour": E["frames_per_survey_hour"], "split": E.get("split"),
            "mode": E.get("mode", "standard"), "modes": modes, "cards": E.get("cards")}


# ---- blinded false-alarm audit (src/detection/fp_audit.py) ------------------------------------
def _audit_model() -> str:
    return load_calibration().data.get("model", SHIPPED_MODEL)


@router.get("/api/audit/items")
def audit_items(annotator: str = Query("", max_length=40)):
    """Blind item list (ids + image paths only) + which ones this annotator already tagged."""
    items = fp_audit.public_items(_audit_model())
    done = fp_audit._latest_tags().get(annotator.strip() or "anon", {}) if annotator else {}
    return {"items": items, "tagged": done, "tags": list(fp_audit.TAGS)}


@router.post("/api/audit/tag")
def audit_tag(body: dict = Body(...)):
    try:
        rec = fp_audit.add_tag(str(body.get("item", "")), str(body.get("tag", "")), str(body.get("annotator", "")))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"saved": rec}


@router.get("/api/audit/summary")
def audit_summary():
    return fp_audit.summary(_audit_model())
