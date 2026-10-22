"""
routes/analyze.py — one frame: See → Prove → Decide, with the agent's mode choice priced against the server
queue, after checking that the recall promise applies to the frame at all (``Reference.check_frame``).
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from .. import samples as samples_mod
from src.agentic.core import Goal, calibrated_modes, choose_mode
from src.agentic.incidents import incident
from src.agentic.pipeline import validate_frame
from src.agentic.provenance import stamp as provenance_stamp
from src.agentic.twin import frame_twin
from src.dashboard.detector_queue import interactive_budget
from .. import service as S

router = APIRouter()


def _interactive_goal(recall: Optional[float], compute_s: Optional[float], ms_per_pass) -> tuple[Goal, Optional[dict]]:
    """The goal of one analyze call. An explicit ``compute_s`` is pure compute; the default interactive
    budget is answer time, so the compute already queued for the network and the measured rest of an answer
    are taken off it first."""
    if compute_s is not None:
        return Goal(recall=recall, compute_s=compute_s), None
    n, wait = S._QUEUE.ahead()
    load = interactive_budget(S.INTERACTIVE_S, n, wait, bool(S._JOBS.counts().get("running")), ms_per_pass,
                              S._QUEUE.overhead_s())
    return Goal(recall=recall, compute_s=max(0.2, load["left_s"])), load


@router.post("/api/analyze")
def analyze(sample: Optional[str] = Query(None), nadir: Optional[str] = Query(None),
            file: Optional[UploadFile] = File(None), simulate: Optional[str] = Query(None),
            recall: Optional[float] = Query(None, ge=0.5, le=0.99, description="recall promise to keep (the goal)"),
            compute_s: Optional[float] = Query(None, ge=0.2, le=60, description="compute budget, s (default interactive)"),
            active: bool = Query(False, description="run active vision even if not validated (EXPERIMENTAL)"),
            images: str = Query("lean", pattern="^(lean|full)$",
                                description="full = also the server-rendered overlay and the plain zoom per find")):
    faults = S._drills(simulate) | ({"opt:active"} if active is True else set())
    if "corrupt_frame" in faults:
        raise S._fail(422, incident("corrupt_frame", "the frame could not be decoded (failure drill)", simulated=True))
    if file is not None:
        try:
            frame_id, frame = S._decode_upload(file)
        except HTTPException as e:
            if e.status_code != 400:
                raise
            raise S._fail(422, incident("corrupt_frame", str(e.detail))) from e
    elif sample:
        p = samples_mod.sample_path(sample)
        if not p or not p.exists():
            raise HTTPException(404, f"sample not found: {sample}")
        frame, frame_id = cv2.imread(str(p)), p.stem
        if frame is None:
            raise HTTPException(400, "could not decode image")
    else:
        raise HTTPException(400, "provide ?sample=ID or a multipart file")
    if nadir and nadir not in ("top", "bottom", "left", "right", "auto"):
        raise HTTPException(400, "nadir must be top|bottom|left|right|auto")
    bad = validate_frame(frame)
    if bad:
        raise S._fail(422, incident(bad, f"{frame_id}: not a usable sonogram"))

    t0 = time.perf_counter()
    pipe = S._pipeline_or_503(faults)
    chk = S._frame_promise(frame)                                       # STUDY-19's check, for one frame
    applies = chk is None or chk["applies"]
    goal, load = _interactive_goal(recall, compute_s, pipe.ms_per_pass())
    modes = calibrated_modes()
    if applies:
        mode, choice = choose_mode(goal, modes, 1, pipe.ms_per_pass())
    else:                                                               # no promise left to buy: the cheapest look
        mode, choice = choose_mode(Goal(compute_s=goal.compute_s), modes[:1], 1, pipe.ms_per_pass())
        choice["why"] = ("the promise does not apply to this frame, so no mode can buy one - the cheapest look "
                         "(standard); a person looks at the whole frame")
    if load and load["calls_ahead"]:
        choice["why"] = (f"the detector is busy ({load['calls_ahead']} call(s) ahead, ~{load['wait_s']:.1f} s) -> "
                         f"{load['left_s']:.1f} s of the {load['budget_s']:g}-s answer budget left: " + choice["why"])
    agent = pipe.agent_for(mode.name if mode else "standard")
    with S._QUEUE.call((mode.passes if mode else 1) * (pipe.ms_per_pass() or 0.0) / 1000.0), S._INFER_LOCK:
        t_in = time.perf_counter()                                    # waiting for the network ends here
        result = pipe.run_frame(frame, frame_id=frame_id, nadir=nadir, faults=faults, agent=agent)
    payload = S._frame_payload(frame, result, full=images == "full")
    payload["guarantees"] = pipe.guarantees(agent)
    payload["applicability"] = chk
    if not applies:
        payload["guarantees"] = S._withdraw_frame_promise(payload["guarantees"], chk)
        payload["incidents"].append(incident("frame_promise_not_applicable", chk["why"], p_value=chk["p_value"],
                                             driver=chk["driver"]))
    payload["agent_choice"] = {"goal": goal.to_dict(), "mode": mode.to_dict() if mode else None, **choice,
                               "promise_applies": applies, "server_load": load}
    payload["provenance"] = provenance_stamp()
    payload["frame_ref"] = {"sample": sample} if (sample and file is None) else S._frame_ref(frame_id)
    try:
        payload["twin"] = frame_twin(frame, result, nadir=nadir)          # 3D: measured geometry only
    except Exception as e:                                              # the twin is a view; never fail analyze
        S.log.exception("frame twin failed")
        payload["twin"] = {"available": False, "reason": f"twin error: {type(e).__name__}"}
        payload["incidents"].append(incident("view_failed", f"3D twin: {type(e).__name__}"))
    t_end = time.perf_counter()
    payload["wall_ms"] = round((t_end - t0) * 1000, 1)                  # the whole answer, server side
    S._QUEUE.note_overhead((t_end - t_in) * 1000 - sum(result.stage_ms.get(k) or 0.0 for k in ("see", "look_harder")))
    S._METRICS.record("analyze", frame_id, result.stage_ms, payload["wall_ms"], result.counts)
    return JSONResponse(payload)
