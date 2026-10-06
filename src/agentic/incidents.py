"""
incidents.py — deliberate failure handling: every failure is named, contained, and told to a person.

A failure in DEPTH never silently changes an outcome and never takes an action. Each one becomes an
**incident** with the same shape everywhere (frame results, surveys, API errors, the studio):

    {code, severity, stage, title, message, fallback, human_action, ...detail}

* ``fallback`` — the safe behaviour the system switched to (always conservative: a failed tool can
  only send a candidate to a *person*, never auto-confirm it; a missing position is withheld, never
  guessed; a failed LLM brief is replaced by the deterministic template);
* ``human_action`` — what the person should do about it.

``CATALOGUE`` is the single list of failure modes the product handles (``docs/failure_handling.md``);
``drills`` lists the ones the studio can trigger on demand (``?simulate=<code>``) so a reviewer can
watch each fallback happen.
"""
from __future__ import annotations

from typing import Optional

SEVERITIES = ("info", "warning", "error")

# code -> (severity, stage, title, default fallback, default human action)
CATALOGUE: dict[str, tuple[str, str, str, str, str]] = {
    "no_detection": (
        "info", "see", "No detections on this frame",
        "frame kept in the survey record; nothing is promoted to a hazard",
        "scan the frame; if you see an object, mark it as a missed object (it becomes a training label)"),
    "low_confidence_only": (
        "info", "decide", "Only low-confidence detections",
        "all finds are LOW-RISK: kept for audit, not queued (the recall promise covers this tier)",
        "optional: open the low-risk finds and override any that look real"),
    "corrupt_frame": (
        "warning", "input", "Corrupt or unreadable frame",
        "frame skipped; the rest of the survey ran normally",
        "re-export the frame from the sonar software and re-run it"),
    "invalid_frame": (
        "warning", "input", "Frame is not a usable sonogram",
        "frame skipped (blank, too small or no signal)",
        "check the sonar export settings for this frame"),
    "missing_gps": (
        "warning", "act", "No GPS for this survey",
        "positions, routes and re-survey passes are withheld; hazards are listed in frame order",
        "attach the recording's GPS (or a track) and re-run to get positions and routes"),
    "invalid_geometry": (
        "warning", "act", "Position failed the geometry check",
        "position withheld for the affected hazard(s); they stay in the queue without a map pin",
        "check the frame's GPS fix / orientation before sending anyone to these hazards"),
    "orientation_unknown": (
        "warning", "stage1", "Sonar geometry not measured",
        "no ground range, shadow or height for this frame; finds still reach a person",
        "set the frame's nadir edge (top / bottom / left / right) and re-run"),
    "model_unavailable": (
        "error", "see", "Detector unavailable",
        "no detection was run and nothing was decided; earlier results are untouched",
        "fetch the weights (python -m src.detection.fetch_model) or restart the server"),
    "tool_failed": (
        "warning", "prove", "An evidence tool failed",
        "the agent fell back to the detector's own confidence; affected finds go to a person, never auto-confirmed",
        "review the affected cards by eye; the trace marks the failed call"),
    "storage_failed": (
        "warning", "act", "Cloud storage upload failed (S3)",
        "results are kept on the server's local disk and all downloads still work",
        "check the bucket / instance role; the next survey retries the upload"),
    "llm_failed": (
        "info", "brief", "LLM brief writer unavailable",
        "the deterministic template brief is served (every number traced to the survey)",
        "none required - the LLM only writes prose and never decides"),
    "view_failed": (
        "info", "act", "A display view failed",
        "the view is hidden; detections, tiers and routes are unaffected",
        "none required"),
}

# failures the studio can trigger on demand to show the fallback (?simulate=<code>)
DRILLS = ("corrupt_frame", "missing_gps", "invalid_geometry", "tool_failed", "model_unavailable",
          "storage_failed", "llm_failed", "no_detection")


def incident(code: str, message: str = "", *, fallback: Optional[str] = None,
             human_action: Optional[str] = None, simulated: bool = False, **detail) -> dict:
    """Build one incident. Unknown codes are allowed (severity warning) so nothing is ever dropped."""
    sev, stage, title, fb, act = CATALOGUE.get(code, ("warning", "system", code.replace("_", " "), "", ""))
    out = {"code": code, "severity": sev, "stage": stage, "title": title,
           "message": message or title, "fallback": fallback or fb, "human_action": human_action or act}
    if simulated:
        out["simulated"] = True
    out.update({k: v for k, v in detail.items() if v is not None})
    return out


def parse_simulate(value: Optional[str]) -> set[str]:
    """``?simulate=a,b`` -> the requested drills (unknown names rejected by the caller)."""
    return {s.strip() for s in (value or "").split(",") if s.strip()}


def catalogue() -> list[dict]:
    return [{"code": c, "severity": v[0], "stage": v[1], "title": v[2], "fallback": v[3],
             "human_action": v[4], "drill": c in DRILLS} for c, v in CATALOGUE.items()]
