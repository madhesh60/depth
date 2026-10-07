"""
pipeline.py — the orchestrator that runs the full See → Prove → Decide → Act loop.

* :meth:`AgenticPipeline.run_frame`  — one frame → :class:`FrameResult` (agent + optional geotag).
* :meth:`AgenticPipeline.run_survey` — many frames → :class:`SurveyResult`: per-frame results,
  survey-level :class:`TrackedObject` hazards, and a human-approved :class:`MissionPlan`
  (recovery route + resurvey list + GPX/GeoJSON/KML/CSV exports).

A ``progress_cb(stage, payload)`` hook is threaded through so the dashboard can light its
See→Prove→Decide→Act stepper live as each frame is processed.

Failure boundaries (``incidents.py``): an unreadable / blank frame is skipped and reported; a frame
whose processing raises is skipped and reported (the survey continues); every position passes a
geometry check before it reaches a map, a route or an export — a failed one is *withheld*, never
guessed. ``faults`` (``?simulate=``) triggers these paths on purpose for the studio's failure drills.

CLI:  python -m src.agentic.pipeline --survey <dir> [--gps synthetic|none] --out runs/survey
"""
from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .agent import ReLookAgent, render
from .geo import PingFix, geotag, synthetic_track, DEFAULT_M_PER_PX
from .incidents import incident
from .mission import export, plan_mission, DEFAULT_SEC_PER_CARD
from .stitch import stitch_boundaries
from .resurvey import merge_repeat_sightings
from .types import Candidate, FrameResult, MissionPlan, SurveyResult, TrackedObject, Verdict

REPO = Path(__file__).resolve().parents[2]
_TRACKED_VERDICTS = (Verdict.CONFIRMED, Verdict.REVIEW)
MAX_GEO_ERROR_M = 2000.0          # a position less certain than this is not useful to send anyone to


def validate_frame(img) -> Optional[str]:
    """None if the frame is a usable sonogram, else why not (corrupt / blank / too small)."""
    if img is None or not hasattr(img, "shape") or img.size == 0:
        return "corrupt_frame"
    if min(img.shape[:2]) < 32:
        return "invalid_frame"
    if float(np.std(img)) < 1.0:
        return "invalid_frame"
    return None


def check_geometry(c: Candidate) -> Optional[str]:
    """None if the position is plausible, else the reason it is withheld."""
    if c.lat is None or c.lon is None:
        return None
    if not (math.isfinite(c.lat) and math.isfinite(c.lon)) or abs(c.lat) > 90 or abs(c.lon) > 180:
        return "non-finite or out-of-range coordinates"
    if c.geo_error_m is not None and (not math.isfinite(c.geo_error_m) or c.geo_error_m > MAX_GEO_ERROR_M):
        return f"position uncertainty above {MAX_GEO_ERROR_M:g} m"
    return None


class AgenticPipeline:
    def __init__(self, agent: Optional[ReLookAgent] = None, m_per_px: float = DEFAULT_M_PER_PX):
        self.agent = agent or ReLookAgent()
        self.m_per_px = m_per_px

    def guarantees(self) -> dict:
        """The promises this run's tiers carry (shown on every report + in the UI)."""
        t = getattr(self.agent.cfg, "tiers", None)
        if t is None:
            return {"mode": "legacy", "note": "no calibrated guarantee (thresholds are the legacy rules)"}
        g = dict(t.guarantees)
        g.update({"mode": "calibrated", "tau_review": t.tau_review, "tau_confirm": t.tau_confirm,
                  "policy": t.policy})
        return g

    # -- one frame ---------------------------------------------------------------------------
    def run_frame(self, frame: np.ndarray, frame_id: str = "frame",
                  fix: Optional[PingFix] = None, nadir: Optional[str] = None,
                  progress_cb: Optional[Callable[[str, dict], None]] = None,
                  faults: Optional[set] = None) -> FrameResult:
        faults = set(faults or ())
        # a real recording's sonar depth is a MEASURED altitude in metres -> heights in metres
        alt_m = fix.depth_m if (fix is not None and not fix.synthetic and fix.depth_m) else None
        result = self.agent.run_frame(frame, frame_id=frame_id, nadir=nadir, progress_cb=progress_cb, altitude_m=alt_m,
                                      faults=faults)
        if fix is not None:
            if "invalid_geometry" in faults:              # failure drill: a corrupted GPS fix
                fix = replace(fix, lat=float("nan"), ping_lat=None)
            bad = 0
            for c in result.candidates:
                geotag(c, fix, result.nadir, result.width, result.height, frame_id, self.m_per_px)
                why = check_geometry(c)
                if why:
                    c.lat = c.lon = c.geo_error_m = None
                    c.position_withheld = why
                    bad += 1
            if bad:
                result.incidents.append(incident("invalid_geometry", f"{frame_id}: {bad} position(s) failed the check",
                                                 frame_id=frame_id, affected=bad,
                                                 simulated="invalid_geometry" in faults))
        if progress_cb:
            progress_cb("act", {"geotagged": fix is not None})
        return result

    # -- many frames -------------------------------------------------------------------------
    def run_survey(self, frames: list[tuple[str, np.ndarray]],
                   track: Optional[dict[str, PingFix]] = None, survey_id: str = "survey",
                   nadir: Optional[str] = None,
                   budget_minutes: Optional[float] = None, sec_per_card: float = DEFAULT_SEC_PER_CARD,
                   progress_cb: Optional[Callable[[str, dict], None]] = None,
                   frame_lock=None, boat_minutes: Optional[float] = None,
                   faults: Optional[set] = None, incidents: Optional[list] = None) -> SurveyResult:
        """``frame_lock`` (optional context manager) is held per frame, not per survey, so a server
        can interleave single-frame requests with a long survey on one shared network. ``incidents``:
        input problems found before the run (e.g. undecodable uploads) - reported with the survey."""
        faults = set(faults or ())
        gps_available = bool(track)
        survey_incidents = list(incidents or [])

        frame_results: list[FrameResult] = []
        for k, (frame_id, image) in enumerate(frames):
            bad = validate_frame(image)
            if bad:
                survey_incidents.append(incident(bad, f"{frame_id}: skipped", frame_id=frame_id))
                if progress_cb:
                    progress_cb("frame_done", {"frame_id": frame_id, "counts": {}})
                continue
            fix = track.get(frame_id) if track else None
            fr_faults = faults if k == 0 else faults - {"tool_failed", "invalid_geometry", "no_detection"}
            try:
                with (frame_lock or nullcontext()):
                    fr = self.run_frame(image, frame_id=frame_id, fix=fix, nadir=nadir, faults=fr_faults)
            except Exception as e:                       # one bad frame never sinks the survey
                survey_incidents.append(incident("tool_failed", f"{frame_id}: processing failed "
                                                 f"({type(e).__name__}: {str(e)[:120]}) - frame skipped",
                                                 frame_id=frame_id))
                if progress_cb:
                    progress_cb("frame_done", {"frame_id": frame_id, "counts": {}})
                continue
            frame_results.append(fr)
            if progress_cb:
                progress_cb("frame_done", {"frame_id": frame_id, "counts": fr.counts})

        # chunk-boundary stitching (survey-level, non-gating): an object cut by the chunk boundary
        # is one hazard, not two. Never changes a verdict or a score.
        # Each stitched pair becomes ONE hazard, represented by the stronger sighting.
        merged_away: set[int] = set()
        also_in: dict[int, str] = {}
        for a, b, _ in stitch_boundaries(frame_results):
            keep, drop = (a, b) if _strength(a) >= _strength(b) else (b, a)
            merged_away.add(id(drop))
            also_in[id(keep)] = b.continuation_of if keep is b else a.continues_in

        tracked: list[TrackedObject] = []
        n = 0
        for fr in frame_results:
            for c in fr.candidates:
                if c.verdict in _TRACKED_VERDICTS and id(c) not in merged_away:
                    n += 1
                    t = _to_tracked(f"H{n:03d}", fr.frame_id, c)
                    if id(c) in also_in:
                        t.also_in.append(also_in[id(c)])
                    tracked.append(t)

        # repeat sightings (another pass / line saw the same object) → one hazard; needs GPS
        merges = 0
        if gps_available:
            tracked, merges = merge_repeat_sightings(tracked)
        mission = plan_mission(tracked, track if gps_available else None, self.guarantees(),
                               budget_minutes, sec_per_card, boat_minutes, merges, incidents=survey_incidents)
        if progress_cb:
            progress_cb("mission", mission.to_dict())
        return SurveyResult(survey_id=survey_id, frames=frame_results, tracked=tracked, mission=mission,
                            track=track if gps_available else None,
                            plan_args={"budget_minutes": budget_minutes, "sec_per_card": sec_per_card,
                                       "boat_minutes": boat_minutes})


_VERDICT_RANK = {Verdict.CONFIRMED: 2, Verdict.REVIEW: 1, Verdict.REJECTED: 0}


def _strength(c: Candidate) -> tuple[int, float]:
    return (_VERDICT_RANK.get(c.verdict, 0), c.evidence.evidence_score if c.evidence else c.conf)


def _to_tracked(oid: str, frame_id: str, c: Candidate) -> TrackedObject:
    ev = c.evidence
    return TrackedObject(
        oid=oid, cls_name=c.cls_name, verdict=c.verdict, conf=round(c.conf, 3),
        evidence_score=ev.evidence_score if ev else 0.0, frame_id=frame_id, bbox=c.bbox,
        lat=c.lat, lon=c.lon, geo_error_m=c.geo_error_m,
        height_m=ev.shadow.height_m if ev else None,
        height_rel=ev.shadow.height_rel if ev else 0.0,
        p_pot=ev.p_pot if ev else None,
        shadow_quality=ev.shadow.quality.value if ev else "none",
        position_withheld=getattr(c, "position_withheld", None),
        p_evidence=ev.p_evidence if ev else None, action=ev.action if ev else None,
        action0=ev.action0 if ev else None, conflict=ev.conflict if ev else None,
        request_resurvey=bool(ev.request_resurvey) if ev else False,
        info_bits=(ev.resurvey_gain or {}).get("info_bits") if ev else None,
    )


# ---- CLI --------------------------------------------------------------------------------------
def _iter_images(path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in exts) if path.is_dir() else [path]


def main():
    ap = argparse.ArgumentParser(description="Run the full See->Prove->Decide->Act loop on a survey.")
    ap.add_argument("--survey", required=True, help="directory of sonar frames")
    ap.add_argument("--gps", choices=["synthetic", "none"], default="synthetic",
                    help="'synthetic' = clearly-labelled demo track; 'none' = honest no-GPS")
    ap.add_argument("--out", default=str(REPO / "runs" / "survey"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    paths = _iter_images(Path(a.survey))
    if a.limit:
        paths = paths[: a.limit]
    frames = [(p.stem, cv2.imread(str(p))) for p in paths]
    frames = [(fid, im) for fid, im in frames if im is not None]
    track = synthetic_track([fid for fid, _ in frames]) if a.gps == "synthetic" else None

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    survey = AgenticPipeline().run_survey(frames, track=track, survey_id=Path(a.survey).name)

    for fr in survey.frames:
        im = dict(frames)[fr.frame_id]
        cv2.imwrite(str(out / f"{fr.frame_id}.jpg"), render(im, fr))
    for fmt in ("json", "geojson", "gpx", "kml", "csv"):
        ext = "json" if fmt == "json" else fmt
        (out / f"mission.{ext}" if fmt != "json" else out / "survey.json").write_text(export(fmt, survey))

    m = survey.mission
    print(f"\nsurvey '{survey.survey_id}': {len(survey.frames)} frames, {len(survey.tracked)} hazards "
          f"({m.counts.get('confirmed',0)} confirmed / {m.counts.get('review',0)} review)")
    print(f"  GPS: {'synthetic demo' if m.gps_synthetic else ('real' if m.gps_available else 'none')}"
          f" | route {len(m.recovery_route)} stops"
          + (f", {m.route_length_m} m" if m.route_length_m is not None else "")
          + f" | human approval required: {m.human_approval_required}")
    print(f"  wrote overlays + survey.json + mission.geojson/gpx/kml/csv -> {out}")


if __name__ == "__main__":
    main()
