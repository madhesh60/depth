"""
pipeline.py — the orchestrator that runs the full See → Prove → Decide → Act loop.

* :meth:`AgenticPipeline.run_frame`  — one frame → :class:`FrameResult` (agent + optional geotag).
* :meth:`AgenticPipeline.run_survey` — many frames → :class:`SurveyResult`: per-frame results,
  survey-level :class:`TrackedObject` hazards, and a human-approved :class:`MissionPlan`
  (recovery route + resurvey list + GPX/GeoJSON/KML/CSV exports).

A ``progress_cb(stage, payload)`` hook is threaded through so the dashboard can light its
See→Prove→Decide→Act stepper live as each frame is processed.

CLI:  python -m src.agentic.pipeline --survey <dir> [--gps synthetic|none] --out runs/survey
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .agent import ReLookAgent, render
from .geo import PingFix, geotag, synthetic_track, DEFAULT_M_PER_PX
from .mission import build_mission, export
from .types import Candidate, FrameResult, MissionPlan, SurveyResult, TrackedObject, Verdict

REPO = Path(__file__).resolve().parents[2]
_TRACKED_VERDICTS = (Verdict.CONFIRMED, Verdict.REVIEW)


class AgenticPipeline:
    def __init__(self, agent: Optional[ReLookAgent] = None, m_per_px: float = DEFAULT_M_PER_PX):
        self.agent = agent or ReLookAgent()
        self.m_per_px = m_per_px

    # -- one frame ---------------------------------------------------------------------------
    def run_frame(self, frame: np.ndarray, frame_id: str = "frame",
                  fix: Optional[PingFix] = None,
                  progress_cb: Optional[Callable[[str, dict], None]] = None) -> FrameResult:
        result = self.agent.run_frame(frame, frame_id=frame_id, progress_cb=progress_cb)
        if fix is not None:
            for c in result.candidates:
                geotag(c, fix, result.nadir, result.width, result.height, frame_id, self.m_per_px)
        if progress_cb:
            progress_cb("act", {"geotagged": fix is not None})
        return result

    # -- many frames -------------------------------------------------------------------------
    def run_survey(self, frames: list[tuple[str, np.ndarray]],
                   track: Optional[dict[str, PingFix]] = None, survey_id: str = "survey",
                   progress_cb: Optional[Callable[[str, dict], None]] = None) -> SurveyResult:
        gps_available = bool(track)
        gps_synthetic = gps_available and any(f.synthetic for f in track.values())

        frame_results: list[FrameResult] = []
        tracked: list[TrackedObject] = []
        n = 0
        for frame_id, image in frames:
            fix = track.get(frame_id) if track else None
            fr = self.run_frame(image, frame_id=frame_id, fix=fix)
            frame_results.append(fr)
            for c in fr.candidates:
                if c.verdict in _TRACKED_VERDICTS:
                    n += 1
                    tracked.append(_to_tracked(f"H{n:03d}", frame_id, c))
            if progress_cb:
                progress_cb("frame_done", {"frame_id": frame_id, "counts": fr.counts})

        mission = build_mission(tracked, gps_available, gps_synthetic)
        if progress_cb:
            progress_cb("mission", mission.to_dict())
        return SurveyResult(survey_id=survey_id, frames=frame_results, tracked=tracked, mission=mission)


def _to_tracked(oid: str, frame_id: str, c: Candidate) -> TrackedObject:
    ev = c.evidence
    return TrackedObject(
        oid=oid, cls_name=c.cls_name, verdict=c.verdict, conf=round(c.conf, 3),
        evidence_score=ev.evidence_score if ev else 0.0, frame_id=frame_id, bbox=c.bbox,
        lat=c.lat, lon=c.lon, geo_error_m=c.geo_error_m,
        height_m=ev.shadow.height_m if ev else None,
        shadow_quality=ev.shadow.quality.value if ev else "none",
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
