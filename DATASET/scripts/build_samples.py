"""
build_samples.py — the 8 demo frames that ship in ``webui/samples/`` (CC BY-SA 4.0, see ATTRIBUTION.md).

The demo must show the DEFAULT model on frames it never learned from or was tuned on. Since EXP-003
became the default (2026-10-06), they come from the **v2b test split**: EXP-003 never trained on it and
never selected or calibrated on it (it verified the recall promise there once). EXP-001 trained on most
of these recordings through v1, so the manifest says so; no labelled crab-pot frame is unseen by both.

Selection is by structure only, never by detector output:

- one recording;
- six consecutive port chunks, so chunk-boundary stitching shows up in the Survey;
- two consecutive starboard chunks;
- every frame has at least one labelled pot and is not a rotated black-bordered copy.

Each sample is flagged ``heldout``, so ``feedback.export`` never turns a demo label into training data.

    python DATASET/scripts/build_samples.py            # Rec9 port 10-15 + starboard 5-6
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tiles import is_rotated_copy  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "DATASET" / "03_yolo_ready_dataset_v2b" / "test"
OUT = REPO / "webui" / "samples"
NOTE = ("Unseen by the default model EXP-003: v2b test split (never trained on, never tuned on). EXP-001 "
        "trained on these recordings through v1, so its results here are optimistic. One un-rotated copy "
        "per frame; consecutive chunks ({rec} port {p0}-{p1}) so chunk-boundary stitching can be "
        "demonstrated. Picked by structure, not by detector output (DATASET/scripts/build_samples.py).")


def find(rec: str, side: str, chunk: int) -> Path:
    pat = re.compile(rf"(?:crabpot_(?:train|valid|test)_)?{re.escape(rec)}_wcp_ss_{side}_0*{chunk}(?:_|\.)")
    hits = [p for p in sorted((SRC / "images").iterdir()) if pat.match(p.name)]
    if len(hits) != 1:
        raise SystemExit(f"{rec} {side} #{chunk}: {len(hits)} matches in {SRC} (need exactly 1)")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--recording", default="Rec9")
    ap.add_argument("--port", default="10-15", help="consecutive port chunks, a-b")
    ap.add_argument("--star", default="5-6", help="consecutive starboard chunks, a-b")
    a = ap.parse_args()
    p0, p1 = map(int, a.port.split("-"))
    s0, s1 = map(int, a.star.split("-"))
    picks = [("port", c) for c in range(p0, p1 + 1)] + [("star", c) for c in range(s0, s1 + 1)]

    old = OUT / "manifest.json"
    if old.exists():                                    # remove the previous set's frames + labels
        for s in json.loads(old.read_text(encoding="utf-8")).get("samples", []):
            (OUT / s["file"]).unlink(missing_ok=True)
            (OUT / "labels" / (Path(s["file"]).stem + ".txt")).unlink(missing_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)

    samples = []
    for i, (side, chunk) in enumerate(picks, 1):
        src = find(a.recording, side, chunk)
        lab = [l for l in (SRC / "labels" / f"{src.stem}.txt").read_text().splitlines() if l.strip()]
        pots = sum(l.split()[0] == "0" for l in lab)
        if not pots or is_rotated_copy(cv2.imread(str(src))):
            raise SystemExit(f"{src.name}: needs >= 1 labelled pot and no rotated border")
        stem = f"{a.recording}_wcp_ss_{side}_{chunk:05d}"
        shutil.copyfile(src, OUT / f"{stem}.jpg")       # bytes unchanged
        (OUT / "labels" / f"{stem}.txt").write_text("\n".join(lab) + "\n")
        word = "starboard" if side == "star" else side
        samples.append({"id": f"sample-{i:02d}", "file": f"{stem}.jpg", "name": f"{a.recording} {word} #{chunk}",
                        "kind": f"crab-pot sonogram · {pots} labelled pot{'s' if pots != 1 else ''}",
                        "recording": a.recording, "side": word, "chunk": chunk, "labelled_pots": pots,
                        "source_file": src.name, "split": "v2b test", "heldout": True, "has_gps": False})
    man = {"note": NOTE.format(rec=a.recording, p0=p0, p1=p1), "license": "CC-BY-SA-4.0",
           "attribution": "ATTRIBUTION.md", "samples": samples}
    old.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(samples)} samples, {sum(s['labelled_pots'] for s in samples)} labelled pots -> {OUT}")


if __name__ == "__main__":
    main()
