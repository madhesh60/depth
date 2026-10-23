"""
infer.py — Stage 2 YOLO inference for marine-debris detection, via ``cv2.dnn``.

Loads the exported ONNX detector with ``cv2.dnn.readNetFromONNX`` (OpenCV 5; no
ultralytics/torch at inference time). Thresholds, NMS IoU and class-aware NMS come from
``models/<MODEL>/calibration.json`` (``calibration.py``) — the single threshold source.

* ``full_frame``  — run the detector on the whole frame (the product path).
* ``roi_guided``  — the retired Stage-1 ROI gate (the STUDY-01 negative result); it is not used by
  the product.

YOLO11/YOLOv8 detect head: the ONNX output is ``(1, 4+nc, N)`` — 4 bbox (cx,cy,w,h in
letterboxed 640-space) followed by ``nc`` class scores, transposed. We letterbox on the
way in and un-letterbox boxes back to original pixels on the way out.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from src.detection.calibration import SHIPPED_MODEL, load_calibration

REPO = Path(__file__).resolve().parents[2]
def _default_onnx() -> Path:
    """``$DEPTH_ONNX``, else ``models/<MODEL>/best.onnx``, else ``runs/<MODEL>/weights/best.onnx``
    (``<MODEL>`` = ``$DEPTH_MODEL``, default ``calibration.SHIPPED_MODEL``). Weights are git-ignored; calibration is tracked."""
    if os.environ.get("DEPTH_ONNX"):
        return Path(os.environ["DEPTH_ONNX"])
    model = os.environ.get("DEPTH_MODEL", SHIPPED_MODEL)
    for p in (REPO / "models" / model / "best.onnx", REPO / "runs" / model / "weights" / "best.onnx"):
        if p.exists():
            return p
    return REPO / "models" / model / "best.onnx"          # absent: where fetch_model installs it (named in the error)


DEFAULT_ONNX = _default_onnx()

# Every runtime threshold comes from ONE file: models/<MODEL>/calibration.json (see
# calibration.py). fishing_gear is the mission-critical, small-object class and runs hot (0.10 —
# EXP-001 error analysis: recall 0.46 @0.25 -> 0.69 @0.10); the others stay at 0.25.
_CAL = load_calibration()
DEFAULT_NAMES = _CAL.names
PER_CLASS_CONF = _CAL.conf
DEFAULT_IOU_NMS = _CAL.iou_nms

_ENGINES = {"auto": "ENGINE_AUTO", "new": "ENGINE_NEW", "classic": "ENGINE_CLASSIC"}
# The cv2.dnn engine the last network actually loaded with: a Net cannot be asked which engine built it,
# so _read_net records it (provenance, /api/health, benchmarks; STUDY-20 measures what it is worth).
# ``choice`` is the evidence when DEPTH_DNN_ENGINE=fastest measured the engines on this machine.
ENGINE: dict = {"requested": None, "loaded": None, "fallback": None, "choice": None}
# DEPTH_DNN_ENGINE=fastest: which engine is faster depends on the machine (STUDY-20: on a Graviton c8g.large the
# classic one, 255 vs 377 ms per forward; on the laptop within noise), so the server measures it at load.
# Same answers = raw outputs within 1 px per box coordinate and 0.01 per class score (STUDY-17's |Δscore|).
_FASTEST = {"rounds": 3, "n": 2, "box_px": 1.0, "score": 0.01}


def configure_runtime(threads: Optional[int] = None) -> dict:
    """Pin OpenCV's thread pool so benchmarks are comparable (``$DEPTH_THREADS``; 0 = OpenCV
    default). Returns the effective settings for provenance logs (:func:`runtime_info`)."""
    n = threads if threads is not None else int(os.environ.get("DEPTH_THREADS", "0") or 0)
    if n > 0:
        cv2.setNumThreads(n)
    return runtime_info()


def runtime_info() -> dict:
    """Threads, and the DNN engine asked for vs the one that loaded (None until a network is loaded)."""
    return {"threads": cv2.getNumThreads(), "engine": ENGINE["loaded"],
            "engine_requested": os.environ.get("DEPTH_DNN_ENGINE", "auto").lower(),
            "engine_fallback": ENGINE["fallback"], "engine_choice": ENGINE["choice"]}


def faster_engine(rounds_ms: dict[str, list[float]], same_answers: bool) -> tuple[str, str]:
    """The rule behind ``fastest``: the classic engine replaces the new one (ENGINE_AUTO's pick) only when both
    give the same answers and the classic one was faster in EVERY interleaved round (STUDY-17's separation rule:
    the per-round ranges do not overlap). Returns (engine, why)."""
    new, classic = rounds_ms["new"], rounds_ms["classic"]
    if not same_answers:
        return "new", "the answers differ - kept the new engine"
    if max(classic) < min(new):
        return "classic", "classic faster in every round, same answers"
    if max(new) < min(classic):
        return "new", "new faster in every round, same answers"
    return "new", "rounds overlap (within noise) - kept the new engine"


def _pick_fastest(nets: dict, imgsz: int) -> tuple[str, dict]:
    """Both engines on one fixed input (seeded noise: the forward's cost does not depend on the pixels; the full
    answer parity on real frames is STUDY-20's): same answers?, then ``rounds`` interleaved rounds (rotated
    order) of ``n`` timed forwards each, and :func:`faster_engine`. Returns (engine, the evidence)."""
    canvas = np.random.default_rng(0).integers(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (imgsz, imgsz), swapRB=True)
    out = {}
    for e, net in nets.items():                               # untimed first forward: allocations + the outputs
        net.setInput(blob)
        out[e] = np.array(net.forward(), copy=True)[0]        # (4+nc, N)
    gap_box = float(np.abs(out["new"][:4] - out["classic"][:4]).max())
    gap_score = float(np.abs(out["new"][4:] - out["classic"][4:]).max())
    same = gap_box <= _FASTEST["box_px"] and gap_score <= _FASTEST["score"]
    names, ms = list(nets), {e: [] for e in nets}
    for r in range(_FASTEST["rounds"]):
        for e in names[r % 2:] + names[:r % 2]:
            t: list[float] = []
            for _ in range(_FASTEST["n"]):
                t0 = time.perf_counter()
                nets[e].setInput(blob)
                nets[e].forward()
                t.append((time.perf_counter() - t0) * 1000)
            ms[e].append(round(float(np.median(t)), 1))
    pick, why = faster_engine(ms, same)
    return pick, {"picked": pick, "why": why, "round_ms": ms, "threads": cv2.getNumThreads(),
                  "same_answers": same, "max_box_gap_px": round(gap_box, 4), "max_score_gap": round(gap_score, 5),
                  "input": f"seeded noise {imgsz}x{imgsz}", "rule": "src.detection.infer.faster_engine"}


def _read_net(onnx_path: Path, imgsz: int = 640):
    """``cv2.dnn.readNetFromONNX`` with the engine from ``$DEPTH_DNN_ENGINE`` (auto|new|classic|fastest).
    ``auto`` (default) does what ENGINE_AUTO does — OpenCV 5's new engine first, the classic one if
    the new one cannot load the model — but explicitly, so the engine that loaded is known
    (:data:`ENGINE`). On EXP-003 ENGINE_AUTO's output is bit-identical to ENGINE_NEW's. ``fastest`` loads
    both, measures them on this machine (:func:`_pick_fastest`, a few seconds) and keeps the winner."""
    want = os.environ.get("DEPTH_DNN_ENGINE", "auto").lower()
    if not hasattr(cv2.dnn, "ENGINE_NEW"):                     # an OpenCV without the engine switch
        ENGINE.update(requested=want, loaded="default", fallback=None, choice=None)
        return cv2.dnn.readNetFromONNX(str(onnx_path))
    forced = os.environ.get("OPENCV_FORCE_DNN_ENGINE")       # OpenCV's own override wins over ours
    errors, nets = [], {}
    for name in {"new": ("new",), "classic": ("classic",)}.get(want, ("new", "classic")):
        try:
            net = cv2.dnn.readNetFromONNX(str(onnx_path), getattr(cv2.dnn, _ENGINES[name]))
        except cv2.error as e:
            errors.append(f"{name}: {str(e).strip().splitlines()[-1][:160]}")
            continue
        if net.empty():
            errors.append(f"{name}: empty network")
            continue
        nets[name] = net
        if want != "fastest" or forced:
            break
    if not nets:
        raise RuntimeError(f"cv2.dnn could not load {onnx_path.name}: " + "; ".join(errors))
    name, choice = next(iter(nets)), None
    if len(nets) == 2:
        name, choice = _pick_fastest(nets, imgsz)
    ENGINE.update(requested=want, loaded=name if not forced else f"forced:{forced}",
                  fallback="; ".join(errors) or None, choice=choice)
    return nets[name]


@dataclass
class Detection:
    """A single Stage-2 detection in ORIGINAL image pixels."""
    bbox: tuple[int, int, int, int]      # x1, y1, x2, y2
    cls_id: int
    cls_name: str
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect ratio and pad to a square ``size``. Returns (canvas, scale, padx, pady)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    padx, pady = (size - nw) // 2, (size - nh) // 2
    canvas[pady:pady + nh, padx:padx + nw] = resized
    return canvas, scale, padx, pady


class YoloOnnxDetector:
    """YOLO11/YOLOv8 detector over ``cv2.dnn``. Reuse one instance across frames."""

    def __init__(
        self,
        onnx_path: str | Path = DEFAULT_ONNX,
        names: Sequence[str] = DEFAULT_NAMES,
        imgsz: Optional[int] = None,
        conf_thres: "float | dict[str, float] | Sequence[float]" = 0.25,
        iou_thres: float = DEFAULT_IOU_NMS,
        class_aware_nms: bool = _CAL.class_aware_nms,
    ):
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX weights not found: {onnx_path}\n"
                f"Get them: python -m src.detection.fetch_model (the published, hash-checked release), "
                f"or export a training run: python -m src.detection.export_onnx"
            )
        self.imgsz = int(imgsz or _CAL.imgsz)          # EXP-001 640; EXP-002 1024 (calibration.json)
        self.net = _read_net(onnx_path, self.imgsz)
        self.engine = ENGINE["loaded"]
        self._onnx_path = str(onnx_path)
        self.names = list(names)
        self.class_aware_nms = class_aware_nms
        self.iou_thres = iou_thres
        # per-class confidence thresholds → array aligned to `names`; `conf_floor` is the min
        # used for the initial keep + NMS, then each detection is filtered by its class threshold.
        if isinstance(conf_thres, dict):
            self.conf_by_class = np.array([conf_thres.get(n, 0.25) for n in self.names], float)
        elif isinstance(conf_thres, (list, tuple, np.ndarray)):
            self.conf_by_class = np.asarray(conf_thres, float)
        else:
            self.conf_by_class = np.full(len(self.names), float(conf_thres))
        self.conf_floor = float(self.conf_by_class.min())
        self.conf_thres = self.conf_floor  # back-compat alias
        # running op-level timings (ms): letterbox+blob, cv2.dnn forward, decode+NMS. One dict shared
        # with every with_conf() view, so re-looks add to it; the benchmark resets it per frame.
        self.prof = {"pre_ms": 0.0, "forward_ms": 0.0, "post_ms": 0.0, "forwards": 0}

    def with_conf(self, conf_thres: "float | dict[str, float]") -> "YoloOnnxDetector":
        """A view of this detector with different confidence thresholds that SHARES the loaded
        network (no second model in memory, one warm-up covers both)."""
        import copy
        other = copy.copy(self)
        if isinstance(conf_thres, dict):
            other.conf_by_class = np.array([conf_thres.get(n, 0.25) for n in self.names], float)
        else:
            other.conf_by_class = np.full(len(self.names), float(conf_thres))
        other.conf_floor = float(other.conf_by_class.min())
        other.conf_thres = other.conf_floor
        return other

    def warmup(self, n: int = 2) -> float:
        """Run ``n`` dummy forwards so the first real request doesn't pay graph/alloc setup.
        Returns the last warm-up latency in ms."""
        import time
        dummy = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        ms = 0.0
        for _ in range(max(1, n)):
            t0 = time.perf_counter()
            self.forward_raw(dummy)
            ms = (time.perf_counter() - t0) * 1000
        return ms

    def forward_raw(self, img: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Letterbox + forward only. Returns (preds (N, 4+nc), scale, padx, pady)."""
        t0 = time.perf_counter()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        canvas, scale, padx, pady = _letterbox(img, self.imgsz)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.imgsz, self.imgsz), swapRB=True)
        t1 = time.perf_counter()
        self.net.setInput(blob)
        out = self.net.forward()                      # (1, 4+nc, N)
        p = self.prof
        p["pre_ms"] += (t1 - t0) * 1000
        p["forward_ms"] += (time.perf_counter() - t1) * 1000
        p["forwards"] += 1
        return out[0].T, scale, padx, pady

    # -- core detector ------------------------------------------------------------------
    def detect(self, img: np.ndarray) -> list[Detection]:
        """Full-frame detection: letterbox → forward → decode → NMS → original px."""
        preds, scale, padx, pady = self.forward_raw(img)
        return self.decode(preds, scale, padx, pady, img.shape[:2])

    def decode(self, preds: np.ndarray, scale: float, padx: int, pady: int,
               hw: tuple[int, int]) -> list[Detection]:
        """Decode raw (N, 4+nc) predictions → per-class-NMS'd detections in original pixels."""
        t0 = time.perf_counter()
        try:
            return self._decode(preds, scale, padx, pady, hw)
        finally:
            self.prof["post_ms"] += (time.perf_counter() - t0) * 1000

    def _decode(self, preds: np.ndarray, scale: float, padx: int, pady: int,
                hw: tuple[int, int]) -> list[Detection]:
        cxcywh = preds[:, :4]
        scores = preds[:, 4:]
        cls_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(scores.shape[0]), cls_ids]
        keep = confs >= self.conf_floor
        if not np.any(keep):
            return []
        cxcywh, cls_ids, confs = cxcywh[keep], cls_ids[keep], confs[keep]

        # letterboxed cxcywh -> original-pixel xywh (top-left)
        cx, cy, bw, bh = cxcywh.T
        x = (cx - bw / 2 - padx) / scale
        y = (cy - bh / 2 - pady) / scale
        w = bw / scale
        h = bh / scale
        boxes = np.stack([x, y, w, h], axis=1)

        idxs = nms(boxes, confs, cls_ids, self.conf_floor, self.iou_thres, self.class_aware_nms)
        if len(idxs) == 0:
            return []

        H, W = hw
        dets: list[Detection] = []
        for i in idxs:
            cid = int(cls_ids[i])
            if confs[i] < self.conf_by_class[cid]:      # per-class confidence gate
                continue
            x1 = int(max(0, boxes[i, 0]))
            y1 = int(max(0, boxes[i, 1]))
            x2 = int(min(W, boxes[i, 0] + boxes[i, 2]))
            y2 = int(min(H, boxes[i, 1] + boxes[i, 3]))
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append(Detection((x1, y1, x2, y2), cid, self.names[cid], float(confs[i])))
        return dets

    # -- ROI-guided gating --------------------------------------------------------------
    def detect_roi_guided(
        self,
        img: np.ndarray,
        rois: Sequence[tuple[int, int, int, int]],
        gate_iou: float = 0.10,
    ) -> list[Detection]:
        """Full-frame detect, then keep only detections spatially backed by a Stage-1 ROI.

        A detection is kept if its center lies inside any ROI, or it overlaps an ROI with
        IoU >= ``gate_iou``. Identical detector to :meth:`detect`, so the delta isolates Stage 1.
        """
        dets = self.detect(img)
        if not rois:
            return []
        return [d for d in dets if _backed_by_roi(d.bbox, rois, gate_iou)]


def nms(boxes_xywh: np.ndarray, confs: np.ndarray, cls_ids: np.ndarray, score_thr: float,
        iou_thr: float, class_aware: bool = True) -> np.ndarray:
    """Non-max suppression. ``class_aware=True`` (default) suppresses only within a class
    (``cv2.dnn.NMSBoxesBatched``), so a crab pot lying next to a wreck box is not deleted by it."""
    if len(confs) == 0:
        return np.empty(0, int)
    b = np.asarray(boxes_xywh, float).tolist()
    c = np.asarray(confs, float).tolist()
    if class_aware:
        idxs = cv2.dnn.NMSBoxesBatched(b, c, np.asarray(cls_ids, int).tolist(), score_thr, iou_thr)
    else:
        idxs = cv2.dnn.NMSBoxes(b, c, score_thr, iou_thr)
    return np.array(idxs, int).flatten()


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _backed_by_roi(box, rois, gate_iou: float) -> bool:
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    for r in rois:
        if r[0] <= cx <= r[2] and r[1] <= cy <= r[3]:
            return True
        if _iou(box, r) >= gate_iou:
            return True
    return False


def draw_detections(img: np.ndarray, dets: Sequence[Detection]) -> np.ndarray:
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(vis, f"{d.cls_name} {d.conf:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    return vis


# -- CLI --------------------------------------------------------------------------------
def _iter_images(path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if path.is_dir():
        yield from sorted(p for p in path.rglob("*") if p.suffix.lower() in exts)
    else:
        yield path


def main():
    p = argparse.ArgumentParser(description="Run the Stage-2 ONNX detector on image(s).")
    p.add_argument("source", help="image file or directory")
    p.add_argument("--onnx", default=str(DEFAULT_ONNX))
    p.add_argument("--mode", choices=["full_frame", "roi_guided"], default="full_frame")
    p.add_argument("--conf", type=float, default=None,
                   help="global conf override; default uses per-class thresholds (PER_CLASS_CONF)")
    p.add_argument("--iou", type=float, default=DEFAULT_IOU_NMS)
    p.add_argument("--out", default=str(REPO / "runs" / "infer_out"),
                   help="directory for annotated overlays")
    p.add_argument("--limit", type=int, default=0, help="max images (0 = all)")
    args = p.parse_args()

    conf = args.conf if args.conf is not None else PER_CLASS_CONF
    det = YoloOnnxDetector(args.onnx, conf_thres=conf, iou_thres=args.iou)
    stage1 = None
    if args.mode == "roi_guided":
        from src.cv_pipeline.pipeline import Stage1Pipeline
        stage1 = Stage1Pipeline()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path in _iter_images(Path(args.source)):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if args.mode == "roi_guided":
            rois = [c.bbox for c in stage1.process(img).candidates]
            dets = det.detect_roi_guided(img, rois)
        else:
            dets = det.detect(img)
        cv2.imwrite(str(out_dir / img_path.name), draw_detections(img, dets))
        print(f"{img_path.name}: {len(dets)} detections")
        n += 1
        if args.limit and n >= args.limit:
            break
    print(f"\n{n} images -> {out_dir}")


if __name__ == "__main__":
    main()
