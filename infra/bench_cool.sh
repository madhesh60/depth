#!/usr/bin/env bash
# bench_cool.sh — the "Best Use of COOL" benchmark of the DEPTH PRODUCT workload on THIS machine.
# infra/bench_fleet.sh runs it on equal-size machines automatically; it also runs by hand:
#
#   FLAVOR=stock LABEL=baseline_x86   PRICE_HOUR=0.1785 ./infra/bench_cool.sh   # c8i.xlarge, Ubuntu 24.04
#   FLAVOR=stock LABEL=graviton_stock PRICE_HOUR=0.1595 ./infra/bench_cool.sh   # c8g.xlarge, Ubuntu 24.04
#   FLAVOR=cool  LABEL=graviton_cool  PRICE_HOUR=0.1695 ./infra/bench_cool.sh   # c8g.xlarge, COOL AMI (+ $0.01/h)
# (B) → (C) isolates COOL on identical hardware; (A) → (C) is the x86 → Graviton + COOL latency and cost story.
#
# What a run records:
#   * all vCPUs: product in both perception modes (the standard look; looking harder = + the 2x2 tile pass,
#     `<label>_lh`) + cv + survey (a real 1-h sonar recording end to end), 3 interleaved repeats,
#     the parity record (every detection per frame), CPU use, peak memory;
#   * the product runs with the cv2.dnn engine the server would pick on this machine (DEPTH_DNN_ENGINE=fastest,
#     as in infra/depth.service; the JSON records which one and why);
#   * 1 thread: product + cv (per-core comparison - an Intel vCPU is a hyper-thread, a Graviton vCPU a core);
#   * STUDY-20: OpenCV 5's two DNN engines (classic vs new) on this machine - speed + same answers;
#   * provenance: COOL only if cv2 loads from /opt/cool (refuses to mislabel); on EC2 the machine's
#     AWS-signed identity (instance type, AMI, Marketplace product code) is saved as <label>__receipt.json.
# Stock runs use a private venv (Ubuntu 24.04 / PEP 668); COOL runs use COOL's interpreter as-is - nothing
# is ever installed into /opt/cool/venvs/* (that could replace the numpy COOL was built against).
set -euo pipefail
cd "$(dirname "$0")/.."

FLAVOR="${FLAVOR:?FLAVOR=stock|cool}"
LABEL="${LABEL:?LABEL=baseline_x86|graviton_stock|graviton_cool}"
PRICE_HOUR="${PRICE_HOUR:-0}"
PRICE_SOURCE="${PRICE_SOURCE:-}"
N="${N:-120}"; N1="${N1:-48}"
REPEATS="${REPEATS:-3}"
THREADS="${THREADS:-0 1}"
WORKLOADS="${WORKLOADS:-product,product_lh,cv,survey}"
WORKLOADS_1T="${WORKLOADS_1T:-product,cv}"
FRAMES="${FRAMES:-webui/samples}"
OPENCV_PIN="${OPENCV_PIN:-5.0.0.93}"
NUMPY_PIN="${NUMPY_PIN:-2.2.6}"

echo "== DEPTH COOL benchmark [$LABEL] flavor=$FLAVOR arch=$(uname -m) vcpus=$(nproc) =="

if [ -n "${BENCH_PY:-}" ]; then
  PY="$BENCH_PY"
elif [ "$FLAVOR" = "stock" ]; then
  VENV="${VENV:-$HOME/.depth-bench-stock}"
  if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install -q --upgrade pip
    "$VENV/bin/python" -m pip install -q "numpy==$NUMPY_PIN" "opencv-python-headless==$OPENCV_PIN"
  fi
  PY="$VENV/bin/python"
elif [ "$FLAVOR" = "cool" ]; then
  PY="${COOL_PY:-$(ls -d /opt/cool/venvs/python_3.1*/bin/python 2>/dev/null | sort -V | tail -1)}"
  [ -x "$PY" ] || { echo "ERROR: no COOL interpreter under /opt/cool/venvs (is this the COOL AMI?)" >&2; exit 2; }
  echo "COOL interpreter: $PY (used as-is; nothing is installed into it)"
else
  echo "ERROR: FLAVOR must be stock or cool" >&2; exit 2
fi

# provenance gate: cv2 + dnn present, and COOL really is COOL
"$PY" - "$FLAVOR" <<'PY'
import sys, cv2, numpy
flavor = sys.argv[1]
f = cv2.__file__ or ""
print(f"cv2 {cv2.__version__} from {f} | numpy {numpy.__version__} | dnn={'yes' if hasattr(cv2, 'dnn') else 'NO'}")
if not hasattr(cv2, "dnn"):
    sys.exit("ERROR: this OpenCV build has no dnn module - the product cannot run on it")
if flavor == "cool" and "/opt/cool" not in f:
    sys.exit("ERROR: flavor=cool but cv2 is not loaded from /opt/cool - refusing to mislabel the run")
PY

M="${DEPTH_MODEL:-EXP-003}"
[ -f "${DEPTH_ONNX:-models/$M/best.onnx}" ] || [ -f "runs/$M/weights/best.onnx" ] \
  || { echo "ERROR: model missing (python -m src.detection.fetch_model)" >&2; exit 2; }

for T in $THREADS; do
  if [ "$T" = "0" ]; then L="$LABEL"; W="$WORKLOADS"; NN="$N"; EXTRA=(--parity)
  else L="${LABEL}_t${T}"; W="$WORKLOADS_1T"; NN="$N1"; EXTRA=(); fi
  DEPTH_DNN_ENGINE="${DEPTH_DNN_ENGINE:-fastest}" "$PY" -m src.bench.product_bench --label "$L" --workloads "$W" --threads "$T" --n "$NN" --repeats "$REPEATS" \
    --frames "$FRAMES" --price-hour "$PRICE_HOUR" --price-source "$PRICE_SOURCE" --out benchmarks "${EXTRA[@]}"
done
# STUDY-20: which cv2.dnn engine loads, what it is worth here, and whether the answers match
"$PY" -m src.bench.dnn_engine --label "$LABEL" --frames "$FRAMES" --threads "${THREADS// /,}" --out benchmarks
echo "== done [$LABEL] — gather benchmarks/*.json from every machine, then: python -m src.bench.compare =="
