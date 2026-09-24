#!/usr/bin/env bash
# bench_cool.sh — the 3-way "Best Use of COOL" benchmark of the DEPTH PRODUCT workload (review M-5).
#
# Same code, same frames (sha256-pinned), same model, three configurations:
#   (A) x86      + stock OpenCV 5   LABEL=baseline_x86    on c7i.xlarge
#   (B) Graviton + stock OpenCV 5   LABEL=graviton_stock  on c8g.xlarge
#   (C) Graviton + COOL (AMI)       LABEL=graviton_cool   on the SAME c8g.xlarge (COOL AMI)
# (B)->(C) isolates COOL on identical hardware; (A)->(C) is the Arm-vs-x86 latency + cost story.
#
#   FLAVOR=stock LABEL=baseline_x86   PRICE_HOUR=0.1785 ./infra/bench_cool.sh
#   FLAVOR=stock LABEL=graviton_stock PRICE_HOUR=0.1595 ./infra/bench_cool.sh
#   FLAVOR=cool  LABEL=graviton_cool  PRICE_HOUR=0.1695 ./infra/bench_cool.sh   # + $0.01/h COOL fee
# (prices: us-east-1 on-demand Linux — verify on the day with the AWS pricing page / CLI)
#
# Fixes vs the old benchmark_graviton.sh (review §3.7):
#   * times the PRODUCT (decode → Stage 1 → cv2.dnn → evidence → render), not the retired Stage-1 ROIs;
#   * Ubuntu 24.04 (PEP 668): stock runs use a private venv, never the system Python;
#   * COOL runs use the COOL interpreter AS-IS — nothing is ever installed into /opt/cool/venvs/*
#     (installing numpy there could replace the build COOL was compiled against);
#   * threads pinned and recorded (THREADS="0 1": all vCPUs, then 1 thread = per-core comparison —
#     an Intel vCPU is a hyper-thread, a Graviton vCPU is a physical core);
#   * provenance: a run is labelled COOL only if cv2 loads from /opt/cool (the script refuses otherwise);
#   * cost: $/1k frames and $ + real-time factor per survey-hour, from PRICE_HOUR.
set -euo pipefail
cd "$(dirname "$0")/.."

FLAVOR="${FLAVOR:?FLAVOR=stock|cool}"
LABEL="${LABEL:?LABEL=baseline_x86|graviton_stock|graviton_cool}"
PRICE_HOUR="${PRICE_HOUR:-0}"
N="${N:-300}"
THREADS="${THREADS:-0 1}"
WORKLOADS="${WORKLOADS:-product cv}"
FRAMES="${FRAMES:-webui/samples}"            # or a synced set: aws s3 sync s3://$S3_BUCKET/bench/frames bench_frames
S3_BUCKET="${S3_BUCKET:-}"
OPENCV_PIN="${OPENCV_PIN:-5.0.0.93}"
NUMPY_PIN="${NUMPY_PIN:-2.2.6}"

echo "== DEPTH COOL benchmark [$LABEL] flavor=$FLAVOR arch=$(uname -m) vcpus=$(nproc) =="

if [ "$FLAVOR" = "stock" ]; then
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

[ -f "${DEPTH_ONNX:-runs/EXP-001/weights/best.onnx}" ] || { echo "ERROR: model missing (set DEPTH_ONNX or run infra/setup_cool_instance.sh)" >&2; exit 2; }

for T in $THREADS; do
  L="$LABEL"; [ "$T" != "0" ] && L="${LABEL}_t${T}"
  for WL in $WORKLOADS; do
    "$PY" -m src.bench.product_bench --label "$L" --workload "$WL" --threads "$T" --n "$N" \
      --frames "$FRAMES" --price-hour "$PRICE_HOUR" --out runs/bench
  done
done

if [ -n "$S3_BUCKET" ]; then
  aws s3 cp runs/bench/ "s3://$S3_BUCKET/benchmarks/" --recursive --exclude "*" --include "${LABEL}*.json"
fi
echo "== done [$LABEL] — collect runs/bench/*.json from all three machines, then: python -m src.bench.compare =="
