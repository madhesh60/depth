#!/usr/bin/env bash
# benchmark_graviton.sh — run the Stage-1 COOL benchmark on one machine and upload the result.
#
# The "Best Use of COOL" case = a reproducible 3-way benchmark of the same workload:
#     (A) x86      + stock OpenCV   -> label baseline_x86       (run on a c7i instance)
#     (B) Graviton + stock OpenCV   -> label graviton_stock     (run on a c7g/c8g instance)
#     (C) Graviton + COOL/KleidiCV  -> label graviton_cool      (run on the SAME c7g/c8g)
# (B)->(C) isolates COOL on identical hardware; (A)->(C) is the headline Arm-vs-x86 story.
#
# Usage (run once per configuration, changing only OPENCV_FLAVOR + LABEL):
#     LABEL=graviton_cool  S3_BUCKET=my-bucket  ./infra/benchmark_graviton.sh
#
# Prereqs on the instance: git, python3.10+, awscli configured (instance role or keys).
set -euo pipefail

LABEL="${LABEL:?set LABEL=baseline_x86|graviton_stock|graviton_cool}"
S3_BUCKET="${S3_BUCKET:-}"                       # optional: upload results here
FRAMES="${FRAMES:-frames}"                       # local dir of sonar frames (>=1000 for the report)
LIMIT="${LIMIT:-1000}"
REPEATS="${REPEATS:-3}"
# OPENCV_FLAVOR: "stock" installs opencv-python-headless; "cool" expects a COOL/KleidiCV
# OpenCV build already on PATH (or pip index) — the benchmark auto-detects KleidiCV in the
# build info and records it, so a run is self-documenting about which flavor actually ran.
OPENCV_FLAVOR="${OPENCV_FLAVOR:-stock}"

echo "== [$LABEL] arch=$(uname -m) flavor=$OPENCV_FLAVOR =="

python3 -m pip install -q --upgrade pip
python3 -m pip install -q numpy pyyaml
if [ "$OPENCV_FLAVOR" = "stock" ]; then
  python3 -m pip install -q opencv-python-headless
fi   # "cool": install the COOL/KleidiCV-accelerated OpenCV wheel here per AWS COOL docs.

# pull frames from S3 if a bucket + prefix is given and the local dir is empty
if [ -n "$S3_BUCKET" ] && [ ! -d "$FRAMES" ]; then
  aws s3 sync "s3://$S3_BUCKET/frames/" "$FRAMES/"
fi

# input manifest hash → pins the exact frame set for reproducibility (10% criterion)
MANIFEST_HASH=$(find "$FRAMES" -type f | sort | sha256sum | cut -d' ' -f1)
echo "input_manifest_sha256=$MANIFEST_HASH"

OUT="runs/bench"
python3 -m src.cv_pipeline.benchmark \
  --images "$FRAMES" --limit "$LIMIT" --repeats "$REPEATS" \
  --label "$LABEL" --out "$OUT"

# stamp the manifest hash into the JSON so the result is self-describing
python3 - "$OUT/$LABEL.json" "$MANIFEST_HASH" <<'PY'
import json, sys
p, h = sys.argv[1], sys.argv[2]
d = json.load(open(p)); d["input_manifest_sha256"] = h
json.dump(d, open(p, "w"), indent=2)
PY

if [ -n "$S3_BUCKET" ]; then
  aws s3 cp "$OUT/$LABEL.json" "s3://$S3_BUCKET/benchmarks/$LABEL.json"
  echo "uploaded s3://$S3_BUCKET/benchmarks/$LABEL.json"
fi
echo "== done [$LABEL] =="
