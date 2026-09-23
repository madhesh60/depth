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
# OPENCV_FLAVOR: "stock" pip-installs a PINNED OpenCV 5 wheel; "cool" expects the COOL AMI's
# OpenCV (COOL is an AMI, NOT a pip wheel — do not pip-install it). We prove which build actually
# ran by PROVENANCE (cv2.__file__ path + version + build info), never by "KleidiCV detected":
# the stock Arm wheel *already bundles KleidiCV*, so that check proves nothing.
OPENCV_FLAVOR="${OPENCV_FLAVOR:-stock}"
OPENCV_PIN="${OPENCV_PIN:-5.0.0.93}"             # OpenCV 5 (rule 1) — pin everything

echo "== [$LABEL] arch=$(uname -m) flavor=$OPENCV_FLAVOR =="

python3 -m pip install -q --upgrade pip
python3 -m pip install -q "numpy>=2" pyyaml       # OpenCV 5 needs numpy>=2
if [ "$OPENCV_FLAVOR" = "stock" ]; then
  python3 -m pip install -q "opencv-python-headless==${OPENCV_PIN}"
else
  # COOL AMI (Ubuntu 24.04): OpenCV lives under /opt/cool/venvs/... — activate that interpreter
  # and run this script with it (e.g. `source /opt/cool/venvs/<env>/bin/activate`). No pip install.
  echo "cool flavor: using the COOL AMI OpenCV under /opt/cool (not pip-installing)."
fi

# --- PROVE which OpenCV actually ran (provenance, not KleidiCV) ------------------------------
PROV=$(python3 - <<'PY'
import cv2, json
info = cv2.getBuildInformation()
print(json.dumps({
    "cv2_version": cv2.__version__,
    "cv2_file": cv2.__file__,
    "is_cool_path": "/opt/cool" in (cv2.__file__ or ""),   # ← the real COOL proof
    "kleidicv_in_build": ("KleidiCV" in info),             # informational only (true on stock Arm too)
    "ipp_in_build": ("Intel IPP" in info),
}))
PY
)
echo "opencv_provenance=$PROV"
IS_COOL_PATH=$(printf '%s' "$PROV" | python3 -c "import json,sys; print(json.load(sys.stdin)['is_cool_path'])")
if [ "$OPENCV_FLAVOR" = "cool" ] && [ "$IS_COOL_PATH" != "True" ]; then
  echo "ERROR: flavor=cool but cv2 is NOT loaded from /opt/cool — refusing to mislabel this run as COOL." >&2
  exit 2
fi

# AMI + instance identity (best-effort via IMDSv2) — records exactly where COOL ran
IMDS_TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)
AMI_ID=$(curl -s -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/ami-id 2>/dev/null || echo "unknown")
INSTANCE_TYPE=$(curl -s -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
echo "ami_id=$AMI_ID instance_type=$INSTANCE_TYPE"

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

# stamp the manifest hash + OpenCV provenance + AMI/instance so the result PROVES where it ran
python3 - "$OUT/$LABEL.json" "$MANIFEST_HASH" "$PROV" "$AMI_ID" "$INSTANCE_TYPE" <<'PY'
import json, sys
p, h, prov, ami, itype = sys.argv[1:6]
d = json.load(open(p))
d["input_manifest_sha256"] = h
d["opencv_provenance"] = json.loads(prov)      # cv2_file / is_cool_path / version — the COOL proof
d["ami_id"] = ami
d["instance_type"] = itype
json.dump(d, open(p, "w"), indent=2)
PY

if [ -n "$S3_BUCKET" ]; then
  aws s3 cp "$OUT/$LABEL.json" "s3://$S3_BUCKET/benchmarks/$LABEL.json"
  echo "uploaded s3://$S3_BUCKET/benchmarks/$LABEL.json"
fi
echo "== done [$LABEL] =="
