#!/bin/bash
# bench_userdata.sh — user data for ONE benchmark machine (infra/bench_fleet.sh fills in the @VARS@).
# Checks out the exact commit, fetches the SHA-256-pinned model + recordings, runs infra/bench_cool.sh,
# uploads the results, its log and the machine's AWS-signed receipt to S3, then TERMINATES itself
# (instance-initiated shutdown = terminate; a 100-minute hard cap backs that up).
set -uo pipefail
exec > /var/log/depth-bench.log 2>&1
shutdown -h +100 "DEPTH benchmark: hard time cap" || true
export HOME=/root DEBIAN_FRONTEND=noninteractive PATH="$PATH:/snap/bin"
B='@BUCKET@'; RUN='@RUN_ID@'; LABEL='@LABEL@'; FLAVOR='@FLAVOR@'; PRICE='@PRICE@'; PRICE_SRC='@PRICE_SOURCE@'
REF='@GIT_REF@'; REGION='@REGION@'
DEST="s3://$B/bench/$RUN/$LABEL"
echo "== DEPTH benchmark $LABEL ($FLAVOR) run $RUN commit $REF price $PRICE/h"

finish() {
  local st=$1
  aws s3 cp --region "$REGION" --only-show-errors /var/log/depth-bench.log "$DEST/bench.log" || true
  if [ -d /opt/depth/benchmarks ]; then
    aws s3 cp --region "$REGION" --only-show-errors --recursive /opt/depth/benchmarks/ "$DEST/" --exclude "*" \
      --include "${LABEL}__*.json" --include "${LABEL}_lh__*.json" --include "${LABEL}_t1__*.json" || true
  fi
  date -u +%FT%TZ > /tmp/depth-status
  aws s3 cp --region "$REGION" --only-show-errors /tmp/depth-status "$DEST/$st" || true
  shutdown -h now
  exit 0
}
trap 'finish FAILED' ERR
set -e

snap wait system seed.loaded || true
command -v aws >/dev/null || snap install aws-cli --classic
apt-get -o DPkg::Lock::Timeout=300 update -y
apt-get -o DPkg::Lock::Timeout=300 install -y git python3-venv
git clone -q https://github.com/madhesh60/depth.git /opt/depth
cd /opt/depth
git checkout -q "$REF"
# the clone carries the committed results: drop them, so only what this machine measures can be uploaded (a failed
# workload must not ship last week's file under today's run, nor another label that shares this prefix)
rm -f benchmarks/*.json

if [ "$FLAVOR" = cool ]; then
  PY=$(ls -d /opt/cool/venvs/python_3.1*/bin/python | sort -V | tail -1)       # COOL's interpreter, as-is
else
  python3 -m venv /opt/benchvenv
  /opt/benchvenv/bin/python -m pip install -q --upgrade pip
  /opt/benchvenv/bin/python -m pip install -q numpy==2.2.6 opencv-python-headless==5.0.0.93
  PY=/opt/benchvenv/bin/python
fi
"$PY" -m src.detection.fetch_model                          # published weights, SHA-256 checked
"$PY" -m src.cv_pipeline.humminbird fetch --large           # 150-s + 1-h recordings, every file SHA-256 pinned
FLAVOR="$FLAVOR" LABEL="$LABEL" PRICE_HOUR="$PRICE" PRICE_SOURCE="$PRICE_SRC" BENCH_PY="$PY" ./infra/bench_cool.sh
finish DONE
