#!/usr/bin/env bash
# setup_cool_instance.sh — turn a fresh COOL AMI instance (Ubuntu 24.04, Graviton, OpenCV 5 under
# /opt/cool) into the live DEPTH server. Idempotent; run as root (the CloudFormation user-data in
# infra/cfn/depth.yaml calls it, or SSM Run Command for an update).
#
#   DEPTH_GIT_REF=<commit|branch>  DEPTH_S3_BUCKET=<bucket>  DEPTH_LOG_GROUP=<log group>  ./infra/setup_cool_instance.sh
#
# What it does:
#   1. installs git + AWS CLI only if missing (nothing else from apt);
#   2. checks out the repo at /opt/depth at EXACTLY $DEPTH_GIT_REF (a commit SHA pins what judges see);
#   3. fetches the ONNX detector with src.detection.fetch_model (published release, SHA-256 checked) —
#      or DEPTH_MODEL_S3 / DEPTH_MODEL_URL + DEPTH_MODEL_SHA256 for a mirror;
#   4. installs the web deps (pure Python, no numpy/opencv) into /opt/depth/pydeps with `pip --target` —
#      the COOL venv itself is NEVER modified;
#   5. CloudWatch agent: ships the server log (one JSON line per analysed frame) + memory/disk metrics;
#   6. installs + starts the systemd unit `depth` (COOL interpreter, threads = vCPUs) and waits until
#      /api/health reports the model warm AND cv2 loaded from /opt/cool — it FAILS otherwise, so the
#      CloudFormation signal means "serving on COOL".
set -euo pipefail

REPO_URL="${DEPTH_REPO_URL:-https://github.com/madhesh60/depth.git}"
REF="${DEPTH_GIT_REF:-${DEPTH_BRANCH:-main}}"
APP=/opt/depth
PORT="${DEPTH_PORT:-8000}"
MODEL_NAME="${DEPTH_MODEL:-EXP-003}"          # models/<name>/: calibration.json tracked, best.onnx fetched
REQUIRE_COOL="${DEPTH_REQUIRE_COOL:-1}"
COOL_PY="${COOL_PY:-$(ls -d /opt/cool/venvs/python_3.1*/bin/python 2>/dev/null | sort -V | tail -1)}"
[ -x "$COOL_PY" ] || { echo "ERROR: COOL interpreter not found under /opt/cool/venvs — use the COOL AMI" >&2; exit 2; }
APT=(apt-get -o DPkg::Lock::Timeout=300)
export PATH="$PATH:/snap/bin"

echo "== [1/6] packages"
command -v git >/dev/null || { "${APT[@]}" update -y && "${APT[@]}" install -y git; }
command -v aws >/dev/null || snap install aws-cli --classic

echo "== [2/6] repo -> $APP @ $REF"
id depth >/dev/null 2>&1 || useradd --system --home "$APP" --shell /usr/sbin/nologin depth
[ -d "$APP/.git" ] || git clone -q "$REPO_URL" "$APP"
git -C "$APP" fetch -q origin "$REF"
git -C "$APP" reset -q --hard FETCH_HEAD
echo "   running commit $(git -C "$APP" rev-parse --short HEAD)"

echo "== [3/6] model"
MODEL="$APP/models/$MODEL_NAME/best.onnx"
mkdir -p "$(dirname "$MODEL")"
if [ -n "${DEPTH_MODEL_S3:-}" ]; then aws s3 cp --only-show-errors "$DEPTH_MODEL_S3" "$MODEL"
elif [ -n "${DEPTH_MODEL_URL:-}" ]; then curl -fsSL "$DEPTH_MODEL_URL" -o "$MODEL"
else ( cd "$APP" && PYTHONPATH="$APP" DEPTH_MODEL="$MODEL_NAME" "$COOL_PY" -m src.detection.fetch_model ); fi
[ -s "$MODEL" ] || { echo "ERROR: no model at $MODEL" >&2; exit 2; }
if [ -n "${DEPTH_MODEL_SHA256:-}" ]; then
  echo "$DEPTH_MODEL_SHA256  $MODEL" | sha256sum -c - || { echo "ERROR: model sha256 mismatch" >&2; exit 2; }
fi
# the raw recording with real per-ping GPS (PINGMapper sample, ~11 MB, every file SHA-256 pinned) -
# optional: the "Raw recording" survey source appears only if this succeeds
( cd "$APP" && PYTHONPATH="$APP" "$COOL_PY" -m src.cv_pipeline.humminbird fetch ) \
  || echo "note: raw-recording fetch skipped - the real-GPS survey source will be hidden"

echo "== [4/6] web deps -> $APP/pydeps (COOL venv untouched)"
# exact tree of the locally tested web stack (none of it depends on numpy/opencv); boto3 lets the job
# store mirror every survey's exports to s3://$DEPTH_S3_BUCKET/results/
if "$COOL_PY" -m pip --version >/dev/null 2>&1; then
  PIP=("$COOL_PY" -m pip install -q --no-deps --target "$APP/pydeps" --upgrade)
else
  # no pip inside the COOL venv (and we never add one): the system pip fetches wheels built for COOL's
  # exact interpreter (version + arch) into the separate pydeps folder
  python3 -m pip --version >/dev/null 2>&1 || "${APT[@]}" install -y python3-pip
  PYV=$("$COOL_PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
  PIP=(python3 -m pip install -q --no-deps --target "$APP/pydeps" --upgrade --only-binary=:all:
       --python-version "$PYV" --implementation cp --platform "manylinux2014_$(uname -m)" --platform "manylinux_2_28_$(uname -m)")
fi
"${PIP[@]}" \
  fastapi==0.127.0 starlette==0.50.0 pydantic==2.13.4 pydantic-core==2.46.4 annotated-types==0.7.0 \
  typing-inspection==0.4.2 anyio==4.9.0 sniffio==1.3.1 idna==3.10 exceptiongroup==1.3.1 \
  uvicorn==0.40.0 h11==0.14.0 click==8.5.0 python-multipart==0.0.21 annotated-doc==0.0.4 \
  typing-extensions==4.15.0 \
  mcp==1.28.1 httpx==0.28.1 httpcore==1.0.7 certifi==2025.1.31 httpx-sse==0.4.3 sse-starlette==3.4.5 \
  pydantic-settings==2.14.1 python-dotenv==1.0.1 jsonschema==4.23.0 jsonschema-specifications==2024.10.1 \
  referencing==0.36.2 rpds-py==0.23.1 attrs==25.3.0 pyjwt==2.13.0 \
  boto3==1.43.98 botocore==1.43.98 s3transfer==0.19.2 jmespath==1.1.0 python-dateutil==2.9.0.post0 \
  urllib3==2.3.0 six==1.17.0
# (mcp declares pyjwt[crypto]; the server path imports neither jwt nor cryptography - checked locally)
PYTHONPATH="$APP:$APP/pydeps" "$COOL_PY" -c "import fastapi, uvicorn, cv2, numpy, boto3, mcp.server.fastmcp; print('deps ok | cv2', cv2.__version__, cv2.__file__)"

# MCP bearer token (remote agents connect to https://<cloudfront>/mcp with it). Generated once,
# root-only, never in the unit file or the repo. Read it later with: sudo cat /etc/depth/mcp.env
install -d -m 700 /etc/depth
if [ ! -s /etc/depth/mcp.env ]; then
  ( umask 077; printf 'DEPTH_MCP_TOKEN=%s\n' "$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')" > /etc/depth/mcp.env )
fi
chmod 600 /etc/depth/mcp.env
# Approver PIN: only people holding it can approve / decline / confirm (agents and API-token holders
# can only ASK). Generated once, root-only; hand it to the judges privately with the submission.
# DEPTH_APPROVALS_OPEN=1 skips it (the studio then says approvals are open to anyone).
if [ "${DEPTH_APPROVALS_OPEN:-0}" != "1" ] && [ ! -s /etc/depth/approver.env ]; then
  ( umask 077; printf 'DEPTH_APPROVER_PIN=%s\n' "$(od -An -N4 -tu4 /dev/urandom | tr -d ' \n' | tail -c 6)" > /etc/depth/approver.env )
fi
[ -f /etc/depth/approver.env ] && chmod 600 /etc/depth/approver.env
# Admin token for webhook management and the operator load test (same handling).
if [ ! -s /etc/depth/admin.env ]; then
  ( umask 077; printf 'DEPTH_ADMIN_TOKEN=%s\n' "$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')" > /etc/depth/admin.env )
fi
chmod 600 /etc/depth/admin.env
mkdir -p "$APP/runs/jobs" /var/log/depth
chown -R depth:depth "$APP/runs" /var/log/depth

echo "== [5/6] CloudWatch agent (logs + memory/disk)"
cat > /etc/logrotate.d/depth <<'EOF'
/var/log/depth/depth.log {
  daily
  rotate 7
  maxsize 50M
  compress
  missingok
  notifempty
  copytruncate
}
EOF
if [ -n "${DEPTH_LOG_GROUP:-}" ]; then
  if [ ! -x /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl ]; then
    ARCH=$(dpkg --print-architecture)                     # arm64 on Graviton
    curl -fsSL -o /tmp/cwagent.deb "https://amazoncloudwatch-agent.s3.amazonaws.com/ubuntu/$ARCH/latest/amazon-cloudwatch-agent.deb"
    dpkg -i -E /tmp/cwagent.deb && rm -f /tmp/cwagent.deb
  fi
  cat > /opt/aws/amazon-cloudwatch-agent/etc/depth.json <<EOF
{"agent": {"metrics_collection_interval": 60, "run_as_user": "root"},
 "logs": {"logs_collected": {"files": {"collect_list": [
   {"file_path": "/var/log/depth/depth.log", "log_group_name": "$DEPTH_LOG_GROUP", "log_stream_name": "{instance_id}"}]}}},
 "metrics": {"namespace": "CWAgent", "append_dimensions": {"InstanceId": "\${aws:InstanceId}"},
   "metrics_collected": {"mem": {"measurement": ["mem_used_percent"]},
                         "disk": {"measurement": ["used_percent"], "resources": ["/"], "drop_device": true}}}}
EOF
  /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -s \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/depth.json
else
  echo "   DEPTH_LOG_GROUP unset - agent skipped (logs stay in /var/log/depth/depth.log)"
fi

echo "== [6/6] systemd unit"
sed -e "s#@COOL_PY@#$COOL_PY#g" -e "s#@APP@#$APP#g" -e "s#@PORT@#$PORT#g" \
    -e "s#@THREADS@#$(nproc)#g" -e "s#@S3_BUCKET@#${DEPTH_S3_BUCKET:-}#g" \
    -e "s#@CORS@#${DEPTH_CORS_ORIGINS:-}#g" -e "s#@MODEL@#$MODEL_NAME#g" \
    "$APP/infra/depth.service" > /etc/systemd/system/depth.service
systemctl daemon-reload
systemctl enable --now depth
systemctl restart depth

for i in $(seq 1 90); do
  H=$(curl -fs "http://127.0.0.1:$PORT/api/health" || true)
  if printf '%s' "$H" | grep -q '"model_loaded":true'; then
    if printf '%s' "$H" | grep -q '"is_cool_path":true'; then
      echo "COOL provenance: cv2 loaded from /opt/cool ✓"
    else
      echo "ERROR: the server runs, but cv2 is NOT loaded from /opt/cool" >&2
      [ "$REQUIRE_COOL" = "1" ] && exit 3
    fi
    [ -s /etc/depth/approver.env ] && echo "approver PIN: sudo cat /etc/depth/approver.env (share it privately)"
    echo "DEPTH is up on :$PORT"; exit 0
  fi
  sleep 2
done
echo "ERROR: DEPTH did not become healthy — journalctl -u depth -n 100; tail /var/log/depth/depth.log" >&2; exit 1
