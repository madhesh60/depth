#!/usr/bin/env bash
# setup_cool_instance.sh — turn a fresh COOL AMI instance (Ubuntu 24.04, Graviton, OpenCV 5 under
# /opt/cool) into the live DEPTH server. Idempotent; run as root (EC2 user-data or SSM Run Command).
#
#   DEPTH_MODEL_S3=s3://<bucket>/models/EXP-001/best.onnx  DEPTH_MODEL_SHA256=<hex>  \
#   DEPTH_S3_BUCKET=<bucket>  ./infra/setup_cool_instance.sh
#
# What it does (review I-1 / X-6):
#   1. installs git + AWS CLI (snap) — nothing else from apt;
#   2. clones/updates the repo at /opt/depth (branch $DEPTH_BRANCH, default main);
#   3. fetches the ONNX model (S3 or URL) and verifies its sha256;
#   4. installs the web deps (fastapi/uvicorn/multipart — pure Python, no numpy/opencv) into
#      /opt/depth/pydeps with `pip --target` — the COOL venv itself is NEVER modified;
#   5. installs + starts the systemd unit `depth` (COOL interpreter, model warmed at startup,
#      threads = vCPUs, restarts on failure) and waits for /api/health to report the model ready
#      AND cv2 loaded from /opt/cool.
set -euo pipefail

REPO_URL="${DEPTH_REPO_URL:-https://github.com/madhesh60/depth.git}"
BRANCH="${DEPTH_BRANCH:-main}"
APP=/opt/depth
PORT="${DEPTH_PORT:-8000}"
MODEL_NAME="${DEPTH_MODEL:-EXP-001}"          # models/<name>/: calibration.json tracked, best.onnx fetched
COOL_PY="${COOL_PY:-$(ls -d /opt/cool/venvs/python_3.1*/bin/python 2>/dev/null | sort -V | tail -1)}"
[ -x "$COOL_PY" ] || { echo "ERROR: COOL interpreter not found under /opt/cool/venvs — use the COOL AMI" >&2; exit 2; }

echo "== [1/5] packages"
command -v git >/dev/null || (apt-get update -y && apt-get install -y git)
command -v aws >/dev/null || snap install aws-cli --classic

echo "== [2/5] repo -> $APP ($BRANCH)"
id depth >/dev/null 2>&1 || useradd --system --home "$APP" --shell /usr/sbin/nologin depth
if [ -d "$APP/.git" ]; then git -C "$APP" fetch -q origin "$BRANCH" && git -C "$APP" reset -q --hard "origin/$BRANCH"
else git clone -q --branch "$BRANCH" "$REPO_URL" "$APP"; fi

echo "== [3/5] model"
MODEL="$APP/models/$MODEL_NAME/best.onnx"
mkdir -p "$(dirname "$MODEL")"
if [ -n "${DEPTH_MODEL_S3:-}" ]; then aws s3 cp --only-show-errors "$DEPTH_MODEL_S3" "$MODEL"
elif [ -n "${DEPTH_MODEL_URL:-}" ]; then curl -fsSL "$DEPTH_MODEL_URL" -o "$MODEL"; fi
[ -s "$MODEL" ] || { echo "ERROR: no model at $MODEL (set DEPTH_MODEL_S3 or DEPTH_MODEL_URL)" >&2; exit 2; }
if [ -n "${DEPTH_MODEL_SHA256:-}" ]; then
  echo "$DEPTH_MODEL_SHA256  $MODEL" | sha256sum -c - || { echo "ERROR: model sha256 mismatch" >&2; exit 2; }
fi

echo "== [4/5] web deps -> $APP/pydeps (COOL venv untouched)"
# exact tree of the locally tested web stack (none of it depends on numpy/opencv)
"$COOL_PY" -m pip install -q --no-deps --target "$APP/pydeps" --upgrade \
  fastapi==0.127.0 starlette==0.50.0 pydantic==2.13.4 pydantic-core==2.46.4 annotated-types==0.7.0 \
  typing-inspection==0.4.2 anyio==4.9.0 sniffio==1.3.1 idna==3.10 exceptiongroup==1.3.1 \
  uvicorn==0.40.0 h11==0.14.0 click==8.5.0 python-multipart==0.0.21 annotated-doc==0.0.4 \
  typing-extensions==4.15.0 \
  mcp==1.28.1 httpx==0.28.1 httpcore==1.0.7 certifi==2025.1.31 httpx-sse==0.4.3 sse-starlette==3.4.5 \
  pydantic-settings==2.14.1 python-dotenv==1.0.1 jsonschema==4.23.0 jsonschema-specifications==2024.10.1 \
  referencing==0.36.2 rpds-py==0.23.1 attrs==25.3.0 pyjwt==2.13.0
# (mcp declares pyjwt[crypto]; the server path imports neither jwt nor cryptography - checked locally)
PYTHONPATH="$APP:$APP/pydeps" "$COOL_PY" -c "import fastapi, uvicorn, cv2, numpy, mcp.server.fastmcp; print('deps ok | cv2', cv2.__version__, cv2.__file__)"

# MCP bearer token (remote agents connect to https://<cloudfront>/mcp with it). Generated once,
# root-only, never in the unit file or the repo. Read it later with: sudo cat /etc/depth/mcp.env
mkdir -p /etc/depth
if [ ! -s /etc/depth/mcp.env ]; then
  ( umask 077; printf 'DEPTH_MCP_TOKEN=%s\n' "$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')" > /etc/depth/mcp.env )
fi
chmod 600 /etc/depth/mcp.env
mkdir -p "$APP/runs/jobs" /var/log/depth
chown -R depth:depth "$APP/runs" /var/log/depth

echo "== [5/5] systemd unit"
sed -e "s#@COOL_PY@#$COOL_PY#g" -e "s#@APP@#$APP#g" -e "s#@PORT@#$PORT#g" \
    -e "s#@THREADS@#$(nproc)#g" -e "s#@S3_BUCKET@#${DEPTH_S3_BUCKET:-}#g" \
    -e "s#@CORS@#${DEPTH_CORS_ORIGINS:-}#g" -e "s#@MODEL@#$MODEL_NAME#g" \
    "$APP/infra/depth.service" > /etc/systemd/system/depth.service
systemctl daemon-reload
systemctl enable --now depth
systemctl restart depth

for i in $(seq 1 60); do
  H=$(curl -fs "http://127.0.0.1:$PORT/api/health" || true)
  if printf '%s' "$H" | grep -q '"model_loaded":true'; then
    printf '%s' "$H" | grep -q '"is_cool_path":true' && echo "COOL provenance: cv2 loaded from /opt/cool ✓" \
      || echo "WARNING: cv2 is NOT loaded from /opt/cool"
    echo "DEPTH is up on :$PORT"; exit 0
  fi
  sleep 2
done
echo "ERROR: DEPTH did not become healthy — journalctl -u depth -n 100" >&2; exit 1
