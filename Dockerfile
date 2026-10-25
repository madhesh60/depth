# DEPTH — the studio, API and MCP server in one CPU-only, torch-free image (OpenCV 5 cv2.dnn).
#
#   docker build -t depth .                                   # fetches the detector (SHA-256 checked)
#   docker run --rm -p 8000:8000 -v depth-data:/data depth    # http://localhost:8000
#
# Stock OpenCV 5 from PyPI (requirements.txt): this image is NOT COOL; /api/health says so (is_cool_path).
# Surveys live in memory, so run ONE replica. Everything the server writes (runs/: jobs, approvals, feedback,
# study, audit tags, webhooks) goes to the /data volume.
# Docs: docs/containers.md

# Behind a TLS-inspecting proxy, pass its CA (used only while building, never stored in the image):
#   docker build --secret id=ca,src=/path/to/ca.pem --build-arg HTTPS_PROXY=... -t depth .
FROM python:3.15-slim-bookworm AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --prefix=/install -r /tmp/requirements.txt

FROM python:3.15-slim-bookworm AS model
# 1 = download the detector from the GitHub Release; 0 = use models/<MODEL>/best.onnx from the build context
ARG FETCH_MODEL=1
ARG DEPTH_MODEL=EXP-003
# 1 = also fetch the 11 MB raw Humminbird recording (real GPS) for the Survey tab
ARG FETCH_RECORDING=1
COPY --from=deps /install /usr/local
WORKDIR /app
COPY src/ src/
COPY models/ models/
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export SSL_CERT_FILE=/run/secrets/ca; fi; \
    mkdir -p DATASET && if [ "$FETCH_MODEL" = "1" ]; then DEPTH_MODEL=$DEPTH_MODEL python -m src.detection.fetch_model; fi \
 && test -f "models/$DEPTH_MODEL/best.onnx" \
 && if [ "$FETCH_RECORDING" = "1" ]; then python -m src.cv_pipeline.humminbird fetch \
      || echo "recording fetch failed - the studio runs without it"; fi

FROM python:3.15-slim-bookworm
ARG GIT_COMMIT=""
ARG DEPTH_MODEL=EXP-003
LABEL org.opencontainers.image.title="DEPTH" \
      org.opencontainers.image.description="Side-scan sonar -> human-approved cleanup plan (OpenCV 5, CPU-only)" \
      org.opencontainers.image.source="https://github.com/madhesh60/depth" \
      org.opencontainers.image.licenses="AGPL-3.0" \
      org.opencontainers.image.revision="${GIT_COMMIT}"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    DEPTH_MODEL=${DEPTH_MODEL} DEPTH_GIT_COMMIT=${GIT_COMMIT} DEPTH_LOG_JSON=1
RUN groupadd --system --gid 10001 depth && useradd --system --uid 10001 --gid depth --home /app depth \
 && mkdir -p /data /app && chown depth:depth /data && ln -s /data /app/runs
COPY --from=deps /install /usr/local
WORKDIR /app
COPY --from=model /app/models/ models/
COPY --from=model /app/DATASET/ DATASET/
COPY src/ src/
COPY webui/ webui/
COPY JUDGES.md README.md LICENSE ./
USER 10001:10001
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4).status == 200 else 1)"
CMD ["python", "-m", "uvicorn", "src.dashboard.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "30"]
