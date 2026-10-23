#!/usr/bin/env bash
# update_server.sh — move the live server to a new commit without SSH and without replacing anything:
# SSM Run Command checks out GIT_REF and re-runs the idempotent setup script, which restarts DEPTH and
# fails unless the server comes back healthy on COOL (a rehearsal server: healthy on its stock OpenCV 5).
#
#   ./infra/update_server.sh               # dry run: prints the command
#   APPLY=1 ./infra/update_server.sh       # GIT_REF defaults to the pushed HEAD
set -euo pipefail
cd "$(dirname "$0")/.."
. infra/lib.sh

STACK="${STACK:-depth}"
GIT_REF="${GIT_REF:-$(git rev-parse HEAD)}"
git fetch -q origin 2>/dev/null || true
git branch -r --contains "$GIT_REF" 2>/dev/null | grep -q . || die "commit $GIT_REF is not on GitHub yet - push it first"
IID=$(stack_output "$STACK" InstanceId)
B=$(stack_output "$STACK" BucketName)
LG=$(stack_output "$STACK" LogGroup)
# a rehearsal server (stock Ubuntu until the COOL AMI swaps in) has no /opt/cool: the setup runs with the stock
# interpreter its user-data installed and without the COOL check - the template's rehearsal exports, read on the box
REH="if [ -s /etc/depth/rehearsal.env ]; then export COOL_PY=/opt/stock-opencv/bin/python DEPTH_REQUIRE_COOL=0; fi"
# the server reads its secrets and settings from SSM under /depth/<stack> (deploy.sh writes them)
CMD="cd /opt/depth && $REH && git fetch -q origin $GIT_REF && git reset -q --hard FETCH_HEAD && DEPTH_GIT_REF=$GIT_REF DEPTH_S3_BUCKET=$B DEPTH_LOG_GROUP=$LG DEPTH_PARAM_PREFIX=/depth/$STACK AWS_REGION=$AWS_REGION bash infra/setup_cool_instance.sh"
echo "instance $IID -> commit ${GIT_REF:0:12}"
echo "+ $CMD"
[ "$APPLY" = "1" ] || { echo "(dry run) APPLY=1 to run it"; exit 0; }

SSM_COMMENT="DEPTH update to ${GIT_REF:0:12}" ssm_run "$IID" "$CMD" | tail -15 || die "update failed"
echo "status: Success"
