#!/usr/bin/env bash
# update_server.sh — move the live server to a new commit without SSH and without replacing anything:
# SSM Run Command checks out GIT_REF and re-runs the idempotent setup script, which restarts DEPTH and
# fails unless the server comes back healthy on COOL.
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
CMD="cd /opt/depth && git fetch -q origin $GIT_REF && git reset -q --hard FETCH_HEAD && DEPTH_GIT_REF=$GIT_REF DEPTH_S3_BUCKET=$B DEPTH_LOG_GROUP=$LG bash infra/setup_cool_instance.sh"
python -I -c 'import json, sys; json.dump({"commands": [sys.argv[1]]}, open(sys.argv[2], "w"))' "$CMD" "$WORK/ssm-update.json"
echo "instance $IID -> commit ${GIT_REF:0:12}"
echo "+ $CMD"
[ "$APPLY" = "1" ] || { echo "(dry run) APPLY=1 to run it"; exit 0; }

CID=$(awsq ssm send-command --instance-ids "$IID" --document-name AWS-RunShellScript \
        --comment "DEPTH update to ${GIT_REF:0:12}" --parameters "file://$WORK/ssm-update.json" \
        --query Command.CommandId --output text)
for i in $(seq 1 90); do
  ST=$(awsq ssm get-command-invocation --command-id "$CID" --instance-id "$IID" --query Status --output text 2>/dev/null || echo Pending)
  case "$ST" in Pending|InProgress|Delayed) sleep 10 ;; *) break ;; esac
done
awsq ssm get-command-invocation --command-id "$CID" --instance-id "$IID" --query StandardOutputContent --output text | tail -15
echo "status: $ST"
[ "$ST" = Success ] || die "update failed - stderr: $(awsq ssm get-command-invocation --command-id "$CID" --instance-id "$IID" --query StandardErrorContent --output text | tail -5)"
