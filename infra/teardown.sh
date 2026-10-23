#!/usr/bin/env bash
# teardown.sh — remove everything the DEPTH demo created. DRY RUN by default (shows what would go);
# APPLY=1 deletes:
#   1. any benchmark-fleet instance still alive (tag role=bench — they normally terminate themselves);
#   2. the stack: instance + disk, Elastic IP, CloudFront, alarms, dashboard, log group, roles, schedule …;
#   3. the retained results bucket, emptied version by version, then deleted;
#   4. the stack's SSM parameters under /depth/<stack>/ (origin secret, approver PIN, tokens);
#   5. the daily EBS snapshots (tag role=backup) - Data Lifecycle Manager keeps them after its policy is gone;
#   6. finally lists anything still tagged project=depth (expected: nothing).
# By hand (not scriptable): cancel the COOL subscription in AWS Marketplace → Manage subscriptions.
# The account Budget is left in place (free).
set -euo pipefail
cd "$(dirname "$0")/.."
. infra/lib.sh

STACK="${STACK:-depth}"
PREFIX="/depth/$STACK"
ACCOUNT=$(awsq sts get-caller-identity --query Account --output text) || die "no AWS credentials"
echo "account $ACCOUNT · region $AWS_REGION · stack $STACK · apply=$APPLY"

say "1/6 benchmark instances"
BENCH=$(awsq ec2 describe-instances --filters Name=tag:project,Values=depth Name=tag:role,Values=bench \
          Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text)
if [ -n "$BENCH" ]; then run ec2 terminate-instances --instance-ids $BENCH >/dev/null; else echo "none"; fi

say "2/6 stack"
BUCKET=""
if awsq cloudformation describe-stacks --stack-name "$STACK" >/dev/null 2>&1; then
  BUCKET=$(stack_resource "$STACK" DataBucket)
  awsq cloudformation list-stack-resources --stack-name "$STACK" \
    --query 'StackResourceSummaries[].[ResourceType,PhysicalResourceId]' --output text | sed 's/^/   /'
  run cloudformation delete-stack --stack-name "$STACK"
  if [ "$APPLY" = "1" ]; then
    echo "waiting (CloudFront takes ~10-15 min to disable and delete)…"
    "$AWS_CLI" cloudformation wait stack-delete-complete --stack-name "$STACK" || die "stack delete did not finish - check the console"
  fi
else
  echo "no stack $STACK"
fi

say "3/6 results bucket ${BUCKET:-(none)}"
if [ -n "$BUCKET" ] && [ "$BUCKET" != None ]; then
  if [ "$APPLY" = "1" ]; then
    while :; do
      awsq s3api list-object-versions --bucket "$BUCKET" --max-items 1000 --output json > "$WORK/versions.json"
      N=$(python -I -c '
import json, sys
d = json.load(open(sys.argv[1]))
objs = [{"Key": o["Key"], "VersionId": o["VersionId"]} for k in ("Versions", "DeleteMarkers") for o in (d.get(k) or [])]
json.dump({"Objects": objs, "Quiet": True}, open(sys.argv[2], "w"))
print(len(objs))' "$WORK/versions.json" "$WORK/delete.json")
      [ "$N" -gt 0 ] || break
      awsq s3api delete-objects --bucket "$BUCKET" --delete "file://$WORK/delete.json" >/dev/null
      echo "   deleted $N object version(s)"
    done
    awsq s3api delete-bucket --bucket "$BUCKET"
    echo "bucket deleted"
  else
    echo "+ empty every object version in s3://$BUCKET, then delete the bucket"
  fi
fi

say "4/6 SSM parameters under $PREFIX/ (secrets)"
PARAMS=$(awsq ssm get-parameters-by-path --path "$PREFIX/" --recursive --query 'Parameters[].Name' --output text 2>/dev/null || true)
if [ -n "$PARAMS" ] && [ "$PARAMS" != None ]; then run ssm delete-parameters --names $PARAMS >/dev/null; echo "$PARAMS"; else echo "none"; fi

say "5/6 daily backup snapshots"
SNAPS=$(awsq ec2 describe-snapshots --owner-ids self --filters Name=tag:project,Values=depth Name=tag:role,Values=backup           --query 'Snapshots[].SnapshotId' --output text 2>/dev/null || true)
if [ -n "$SNAPS" ] && [ "$SNAPS" != None ]; then
  for sn in $SNAPS; do run ec2 delete-snapshot --snapshot-id "$sn"; done
else
  echo "none"
fi

say "6/6 anything still tagged project=depth"
LEFT=$(awsq resourcegroupstaggingapi get-resources --tag-filters Key=project,Values=depth \
         --query 'ResourceTagMappingList[].ResourceARN' --output text 2>/dev/null || true)
[ -z "$LEFT" ] && echo "nothing left" || printf '%s\n' $LEFT
echo
echo "By hand: AWS Marketplace -> Manage subscriptions -> COOL -> Cancel subscription (stops the software fee)."
[ "$APPLY" = "1" ] || echo "(dry run) APPLY=1 ./infra/teardown.sh to delete"
