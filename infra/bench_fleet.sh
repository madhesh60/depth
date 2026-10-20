#!/usr/bin/env bash
# bench_fleet.sh — the whole COOL benchmark as ONE command (reproduce it in your own account: ~40 min, < $1).
# Launches machines of EQUAL size that run the same benchmark — same commit, frames, model and recordings,
# all SHA-256 pinned — then upload results + their AWS-signed receipts to S3 and TERMINATE THEMSELVES:
#
#   A  baseline_x86         c8i.xlarge   Ubuntu 24.04 amd64 + stock OpenCV 5 (pip)
#   B  graviton_stock       c8g.xlarge   Ubuntu 24.04 arm64 + stock OpenCV 5 (pip)
#   C  graviton_cool        c8g.xlarge   COOL AMI (OpenCV 5 from /opt/cool)
#   D  graviton_cool_large  c8g.large    COOL AMI (the demo's size: does the cheapest size meet the target?)
#
# Prices are read from the AWS Price List API at launch (+ COOL's $0.01/h software fee) and stored in every
# result. Needs the stack from infra/deploy.sh (bucket, no-ingress security group, write-only bench role).
#
#   ./infra/bench_fleet.sh                       # plan (dry run)
#   APPLY=1 ./infra/bench_fleet.sh               # run, wait, download into benchmarks/
#   TARGETS="A C" APPLY=1 ./infra/bench_fleet.sh # a subset (e.g. under a small vCPU quota)
#   then: python -m src.bench.compare            # → docs/reports/cool_benchmark.md
set -euo pipefail
cd "$(dirname "$0")/.."
. infra/lib.sh

STACK="${STACK:-depth}"
TARGETS="${TARGETS:-A B C D}"
COOL_FEE="${COOL_FEE:-0.01}"
GIT_REF="${GIT_REF:-$(git rev-parse HEAD)}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
git fetch -q origin 2>/dev/null || true
git branch -r --contains "$GIT_REF" 2>/dev/null | grep -q . || die "commit $GIT_REF is not on GitHub yet - push it first"

say "stack outputs"
B=$(stack_output "$STACK" BucketName); PROFILE=$(stack_output "$STACK" BenchInstanceProfile)
SG=$(stack_output "$STACK" BenchSecurityGroup); SUBNET=$(stack_output "$STACK" SubnetId)
[ -n "$B" ] && [ "$B" != None ] || die "stack $STACK not found - run infra/deploy.sh first"
COOL_CODE="${COOL_PRODUCT_CODE:-aajkmdd4qo3r7yhqg61a7aah9}"     # Cloud Optimized OpenCV for AWS Graviton4
COOL_AMI="${COOL_AMI_ID:-$(awsq ec2 describe-images --owners aws-marketplace --filters "Name=product-code,Values=$COOL_CODE" \
            --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)}"
cool_ok() {   # subscribed? a dry-run launch of the COOL AMI is the cheapest exact test
  "$AWS_CLI" ec2 run-instances --dry-run --image-id "$COOL_AMI" --instance-type "$1" 2>&1 | tr -d '\r' | grep -q DryRunOperation
}
U_ARM=$(awsq ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id --query Parameter.Value --output text)
U_X86=$(awsq ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id --query Parameter.Value --output text)
echo "bucket $B · subnet $SUBNET · sg $SG · profile $PROFILE · run $RUN_ID · commit ${GIT_REF:0:12}"
echo "AMIs: COOL $COOL_AMI · Ubuntu 24.04 arm64 $U_ARM · amd64 $U_X86"

declare -A TYPE=([A]=c8i.xlarge [B]=c8g.xlarge [C]=c8g.xlarge [D]=c8g.large)
declare -A AMI=([A]=$U_X86 [B]=$U_ARM [C]=$COOL_AMI [D]=$COOL_AMI)
declare -A LABEL=([A]=baseline_x86 [B]=graviton_stock [C]=graviton_cool [D]=graviton_cool_large)
declare -A FLAVOR=([A]=stock [B]=stock [C]=cool [D]=cool)
TODAY=$(date -u +%Y-%m-%d)
LAUNCHED=()

say "launch"
for t in $TARGETS; do
  it=${TYPE[$t]}
  if [ "${FLAVOR[$t]}" = cool ] && ! cool_ok "$it"; then
    echo "$t ${LABEL[$t]}: SKIPPED - not subscribed to COOL yet (AWS Marketplace -> Subscribe), or $COOL_AMI unavailable"
    continue
  fi
  base=$(price_of "$it")
  if [ "${FLAVOR[$t]}" = cool ]; then
    price=$(python -I -c "print(round($base + $COOL_FEE, 5))"); src="AWS Price List API $TODAY ($it on-demand Linux $AWS_REGION \$$base/h) + COOL software fee \$$COOL_FEE/h"
  else
    price=$base; src="AWS Price List API $TODAY ($it on-demand Linux $AWS_REGION)"
  fi
  rootdev=$(awsq ec2 describe-images --image-ids "${AMI[$t]}" --query 'Images[0].RootDeviceName' --output text)
  snap=$(awsq ec2 describe-images --image-ids "${AMI[$t]}" --query "Images[0].BlockDeviceMappings[?DeviceName=='$rootdev'].Ebs.VolumeSize | [0]" --output text)
  gib=$(( snap > 20 ? snap : 20 ))
  ud="$WORK/bench-${LABEL[$t]}.sh"
  sed -e "s#@BUCKET@#$B#g" -e "s#@RUN_ID@#$RUN_ID#g" -e "s#@LABEL@#${LABEL[$t]}#g" -e "s#@FLAVOR@#${FLAVOR[$t]}#g" \
      -e "s#@PRICE@#$price#g" -e "s#@PRICE_SOURCE@#$src#g" -e "s#@GIT_REF@#$GIT_REF#g" -e "s#@REGION@#$AWS_REGION#g" \
      infra/bench_userdata.sh > "$ud"
  echo "$t ${LABEL[$t]}: $it · ${AMI[$t]} · \$$price/h · root ${gib} GiB"
  ARGS=(ec2 run-instances --image-id "${AMI[$t]}" --instance-type "$it" --count 1
        --iam-instance-profile "Name=$PROFILE"
        --network-interfaces "DeviceIndex=0,SubnetId=$SUBNET,Groups=$SG,AssociatePublicIpAddress=true,DeleteOnTermination=true"
        --metadata-options "HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=1"
        --block-device-mappings "DeviceName=$rootdev,Ebs={VolumeSize=$gib,VolumeType=gp3,Encrypted=true,DeleteOnTermination=true}"
        --instance-initiated-shutdown-behavior terminate
        --user-data "file://$ud"
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=depth-bench-${LABEL[$t]}},{Key=project,Value=depth},{Key=role,Value=bench},{Key=run,Value=$RUN_ID}]"
                             "ResourceType=volume,Tags=[{Key=project,Value=depth},{Key=role,Value=bench},{Key=run,Value=$RUN_ID}]"
        --query 'Instances[0].InstanceId' --output text)
  if [ "$APPLY" = "1" ]; then
    if iid=$(awsq "${ARGS[@]}"); then echo "   -> $iid"; LAUNCHED+=("$t")
    else echo "   launch FAILED (vCPU quota? re-run later with TARGETS=\"$t\")"; fi
  else
    printf '+ aws %s\n' "${ARGS[*]}"
  fi
done
[ "$APPLY" = "1" ] || { echo "(dry run) APPLY=1 ./infra/bench_fleet.sh to launch"; exit 0; }

say "waiting for results in s3://$B/bench/$RUN_ID/ (each machine terminates itself when done)"
declare -A DONE=()
for i in $(seq 1 110); do
  LIST=$(awsq s3 ls "s3://$B/bench/$RUN_ID/" --recursive 2>/dev/null || true)
  for t in "${LAUNCHED[@]}"; do
    [ -n "${DONE[$t]:-}" ] && continue
    if printf '%s\n' "$LIST" | grep -q "/${LABEL[$t]}/DONE$"; then DONE[$t]=DONE; echo "$(date -u +%H:%M) ${LABEL[$t]} done"
    elif printf '%s\n' "$LIST" | grep -q "/${LABEL[$t]}/FAILED$"; then DONE[$t]=FAILED; echo "$(date -u +%H:%M) ${LABEL[$t]} FAILED"; fi
  done
  [ ${#DONE[@]} -eq ${#LAUNCHED[@]} ] && break
  sleep 60
done

say "download"
mkdir -p benchmarks "$WORK/bench-logs/$RUN_ID"
for t in "${LAUNCHED[@]}"; do
  awsq s3 cp "s3://$B/bench/$RUN_ID/${LABEL[$t]}/" benchmarks/ --recursive --exclude "*" --include "*.json" --only-show-errors || true
  awsq s3 cp "s3://$B/bench/$RUN_ID/${LABEL[$t]}/bench.log" "$WORK/bench-logs/$RUN_ID/${LABEL[$t]}.log" --only-show-errors || true
  echo "${LABEL[$t]}: ${DONE[$t]:-TIMEOUT} (log: $WORK/bench-logs/$RUN_ID/${LABEL[$t]}.log)"
done

say "safety net: terminate anything of this run still alive"
LEFT=$(awsq ec2 describe-instances --filters "Name=tag:run,Values=$RUN_ID" Name=instance-state-name,Values=pending,running,stopping,stopped \
         --query 'Reservations[].Instances[].InstanceId' --output text)
[ -n "$LEFT" ] && awsq ec2 terminate-instances --instance-ids $LEFT --query 'TerminatingInstances[].InstanceId' --output text || echo "none left"
echo "next: python -m src.bench.compare"
