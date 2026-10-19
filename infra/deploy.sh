#!/usr/bin/env bash
# deploy.sh — create or update the DEPTH demo stack (infra/cfn/depth.yaml) on Graviton4 + COOL.
#
# DRY RUN by default: resolves every input (COOL AMI + its Marketplace product code, an AZ that offers the
# instance types, the CloudFront prefix list), validates the template with CloudFormation, creates an inert
# CHANGE SET and prints it. Nothing that costs money is created. APPLY=1 executes the change set, waits until
# the server signals "serving on COOL", then checks the live URL and verifies the COOL receipt.
#
#   ALERT_EMAIL=you@example.com ./infra/deploy.sh                     # plan
#   APPLY=1 ALERT_EMAIL=you@example.com ./infra/deploy.sh             # deploy (~15-25 min)
#   APPLY=1 INSTANCE_TYPE=c8g.xlarge ./infra/deploy.sh                # resize (stop/start, same address)
#   COOL_AMI_ID=ami-... ./infra/deploy.sh                             # pin the AMI if the lookup is ambiguous
# Code updates on the running server: ./infra/update_server.sh (SSM, no SSH, no replacement).
set -euo pipefail
cd "$(dirname "$0")/.."
. infra/lib.sh

STACK="${STACK:-depth}"
SECRET_PARAM="${SECRET_PARAM:-/depth/$STACK/origin-secret}"
TEMPLATE=infra/cfn/depth.yaml

say "identity"
ACCOUNT=$(awsq sts get-caller-identity --query Account --output text) || die "no AWS credentials (aws login --profile hackathon)"
echo "account $ACCOUNT · region $AWS_REGION · profile ${AWS_PROFILE:-env} · cli $AWS_CLI · apply=$APPLY"
BUDGETS=$(awsq budgets describe-budgets --account-id "$ACCOUNT" --query 'Budgets[].BudgetName' --output text 2>/dev/null || true)
[ -n "$BUDGETS" ] && echo "cost guard: budget(s) $BUDGETS" || echo "WARNING: no AWS Budget in this account - create one before deploying"

STATUS=$(awsq cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo NONE)
echo "stack $STACK: $STATUS"
PARAMS=()

if [ "$STATUS" = NONE ] || [ "$STATUS" = REVIEW_IN_PROGRESS ]; then
  : "${ALERT_EMAIL:?set ALERT_EMAIL (alarm emails) for the first deploy}"
  INSTANCE_TYPE="${INSTANCE_TYPE:-c8g.large}"
  STOP_AT="${STOP_AT:-2026-11-10T06:30:00}"
  GIT_REF="${GIT_REF:-$(git rev-parse HEAD)}"
  git fetch -q origin 2>/dev/null || true
  git branch -r --contains "$GIT_REF" 2>/dev/null | grep -q . || die "commit $GIT_REF is not on GitHub yet - push it first (the server clones it)"

  say "COOL AMI"
  AMI="${COOL_AMI_ID:-}"
  if [ -z "$AMI" ]; then
    awsq ec2 describe-images --owners aws-marketplace --filters "Name=architecture,Values=arm64" \
      "Name=name,Values=*COOL*,*cool*,*Cloud*Optimized*OpenCV*" \
      --query 'reverse(sort_by(Images,&CreationDate))[].[ImageId,CreationDate,Name]' --output text > "$WORK/cool-amis.txt"
    cat "$WORK/cool-amis.txt"
    AMI=$(head -1 "$WORK/cool-amis.txt" | cut -f1)
    [ -n "$AMI" ] || die "no COOL AMI visible - subscribe to the COOL Graviton4 listing first, or set COOL_AMI_ID"
    echo "picked the newest: $AMI (set COOL_AMI_ID to override)"
  fi
  read -r ARCH ROOTDEV CODES < <(awsq ec2 describe-images --image-ids "$AMI" \
    --query 'Images[0].[Architecture,RootDeviceName,join(`","`,ProductCodes[].ProductCodeId || `[]`)]' --output text)
  SNAP_GIB=$(awsq ec2 describe-images --image-ids "$AMI" \
    --query "Images[0].BlockDeviceMappings[?DeviceName=='$ROOTDEV'].Ebs.VolumeSize | [0]" --output text)
  echo "AMI $AMI · $ARCH · root $ROOTDEV ${SNAP_GIB} GiB · Marketplace product code(s): ${CODES:-none}"
  [ "$ARCH" = arm64 ] || die "$AMI is $ARCH, not arm64 (Graviton)"
  [ -n "$CODES" ] && [ "$CODES" != None ] || die "$AMI carries no Marketplace product code - not the COOL listing"
  ROOT_GIB=$(( SNAP_GIB > 20 ? SNAP_GIB : 20 ))

  say "network"
  VPC=$(awsq ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  [ "$VPC" != None ] || die "no default VPC in $AWS_REGION"
  # one AZ that offers the demo type AND the benchmark fleet's types (the fleet launches in the same subnet)
  AZ=$(awsq ec2 describe-instance-type-offerings --location-type availability-zone \
        --filters "Name=instance-type,Values=$INSTANCE_TYPE,c8g.large,c8g.xlarge,c8i.xlarge" \
        --query 'InstanceTypeOfferings[].[Location,InstanceType]' --output text \
      | sort -u | cut -f1 | sort | uniq -c | awk -v n="$(printf '%s\n' "$INSTANCE_TYPE" c8g.large c8g.xlarge c8i.xlarge | sort -u | wc -l)" '$1==n {print $2}' | head -1)
  [ -n "$AZ" ] || die "no AZ in $AWS_REGION offers $INSTANCE_TYPE + c8g.large + c8g.xlarge + c8i.xlarge"
  SUBNET=$(awsq ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC" "Name=availability-zone,Values=$AZ" \
            Name=default-for-az,Values=true --query 'Subnets[0].SubnetId' --output text)
  PL=$(awsq ec2 describe-managed-prefix-lists --filters Name=prefix-list-name,Values=com.amazonaws.global.cloudfront.origin-facing \
        --query 'PrefixLists[0].PrefixListId' --output text)
  echo "vpc $VPC · subnet $SUBNET ($AZ) · CloudFront origin-facing prefix list $PL"

  say "origin secret (SSM parameter $SECRET_PARAM - the value is never printed or templated)"
  if awsq ssm get-parameter --name "$SECRET_PARAM" --query Parameter.Name --output text >/dev/null 2>&1; then
    echo "exists"
  else
    # created even in a dry run: free, and CloudFormation must resolve it to build the change set
    awsq ssm put-parameter --name "$SECRET_PARAM" --type String --value "$(rand_hex 24)" \
      --description "DEPTH: CloudFront X-Origin-Verify value (origin access token)" --tags Key=project,Value=depth >/dev/null
    echo "created"
  fi

  PARAMS=(CoolAmiId="$AMI" InstanceType="$INSTANCE_TYPE" RootDeviceName="$ROOTDEV" RootVolumeGiB="$ROOT_GIB"
          GitRef="$GIT_REF" VpcId="$VPC" SubnetId="$SUBNET" CloudFrontPrefixListId="$PL"
          OriginSecretParam="$SECRET_PARAM" AlertEmail="$ALERT_EMAIL" StopAt="$STOP_AT")
else
  # update: only what you set changes; every other parameter keeps its value (no accidental replacement)
  [ -n "${INSTANCE_TYPE:-}" ] && PARAMS+=(InstanceType="$INSTANCE_TYPE")
  [ -n "${STOP_AT:-}" ] && PARAMS+=(StopAt="$STOP_AT")
  [ -n "${ALERT_EMAIL:-}" ] && PARAMS+=(AlertEmail="$ALERT_EMAIL")
  [ -n "${COOL_AMI_ID:-}" ] && { echo "WARNING: a new AMI REPLACES the instance"; PARAMS+=(CoolAmiId="$COOL_AMI_ID"); }
fi

say "template"
awsq cloudformation validate-template --template-body "file://$TEMPLATE" --query 'Description' --output text | cut -c1-120

DEPLOY=(cloudformation deploy --stack-name "$STACK" --template-file "$TEMPLATE" --capabilities CAPABILITY_IAM
        --tags project=depth --no-fail-on-empty-changeset)
[ ${#PARAMS[@]} -gt 0 ] && DEPLOY+=(--parameter-overrides "${PARAMS[@]}")

if [ "$APPLY" != "1" ]; then
  say "change set (dry run - nothing is executed)"
  OUT=$("$AWS_CLI" "${DEPLOY[@]}" --no-execute-changeset | tr -d '\r') || die "change set failed: $OUT"
  CS=$(printf '%s\n' "$OUT" | grep -oE 'arn:aws[a-z-]*:cloudformation:[^ ]+changeSet/[^ ]+' | head -1 || true)
  if [ -n "$CS" ]; then
    awsq cloudformation describe-change-set --change-set-name "$CS" \
      --query 'Changes[].ResourceChange.[Action,LogicalResourceId,ResourceType,Replacement]' --output table
    echo "review above; then: APPLY=1 ./infra/deploy.sh"
  else
    printf '%s\n' "$OUT"
  fi
  exit 0
fi

say "deploying (the instance signals only once DEPTH serves on COOL; ~15-25 min)"
"$AWS_CLI" "${DEPLOY[@]}" --disable-rollback | tr -d '\r'
URL=$(stack_output "$STACK" Url)
IID=$(stack_output "$STACK" InstanceId)
echo "live: $URL · instance $IID · dashboard: $(stack_output "$STACK" DashboardUrl)"

say "live check through CloudFront"
for i in $(seq 1 60); do
  H=$(curl -fsS "$URL/api/health" 2>/dev/null || true)
  if printf '%s' "$H" | grep -q '"is_cool_path":true'; then echo "$URL/api/health: serving on COOL"; break; fi
  sleep 10
done
python -m src.bench.verify_cool "$URL" || echo "WARNING: the COOL receipt did not verify - see above"
echo "shell (no SSH): aws ssm start-session --target $IID"
echo "SNS: confirm the subscription email sent to the alert address"
