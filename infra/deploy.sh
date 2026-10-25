#!/usr/bin/env bash
# deploy.sh — create or update the DEPTH demo stack (infra/cfn/depth.yaml) on Graviton4 + COOL.
#
# DRY RUN by default: resolves every input (the COOL AMI by its Marketplace product code, an AZ that offers
# the instance types, the CloudFront prefix list), checks the COOL subscription with a dry-run launch,
# validates the template with CloudFormation, creates an inert CHANGE SET and prints it. Nothing that costs
# money is created. APPLY=1 executes the change set, waits until the server signals that it is serving,
# then checks the live URL and the COOL receipt.
#
#   ALERT_EMAIL=you@example.com ./infra/deploy.sh                     # plan
#   APPLY=1 ALERT_EMAIL=you@example.com ./infra/deploy.sh             # deploy (~15-25 min)
#   REHEARSAL=1 APPLY=1 ALERT_EMAIL=… ./infra/deploy.sh               # dress rehearsal: stock Ubuntu, NOT COOL
#   REHEARSAL=0 APPLY=1 ./infra/deploy.sh                             # rehearsal -> COOL (replaces the instance;
#                                                                     #   the Elastic IP and CloudFront stay)
#   APPLY=1 INSTANCE_TYPE=c8g.xlarge ./infra/deploy.sh                # resize (stop/start, same address)
#   ENABLE_CDN=0 APPLY=1 ./infra/deploy.sh                            # no CloudFront (a new account waits for AWS
#                                                                     #   Support to verify it): the origin stays
#                                                                     #   closed; the checks run on the box over SSM
#   ENABLE_CDN=1 APPLY=1 ./infra/deploy.sh                            # add CloudFront later (same server)
#   PUBLIC_HTTPS=1 APPLY=1 ./infra/deploy.sh                          # no CloudFront yet: HTTPS on the Elastic IP
#                                                                     #   (Caddy + Let's Encrypt, <ip>.sslip.io;
#                                                                     #   443 opens, 8000 stays closed)
# Code updates on the running server: ./infra/update_server.sh (SSM, no SSH, no replacement).
set -euo pipefail
cd "$(dirname "$0")/.."
. infra/lib.sh

STACK="${STACK:-depth}"
SECRET_PARAM="${SECRET_PARAM:-/depth/$STACK/origin-secret}"
COOL_CODE="${COOL_PRODUCT_CODE:-aajkmdd4qo3r7yhqg61a7aah9}"     # Cloud Optimized OpenCV for AWS Graviton4
TEMPLATE=infra/cfn/depth.yaml
case "${ENABLE_CDN:-}" in 0) CDN=false ;; 1|"") CDN=true ;; *) die "ENABLE_CDN must be 0 or 1" ;; esac
case "${PUBLIC_HTTPS:-}" in 0) HTTPS=false ;; 1) HTTPS=true ;; "") HTTPS="" ;; *) die "PUBLIC_HTTPS must be 0 or 1" ;; esac

# resolve_ami: sets AMI REH ROOTDEV ROOT_GIB (+ checks arm64, the COOL product code and the subscription)
resolve_ami() {
  say "server image"
  if [ "${REHEARSAL:-0}" = "1" ]; then
    REH=true
    AMI=$(awsq ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id \
          --query Parameter.Value --output text)
    echo "REHEARSAL: stock Ubuntu 24.04 arm64 $AMI + stock OpenCV 5 (pip) - NOT COOL"
  else
    REH=false
    AMI="${COOL_AMI_ID:-}"
    if [ -z "$AMI" ]; then
      AMI=$(awsq ec2 describe-images --owners aws-marketplace --filters "Name=product-code,Values=$COOL_CODE" \
            --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)
      [ -n "$AMI" ] && [ "$AMI" != None ] || die "no Marketplace AMI carries the COOL product code $COOL_CODE"
      echo "COOL AMI = newest image with product code $COOL_CODE"
    fi
  fi
  local arch codes snap name
  read -r arch ROOTDEV codes < <(awsq ec2 describe-images --image-ids "$AMI" \
    --query 'Images[0].[Architecture,RootDeviceName,join(`","`,ProductCodes[].ProductCodeId || `[]`)]' --output text)
  name=$(awsq ec2 describe-images --image-ids "$AMI" --query 'Images[0].Name' --output text)
  snap=$(awsq ec2 describe-images --image-ids "$AMI" \
    --query "Images[0].BlockDeviceMappings[?DeviceName=='$ROOTDEV'].Ebs.VolumeSize | [0]" --output text)
  echo "$AMI ($name) · $arch · root $ROOTDEV ${snap} GiB · product code(s): ${codes:-none}"
  [ "$arch" = arm64 ] || die "$AMI is $arch, not arm64 (Graviton)"
  if [ "$REH" = false ]; then
    case ",$codes," in *",$COOL_CODE,"*) ;; *) die "$AMI does not carry the COOL product code $COOL_CODE";; esac
    local probe
    probe=$("$AWS_CLI" ec2 run-instances --dry-run --image-id "$AMI" --instance-type "${INSTANCE_TYPE:-c8g.large}" 2>&1 | tr -d '\r' || true)
    case "$probe" in
      *DryRunOperation*) echo "COOL subscription: OK (a dry-run launch is allowed)" ;;
      *OptInRequired*)   die "not subscribed to COOL yet - AWS Marketplace -> 'Cloud Optimized OpenCV for AWS Graviton4' -> Subscribe, then re-run" ;;
      *)                 echo "WARNING: dry-run launch answered: ${probe:0:200}" ;;
    esac
  fi
  ROOT_GIB=$(( snap > 20 ? snap : 20 ))
}

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
  resolve_ami

  say "network"
  VPC=$(awsq ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  [ "$VPC" != None ] || die "no default VPC in $AWS_REGION"
  # one AZ that offers the demo type AND the benchmark fleet's types (the fleet launches in the same subnet)
  NEED=$(printf '%s\n' "$INSTANCE_TYPE" c8g.large c8g.xlarge c8i.xlarge | sort -u | wc -l | tr -d ' ')
  AZ=$(awsq ec2 describe-instance-type-offerings --location-type availability-zone \
        --filters "Name=instance-type,Values=$INSTANCE_TYPE,c8g.large,c8g.xlarge,c8i.xlarge" \
        --query 'InstanceTypeOfferings[].[Location,InstanceType]' --output text \
      | sort -u | cut -f1 | sort | uniq -c | awk -v n="$NEED" '$1==n {print $2}' | head -1)
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

  PARAMS=(CoolAmiId="$AMI" Rehearsal="$REH" InstanceType="$INSTANCE_TYPE" RootDeviceName="$ROOTDEV"
          RootVolumeGiB="$ROOT_GIB" GitRef="$GIT_REF" VpcId="$VPC" SubnetId="$SUBNET" CloudFrontPrefixListId="$PL"
          OriginSecretParam="$SECRET_PARAM" AlertEmail="$ALERT_EMAIL" StopAt="$STOP_AT" EnableCdn="$CDN"
          PublicHttps="${HTTPS:-false}")
else
  # update: only what you set changes; every other parameter keeps its value (no accidental replacement)
  if [ -n "${REHEARSAL:-}" ] || [ -n "${COOL_AMI_ID:-}" ]; then
    resolve_ami
    echo "NOTE: a new server image REPLACES the instance: the new one must signal before the old one goes;"
    echo "      the Elastic IP, CloudFront, bucket and alarms stay"
    GIT_REF="${GIT_REF:-$(git rev-parse HEAD)}"
    git fetch -q origin 2>/dev/null || true
    git branch -r --contains "$GIT_REF" 2>/dev/null | grep -q . || die "commit $GIT_REF is not on GitHub yet - push it first"
    PARAMS+=(CoolAmiId="$AMI" Rehearsal="$REH" RootDeviceName="$ROOTDEV" RootVolumeGiB="$ROOT_GIB" GitRef="$GIT_REF")
  fi
  [ -n "${INSTANCE_TYPE:-}" ] && PARAMS+=(InstanceType="$INSTANCE_TYPE")
  [ -n "${STOP_AT:-}" ] && PARAMS+=(StopAt="$STOP_AT")
  [ -n "${ALERT_EMAIL:-}" ] && PARAMS+=(AlertEmail="$ALERT_EMAIL")
  [ -n "${ENABLE_CDN:-}" ] && PARAMS+=(EnableCdn="$CDN")
  [ -n "${HTTPS:-}" ] && PARAMS+=(PublicHttps="$HTTPS")
fi
[ -n "${APPROVER_EMAIL:-}" ] && PARAMS+=(ApproverEmail="$APPROVER_EMAIL")

# cost anomaly monitor: an account holds one AWS-services monitor (AWS makes one for new accounts) - reuse it. Asked
# only while the stack does not carry the parameter yet, so a later run never swaps the stack's own monitor away.
HAS_CM=0
[ "$STATUS" != NONE ] && HAS_CM=$(awsq cloudformation describe-stacks --stack-name "$STACK" \
  --query "length(Stacks[0].Parameters[?ParameterKey=='CostMonitorArn'])" --output text)
if [ "$HAS_CM" = 0 ]; then
  CM=$(awsq ce get-anomaly-monitors --output text --query \
    "AnomalyMonitors[?MonitorType=='DIMENSIONAL' && MonitorDimension=='SERVICE' && MonitorName!='$STACK-services'].MonitorArn | [0]" \
    2>/dev/null || true)
  [ "$CM" = None ] && CM=""
  echo "cost anomaly monitor: ${CM:-none yet - the stack creates one}"
  PARAMS+=(CostMonitorArn="$CM")
fi

# secrets the server reads at setup: SecureString (AWS-managed key aws/ssm), created once, never printed; the value
# goes through stdin into a root-only file that is deleted right after (never on a command line)
ensure_secrets() {
  local name desc f
  for name in approver-pin mcp-token admin-token; do
    if awsq ssm get-parameter --name "/depth/$STACK/$name" --query Parameter.Name --output text >/dev/null 2>&1; then
      echo "   /depth/$STACK/$name: exists"; continue
    fi
    case "$name" in
      approver-pin) desc="DEPTH approver PIN - people decide with it, agents can only ask" ;;
      mcp-token)    desc="DEPTH MCP bearer token - remote agents" ;;
      admin-token)  desc="DEPTH admin token - webhooks and the load test" ;;
    esac
    f="$WORK/param-$name.json"
    { [ "$name" = approver-pin ] && python -I -c 'import secrets; print(f"{secrets.randbelow(10**6):06d}", end="")' \
      || rand_hex 24; } | ( umask 077; python -I -c 'import json, sys
json.dump({"Name": sys.argv[1], "Value": sys.stdin.read(), "Type": "SecureString", "Description": sys.argv[2],
           "Tags": [{"Key": "project", "Value": "depth"}]}, open(sys.argv[3], "w"))' "/depth/$STACK/$name" "$desc" "$f" )
    awsq ssm put-parameter --cli-input-json "file://$f" >/dev/null; rm -f "$f"
    echo "   /depth/$STACK/$name: created"
  done
}

say "template"
SIZE=$(wc -c < "$TEMPLATE" | tr -d ' ')
[ "$SIZE" -le 51200 ] || die "$TEMPLATE is $SIZE bytes - over CloudFormation's 51,200-byte inline limit: trim Metadata text"
awsq cloudformation validate-template --template-body "file://$TEMPLATE" --query 'Description' --output text | cut -c1-120
echo "$SIZE bytes"

DEPLOY=(cloudformation deploy --stack-name "$STACK" --template-file "$TEMPLATE" --capabilities CAPABILITY_IAM
        --tags project=depth --no-fail-on-empty-changeset)
[ ${#PARAMS[@]} -gt 0 ] && DEPLOY+=(--parameter-overrides "${PARAMS[@]}")

if [ "$APPLY" != "1" ]; then
  say "change set (dry run - nothing is executed)"
  OUT=$("$AWS_CLI" "${DEPLOY[@]}" --no-execute-changeset 2>&1 | tr -d '\r') || die "change set failed: $OUT"
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

say "secrets (SSM SecureString under /depth/$STACK/ - read the PIN with ./infra/secrets.sh pin)"
ensure_secrets

say "cost allocation tag 'project' (the bill per project in Cost Explorer; free)"
awsq ce update-cost-allocation-tags-status --cost-allocation-tags-status TagKey=project,Status=Active >/dev/null 2>&1 \
  && echo "active" || echo "not yet - a tag can be activated up to 24 h after it first appears on a bill; re-run later"

# ROLLBACK=1: a failed update rolls back (safer for operations changes); the default keeps a failed instance
# alive for debugging its bootstrap, as before
ROLLBACK_FLAG=(--disable-rollback); [ "${ROLLBACK:-0}" = "1" ] && ROLLBACK_FLAG=()
say "deploying (the instance signals only once DEPTH serves; ~15-25 min)"
"$AWS_CLI" "${DEPLOY[@]}" "${ROLLBACK_FLAG[@]}" | tr -d '\r'
URL=$(stack_output "$STACK" Url)                  # empty while CloudFront is off
IID=$(stack_output "$STACK" InstanceId)
echo "instance $IID · dashboard: $(stack_output "$STACK" DashboardUrl)"
HTTPS_URL=$(stack_output "$STACK" HttpsUrl)       # PublicHttps=true: Caddy on the Elastic IP
if { [ -z "$URL" ] || [ "$URL" = None ]; } && [ -n "$HTTPS_URL" ] && [ "$HTTPS_URL" != None ]; then
  HOST="${HTTPS_URL#https://}"
  say "HTTPS on the Elastic IP without CloudFront: Caddy + Let's Encrypt for $HOST"
  # Ubuntu's own caddy package first, Caddy's signed apt repository as the fallback; Caddy gets the certificate
  # itself (TLS-ALPN on 443) and proxies to the app on loopback, flushing at once (MCP streams). It sends the origin
  # secret as CloudFront does: uvicorn trusts X-Forwarded-For from loopback, so the app sees every proxied request
  # as the visitor's, and the origin guard admits only the secret or an on-box caller (without it: 403 for all).
  # The value is read on the box, never in this command; the Caddyfile is root:caddy 0640.
  SSM_COMMENT="DEPTH HTTPS (Caddy)" ssm_run "$IID" "set -e
export DEBIAN_FRONTEND=noninteractive
if ! command -v caddy >/dev/null; then
  apt-get -o DPkg::Lock::Timeout=300 update -y >/dev/null
  apt-get -o DPkg::Lock::Timeout=300 install -y caddy >/dev/null 2>&1 || {
    curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list
    apt-get -o DPkg::Lock::Timeout=300 update -y >/dev/null; apt-get -o DPkg::Lock::Timeout=300 install -y caddy >/dev/null; }
fi
. /etc/depth/origin.env
[ -n \"\${DEPTH_ORIGIN_SECRET:-}\" ] || { echo 'no DEPTH_ORIGIN_SECRET in /etc/depth/origin.env' >&2; exit 1; }
install -m 640 -o root -g caddy /dev/null /etc/caddy/Caddyfile.new
printf '%s {\n\treverse_proxy 127.0.0.1:8000 {\n\t\theader_up X-Origin-Verify %s\n\t\tflush_interval -1\n\t}\n\theader Strict-Transport-Security \"max-age=31536000\"\n}\n' '$HOST' \"\$DEPTH_ORIGIN_SECRET\" > /etc/caddy/Caddyfile.new
mv -f /etc/caddy/Caddyfile.new /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null
systemctl enable caddy >/dev/null 2>&1; systemctl restart caddy
want='DEPTH_PUBLIC_URL=$HTTPS_URL'
( umask 022; printf '%s\n' \"\$want\" > /etc/depth/public.env ); systemctl restart depth
echo \"caddy \$(caddy version | cut -d' ' -f1) serving $HOST; DEPTH restarted\""
  URL="$HTTPS_URL"
fi

if [ -n "$URL" ] && [ "$URL" != None ]; then
  say "public address -> the server (/etc/depth/public.env: links in webhooks and briefs)"
  SSM_COMMENT="DEPTH public address" ssm_run "$IID" "want='DEPTH_PUBLIC_URL=$URL'
[ \"\$(cat /etc/depth/public.env 2>/dev/null)\" = \"\$want\" ] && echo unchanged && exit 0
( umask 022; printf '%s\n' \"\$want\" > /etc/depth/public.env ); systemctl restart depth; echo 'set; DEPTH restarted'"

  say "live check: $URL"
  LIVE=0
  for i in $(seq 1 60); do
    H=$(curl -fsS "$URL/api/health" 2>/dev/null || true)
    if printf '%s' "$H" | grep -q '"model_loaded":true'; then
      printf '%s' "$H" | grep -q '"is_cool_path":true' && echo "$URL/api/health: serving on COOL" \
        || echo "$URL/api/health: serving (NOT on COOL - rehearsal?)"
      LIVE=1; break
    fi
    sleep 10
  done
  [ "$LIVE" = 1 ] || echo "WARNING: $URL/api/health did not answer in 10 min - last reply: $(curl -sS "$URL/api/health" 2>&1 | cut -c1-200)"
  python -m src.bench.verify_cool "$URL" || echo "(the COOL receipt did not verify - expected only in a rehearsal)"
else
  say "on-box check over SSM (no CloudFront: the origin is closed to the internet)"
  SSM_COMMENT="DEPTH health" ssm_run "$IID" "curl -fsS http://127.0.0.1:8000/api/health" | cut -c1-400
  SSM_COMMENT="DEPTH COOL receipt" ssm_run "$IID" "curl -fsS http://127.0.0.1:8000/api/cool" > "$WORK/receipt.json"
  python -m src.bench.verify_cool "$WORK/receipt.json" \
    || echo "(the COOL receipt did not verify - expected only in a rehearsal)"
  echo "receipt saved: $WORK/receipt.json · add CloudFront once AWS verifies the account: ENABLE_CDN=1 APPLY=1 ./infra/deploy.sh"
fi
echo "shell (no SSH): aws ssm start-session --target $IID"
echo "SNS: confirm the two subscription emails (alerts, approvals) - AWS sends each once"
[ "$STATUS" = NONE ] || echo "next: APPLY=1 ./infra/update_server.sh - the running server re-reads its secrets and settings from SSM"
