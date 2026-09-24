#!/usr/bin/env bash
# deploy_aws.sh — stand up the live DEPTH demo on AWS (review I-1 / C-1). DRY RUN BY DEFAULT:
# it prints every AWS CLI call; set APPLY=1 to execute. Nothing here has been run yet.
#
# Architecture (one product, one deploy target — see infra/README.md):
#   Budgets alarm → S3 bucket (models · uploads 7-day lifecycle · results · benchmarks)
#   → IAM role (SSM + CloudWatch agent + this bucket only) → security group (port 8000 from the
#   CloudFront origin-facing prefix list ONLY; no SSH — access is SSM Session Manager)
#   → EC2 c8g.xlarge from the COOL AMI (IMDSv2 required; user-data = setup_cool_instance.sh)
#   → CloudFront (HTTPS, caching disabled, origin port 8000) → StatusCheck alarm (auto-recover + email).
#
# Prereqs (manual, once):
#   1. Subscribe to "COOL - Cloud Optimized OpenCV Library" in AWS Marketplace (Graviton listing)
#      and copy the AMI id for us-east-1 → COOL_AMI_ID.
#   2. aws login / credentials for profile $AWS_PROFILE.
#   3. Upload the model once: aws s3 cp runs/EXP-001/weights/best.onnx s3://$BUCKET/models/EXP-001/
#
#   AWS_PROFILE=hackathon COOL_AMI_ID=ami-xxxx ALERT_EMAIL=you@example.com ./infra/deploy_aws.sh        # dry run
#   APPLY=1 AWS_PROFILE=hackathon COOL_AMI_ID=ami-xxxx ALERT_EMAIL=you@example.com ./infra/deploy_aws.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${COOL_AMI_ID:?set COOL_AMI_ID (from the COOL Marketplace listing, us-east-1)}"
: "${ALERT_EMAIL:?set ALERT_EMAIL for the budget + health alarms}"
export AWS_PROFILE="${AWS_PROFILE:-hackathon}"
export AWS_REGION="${AWS_REGION:-us-east-1}" AWS_DEFAULT_REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c8g.xlarge}"
BUDGET_USD="${BUDGET_USD:-40}"
NAME="${NAME:-depth}"
APPLY="${APPLY:-0}"
AWS="${AWS_CLI:-aws}"

run() {                       # print always; execute only with APPLY=1
  printf '+ %s\n' "$*" >&2
  if [ "$APPLY" = "1" ]; then "$@"; fi
}
out() {                       # like run, but captures stdout (placeholder in dry run)
  printf '+ %s\n' "$*" >&2
  if [ "$APPLY" = "1" ]; then "$@"; else echo "<dry-run:$1-$2>"; fi
}

ACCOUNT=$(out $AWS sts get-caller-identity --query Account --output text)
BUCKET="${BUCKET:-$NAME-$ACCOUNT-$AWS_REGION}"
echo "== account=$ACCOUNT region=$AWS_REGION bucket=$BUCKET instance=$INSTANCE_TYPE apply=$APPLY"

echo "== [1/8] budget alarm (before anything costs money)"
cat > /tmp/depth-budget.json <<EOF
{"BudgetName":"$NAME-monthly","BudgetLimit":{"Amount":"$BUDGET_USD","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}
EOF
cat > /tmp/depth-notify.json <<EOF
[{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":50,"ThresholdType":"PERCENTAGE"},
  "Subscribers":[{"SubscriptionType":"EMAIL","Address":"$ALERT_EMAIL"}]},
 {"Notification":{"NotificationType":"FORECASTED","ComparisonOperator":"GREATER_THAN","Threshold":90,"ThresholdType":"PERCENTAGE"},
  "Subscribers":[{"SubscriptionType":"EMAIL","Address":"$ALERT_EMAIL"}]}]
EOF
run $AWS budgets create-budget --account-id "$ACCOUNT" --budget file:///tmp/depth-budget.json \
    --notifications-with-subscribers file:///tmp/depth-notify.json || true

echo "== [2/8] S3 bucket (private, uploads expire after 7 days)"
run $AWS s3api create-bucket --bucket "$BUCKET" || true
run $AWS s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
cat > /tmp/depth-lifecycle.json <<'EOF'
{"Rules":[{"ID":"uploads-7d","Filter":{"Prefix":"uploads/"},"Status":"Enabled","Expiration":{"Days":7}}]}
EOF
run $AWS s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration file:///tmp/depth-lifecycle.json

echo "== [3/8] IAM role + instance profile (SSM, CloudWatch agent, this bucket only)"
cat > /tmp/depth-trust.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF
cat > /tmp/depth-s3.json <<EOF
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject"],"Resource":"arn:aws:s3:::$BUCKET/*"},
 {"Effect":"Allow","Action":["s3:ListBucket"],"Resource":"arn:aws:s3:::$BUCKET"}]}
EOF
run $AWS iam create-role --role-name "$NAME-ec2" --assume-role-policy-document file:///tmp/depth-trust.json || true
run $AWS iam attach-role-policy --role-name "$NAME-ec2" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
run $AWS iam attach-role-policy --role-name "$NAME-ec2" --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
run $AWS iam put-role-policy --role-name "$NAME-ec2" --policy-name "$NAME-bucket" --policy-document file:///tmp/depth-s3.json
run $AWS iam create-instance-profile --instance-profile-name "$NAME-ec2" || true
run $AWS iam add-role-to-instance-profile --instance-profile-name "$NAME-ec2" --role-name "$NAME-ec2" || true

echo "== [4/8] security group: 8000 from CloudFront origin-facing IPs only (no SSH)"
VPC=$(out $AWS ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(out $AWS ec2 create-security-group --group-name "$NAME-sg" --description "DEPTH: 8000 from CloudFront only" \
     --vpc-id "$VPC" --query GroupId --output text)
PL=$(out $AWS ec2 describe-managed-prefix-lists \
     --filters Name=prefix-list-name,Values=com.amazonaws.global.cloudfront.origin-facing \
     --query 'PrefixLists[0].PrefixListId' --output text)
run $AWS ec2 authorize-security-group-ingress --group-id "$SG" \
    --ip-permissions "IpProtocol=tcp,FromPort=8000,ToPort=8000,PrefixListIds=[{PrefixListId=$PL}]"

echo "== [5/8] EC2 $INSTANCE_TYPE from the COOL AMI (IMDSv2, user-data = setup script)"
cat > /tmp/depth-userdata.sh <<EOF
#!/bin/bash
set -e
export DEPTH_MODEL_S3=s3://$BUCKET/models/EXP-001/best.onnx DEPTH_S3_BUCKET=$BUCKET
${DEPTH_MODEL_SHA256:+export DEPTH_MODEL_SHA256=$DEPTH_MODEL_SHA256}
command -v git >/dev/null || (apt-get update -y && apt-get install -y git)
git clone -q https://github.com/madhesh60/depth.git /opt/depth || git -C /opt/depth pull -q
bash /opt/depth/infra/setup_cool_instance.sh > /var/log/depth-setup.log 2>&1
EOF
IID=$(out $AWS ec2 run-instances --image-id "$COOL_AMI_ID" --instance-type "$INSTANCE_TYPE" \
      --iam-instance-profile Name="$NAME-ec2" --security-group-ids "$SG" \
      --metadata-options HttpTokens=required,HttpEndpoint=enabled \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=20,VolumeType=gp3}' \
      --user-data file:///tmp/depth-userdata.sh \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=project,Value=depth}]" \
      --query 'Instances[0].InstanceId' --output text)
run $AWS ec2 wait instance-running --instance-ids "$IID"
DNS=$(out $AWS ec2 describe-instances --instance-ids "$IID" --query 'Reservations[0].Instances[0].PublicDnsName' --output text)

echo "== [6/8] CloudFront (HTTPS → EC2:8000, caching disabled, all viewer headers except Host)"
cat > /tmp/depth-cf.json <<EOF
{"CallerReference":"$NAME-$(date +%s)","Comment":"DEPTH live demo","Enabled":true,
 "Origins":{"Quantity":1,"Items":[{"Id":"ec2","DomainName":"$DNS",
   "CustomOriginConfig":{"HTTPPort":8000,"HTTPSPort":443,"OriginProtocolPolicy":"http-only","OriginReadTimeout":60}}]},
 "DefaultCacheBehavior":{"TargetOriginId":"ec2","ViewerProtocolPolicy":"redirect-to-https",
   "AllowedMethods":{"Quantity":7,"Items":["GET","HEAD","OPTIONS","PUT","POST","PATCH","DELETE"],
     "CachedMethods":{"Quantity":2,"Items":["GET","HEAD"]}},
   "CachePolicyId":"4135ea2d-6df8-44a3-9df3-4b5a84be39ad",
   "OriginRequestPolicyId":"b689b0a8-53d0-40ab-baf2-68738e2966ac","Compress":true},
 "PriceClass":"PriceClass_100"}
EOF
CF=$(out $AWS cloudfront create-distribution --distribution-config file:///tmp/depth-cf.json \
     --query 'Distribution.DomainName' --output text)

echo "== [7/8] alarms: status check → auto-recover + email"
TOPIC=$(out $AWS sns create-topic --name "$NAME-alerts" --query TopicArn --output text)
run $AWS sns subscribe --topic-arn "$TOPIC" --protocol email --notification-endpoint "$ALERT_EMAIL"
run $AWS cloudwatch put-metric-alarm --alarm-name "$NAME-status-check" --namespace AWS/EC2 \
    --metric-name StatusCheckFailed_System --dimensions Name=InstanceId,Value="$IID" \
    --statistic Maximum --period 60 --evaluation-periods 2 --threshold 1 --comparison-operator GreaterThanOrEqualToThreshold \
    --alarm-actions "arn:aws:automate:$AWS_REGION:ec2:recover" "$TOPIC"

echo "== [8/8] done"
echo "Live URL (after CloudFront deploys, ~5-10 min): https://$CF"
echo "Health: curl https://$CF/api/health   (expect model_loaded:true, is_cool_path:true)"
echo "Shell:  aws ssm start-session --target $IID"
echo "Bench:  (in the SSM shell) cd /opt/depth && sudo -u depth FLAVOR=cool LABEL=graviton_cool PRICE_HOUR=0.1695 ./infra/bench_cool.sh"
