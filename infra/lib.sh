# lib.sh — shared helpers for the infra scripts (sourced, not run). Works in Linux / AWS CloudShell and in
# Git Bash on Windows (where the AWS CLI is a Windows .exe that prints CRLF line ends).

# Git Bash rewrites arguments that start with "/" into Windows paths before calling a Windows program, so the
# SSM name /aws/service/... reached aws.exe as C:/Program Files/Git/aws/service/... - never rewrite anything
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'

# credentials: the named profile when it exists (laptop), else whatever the environment provides (CloudShell)
if [ -z "${AWS_PROFILE:-}" ] && grep -qs '^\[profile hackathon\]' "$HOME/.aws/config"; then
  export AWS_PROFILE=hackathon
fi
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="$AWS_REGION"

if [ -z "${AWS_CLI:-}" ]; then
  if command -v aws >/dev/null 2>&1; then
    AWS_CLI=aws
  elif [ -n "${LOCALAPPDATA:-}" ] && [ -x "$(cygpath -u "$LOCALAPPDATA" 2>/dev/null)/Programs/Amazon/AWSCLIV2/aws.exe" ]; then
    AWS_CLI="$(cygpath -u "$LOCALAPPDATA")/Programs/Amazon/AWSCLIV2/aws.exe"
  else
    echo "ERROR: AWS CLI v2 not found (set AWS_CLI=/path/to/aws)" >&2; exit 1
  fi
fi
APPLY="${APPLY:-0}"
WORK="runs/infra"                                   # git-ignored scratch (relative: the Windows CLI can read it)
mkdir -p "$WORK"

awsq() { "$AWS_CLI" "$@" | tr -d '\r'; }            # query helper: never a trailing CR in a variable
run()  {                                            # print always; execute only with APPLY=1
  printf '+ aws %s\n' "$*" >&2
  if [ "$APPLY" = "1" ]; then "$AWS_CLI" "$@" | tr -d '\r'; fi
}
die()  { echo "ERROR: $*" >&2; exit 1; }
say()  { printf '\n== %s\n' "$*"; }
rand_hex() { head -c "${1:-24}" /dev/urandom | od -An -tx1 | tr -d ' \n'; }

# us-east-1 on-demand $/h (Linux, shared tenancy) from the AWS Price List API
price_of() {
  awsq pricing get-products --region us-east-1 --service-code AmazonEC2 --output json --filters \
    "Type=TERM_MATCH,Field=instanceType,Value=$1" "Type=TERM_MATCH,Field=regionCode,Value=$AWS_REGION" \
    "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" "Type=TERM_MATCH,Field=tenancy,Value=Shared" \
    "Type=TERM_MATCH,Field=preInstalledSw,Value=NA" "Type=TERM_MATCH,Field=capacitystatus,Value=Used" \
    "Type=TERM_MATCH,Field=licenseModel,Value=No License required" \
  | python -I -c '
import json, sys
for item in json.load(sys.stdin).get("PriceList", []):
    for term in json.loads(item)["terms"]["OnDemand"].values():
        for dim in term["priceDimensions"].values():
            usd = float(dim["pricePerUnit"]["USD"])
            if usd > 0:
                print(usd); sys.exit(0)
sys.exit("no on-demand price found")'
}

stack_output() {   # stack_output <stack> <OutputKey>
  awsq cloudformation describe-stacks --stack-name "$1" --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text
}
stack_param() {    # stack_param <stack> <ParameterKey>
  awsq cloudformation describe-stacks --stack-name "$1" --query "Stacks[0].Parameters[?ParameterKey=='$2'].ParameterValue" --output text
}
stack_resource() { # stack_resource <stack> <LogicalId> - physical id; works when a failed stack has no outputs
  awsq cloudformation describe-stack-resource --stack-name "$1" --logical-resource-id "$2" \
    --query StackResourceDetail.PhysicalResourceId --output text 2>/dev/null || true
}

# ssm_run <instance-id> <shell command>: run it as root over SSM Run Command (no SSH), wait for the end, print
# its stdout; non-zero (stderr tail on stderr) unless it succeeded. SSM_COMMENT labels it in the console.
ssm_run() {
  local iid="$1" cmd="$2" cid st=Pending i
  python -I -c 'import json, sys; json.dump({"commands": [sys.argv[1]]}, open(sys.argv[2], "w"))' "$cmd" "$WORK/ssm-cmd.json"
  cid=$(awsq ssm send-command --instance-ids "$iid" --document-name AWS-RunShellScript --comment "${SSM_COMMENT:-DEPTH}" \
          --parameters "file://$WORK/ssm-cmd.json" --query Command.CommandId --output text) || return 1
  for i in $(seq 1 90); do
    st=$(awsq ssm get-command-invocation --command-id "$cid" --instance-id "$iid" --query Status --output text 2>/dev/null || echo Pending)
    case "$st" in Pending|InProgress|Delayed) sleep 10 ;; *) break ;; esac
  done
  awsq ssm get-command-invocation --command-id "$cid" --instance-id "$iid" --query StandardOutputContent --output text
  [ "$st" = Success ] && return 0
  echo "SSM command $st: $(awsq ssm get-command-invocation --command-id "$cid" --instance-id "$iid" \
        --query StandardErrorContent --output text | tail -5)" >&2
  return 1
}
