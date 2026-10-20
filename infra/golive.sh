#!/usr/bin/env bash
# golive.sh — the remaining AWS work as two commands, so COOL can land at the last minute.
#
#   PHASE=now  APPLY=1 ./infra/golive.sh   # needs no COOL: benchmarks A (x86) + B (Graviton stock),
#                                          # load test on the running (rehearsal) server
#   PHASE=cool APPLY=1 ./infra/golive.sh   # the moment the COOL subscription is accepted:
#                                          # swap in the COOL AMI -> verify the receipt -> benchmarks C + D
#                                          # -> load test on COOL -> docs/cool_benchmark.md
#
# Without APPLY=1 it only prints the plan (each step's own dry run). Results land in benchmarks/ and docs/;
# commit them afterwards. Needs `aws login --profile hackathon`.
set -euo pipefail
cd "$(dirname "$0")/.."
. infra/lib.sh

STACK="${STACK:-depth}"
PHASE="${PHASE:-}"
[ "$PHASE" = now ] || [ "$PHASE" = cool ] || die "set PHASE=now or PHASE=cool"
"$AWS_CLI" sts get-caller-identity >/dev/null 2>&1 || die "AWS session expired: aws login --profile hackathon"

load_test() {   # load_test <label>: run on the box over SSM (the origin is closed to the internet without CloudFront)
  local iid label="$1"
  iid=$(stack_output "$STACK" InstanceId)
  say "load test on $iid ($label) - pass mark in src/bench/load_test.py"
  if [ "$APPLY" != "1" ]; then echo "(dry run) would run src.bench.load_test on the box"; return; fi
  SSM_COMMENT="DEPTH load test" ssm_run "$iid" "cd /opt/depth && set -a && . /etc/depth/admin.env && set +a && \
python3 -m src.bench.load_test http://127.0.0.1:8000 --label $label --admin-token \"\$DEPTH_ADMIN_TOKEN\" \
--out /tmp >/tmp/load_$label.log 2>&1; tail -5 /tmp/load_$label.log 1>&2; gzip -9c /tmp/load_$label.json | base64 -w0" \
    > "$WORK/load_$label.b64" || echo "load test FAILED - see the SSM output above"
  # SSM returns at most 24,000 characters of output: the JSON travels gzipped + base64
  mkdir -p benchmarks/load
  python -I -c 'import base64,gzip,json,sys; d=json.loads(gzip.decompress(base64.b64decode(open(sys.argv[1]).read())))
open(sys.argv[2],"w").write(json.dumps(d,indent=2)); print("passed:", d.get("passed"), d.get("checks"))' \
    "$WORK/load_$label.b64" "benchmarks/load/load_test__$label.json" || echo "(no JSON result - see $WORK/load_$label.b64)"
}

if [ "$PHASE" = now ]; then
  say "1/2 benchmarks without COOL: A (c8i.xlarge x86) + B (c8g.xlarge Graviton stock)"
  TARGETS="A B" ./infra/bench_fleet.sh
  load_test c8g_large_rehearsal
  if [ "$APPLY" = "1" ]; then python -m src.bench.compare; fi
  say "done - commit benchmarks/ + docs/cool_benchmark.md; later: PHASE=cool APPLY=1 ./infra/golive.sh"
  exit 0
fi

say "1/4 COOL subscription"
AMI_CHECK=$("$AWS_CLI" ec2 describe-images --image-ids "${COOL_AMI_ID:-ami-033e481a24f94c8cb}" \
  --query 'Images[0].ImageId' --output text 2>&1 | tr -d '\r' || true)
echo "COOL AMI visible: $AMI_CHECK (deploy.sh re-checks the subscription itself)"

say "2/4 rehearsal -> COOL (replaces only the instance; ~15-25 min)"
REHEARSAL=0 ./infra/deploy.sh
if [ "$APPLY" = "1" ]; then
  python -m src.bench.verify_cool runs/infra/receipt.json --md docs/cool_verification.md \
    --json runs/infra/verify.json || die "the COOL receipt did not verify - stop here and look"
fi

say "3/4 benchmarks on COOL: C (c8g.xlarge) + D (c8g.large)"
TARGETS="C D" ./infra/bench_fleet.sh

load_test c8g_large_cool
say "4/4 report"
if [ "$APPLY" = "1" ]; then python -m src.bench.compare; fi
echo "done - commit benchmarks/ docs/cool_benchmark.md docs/cool_verification.md; pin the product code in verify_cool.COOL_PRODUCT_CODES"
