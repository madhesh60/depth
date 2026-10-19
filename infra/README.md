# infra/ — DEPTH on AWS Graviton4 + COOL: deploy it, prove it, measure it, remove it

**One product, one deploy target.** The live demo is the FastAPI app (`src/dashboard/app.py`) running the
full agent on **one EC2 Graviton4 instance launched from the COOL Marketplace AMI** (OpenCV 5 under
`/opt/cool`), behind CloudFront for HTTPS. Everything is one CloudFormation stack
([`cfn/depth.yaml`](cfn/depth.yaml)); the benchmark is one command that launches self-terminating machines.
The earlier Lambda / API Gateway / DynamoDB / Amplify / SageMaker plan stays **retired**: COOL ships as an
AMI, so only EC2 can run the product on COOL.

_Status: written, linted (cfn-lint 1.57.2, 0 findings) and dry-run checked; the AWS runs are pending._

## Architecture

```
judge ─HTTPS─► CloudFront (all edges, free tier) ── http :8000, X-Origin-Verify ──► Elastic IP
                │  /vendor/* /img/* /samples/* cached at the edge                 │
                │  /api/* and /mcp uncached; security headers added              ▼
                │                                           EC2 c8g.large · COOL AMI (Graviton4)
                │                                           SG: 8000 from CloudFront's prefix list only,
                │                                               no SSH; IMDSv2; encrypted gp3
                │                                           systemd `depth` → COOL's python →
                │                                           STAGE 1 → SEE (cv2.dnn) → PROVE → DECIDE → ACT
                ▼                                           /api/cool = AWS-signed COOL receipt
 CloudWatch: per-stage latency (log metric filters), CPU/memory, dashboard, 4 alarms → SNS email
 S3 (versioned, TLS-only): results/ (30 d) · uploads/ (7 d) · bench/            SSM: shell + updates
 EventBridge Scheduler: stops the instance 2026-11-10 · account Budget $40/month (outside the stack)

 bench_fleet.sh ──► 4 self-terminating machines (c8i.xlarge x86 · c8g.xlarge stock · c8g.xlarge COOL ·
                    c8g.large COOL) → results + receipts in S3 → benchmarks/ → src.bench.compare
```

Scaling story (described, not built): survey jobs → SQS → an Auto Scaling group of the same COOL AMI
(Spot) → S3. Same code, more workers; the job store already persists results.

## Files

| File | Role |
|---|---|
| [`cfn/depth.yaml`](cfn/depth.yaml) | The whole demo as one stack (design reasons recorded per resource in `Metadata`). |
| [`deploy.sh`](deploy.sh) | Resolves the COOL AMI (checks arm64 + Marketplace product code + the subscription), an AZ that offers every type, the CloudFront prefix list; validates; prints the change set. `APPLY=1` deploys and verifies the live receipt. `REHEARSAL=1` / `ENABLE_CDN=0`: see the checklist. |
| [`setup_cool_instance.sh`](setup_cool_instance.sh) | On the instance: exact commit, SHA-256-checked model, web deps into `/opt/depth/pydeps` (**the COOL venv is never modified**), CloudWatch agent, systemd. Fails unless `/api/health` shows the model warm **and** `cv2` from `/opt/cool` — so the CloudFormation signal means "serving on COOL". |
| [`depth.service`](depth.service) | systemd unit: COOL interpreter, threads = vCPUs, restart on failure, log file for the agent. |
| [`update_server.sh`](update_server.sh) | New commit on the live server through SSM Run Command (no SSH, no replacement). |
| [`bench_fleet.sh`](bench_fleet.sh) + [`bench_userdata.sh`](bench_userdata.sh) | The whole COOL benchmark as one command (see below). |
| [`bench_cool.sh`](bench_cool.sh) | The benchmark on one machine (the fleet runs it; it also runs by hand). |
| [`teardown.sh`](teardown.sh) | Removes everything (stack, retained bucket with every version, parameter) and lists leftovers. |
| [`lib.sh`](lib.sh) | Shared helpers: AWS CLI detection (Linux, CloudShell, Git Bash on Windows), Price List API lookup, `ssm_run` (a root command on the server over SSM). |

## Verified COOL — anyone can check it

`GET https://<demo>/api/cool` returns the **COOL execution receipt** (`src/bench/cool_attest.py`):

1. the **AWS-signed instance identity document** exactly as IMDSv2 serves it: AMI id, **Marketplace product
   code** (the COOL listing), `architecture: arm64`, instance type, region. AWS signs it; the server cannot
   forge it;
2. every OpenCV shared object **mapped into the server process** (`/proc/self/maps`) and whether all of them
   live under `/opt/cool` (no second OpenCV beside it);
3. the COOL build **read from the machine** (version files, packages, `cv2.getBuildInformation()` lines).

```bash
python -m src.bench.verify_cool https://<demo>          # PASS/FAIL per check, exit 0 only if verified
python -m src.bench.verify_cool https://<demo> --analyze # + a live analysis names the same signed document
```

The verifier checks the RSA-SHA256 signature against AWS's published us-east-1 certificate (bundled; the
tests check its self-signature) and the PKCS7 RSA-2048 signature with `openssl -nointern` (AWS's documented
command omits `-nointern`, which would accept a forged signature that carries its own certificate). Every
`/api/analyze` result and every exported report carries the signed document's SHA-256 in its provenance;
each benchmark JSON comes with its machine's receipt (`benchmarks/<label>__receipt.json`).

_Trust model:_ AWS's signature proves which machine it is. The library map is reported by the open-source
server code; re-run the fleet in your own account to see the same on a fresh machine.

## The benchmark — one command, equal sizes, self-terminating

```bash
APPLY=1 ./infra/bench_fleet.sh        # ~40 min; downloads benchmarks/*.json; machines terminate themselves
python -m src.bench.compare           # → docs/cool_benchmark.md + docs/img/cool_benchmark.png
```

| target | machine | software | role |
|---|---|---|---|
| A `baseline_x86` | c8i.xlarge (Intel, same generation) | Ubuntu 24.04 + stock OpenCV 5.0.0.93 (pip) | x86 baseline |
| B `graviton_stock` | c8g.xlarge | Ubuntu 24.04 + stock OpenCV 5.0.0.93 (pip) | Graviton without COOL |
| C `graviton_cool` | c8g.xlarge | COOL AMI (OpenCV 5 from `/opt/cool`) | **B → C = COOL's own effect** |
| D `graviton_cool_large` | c8g.large | COOL AMI | the demo's size |

Every machine runs the same commit, frames, model and recordings (all SHA-256 pinned) and records:

* `product` — exactly what `/api/analyze` runs, per stage **and** per op (decode · stage1 · dnn_pre ·
  dnn_forward · dnn_post · evidence · render) so it shows where COOL acts;
* `cv` — the product path's OpenCV image ops alone; `survey` — a **real 1-hour Humminbird recording** end to
  end: measured real-time factor and $ per survey-hour (not modelled);
* 3 interleaved repeats, CPU utilisation, peak memory, all vCPUs and 1 thread;
* parity — every detection, so `compare` checks COOL changes speed, not answers (class, IoU ≥ 0.9,
  |Δscore| ≤ 0.01, same tier);
* price from the **AWS Price List API** at launch (+ COOL's $0.01/h fee), and the machine's signed receipt.

Load test (pass mark fixed in the code before any run — 0 errors, analyze p95 ≤ 3 s with 4 clients back
to back + 1 survey, health p95 ≤ 1 s): `python -m src.bench.load_test https://<demo> --label c8g_large`.

## Security and responsible operation

| concern | how |
|---|---|
| network | origin port 8000 open only to CloudFront's origin-facing prefix list; no port 22; egress 443/80 only |
| origin bypass | CloudFront adds `X-Origin-Verify`; the app refuses requests without it (`origin_guard.py`); the value lives in an SSM parameter, never in the template |
| access | SSM Session Manager only (no key pair); IMDSv2 required, hop limit 1 |
| identity | instance role: this bucket, this log group, this stack's signal, one parameter; the fleet role may only write `bench/` |
| data | S3 encrypted, TLS-only, public access blocked, versioned; uploads expire in 7 days, results in 30; EBS encrypted |
| abuse | per-client rate limits keyed on the **last** `X-Forwarded-For` hop (the one CloudFront appends); survey queue cap; security headers at the edge |
| failures | system check → automatic recovery; instance check → reboot; CPU > 85 % and edge 5xx > 5 % → email |
| cost | account Budget ($40/month, 25/50/100 % + forecast alerts); scheduled stop 2026-11-10; `teardown.sh` removes everything; every resource tagged `project=depth` |
| deliberate omissions | WAF ($5+/month — the app rate-limits), customer KMS keys ($1/month each — AWS-managed encryption at rest instead), CloudFront access logs (metrics + origin JSON logs instead); each recorded in the template |

## Cost (us-east-1, on-demand, as planned — the measured bill goes in the report)

| item | rate | to 9 Nov (~21 days) |
|---|---|---|
| c8g.large | ≈ $0.08/h | ≈ $40 |
| public IPv4 (Elastic IP) | $0.005/h | ≈ $2.5 |
| 20 GB gp3 | $0.08/GB-month | ≈ $1.1 |
| CloudFront, S3, CloudWatch (6 custom metrics, 4 alarms, 1 dashboard), SNS, Scheduler, SSM | free tier | ≈ $0 |
| COOL software fee (Marketplace, card) | $0.01/h after the 7-day trial | ≈ $3.5 |
| benchmark fleet (4 machines, ~40 min) | — | < $1 |

## Deploy checklist

1. `aws login --profile hackathon`; subscribe to the COOL Graviton4 listing in AWS Marketplace.
2. `ALERT_EMAIL=… ./infra/deploy.sh` → read the change set → `APPLY=1 ALERT_EMAIL=… ./infra/deploy.sh`.
3. Confirm the SNS email. `python -m src.bench.verify_cool https://<demo>` → VERIFIED.
4. `APPLY=1 ./infra/bench_fleet.sh` → `python -m src.bench.compare`; the load test; commit `benchmarks/`.
5. After judging: the instance stops itself on 2026-11-10; then `APPLY=1 ./infra/teardown.sh` and cancel the
   COOL subscription.

Two waits a new account can hit, and how the stack keeps moving through them:

* **COOL subscription pending** → `REHEARSAL=1` runs the whole stack on stock Ubuntu 24.04 arm64 + stock
  OpenCV 5 (pip). `/api/health` and the receipt say *rehearsal* and the verifier fails it, so a rehearsal can
  never pass for COOL. `REHEARSAL=0 APPLY=1 ./infra/deploy.sh` then swaps in the COOL AMI. Only the instance
  is replaced; the Elastic IP, CloudFront, bucket and alarms stay.
* **CloudFront not yet allowed** → AWS answers *"Your account must be verified before you can add new
  CloudFront resources"* until AWS Support verifies the account (a support case, Account and billing).
  Until then, `ENABLE_CDN=0` deploys everything else. The origin stays closed to the internet, and `deploy.sh`
  checks the server on the box over SSM. It saves the receipt to `runs/infra/receipt.json` and verifies it
  from the file. `ENABLE_CDN=1 APPLY=1 ./infra/deploy.sh` adds CloudFront later to the same server, and writes
  the public address to it.
