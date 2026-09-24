# infra/ — deploy DEPTH on Graviton + COOL, and prove it with the 3-way benchmark

**One product, one deploy target.** The live demo is the FastAPI app (`src/dashboard/app.py`) running
the full agent on an **EC2 Graviton instance launched from the COOL AMI** (OpenCV 5 under
`/opt/cool`), behind CloudFront for HTTPS. The earlier Lambda / API Gateway / DynamoDB / Amplify /
SageMaker plan is **retired** (review §3.7/§8): Lambda could not use COOL (COOL is an AMI) and ran a
different, detection-only product with its own thresholds. Status: **scripts written and dry-run
checked; not yet executed** (AWS work is scheduled for a dedicated day).

## Architecture

```
Judge ──HTTPS──► CloudFront ──:8000 (CloudFront origin-facing IPs only)──► EC2 c8g.xlarge · COOL AMI
                                                                        systemd `depth` (COOL python)
                                                                        FastAPI + job queue
   STAGE 1 canonicalise (OpenCV/COOL) → SEE YOLO11 cv2.dnn → PROVE evidence → DECIDE guaranteed
   tiers → ACT geotag · routes · exports
S3  (models · uploads 7-day lifecycle · results/<survey> · benchmarks)   ◄── job results mirrored
CloudWatch (status-check alarm → auto-recover + email · JSON frame logs)  Budgets (50% / 90% alerts)
SSM Session Manager (no SSH port)                                        IAM role scoped to the bucket
```

Scaling story (described, not built): survey jobs → SQS → Auto Scaling group of the same COOL AMI
workers (Spot) → S3. Same code, more workers.

## Files

| File | Role |
|---|---|
| `deploy_aws.sh` | One command, **dry-run by default** (`APPLY=1` executes): budget alarm → S3 → IAM → security group → EC2 (COOL AMI, IMDSv2, user-data) → CloudFront → alarms. Run it from **AWS CloudShell** or Linux/WSL. |
| `setup_cool_instance.sh` | On the instance (root): repo → model from S3 (sha256-checked) → web deps into `/opt/depth/pydeps` with `pip --target` (**the COOL venv is never modified**) → systemd unit → waits until `/api/health` shows `model_loaded` and `is_cool_path`. Idempotent. |
| `depth.service` | systemd unit template: COOL interpreter, `DEPTH_THREADS` = vCPUs, restart on failure, JSON frame logs. |
| `bench_cool.sh` | The 3-way COOL benchmark of the **product** workload (`src/bench/product_bench.py`). |

The pinned web stack in `setup_cool_instance.sh` was verified locally: a clean venv holding only
`numpy` + `opencv-python-headless` (standing in for the COOL venv) plus the `--target` deps imports
and serves the app (`/api/health` → model warm in 0.3 s; `/api/analyze` works).

## The 3-way COOL benchmark (what the award measures)

Same code, same frames (content sha256), same model (sha256), three machines:

```bash
# (A) x86 + stock OpenCV 5 — c7i.xlarge (Ubuntu 24.04)
FLAVOR=stock LABEL=baseline_x86   PRICE_HOUR=0.1785 ./infra/bench_cool.sh
# (B) Graviton + stock OpenCV 5 — c8g.xlarge
FLAVOR=stock LABEL=graviton_stock PRICE_HOUR=0.1595 ./infra/bench_cool.sh
# (C) Graviton + COOL — the SAME c8g.xlarge on the COOL AMI (+ $0.01/h COOL fee)
FLAVOR=cool  LABEL=graviton_cool  PRICE_HOUR=0.1695 ./infra/bench_cool.sh
# then, with all runs/bench/*.json in one place:
python -m src.bench.compare        # → docs/cool_benchmark.md + docs/img/cool_benchmark.png
```
_(Prices: us-east-1 on-demand Linux as remembered on 2026-09-24 — confirm on the pricing page on the day.)_

What each run records and why (review §3.7 fixes):

* **the product workload**, per stage — decode → Stage 1 → `cv2.dnn` → evidence/tiers → render —
  plus a `cv` workload (all OpenCV image ops, no network) that isolates where COOL/KleidiCV acts;
* **threads pinned** (`THREADS="0 1"`: all vCPUs, then one thread — an Intel vCPU is a hyper-thread,
  a Graviton vCPU is a physical core);
* **provenance** — a run is labelled COOL only if `cv2.__file__` is under `/opt/cool`; the script
  refuses to mislabel. The stock runs use a private venv (Ubuntu 24.04 / PEP 668), the COOL run
  uses COOL's interpreter as-is;
* **cost** — $/1,000 frames and, per **survey-hour** of sonar, compute seconds, real-time factor and
  $ (frames per survey-hour = ping rate × 3600 / pings per frame × channels; assumptions recorded).

Local reference (this laptop, x86, stock OpenCV 5.0.0): see `docs/cool_benchmark.md`.

## Deploy checklist (the AWS day)

1. `aws login` (profile `hackathon`, us-east-1); subscribe to the COOL Graviton listing; note the AMI id.
2. `aws s3 cp runs/EXP-001/weights/best.onnx s3://<bucket>/models/EXP-001/` (+ record its sha256).
3. `AWS_PROFILE=hackathon COOL_AMI_ID=… ALERT_EMAIL=… ./infra/deploy_aws.sh` → read the plan →
   `APPLY=1 …` → wait for CloudFront → `curl https://<cf>/api/health` shows `is_cool_path: true`.
4. SSM into the instance → `bench_cool.sh` (C). Launch a c7i.xlarge for (A) and a stock Ubuntu c8g
   for (B), run, **terminate them**. `python -m src.bench.compare`.
5. Leave the demo instance running for judging (27 Oct – 9 Nov): ≈ $0.17/h ⇒ ≈ $60 for the window;
   the budget alarm is set before anything else is created.
