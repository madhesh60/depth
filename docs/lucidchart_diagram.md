# Architecture diagram — Lucidchart source

The as-built deployment ([architecture.md](architecture.md), [infra/README.md](../infra/README.md)), written
so you can paste it into Lucidchart: **Insert → Diagram as code → Mermaid**, paste the block, then style it.
Export as SVG to `docs/img/architecture.svg` when done.

Labels match the code. Two items carry a status because they are not live yet: CloudFront (waiting for AWS
account verification) and the COOL AMI (waiting for the Marketplace subscription). Until then the same
server runs stock OpenCV 5 in *rehearsal* mode, and the receipt says so.

```mermaid
flowchart LR
  subgraph IN[Inputs]
    SON[Sonogram images]
    RAW[Raw Humminbird recording<br/>.DAT/.SON/.IDX · per-ping GPS]
  end

  subgraph AWS[AWS us-east-1 · one CloudFormation stack]
    CF[CloudFront HTTPS<br/>X-Origin-Verify header<br/><i>pending account verification</i>]
    subgraph EC2[EC2 c8g.large Graviton4 · COOL AMI · Elastic IP · systemd]
      direction TB
      S0[Sensor cross-check<br/>seabed track vs depth sounder]
      S1[Stage 1 canonicalise<br/>bottom track · slant→ground]
      S2[See: YOLO11 ONNX<br/>cv2.dnn · OpenCV 5 /opt/cool]
      S3[Prove: shadow · height · water column]
      S4[Decide: calibrated tiers<br/>≥78% of pots reach a person · ≥84% looking harder]
      S5[Act: geotag · stitch · routes<br/>re-survey passes · mission brief]
      S0 --> S1 --> S2 --> S3 --> S4 --> S5
    end
    S3B[(S3: jobs · exports · benchmarks)]
    CW[CloudWatch logs · metrics<br/>dashboard · 4 alarms]
    SNS[SNS alert email]
    SSM[SSM: secrets · shell · no SSH]
    SCH[EventBridge Scheduler<br/>auto-stop 2026-11-10]
    BED[Bedrock Claude<br/>optional brief writer]
  end

  subgraph HUM[Human loop]
    STU[Studio: Analyze · Survey · Study · Audit]
    PER[Named person ✓ ✕ recovered<br/>→ agent re-plans]
  end

  subgraph OUT[Connect]
    MCP[MCP server /mcp<br/>agents can only ask]
    OGC[OGC API – Features<br/>QGIS / ArcGIS]
    WH[HMAC-signed webhooks]
    EMB[&lt;depth-hazards&gt; web component]
    EXP[GeoJSON · GPX · KML · CSV]
  end

  SON --> CF
  RAW --> CF
  CF --> EC2
  EC2 --> S3B
  EC2 --> CW --> SNS
  SSM --> EC2
  SCH --> EC2
  S5 -.-> BED
  S5 --> STU --> PER --> S5
  S5 --> MCP & OGC & WH & EMB & EXP
```

## Benchmark fleet (second, smaller diagram)

```mermaid
flowchart LR
  BF[infra/bench_fleet.sh<br/>same commit · frames · model · SHA-256 pinned]
  BF --> A[A c8i.xlarge x86<br/>stock OpenCV 5]
  BF --> B[B c8g.xlarge Graviton<br/>stock OpenCV 5]
  BF --> C[C c8g.xlarge Graviton<br/>COOL]
  BF --> D[D c8g.large Graviton<br/>COOL · demo size]
  A & B & C & D -->|results + AWS-signed receipt, then self-terminate| S3[(S3)]
  S3 --> CMP[src.bench.compare → docs/reports/cool_benchmark.md]
```
