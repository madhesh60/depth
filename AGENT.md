> **📌 This is the original grant proposal / competition brief — a historical record, not the
> current design.** The as-built system is in [`architecture.md`](architecture.md): the taxonomy is
> 4 classes for EXP-001 (`rope_line` merged into `fishing_gear`) and 2 sonar-only classes for
> EXP-002; the classical-CV ROI gate was **retired** (STUDY-01) and Stage 1 is now **sonar
> canonicalisation** in the product path (bottom tracking, STUDY-08); the AWS plan is **one EC2
> Graviton instance on the COOL AMI behind CloudFront** — Lambda / API Gateway / DynamoDB / Amplify /
> SageMaker below are superseded. Current state: [`README.md`](README.md),
> [`experiments.md`](experiments.md), [`progress.md`](progress.md).

**PS**

Background The accumulation of anthropogenic (man-made) debris in marine ecosystems poses a critical threat to global biodiversity. Among the most destructive types of pollution are ‘ghost nets’—abandoned, lost, or discarded fishing gear. These nets continuously trap and kill marine life,destroy coral reefs, and damage commercial vessel propellers.  
  
Because the ocean is vast and dark, marine conservationists and underwater technologists rely on Side Scan Sonar (SSS) instruments. These sensors are towed behind ships or mounted on Autonomous Underwater Vehicles (AUVs) to create detailed acoustic maps of the seafloor.However, manual inspection of thousands of kilometers of sonar logs is incredibly slow, tedious, and prone to human error. Debris can easily blend into natural geological features like rock formations, sand ripples, and marine ridges. Automating this process via computer vision is essential for efficient ocean cleanup operations.  
  
• Description Participants must develop an end-to-end automated computer vision pipeline capable of ingesting side-scan sonar imagery, identifying man-made debris against a complex natural background, and generating actionable localized data.The software system must be robust enough to handle the core challenges inherent to acoustic imagery: high speckle noise, varying pixel resolutions, acoustic shadows, and data dropouts caused by underwater vehicle motion (heave, pitch, and roll). The primary objective is to build an algorithm that reliably separates natural seafloor topology from artificial anomalies. The final solution should be optimized to run efficiently, potentially allowing deployment on edge devices or onboard a marine drone without requiring heavy cloud computing dependencies.  
• Expected Solution Teams are expected to deliver a functional, modular software prototype containing the following core components:  
• Object Detection / Semantic Segmentation Model: An AI/ML architecture (such as YOLO,Faster R-CNN, or U-Net) trained to detect and draw bounding boxes or pixel-level masks around man-made objects (including shipwrecks, pipes, cylinders, and entangled debris nets).  
• Confidence Scoring & Noise Filtering Module: An algorithmic pipeline or pre-processing filter that minimizes false positives caused by natural acoustic shadows or rock clusters,outputting a clear confidence score (0% to 100%) for every detected anomaly.  
• Anomalous Reporting & Geotagging Engine: A data-parsing script or lightweight dashboard interface that reads sonar metadata (such as coordinate files or ping headers) to output a structured report (JSON or CSV format). This report must detail the exact location (latitude/longitude), bounding dimensions, and classification of each detected hazard.  
• User Interface (UI) Dashboard: A visual interface where a user can upload a raw sonar image log, view the AI models' detections overlaid on the map in real-time, and download the generated anomaly reports.


| Field                | Details                                                           |
| -------------------- | ----------------------------------------------------------------- |
| **Team Name**        | Syndicate                                                         |
| **Submission Date**  | August 26, 2026                                                   |
| **Grant Requested**  | AWS Cloud Compute Grant ($150) - WON - COMPLETED  in this stage   |
| **Competition Path** | Primary: Best Use of COOL Award · Secondary: Agentic Vision Award |

---

## 1. Team Name

**Syndicate**

---

## 2. Problem Statement and Real-World Impact

### Problem

Abandoned, lost, and discarded fishing gear — collectively termed **ghost gear** by the United Nations Environment Programme — constitutes one of the most destructive and least addressed forms of marine pollution. An estimated **640,000 tonnes** of ghost nets enter the world's oceans annually (FAO, 2024), where they continue to entangle and kill marine life indefinitely, smother coral reef ecosystems, and create collision hazards for commercial vessel propellers and rudders.

Marine conservation teams and Autonomous Underwater Vehicle (AUV) operators rely on **Side-Scan Sonar (SSS)** to survey the seafloor for submerged debris. However, manual interpretation of sonar imagery remains the operational bottleneck:

- **Slow:** A trained sonar analyst requires **4–8 hours** to review a single survey mission's imagery, which may span hundreds of frames.
- **Error-prone:** Submerged debris frequently blends visually with natural seafloor features — rock formations, sand ripples, geological ridges, and kelp beds — leading to high miss rates under fatigue.
- **Subjective:** Classification consistency varies between analysts, producing unreliable data for longitudinal debris monitoring.
- **Expensive:** Dedicated sonar analysts command specialist salaries, pricing out smaller NGOs and research teams from conducting regular surveys.

**No scalable, automated system exists** that can process side-scan sonar imagery in near-real-time, classify marine debris with pixel-level precision, and produce geotagged hazard reports suitable for direct cleanup dispatch — all while remaining deployable at the edge on low-power hardware aboard survey vessels.

### Real-World Impact

| Impact Dimension | Measurable Outcome |
|---|---|
| **Operational Efficiency** | Reduces sonar review time from 4–8 hours to under 10 minutes per survey mission |
| **Wildlife Conservation** | Enables faster ghost net removal; an estimated 100,000+ marine animals die annually from entanglement (Ocean Conservancy) |
| **Reef Ecosystem Protection** | Early debris detection prevents progressive smothering of coral formations |
| **Maritime Safety** | Real-time hazard flagging during active surveys prevents vessel propeller collisions |
| **Democratized Access** | Eliminates the need for specialist sonar analysts, lowering the cost barrier for small NGOs and research teams |
| **Policy Support** | Structured, geotagged detection data supports evidence-based marine debris regulation (UN SDG 14 — Life Below Water) |

---

## 3. Planned OpenCV 5 Image and Video Analysis

The pipeline employs OpenCV 5 as the **core vision processing engine** across a **two-stage detection architecture**: a lightweight classical computer vision stage that generates candidate anomaly regions, followed by a deep learning verification stage. This hybrid design achieves two critical objectives simultaneously:

1. **Edge-deployable speed** — The classical stage runs on CPU alone, enabling real-time processing aboard survey vessels without GPU hardware.
2. **High precision** — The deep learning stage eliminates false positives from the classical stage, ensuring that only verified debris reaches the final report.

### Stage 1 — Classical Pre-Processing and Candidate Generation (OpenCV 5)

| Step | OpenCV 5 Technique | Purpose |
|---|---|---|
| **1. Color/Intensity Space Transformation** | `cv2.cvtColor()` — BGR → HSV / CIELAB | Isolates structural brightness (Value / L-channel) from illumination-dependent color. Submerged debris reflects acoustic signal more strongly than the surrounding seabed, making it separable in the brightness channel regardless of depth or lighting conditions. |
| **2. Speckle Noise Reduction** | `cv2.fastNlMeansDenoising()`, `cv2.medianBlur()` | Suppresses high-frequency speckle noise inherent to acoustic sonar returns without eroding the edges of true objects — critical for preserving debris boundaries. |
| **3. Adaptive Segmentation** | `cv2.adaptiveThreshold()`, `cv2.threshold()` with `THRESH_OTSU` | Automatically recalculates the optimal foreground/background cutoff per image region, compensating for acoustic shadow gradients and vehicle-motion-induced intensity drift (heave, pitch, roll). |
| **4. Morphological Structural Filtering** | `cv2.morphologyEx()` with `MORPH_OPEN` (erosion → dilation) | Removes small transient pixel clusters from sensor noise and natural clutter while preserving large, dense clusters consistent with debris accumulation. |
| **5. Geometric Contour Classification** | `cv2.findContours()`, aspect ratio, solidity, extent, `cv2.contourArea()` | Filters candidate regions by geometric regularity — rejecting irregular natural formations (rock clusters, sand ripples) that do not match the compactness or elongation profile of man-made objects (pipes, cylinders, netting bundles). |

> **Performance Note:** This stage runs entirely on CPU, producing a reduced set of candidate Regions of Interest (ROIs) that constrains the search space for Stage 2, reducing deep learning inference cost by an estimated **60–80%** compared to full-frame inference.

### Stage 2 — Deep Learning Verification (OpenCV 5 + YOLOv8/YOLO11-Seg)

| Component | Implementation | Details |
|---|---|---|
| **Model Architecture** | YOLOv8-seg / YOLO11-seg (fine-tuned) | Pixel-level instance segmentation and multi-class classification |
| **Inference via OpenCV 5** | `cv2.dnn.readNetFromONNX()` | Model loaded in ONNX format through the OpenCV 5 DNN module |
| **ROI Cropping** | `cv2.boundingRect()` | Extract candidate regions from Stage 1 for targeted inference |
| **Mask Compositing** | `cv2.addWeighted()` | Overlay segmentation masks on original sonar frames |
| **Detection Visualization** | `cv2.polylines()`, `cv2.putText()`, `cv2.rectangle()` | Draw classification labels, confidence scores, and bounding polygons for the dashboard |
| **Confidence Filtering** | Configurable threshold (default: 0.5) | Detections below threshold are discarded — satisfying the noise filtering requirement |

**Classification Taxonomy:**

| Class ID | Category | Description |
|---|---|---|
| 0 | **Ghost Net** | Abandoned fishing nets, trawl fragments, and monofilament tangles |
| 1 | **Pipe / Cylinder** | Discarded industrial piping, barrels, and cylindrical debris |
| 2 | **Structural Fragment** | Shipwreck components, hull plates, and structural metal debris |
| 3 | **Rope / Line** | Discarded rope, anchor chains, and longline fishing gear |
| 4 | **Natural Formation** | Rock clusters, geological ridges (negative/control class) |

### Stage 3 — Geotagging and Structured Reporting

- Sonar ping metadata (latitude, longitude, timestamp, vehicle heading, depth) is parsed alongside each frame.
- Confirmed detections are serialized into structured **JSON/CSV records** containing: geographic coordinates, bounding dimensions, mask area (m²), classification label, confidence score, and source frame reference.
- Reports are immediately uploadable to GIS platforms (QGIS, ArcGIS) for spatial analysis and cleanup route planning.

---

## 4. Planned AWS Architecture and Services

### Service Architecture

| Layer | AWS Service | Role in Pipeline |
|---|---|---|
| **Data Ingestion** | Amazon S3 | Stores raw sonar image logs, processed frames, model artifacts, and generated reports. Organized by survey mission under prefixes: `/raw/`, `/processed/`, `/models/`, `/reports/`. |
| **Model Training** | Amazon SageMaker | Fine-tunes the YOLO segmentation model on labeled sonar debris datasets. Manages experiment tracking, hyperparameter tuning, and model versioning via SageMaker Model Registry. |
| **Inference API** | AWS Lambda + Amazon API Gateway | Serverless endpoint that accepts uploaded sonar frames, executes the Stage 1 → Stage 2 pipeline, and returns structured detection results as JSON. |
| **Optimized CV Compute** | AWS Graviton via COOL | Executes the Stage 1 classical OpenCV workload (denoising, thresholding, morphological filtering) on Arm-based Graviton instances for improved throughput and lower cost per frame. |
| **Detection Storage** | Amazon DynamoDB | Stores structured per-detection records (location, dimensions, classification, confidence) with GSI indexes for geographic and temporal queries. |
| **Report Generation** | Amazon S3 + Lambda | Generates downloadable JSON/CSV/GeoJSON hazard reports on demand. |
| **Dashboard Hosting** | AWS Amplify | Hosts the web-based upload, map visualization, and report download interface. |
| **Monitoring** | Amazon CloudWatch | Logs pipeline latency, error rates, inference throughput, and resource utilization for reproducible benchmarking. |

### COOL Path Integration

The team will execute the **Stage 1 classical OpenCV workload on AWS Graviton via COOL** and produce reproducible benchmarks:

| Benchmark Metric | Measurement Method |
|---|---|
| Per-frame latency (ms) | CloudWatch custom metrics, averaged over 1,000+ frames |
| Throughput (frames/second) | Sustained processing rate under continuous load |
| CPU utilization (%) | CloudWatch EC2 metrics |
| Cost per 1,000 frames ($) | AWS Cost Explorer, normalized by instance type |

All measurements will be reported for both **Graviton (Arm via COOL)** and an equivalent **x86 baseline instance**, with identical input data and pipeline configuration.

---

## 5. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SONAR / AUV DATA SOURCE                        │
│                    (Side-Scan Sonar logs + ping metadata)               │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ upload
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Amazon S3                                     │
│              /raw/  ·  /processed/  ·  /models/  ·  /reports/           │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ S3 Event Notification
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    AWS Lambda + API Gateway                              │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  STAGE 1: Classical OpenCV 5 Pre-Processing  (via COOL / Graviton) │ │
│  │  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐  ┌────────┐ │ │
│  │  │ cvtColor │→│ denoise  │→│  Otsu   │→│ morphOp  │→│contours│ │ │
│  │  │ HSV/LAB  │ │ NLMeans  │ │threshold│ │  OPEN    │ │ filter │ │ │
│  │  └─────────┘  └──────────┘  └────────┘  └──────────┘  └────────┘ │ │
│  └─────────────────────────────────┬──────────────────────────────────┘ │
│                                    │ candidate ROIs                     │
│  ┌─────────────────────────────────▼──────────────────────────────────┐ │
│  │  STAGE 2: YOLOv8/11-Seg Instance Segmentation (via OpenCV 5 DNN)   │ │
│  │  → Classification: ghost_net | pipe | fragment | rope | natural    │ │
│  │  → Confidence scoring + NMS filtering                              │ │
│  └─────────────────────────────────┬──────────────────────────────────┘ │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │ verified detections
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Amazon DynamoDB                                │
│           (geotagged detections: location, class, confidence)           │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
              Outputs: Dashboard (Amplify) · Reports (JSON/CSV/GeoJSON) · CloudWatch
```

### Data Flow Summary

1. **Ingest:** Raw side-scan sonar frames and ping metadata are uploaded to Amazon S3, organized by survey mission.
2. **Trigger:** S3 Event Notifications invoke an AWS Lambda function for each new upload.
3. **Stage 1 (COOL):** Lambda executes the classical OpenCV 5 preprocessing pipeline on **AWS Graviton via COOL** — color space conversion, denoising, adaptive thresholding, morphological filtering, and geometric contour classification — producing candidate ROIs.
4. **Stage 2 (DL):** Candidate ROIs are passed to a fine-tuned YOLOv8/YOLO11-seg model loaded via `cv2.dnn.readNetFromONNX()` for pixel-level instance segmentation and multi-class classification.
5. **Store:** Verified detections (geographic coordinates, classification, confidence, mask area) are written to Amazon DynamoDB.
6. **Serve:** AWS Amplify hosts the web dashboard for uploading sonar logs, viewing detection overlays on an interactive map, and downloading structured hazard reports in JSON/CSV/GeoJSON formats.
7. **Monitor:** Amazon CloudWatch tracks per-frame latency, throughput (FPS), error rates, and resource utilization for reproducible benchmarking.

---

## 6. Target Users and Beneficiaries

| User Group | Use Case |
|---|---|
| **Marine Conservation NGOs** (e.g., Ghost Fishing Foundation, Ocean Conservancy, The Ocean Cleanup) | Prioritize debris removal operations by location, volume, and debris type using geotagged detection reports |
| **AUV / ROV Survey Operators** | Automate post-mission sonar log review, replacing hours of manual frame-by-frame analysis with instant AI-powered triage |
| **Coast Guard and Fisheries Authorities** | Generate standardized hazard reports for regulatory compliance, vessel safety advisories, and cleanup dispatch |
| **Marine Research Institutions** | Access longitudinal, structured debris accumulation datasets for peer-reviewed environmental research |
| **Port Authorities and Harbor Management** | Monitor harbor and channel debris in real-time to maintain safe navigation and prevent propeller damage |
| **International Environmental Bodies** (e.g., UNEP, IMO, FAO) | Data-driven reporting on SDG 14 (Life Below Water) progress with quantified debris density metrics |

---

## 7. Proposed Evaluation Method and Judge Demonstration

### Quantitative Evaluation Metrics

| Metric | Target | Measurement Method |
|---|---|---|
| **mAP@0.5** (Mean Average Precision) | ≥ 0.70 | COCO evaluation protocol on held-out annotated sonar test set |
| **mAP@0.5:0.95** | ≥ 0.45 | COCO evaluation protocol (stricter IoU range) |
| **Precision** | ≥ 0.80 | Per-class and aggregate, on test set |
| **Recall** | ≥ 0.70 | Per-class and aggregate, on test set |
| **False Positive Reduction** | ≥ 60% reduction | Measured before and after Stage 1 candidate filtering vs. full-frame inference |
| **Per-Frame Latency** | < 300ms | CloudWatch metrics on Graviton (COOL) vs. x86 baseline |
| **Throughput** | ≥ 5 FPS | Sustained processing rate under continuous load |
| **Edge Feasibility** | Confirmed deployable | Inference benchmarked on representative low-power device profile |

### Datasets

| Dataset | Type | Size | Source |
|---|---|---|---|
| **Marine Debris Archive** | Side-scan sonar | 1,800+ annotated images | NOAA / research archives |
| **Trash-ICRA19** | Forward-looking sonar + underwater RGB | 5,700+ images | J-EDI dataset |
| **UATD (Underwater Acoustic Target Detection)** | Sonar imagery | 9,000+ images | Academic benchmark dataset |
| **Custom Annotated Set** | Side-scan sonar | ~500 frames (manually annotated during build phase) | Collected from open AUV survey datasets |

### Evaluation Protocol

All experiments will follow a rigorous, reproducible evaluation methodology:

1. **Data Partitioning:** An 80/10/10 stratified split across training, validation, and test sets, ensuring balanced representation of all debris classes and natural formation controls.
2. **Stage-Wise Ablation Study:** Detection performance will be reported independently for Stage 1 (classical CV), Stage 2 (deep learning), and the combined pipeline to quantify the marginal contribution of each processing stage.
3. **COOL Infrastructure Benchmark:** The identical pipeline will be executed on AWS Graviton (via COOL) and an equivalent x86 instance under matched input data and configuration, with all measurements independently reproducible.
4. **Failure Mode Analysis:** False positives (natural formations misclassified as debris) and false negatives (camouflaged or occluded debris) will be explicitly documented, categorized, and presented with representative examples.

### Judge Demonstration Plan

| Demo Component | Format | What Judges Will See |
|---|---|---|
| **Live Dashboard Demo** | Web endpoint (AWS Amplify) or scheduled screen-share | Upload a raw sonar log → real-time detection overlays appear on an interactive map → download the generated hazard report (JSON/CSV/GeoJSON) |
| **Pipeline Transparency View** | Side-by-side comparison | Raw sonar frame → Stage 1 candidate masks → Stage 2 final classifications with confidence scores — making the pipeline's reasoning fully inspectable |
| **COOL Benchmark Dashboard** | CloudWatch metrics | Side-by-side latency/throughput graphs: Graviton (COOL) vs. x86 baseline |
| **5-Minute Submission Video** | Pre-recorded | Team introduction, problem narrative, live application walkthrough, architecture explanation, principal results, and documented failure cases |

---

## 8. Focus Path

### Primary: Best Use of COOL Award

The Stage 1 classical OpenCV workload — comprising denoising, adaptive thresholding, morphological filtering, and contour analysis — will be executed on **AWS Graviton via the Cloud-Optimized OpenCV Library (COOL)**. The team will produce:

- Reproducible latency and throughput benchmarks against an equivalent x86 instance.
- Documented COOL version, Graviton instance type, and deployment configuration.
- Evidence that COOL executes the claimed core workload (Stage 1 preprocessing).

### Secondary (Stretch Goal): Agentic Vision Award

If development progresses ahead of schedule, the pipeline will be extended with an **agentic feedback loop**: low-confidence detections (below a secondary threshold) will trigger an automated re-scan request — adjusting sonar gain parameters or requesting a follow-up AUV pass over the flagged area — rather than simply logging the uncertain detection. This closes the perception → decision → action loop required for the Agentic Vision path.

---

## 9. Team Bio

| Member | Role | Skills | Competition Experience |
|---|---|---|---|
| **Madhesh** | Solo Participant — Computer Vision & Cloud AI Engineer | Python, OpenCV, PyTorch, YOLOv8/YOLO11, TensorFlow, AWS (S3, Lambda, SageMaker), Docker | **TESCO Retail Hackathon** — Built an OpenCV-based retail shelf audit system for automated visual merchandising compliance (planogram verification, alcohol regulation checks, rule-matching). The classical CV → DL pipeline mirrors the two-stage architecture proposed here. |


---

*Prepared for submission to the OpenCV AI Competition 2026 — AWS Cloud Compute Grant Review.*  
*Team Syndicate · August 2026*