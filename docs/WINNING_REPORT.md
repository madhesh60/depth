# How to Win: Marine Debris Sonar Project

**OpenCV AI Competition 2026 · Team Syndicate (Madhesh) · Full report**

- **Written:** 22 Sep 2026
- **Hard deadline:** 26 Oct 2026, 11:59 PM Pacific = **27 Oct 2026, 12:29 PM India time**
- **Days left:** **34**

**What this report is based on**

- Your project folder `MARINE_DEBRIS` (code, dataset, raw archives, training results)
- Your GitHub repo `madhesh60/depth` (latest commit `8170eb3`)
- The official rules (Devpost + opencv.org)
- Tests I ran myself (OpenCV 5 + your real trained model)
- Web research (COOL, OpenCV 5, other teams, published work)

---

## Words used a lot (plain meaning)

| Word | Plain meaning | Example |
|---|---|---|
| Recall | Out of all real objects, how many you catch | 10 crab pots, you find 7 → recall 0.7 |
| Precision | Out of everything you flag, how many are real | You flag 10, 6 are real → precision 0.6 |
| False alarm | Model says "debris" but it's nothing | A rock marked as a crab pot |
| mAP | One overall score for a detector (1.0 = perfect) | 0.82 = good, 0.45 = weak |
| Validation set | The practice test | You may tune settings on it |
| Test set | The final exam | Look at it once, at the end |
| Side-scan sonar | A sonar towed or mounted on a boat that "paints" the seabed on both sides | Humminbird side imaging |
| Forward-looking sonar | A sonar that looks ahead like a torch beam (fan-shaped image) | UATD dataset |
| AMI | A ready-made server image on AWS: you start it and everything is already installed | COOL comes as an AMI |
| Graviton | Amazon's own Arm chips (cheaper per hour than Intel/AMD servers) | c8g servers |
| COOL | OpenCV build tuned by OpenCV/AWS to run faster on Graviton | The COOL award |
| KleidiCV | Arm's speed-up library for common OpenCV steps | Already inside normal OpenCV 5 for Arm |
| Agent | A program that looks at results and picks its next step by itself | "Unsure? Zoom in and check again" |
| GPX / KML / GeoJSON | Map files that phones, GPS units and map apps can open | Waypoints for a cleanup boat |

---

## 0. The short version

**In one line:** your research work is strong, but judges score most on four things that are missing or broken right now: a live system on AWS, real use of OpenCV 5, honest numbers, and a clearly new idea.

**Top 10 actions, in order**

1. Use **OpenCV 5 for real**: pin `opencv-python-headless==5.0.0.93` and pin every other version.
2. **Fix the dataset.** 955 pictures of real pipelines and platforms are labelled "empty seabed". 1,547 real empty crab-pot frames were thrown away.
3. **Stop naming classes what they are not.** "natural_formation" is really photos of fish and plants. "pipe_cylinder" is really test objects seen by a different sonar type.
4. **Make it a side-scan-sonar product** focused on ghost fishing gear (crab pots) and wreck debris. Report numbers per data source.
5. **Keep the official crab-pot test split untouched.** Then you can compare head-to-head with the published GhostVision paper.
6. **Get on AWS this week.** One Graviton (c8g) server running the COOL AMI hosts your whole app. Don't wait for credits.
7. **Build the novel loop: See → Prove → Decide → Act** (shadow proof, a re-look agent, and a cleanup plan). This is your innovation and your Agentic award entry.
8. **Redo the COOL speed test** so it proves COOL itself (not just "KleidiCV found"), and report **cost per survey-hour**.
9. **Build a simple web page with sample-data buttons**, so a judge sees results in 30 seconds.
10. **Freeze features on 21 Oct. Submit on 25 Oct**, a day early.

---

## 1. Where you stand today

| Item | Status |
|---|---|
| Grant ($150) | Won. The 30-min check-in must happen **by 2 Oct** to unlock the second half. |
| Dataset | Built, audited, leakage fixed, but 3 label bugs remain (section 4B) |
| Model | YOLO11s: test mAP@0.5 = 0.822 overall, but only **0.45 on crab pots** (your main target) |
| OpenCV version actually used | 4.12 (the rules require **5**) |
| AWS | Code written in `infra/`, **nothing deployed** |
| COOL speed test | Script written, **never run** on Graviton |
| Web dashboard | Not started |
| Geotagging | Fake: one lat/lon is copied onto every object |
| Agent loop | Not started |
| Report + video | Not started |

**Key dates**

| Date | What |
|---|---|
| 2 Oct | Last day for the grant check-in |
| 21 Oct | Feature freeze (my suggestion) |
| 25 Oct | Submit (my suggestion) |
| 26 Oct, 11:59 PM PT (27 Oct, 12:29 PM IST) | Hard deadline |
| 27 Oct – 9 Nov | Judging. **Your demo link must stay live the whole time.** |
| 10 Nov | Winners announced on OpenCV Live |

---

## 2. How you will be judged

### Main prize (1st $5,000 · 2nd $3,000 · 3rd $2,000)

| What judges score | Weight | What it means for you |
|---|---|---|
| Technical execution | 30% | Does it work end to end? Is the ML sound? Is OpenCV 5 doing real work? |
| Innovation | 20% | What is new? Many teams share your idea (Smart India Hackathon problem 26057). |
| Real-world impact | 20% | Would a cleanup crew really use it? Show proof, not claims. |
| User experience | 10% | Can a judge use it in one minute? |
| Documentation & presentation | 10% | Report, README, video |
| Cloud delivery, reproducibility, responsible operation | 10% | Runs on AWS, can be rebuilt, safe and honest |

**Tie-break:** Technical execution first, then Real-world impact.

### Extra awards ($1,000 each)

You can win one main prize **and** both extra awards.

- **Best Use of COOL:**

  | Criterion | Weight |
  |---|---|
  | Verified COOL on Graviton | 30% |
  | Architecture | 25% |
  | Measured performance / value | 20% |
  | Innovation | 15% |
  | Reproducibility | 10% |

- **Agentic Vision:**

  | Criterion | Weight |
  |---|---|
  | OpenCV 5 + agent doing real work | 30% |
  | Orchestration & autonomy | 25% |
  | Task success | 20% |
  | Failure handling & human control | 15% |
  | User experience | 10% |

  A chatbot that only explains results does **not** count. The picture results must change the agent's next step.

### Must-submit items

- Technical report covering: problem, users, architecture, OpenCV 5 use, AWS deployment, evaluation, limits, responsible use
- Code repo with pinned versions, plus build, deploy and test steps
- Architecture diagram showing OpenCV 5 + AWS (+ COOL and the agent, if you enter those awards)
- Working web link (or a booked live screen-share)
- Video, 5 minutes or less
- Evaluation, including failure cases

The COOL and Agentic awards each need extra evidence (see section 13).

### What past winners had in common

The 2023 OpenCV competition winners were **working, real-world systems** (a farm robot, a guide-dog robot, an exoplanet tool). A real demo and polish matter more than a long feature list.

---

## 3. What is good (keep these)

1. **Real problem, right category.** Lost fishing gear keeps killing sea life, and "environmental monitoring" is on the organisers' wish list.
2. **You won the grant.** Judges already rated your proposal as strong.
3. **You work like a researcher.** Every experiment has a guess, settings, results and a "next step" (`experiments.md`). Judges love this.
4. **You publish failures.** STUDY-01 showed your Stage 1 filter fires *more* on empty seabed (60 regions per frame) than on frames with debris (8). Saying that openly builds trust.
5. **You fixed data leakage.** Frames from the same recording no longer sit in both training and test (0 shared).
6. **Sharp error analysis.** You found that 91% of missed crab pots are tiny (under 10% of the frame width). That's the right diagnosis.
7. **Clean, settings-driven code.** Stage 1 times every step. The benchmark records the OpenCV build and a fingerprint (hash) of the input files.
8. **Small, torch-free inference.** `infer.py` and `lambda_handler.py` use only OpenCV's `cv2.dnn`, which keeps deployment small and simple.
9. **Your model works on OpenCV 5.** I ran your real `best.onnx` with OpenCV 5.0.0 on real sonar frames. It loaded and detected objects.
10. **Cost awareness.** Your infra notes already say "shut test servers down after each run".
11. **Tidy git history.** Small commits, each labelled with its type.
12. **Honest tiling study (STUDY-02).** You reported both the gain (+8 points recall) and the cost (big objects get worse).

---

## 4. Problems, ranked by damage

**Legend:** 🔴 can sink you · 🟠 costs many points · 🟡 costs some points

### 4A. Rule risks

1. 🔴 **You use OpenCV 4.12, not 5.**
   - Rule 1 says OpenCV 5 must do real work. `requirements.txt` says `opencv-python>=4.10`, and your benchmark file says 4.12.
   - OpenCV 5.0.0.93 is on pip now.
   - **Fix:** section 5.1.

2. 🔴 **Nothing runs on AWS.**
   - Rule 2 says a meaningful part must run on AWS.
   - `infra/` is written but not deployed, and the README still says "AWS CLI not installed".
   - **Fix:** section 5.6, starting this week.

3. 🔴 **Claims that don't match reality.** The rules let judges reject entries that "misrepresent capabilities or benchmarks". Right now:

   | Claim | What is actually true |
   |---|---|
   | "natural_formation = rock clusters, geological ridges" | It is ICRA19's **"bio"** class: fish, plants and sea life in normal **camera photos**. 267 of its 269 test boxes are camera photos. |
   | "pipe_cylinder = industrial piping" | All 65 test boxes are UATD **"cylinder" test objects**, seen with a **forward-looking** sonar (a different sonar type). |
   | "side-scan sonar" | UATD is forward-looking sonar. TrashCan and ICRA19 are camera photos. |
   | "geotagged hazard reports" | The Lambda copies one lat/lon onto every object. |
   | "COOL detected" | Normal pip OpenCV 5 for Arm already contains KleidiCV, so this check passes without COOL. |
   | "edge-deployable" | Never measured on an edge device. |

   - **Fix:** rename and narrow the scope (sections 5.2 and 5.10). Honest beats impressive.

### 4B. Data problems (these hurt the model itself)

4. 🔴 **955 pictures of real man-made objects are labelled "empty seabed".**
   - It's like showing a child a photo of a pipe and saying "there's nothing here", 900 times over.
   - **How it happened:**
     - `step3_remap.py` gave each Marine PULSE pipeline or platform picture one box covering the whole image.
     - The v1 builder then deleted any box bigger than 95% of the image.
     - Those pictures were left with no boxes, so they became "background".
   - **Counts:**

     | Split | Pipeline / cable | Engineering platform |
     |---|---:|---:|
     | Train | 836 | 61 |
     | Test | 0 | 57 |
     | Val | 1 | 0 |

   - **Extra damage:** all 57 "background" test images are platforms. So a **correct** detection there is counted as a false alarm.
   - Your `docs/references.md` cites these same frames as proof of "extensive negative sampling". That claim must change.

5. 🟠 **1,547 real empty crab-pot frames were thrown away** (1,430 train · 53 valid · 64 test).
   - `step3_remap.py` skips any image with no pot.
   - These frames are the best possible "empty seabed" examples for your biggest error, which is empty seabed flagged as fishing gear (79% of those false alarms).

6. 🟠 **Class names hide what the data really is.** Here is the real test-set makeup, counted from your label files:

   | Class (test boxes) | Where the boxes really come from |
   |---|---|
   | fishing_gear (952) | 933 crab pots (side-scan sonar) + 19 camera photos of nets |
   | pipe_cylinder (65) | 65 UATD cylinders (forward-looking sonar test objects) |
   | structural_fragment (501) | 422 UATD test objects (bucket, plane model, cage, tyre) + 42 KLSG shipwreck + 12 AI4Shipwrecks + 25 camera photos |
   | natural_formation (269) | 267 ICRA19 "bio" camera photos + 2 sonar |

   - So the 0.822 headline mostly measures two things: finding placed test objects in another sonar type, and spotting sea life in camera photos.
   - The one real side-scan debris class (crab pots) scores 0.45.

7. 🟠 **Mixed sensors.**
   - Side-scan sonar, forward-looking sonar and camera photos look nothing alike.
   - The model learned shortcuts, for example "camera photo → natural_formation".
   - Debris in camera photos: **0 of 44 found**.

8. 🟡 **You broke the official crab-pot test split.**
   - The Hugging Face crab-pot dataset has a fixed test split of 398 images. It appears to match the GhostVision paper's test set (399 images).
   - Your re-split mixed these frames into training, so you can't compare fairly with the published result.

9. 🟡 **Licences are not listed anywhere.**
   - **Crab-pot data:** CC-BY-SA-4.0. You must credit it and share changes under the same licence. The page also mentions GPL, so confirm with the authors.
   - **UATD:** CC BY 4.0.
   - **TrashCan and ICRA19:** their originals come from JAMSTEC deep-sea video. TrashCan's original terms are custom: research use, credit JAMSTEC, and commercial use needs permission. This holds even though the Roboflow copies say CC BY 4.0.
   - **AI4Shipwrecks and Marine PULSE:** licence not stated on their pages. Check before you publish.
   - **Ultralytics YOLO11:** AGPL-3.0. If you serve it on a website, your full source must be open under AGPL. Your repo is public, so just add an AGPL `LICENSE` file.

### 4C. Evaluation problems

10. 🟠 **Thresholds were tuned on the test set.**
    - The fishing_gear setting (0.10) was chosen by looking at test results.
    - It's like practising on the real exam paper and then reporting your exam score.
    - **Fix:** pick thresholds on validation and look at test only once.
11. 🟠 **One overall score hides the truth.** Always show numbers per source and per sensor.
12. 🟡 **Overfitting.**
    - Validation mAP peaked at **0.704 (epoch 16)** and fell to **0.663 (epoch 40)** while training loss kept falling.
    - The model was memorising training pictures. More epochs won't help.
13. 🟡 **Small studies.**
    - The tiling study used 60 frames; the ROI study used 300 frames.
    - That's fine for choosing a direction, but too small for final claims. Use the full validation set and add error ranges.
14. 🟡 **Speed numbers come from the wrong machine.**
    - "11.5 ms per frame" was measured on a Kaggle GPU.
    - On a normal CPU, your model took **~212 ms per frame** (my test, 2 cores). Your live system will run on CPU.
    - Measure on the real server.

### 4D. System, AWS and COOL problems

15. 🔴 **COOL cannot run on Lambda.**
    - COOL is sold only as an **AMI** (Ubuntu 24.04, with Python set up in `/opt/cool/venvs/`).
    - Your plan runs Stage 1 "on Graviton via COOL" inside Lambda, which is impossible.
    - Your script also says "install the COOL wheel", but no such wheel exists.
16. 🔴 **Your COOL test can't prove COOL.**
    - Normal `opencv-python-headless==5.0.0.93` for Arm already includes **KleidiCV 26.03**. I opened the wheel and checked.
    - So "Graviton normal vs Graviton COOL" may show only a small difference.
    - And "KleidiCV detected = yes" will be true for **both**.
17. 🟠 **Stage 1 no longer does anything useful for the product.**
    - After STUDY-01 it doesn't feed the detector. It now exists only to be timed.
    - `config.py` even says the steps were chosen so the COOL benchmark shows a speed-up.
    - A judge will ask "why does this step exist?", and architecture is 25% of the COOL score.
18. 🟠 **The Lambda says it runs Stage 1, but it doesn't.** Its description promises preprocessing; the code never calls it.
19. 🟠 **No dashboard, no report files, no map.** These are core to your pitch and to user experience (10%).
20. 🟡 **Too many AWS services planned** for one person in 34 days: SageMaker, DynamoDB, Amplify, API Gateway, Lambda and CloudWatch. "Meaningful" beats "many".
21. 🟡 **Judges can't get the model.** `runs/` is ignored by git, and `best.onnx` isn't hosted anywhere public.
22. 🟡 **Missing pieces:**
    - No tests.
    - No setup script.
    - No `export_onnx.py`, even though your code tells users to run it.

### 4E. Innovation and competition problems

23. 🔴 **Your idea is not unique.**
    - It is Smart India Hackathon 2026 problem **26057**.
    - Many teams have public repos for it: SonarSentinel, SONARIS, SIH26057-Marine-debris-Detection, SIH-2026-SSS-project, and more.
    - Your `AGENT.md` starts with the SIH problem text, word for word.
24. 🟠 **Published work already does the core of it.**
    - **GhostVision** (University of Delaware, paper published May 2026):
      - Detects derelict crab pots in side-scan sonar using YOLO / RF-DETR.
      - Reaches **F1 ≈ 0.71–0.73**. F1 is one score that balances recall and precision.
      - Runs about **10× faster than real time**.
      - Geotags detections with PINGMapper.
      - Uses the **same crab-pot dataset** you use.
    - **Marinedetect** (GitHub):
      - Already estimates object height from the sonar shadow.
      - Scores crab-pot mAP@0.5 = 0.465, close to your 0.45.
    - **Meaning:** "YOLO on sonar + dashboard" is only the baseline. You need a clear extra (section 6).

### 4F. Docs and repo problems

25. 🟡 **README.** It still has a "Why this wins" section (reads like selling to the judges) and the line "AWS CLI not installed".
26. 🟡 **`AGENT.md`.** It has broken characters (`â€˜ghost netsâ€™`) and the copied SIH text. Move the proposal to `docs/proposal.md` and clean it up.
27. 🟡 **Repo name.** `depth` says nothing about the project. Rename it, for example to `ghostgear-sonar`; GitHub redirects the old links.
28. 🟡 **Missing documents.** No `LICENSE`, no dataset card, no model card, no data-licence table.
29. 🟡 **Loose files.** `data_exp001.yaml` isn't tracked by git. Large zips sit in the root; git ignores them, which is fine, but keep the root tidy.

---

## 5. Changes to make (exact steps)

### 5.1 Versions (day 1, about 30 minutes)

Split your requirements into two files, so the live server stays small.

**`requirements.txt`** (live app + inference):

```
opencv-python-headless==5.0.0.93
numpy==<the exact version you test with>
fastapi==<pin>
uvicorn==<pin>
python-multipart==<pin>
boto3==<pin>
```

**`requirements-train.txt`** (Kaggle only):

- `ultralytics==8.4.158`
- `onnx` and `onnxslim`, pinned to exact versions too

**Then:**

1. Always use `==`, never `>=`.
2. Re-run `infer.py` and the Stage 1 benchmark on OpenCV 5.
3. Update every speed number in the docs.

### 5.2 Dataset v2 (days 1–3)

**Goal:** honest, side-scan only, bug-free.

| Change | Why |
|---|---|
| Drop the camera photos (TrashCan, ICRA19) | Not your sensor, and they create shortcuts |
| Take UATD out of the main model (or keep it only as a separate "forward-looking sonar" test) | Different sonar type; placed test objects, not real debris |
| Marine PULSE: remove the pipeline/platform images from "background" (or drop them) | They contain real objects |
| Marine PULSE: keep "seabed surface" and "underwater residual mound" as background | Real natural seabed in sonar. **This** is your true "natural formation" test. |
| Add back the 1,547 empty crab-pot frames as background | Real empty seabed from the same sonar |
| Keep the **official crab-pot test split** (398 images) locked; never train on it | Enables the head-to-head with GhostVision |
| Classes: `ghost_gear` (crab pots) and `wreck_debris` (KLSG shipwreck + AI4Shipwrecks) | The names match the data |
| Write `docs/dataset_card.md` (sources, sensor, licence, counts) | Documentation + responsible use |

**A new metric you get for free:** *false alarms per natural-seabed image* (seabed surface + mounds). It directly proves the claim "we separate natural seabed from debris".

### 5.3 Training (EXP-002 on v2, not v1)

- Don't spend 8–10 Kaggle hours on the buggy v1. Train on v2.
- `imgsz 1024`, about 25–30 epochs, `patience 8`. Your curve peaked early.
- Help small objects: set `copy_paste` above 0 (see idea 6.7) and keep mosaic.
- Keep `best.pt` chosen by the validation score.

### 5.4 Evaluation (write `src/detection/evaluate.py`)

1. **Thresholds from validation only.** Look at the test set **once**.
2. **Tables** per class, per source, per sensor and per object size (small / medium / large).
3. **Head-to-head** on the official crab-pot test split vs GhostVision (F1, precision, recall).
4. **Error ranges.**
   - Re-draw the test images randomly 1,000 times (this is called "bootstrap") and report the spread.
   - Example: "recall 0.74 (0.70–0.78)".
5. **Failure gallery:** 12 missed pots and 12 false alarms, each with a one-line reason.

### 5.5 Stage 1 becomes real sonar cleanup (so COOL speeds up useful work)

Your crab-pot frames are PINGMapper exports named like `Rec14_wcp_ss_port_00025`. In PINGMapper names:

- **"wcp"** means the water column is still in the picture.
- **"port"** tells you which side of the boat the image shows.

So you can do real sonar processing:

1. **Bottom tracking.** Find the first strong echo on each ping. That gives the sonar's height above the seabed (altitude).
2. **Water-column removal.** Cut away the empty water above the seabed.
3. **Slant-range correction.** Stretch pixels so a distance on the image equals a distance on the seabed (`cv2.remap` / `cv2.resize`).
4. **Gain normalisation.** Even out brightness along the track (per-row normalisation, or CLAHE, a local contrast booster).
5. **Tiling.** Cut the frame into zoomed tiles so tiny targets look bigger. STUDY-02 already showed +8 points recall.

**Then measure:** does detection on cleaned frames beat raw frames on validation? Keep only the steps that help.

**Result:** COOL now speeds up work the product really needs, and bottom tracking gives the altitude that idea 6.1 needs.

### 5.6 AWS (start this week; don't wait for credits)

**Simple, strong setup:**

```
Judge's browser
   │  HTTPS (CloudFront in front of the server gives a free https link)
   ▼
EC2 c8g.large (Graviton4) running the COOL AMI   ◄── the live demo
   • FastAPI web app + simple web page
   • OpenCV 5 (COOL build): cleanup → tiles → YOLO (cv2.dnn) → shadow proof → agent
   │                              │
   ▼                              ▼
S3 (uploads, results,          CloudWatch (logs, latency, "site down" alarm)
 model, benchmark files)       AWS Budgets alarm (e.g. $40)
   ▲
Kaggle training → best.onnx → S3
Optional: an Amazon Bedrock model writes the plain-English mission brief
```

**Cost**

- c8g.large costs about **$0.08/hour** (about $58/month) on demand.
- COOL adds a small hourly software fee. There is a 7-day free trial; check the Marketplace page.
- Keeping it up for the 14 judging days costs about **$27** plus the software fee. That fits inside your grant plus free AWS credits.

**Day-1 check on the COOL server**

1. Activate the COOL Python in `/opt/cool/venvs/...`.
2. Run:
   ```
   python -c "import cv2; print(cv2.__version__, cv2.__file__); print(cv2.dnn)"
   ```
3. If `cv2.dnn` exists, **your whole pipeline, neural network included, runs inside COOL**. That's the cleanest COOL story.

**Keep it running**

- Run the app as a `systemd` service. systemd is Linux's helper that restarts a program after a crash.
- Add a `/health` page and a CloudWatch alarm on it.

**Security**

- Private S3 bucket.
- An IAM role on the server (a permission badge, so no passwords or keys sit on it).
- Upload size limit, for example 20 MB.
- Uploads auto-deleted after 7 days (an S3 lifecycle rule).

**Drop from the plan unless you have spare time:** SageMaker, DynamoDB, Amplify and Lambda. Keep the Lambda code as an optional batch path, clearly marked "does not use COOL".

### 5.7 COOL benchmark v2 (your Best Use of COOL case)

Use three setups with the same frames (at least 1,000, fingerprinted with a hash), the same thread count, a warm-up, and 3 repeats:

| Label | Machine | OpenCV |
|---|---|---|
| A | c7i.large (Intel x86) | pip `opencv-python-headless==5.0.0.93` (includes Intel IPP) |
| B | c8g.large (Graviton4) | Same pip wheel (already includes KleidiCV 26.03) |
| C | Same c8g.large | COOL AMI (COOL 3.1, built on OpenCV 5.0) |

**Fairness note:** on Intel, 2 vCPUs are 1 physical core running two threads. On Graviton, 2 vCPUs are 2 real cores. Compare at equal price too, and say which comparison you use.

**Measure three things:**

1. Stage 1 cleanup on its own
2. The full pipeline, including YOLO
3. Throughput with all cores busy

**Report:** p50/p95 speed, frames per second, **cost per 1,000 frames**, and **cost per survey-hour**.

**Formulas:**

- Cost per 1,000 frames = price per hour ÷ (frames per second × 3,600) × 1,000
  - Example: $0.08/hour at 20 frames/s → 0.08 ÷ 72,000 × 1,000 ≈ **$0.0011**
- Real-time factor = processing time ÷ recording time
  - GhostVision processed data in about 9–10% of survey time. Match or beat that on Graviton, and say so.

**Prove COOL ran:**

- Record `cv2.__file__` (it should point under `/opt/cool/`), the COOL version, the AMI ID, the instance type and the build-info text.
- Stop relying on "KleidiCV found".

**Be honest if B ≈ C.** For example: "Normal OpenCV 5 already includes KleidiCV; COOL adds X% on top for these steps." Honest numbers earn points; hidden ones get caught.

### 5.8 The web page (user experience is 10%, and it carries your demo)

- **Sample buttons:** "Try a sample survey". Judges don't have sonar files.
- **Upload:** a single frame, a zip, or a PINGMapper export.
- **Results view:** boxes on the image, plus an **evidence card** per object with:
  - a zoomed crop
  - the shadow overlay
  - the estimated height
  - the confidence before and after re-look
  - the agent's decision
- **Agent trace panel:** each tool call, in order, with timing.
- **Map (Leaflet):**
  - Shown when the file has GPS.
  - Otherwise a clear "no GPS in this file" message.
- **Downloads:** JSON, CSV, GeoJSON, KML, GPX.
- **Human review queue:** approve / reject buttons whose answers are saved to S3 (see idea 6.6).
- **Footer:** shows `cv2.__version__` and "running on Graviton + COOL", so judges see it live.

### 5.9 Real geotagging (only honest GPS)

- **Use PINGMapper** (free) on real Humminbird recordings.
  - It exports per-ping latitude, longitude, heading and depth, plus the sonar images.
  - It comes with small and large **test recordings** you can use.
- **Turn a detection into lat/lon:**
  1. The ping number gives the boat's position.
  2. The distance across the track (after slant correction) gives a sideways offset, using the heading.
  3. Combine the two into lat/lon, and show an error estimate (for example "± a few metres").
- **The Hugging Face crab-pot images have no GPS.** Say so in the web page and the report.
- Check GhostVision's GitHub / Zenodo pages for sample recordings that do contain pots.

### 5.10 Docs and repo

- **Repo:** rename it (for example `ghostgear-sonar`) and add a `LICENSE` file (AGPL-3.0, because of Ultralytics).
- **README for judges:**
  - one-line pitch
  - demo link
  - 60-second quickstart
  - honest results table
  - architecture picture
  - limits
  - Remove "Why this wins".
- **Proposal:** move it to `docs/proposal.md`, remove the SIH text and fix the broken characters.
- **New docs:** `docs/dataset_card.md`, `docs/model_card.md` and `docs/responsible_use.md`.
- **Model file:** host `best.onnx` (as a GitHub Release, or a public S3 link) with its hash.

### 5.11 Tests (quick wins for technical execution and reproducibility)

Write `pytest` tests for:

- YOLO output decoding and NMS (removing duplicate boxes)
- The letterbox maths (resizing with padding)
- The shadow check, on a synthetic image
- The geotag maths
- An API smoke test: upload a sample and get "200 OK"

Optional: GitHub Actions, so the tests run on every push.

---

## 6. Novel ideas that can make you win

**In one line:** don't just detect. Do **See → Prove → Decide → Act**:

- The model must show **physical proof** for each find.
- An **agent** decides what to do when unsure.
- The output is a **cleanup plan**, not just boxes.

**Real-life picture:** a doctor doesn't trust one blurry X-ray. They zoom in, check a second sign, take another picture if needed, and then write a treatment plan. Your system should work the same way.

### 6.1 "Show your shadow" proof (headline innovation · 3–4 days)

**What it is**

- In side-scan sonar, a real object standing on the seabed makes a **bright echo followed by a dark shadow**, because sound can't pass through it.
- Flat seabed patches and noise don't make a proper shadow.
- Streetlight example: a person standing under a lamp casts a shadow; a drawing on the ground does not.

**How to build it (OpenCV)**

1. For each detection, look at a strip of pixels just beyond the object, on the side away from the sonar track.
   - The name tells you port or starboard.
   - Check on 10 images which side of the image is "far".
2. Measure how much darker that strip is than the nearby seabed, and how long the dark run is.

**Height from the shadow:** `h = H × Ls ÷ (G + Ls)`

| Symbol | Meaning |
|---|---|
| H | Sonar height above the seabed (from bottom tracking, section 5.5) |
| G | Seabed distance from the track to the object |
| Ls | Shadow length |

- A crab pot is roughly knee-high.
- A "crab pot" 5 m tall is really a rock.
- Zero height means flat texture.

**How to use it**

- Keep low-confidence detections only when they have a proper shadow and a believable height.
- This lets you run the detector at a low threshold (more pots found) without drowning in false alarms.

**How to prove it:** on validation, compare precision and recall at a low threshold with and without the shadow check. Show the shadow on every evidence card.

**Why it's yours:** Marinedetect computes height. You use the shadow as **proof that makes the decision**, combined with an agent and a cleanup plan.

### 6.2 "Re-look" agent (Agentic Vision award · 4–5 days)

**What:** when the detector is unsure, an agent picks the next tool, like a person squinting and zooming in.

**Tools:**

- `detect`
- `zoom_relook` (a 2× tile around the spot)
- `enhance_contrast`
- `shadow_check`
- `estimate_height`
- `match_other_pass`
- `ask_human`
- `plan_resurvey`
- `plan_recovery`

**Start with simple rules you can test:**

```
for each detection:
    if confidence is high and a shadow is found:        ACCEPT
    elif confidence is medium or low:
        relook = zoom_relook + enhance_contrast          # new picture → new decision
        if relook confidence is high and shadow found:  ACCEPT
        elif seen in another pass nearby:               ACCEPT (2 sightings)
        else:                                           ASK_HUMAN (with evidence card)
    else:                                               DROP

mission level:
    unsure clusters  → plan re-survey lines (GPX)  → a human must approve
    confirmed pots   → recovery route (GPX/KML)    → a human must approve
```

Tune every "high / medium / low" number on validation.

**Optional LLM layer**

- An Amazon Bedrock model reads the JSON results and writes the mission brief, or proposes the re-survey order, using only the tools above.
- The rule-based core stays in charge of safety.

**Evidence the award asks for:**

- An agent diagram
- Saved traces (JSON) showing a re-look that **changed** the decision
- An evaluation covering:
  - recall and precision before vs after the agent
  - tool calls per frame
  - time per frame
  - failure cases
  - human-approval gates

**Bonus:** expose the tools as an **MCP server**, a standard plug that lets any AI agent (Claude, Kiro and others) call your sonar tools. The organisers list MCP as welcome.

### 6.3 "Cost per survey-hour" (COOL award + impact · 1 day after 5.7)

- Turn speed into money a cleanup group understands: "One Graviton server processes an hour of sonar recording in X minutes for $Y; the Intel server costs $Z."
- Add the real-time factor, and compare with GhostVision's published "about 10× faster than real time".

### 6.4 Head-to-head with published work (1 day)

- Evaluate on the official crab-pot test split, and put your number next to GhostVision's F1 (≈ 0.71–0.73) in one honest table.
- Win, tie or lose, say it and explain why. Judges trust teams who compare against real baselines.

### 6.5 From pixels to a cleanup plan (Physical AI "act" · 2 days, needs GPS from 5.9)

- **Output files a boat crew can load:**
  - a GeoJSON / KML hazard map
  - GPX waypoints for re-survey lines
  - a shortest-order recovery route for the confirmed pots ("always go to the nearest next pot" is enough)
- A human approves before anything is "dispatched".

### 6.6 "Every correction teaches the model" (1 day)

- The approve / reject clicks in the review queue are saved to S3 as new labels.
- Show a small experiment: retrain with 100 corrected frames → fewer false alarms.
- Real-life example: a spam filter that improves every time you click "not spam".

### 6.7 Physics-aware copy-paste (2 days · stretch)

- Cut real crab pots **together with their shadows** out of training frames.
- Blend them into the 1,547 empty seabed frames with `cv2.seamlessClone`, keeping the correct direction.
- That gives more tiny targets to learn from, which means better recall on the class that matters.

### 6.8 Proof of removal (stretch)

- After a cleanup, a second survey over the same GPS spots should show "no pot here now".
- Compare before/after detections by location, and mark each pot "removed" or "still there".

### What to pick (34 days, solo)

| Priority | Ideas |
|---|---|
| Must | 6.1 + 6.2 + 6.3 + 6.4 |
| If GPS works | 6.5 + 6.6 |
| Only if ahead | 6.7 + 6.8 |

---

## 7. Final system (what the judges will see)

```
Raw sonar (PINGMapper export or sample frames)
   │
   ▼  SEE: OpenCV 5 (COOL on Graviton)
Stage 1 cleanup: bottom track → cut water column → slant-range fix → even brightness → tiles
   │
   ▼
YOLO11 (ONNX) through cv2.dnn  →  candidate objects + confidence
   │
   ▼  PROVE: OpenCV 5
Shadow check + height estimate  →  evidence card per object
   │
   ▼  DECIDE: agent
accept / re-look (zoom, contrast) / second pass / ask a human
   │
   ▼  ACT
GeoJSON · KML · CSV map  +  GPX re-survey lines  +  recovery route  (a human approves)
   │
   ▼
Web page on EC2 c8g (COOL) · results in S3 · logs and alarms in CloudWatch
```

---

## 8. Scorecard: now vs after this plan

This is **my rough guess, not an official score**.

| Criterion (weight) | Now | After plan | Main reason |
|---|---:|---:|---|
| Technical execution (30) | ~12 | ~25 | Working system, OpenCV 5, fixed data, honest evaluation |
| Innovation (20) | ~7 | ~16 | Shadow proof + re-look agent + survey-hour cost |
| Real-world impact (20) | ~10 | ~16 | GhostVision comparison, real GPS, cleanup plan |
| User experience (10) | ~1 | ~8 | Sample buttons, evidence cards, downloads |
| Documentation (10) | ~6 | ~9 | Report, cards, honest README, video |
| Cloud + responsible (10) | ~1 | ~8 | Live on Graviton + COOL, alarms, security, stated limits |
| **Total (100)** | **~37** | **~82** | |

| Extra award | Now | After plan |
|---|---:|---:|
| COOL | ~10 | ~75 |
| Agentic Vision | 0 | ~70 |

---

## 9. The 34-day plan

### Week 1 (22–28 Sep): foundations

- [ ] Pin OpenCV 5; re-run inference and Stage 1 on it
- [ ] Build dataset v2 (5.2) and its dataset card; lock the official crab-pot test split
- [ ] Start EXP-002 on v2 (Kaggle "Save & Run All")
- [ ] AWS setup:
  - [ ] Account, budget alarm, IAM role, S3 bucket
  - [ ] Start the c8g COOL AMI and confirm `cv2.dnn` exists in COOL
  - [ ] Stop the server when idle
- [ ] Book the grant check-in (it must be done by 2 Oct)

### Week 2 (29 Sep – 5 Oct): make it right

- [ ] Grant check-in (talking points in section 10)
- [ ] `evaluate.py`: validation-tuned thresholds, per-source tables, bootstrap ranges, failure gallery
- [ ] Stage 1 sonar cleanup (5.5); keep only the steps that help on validation
- [ ] Shadow proof v1 (6.1) + its evaluation
- [ ] GhostVision head-to-head (6.4)

### Week 3 (6–12 Oct): make it usable

- [ ] FastAPI app + web page (5.8) with sample buttons and evidence cards
- [ ] Deploy on the c8g COOL server; systemd, CloudWatch, HTTPS link
- [ ] PINGMapper geotag path (5.9); map + downloads

### Week 4 (13–19 Oct): make it win

- [ ] Re-look agent + trace view + agent evaluation (6.2)
- [ ] COOL 3-way benchmark + charts + cost per survey-hour (5.7, 6.3)
- [ ] GPX / KML cleanup plan (6.5); review queue that saves labels (6.6)

### Week 5 (20–26 Oct): ship

- [ ] **21 Oct: feature freeze.** Only bug fixes after this.
- [ ] Technical report, architecture and agent diagrams, README, cards
- [ ] **23–24 Oct:** record the video
- [ ] **25 Oct:** submit on Devpost
- [ ] Keep the server up until 9 Nov and check it every day

---

## 10. Grant check-in talking points (before 2 Oct)

1. **Problem:** derelict crab pots / ghost gear in side-scan sonar. Manual review is slow.
2. **Honest progress:**
   - dataset audited and rebuilt
   - leakage fixed
   - baseline trained
   - error analysis done
   - a negative result published (STUDY-01)
3. **What we learned:** crab pots are tiny, and mixed sensors inflate scores. We're now side-scan only, with honest per-source numbers.
4. **Plan:**
   - OpenCV 5 + COOL on Graviton running the full pipeline
   - shadow proof + re-look agent
   - a cleanup plan as the output
5. **Questions for them:**
   - Does COOL's OpenCV build include `cv2.dnn`?
   - What evidence do they want for "COOL executes the core workload"?

---

## 11. Video (5 minutes max)

| Time | Show |
|---|---|
| 0:00–0:20 | A real sonar frame with a crab pot, plus one line on why ghost gear matters |
| 0:20–0:50 | Who uses it (cleanup crews, surveyors) and their pain (hours of manual review) |
| 0:50–2:30 | **Live demo:** sample survey → detections → evidence card with shadow + height → the agent re-looks an unsure spot → map → download GPX / GeoJSON |
| 2:30–3:20 | Architecture: OpenCV 5 + COOL on Graviton, S3, CloudWatch, the agent loop |
| 3:20–4:20 | Results: honest table, GhostVision comparison, COOL cost chart, 2 failure cases |
| 4:20–5:00 | Impact, limits (trained on one bay's crab pots and one sonar brand), responsible use, you |

**Tip:** record the demo on the real AWS link, not on localhost.

---

## 12. Technical report outline (matches the required sections)

1. **Problem and users**
2. **Data:** sources, sensors, licences, the v1 → v2 fixes, and the 3 bugs you found and fixed. Showing you caught your own bugs is a strength.
3. **Architecture diagram:** where OpenCV 5 runs, where COOL runs, where AWS is used
4. **OpenCV 5 implementation:** cleanup, `cv2.dnn`, shadow proof, drawing
5. **Agent loop:** tools, rules, human control, traces
6. **AWS deployment:** services, cost, security, uptime
7. **Evaluation:** per-source tables, GhostVision comparison, agent evaluation, COOL benchmark, error ranges
8. **Failure cases and limits**
9. **Responsible use:**
   - a human approves before any dispatch
   - never publish exact coordinates of protected wrecks (war graves, heritage sites)
   - no personal data
   - uploads deleted after 7 days
   - licences respected
   - model limits stated
10. **How to reproduce:** commands, versions, file hashes

---

## 13. Submission checklist

### Everyone

- [ ] OpenCV 5.0.0.93 pinned and used in the live app (version shown in the page footer)
- [ ] Live HTTPS link, up from 27 Oct to 9 Nov, with sample buttons
- [ ] Public repo (renamed) with LICENSE, pinned requirements, setup script and tests
- [ ] `best.onnx` downloadable, with its hash
- [ ] Architecture diagram with OpenCV 5 + AWS + COOL + the agent
- [ ] Technical report (section 12)
- [ ] Video, 5:00 or less
- [ ] Evaluation, including failure cases

### Best Use of COOL (extra)

- [ ] COOL version, AMI ID, instance type
- [ ] Proof that COOL ran the core workload (`cv2.__file__` path, logs)
- [ ] 3-way benchmark files + charts + a one-command rerun + the input hash
- [ ] Cost per 1,000 frames and per survey-hour

### Agentic Vision (extra)

- [ ] Agent diagram (perceive → decide → act)
- [ ] Traces where a re-look changed the decision
- [ ] Agent evaluation: task success, failure handling, logging, human control

---

## 14. Risks and backups

| Risk | Backup |
|---|---|
| COOL's build has no `cv2.dnn` | Run Stage 1 + shadow proof in COOL and YOLO in stock OpenCV 5 in the same app; say so clearly |
| COOL is about as fast as stock | Report it honestly; lead with Graviton vs x86 cost and full-pipeline cost per survey-hour |
| EXP-002 doesn't lift recall | Use tiling + shadow proof at a low threshold, and report both |
| No crab-pot recording with GPS | Show geotagging on PINGMapper's test recording; label the crab-pot frames "no GPS" |
| AWS bill surprise | Budget alarm; stop test servers; only one small c8g for the demo |
| Server goes down during judging | systemd auto-restart; CloudWatch alarm to your email; check daily |
| Kaggle GPU time runs out | Smaller model or fewer epochs, or a few hours on an AWS GPU (g5 / g6) paid from credits |
| Time runs out | Drop 6.7 / 6.8 first, then 6.5. **Never drop the live demo, the report or the video.** |

---

## 15. What NOT to do

- Don't add AWS services just to list them.
- Don't show one overall mAP without the per-source table.
- Don't tune anything on the test set again.
- Don't train on v1 again; fix the data first.
- Don't claim "edge" or "real-time" without a measurement.
- Don't let the demo depend on a cold serverless start.
- Don't publish exact coordinates of protected shipwrecks.
- Don't keep the SIH problem text in the repo.

---

## 16. Facts I checked myself (evidence log)

| Check | Result |
|---|---|
| OpenCV 5 on pip | `opencv-python` 5.0.0.93 is available (OpenCV 5.0.0 released 19 Aug 2026) |
| Your `best.onnx` on OpenCV 5.0.0 | Loads and runs: 15 crab-pot frames, 43 detections, ~212 ms per frame on 2 CPU cores |
| Generic YOLO11s ONNX on OpenCV 5 | Output matches ONNX Runtime (max difference ~0.001). New engine ~217 ms, classic ~252 ms, ONNX Runtime ~105 ms (2 cores) |
| Normal OpenCV 5 wheel for Arm | Build info: "Custom HAL: YES (carotene 0.0.1, KleidiCV 26.03)" |
| Normal OpenCV 5 wheel for x86 | Uses Intel IPP |
| COOL | AMI on AWS Marketplace: COOL 3.1 built on OpenCV 5.0, Ubuntu 24.04, Python in `/opt/cool/venvs/`, Graviton4 c8g / m8g / r8g |
| Marine PULSE "background" bug | Empty labels on 836 + 61 train, 57 test and 1 val images of pipelines / platforms |
| Empty crab-pot frames skipped | 1,430 train + 53 valid + 64 test = 1,547 |
| Where test boxes come from | See the table in 4B, item 6 |
| ICRA19 → natural_formation | Class "bio" was mapped to natural_formation (`DATASET/scripts/data_injection.py`) |
| UATD mapping | cylinder → pipe; bucket, plane, square cage, tyre → structural (`data_injection.py`) |
| Overfitting | Val mAP@0.5: 0.704 at epoch 16 → 0.663 at epoch 40 (`runs/EXP-001/results.csv`) |
| `AGENT.md` shown as "changed" | Only the Windows line endings changed |

---

## 17. Sources

- Competition rules: [Devpost](https://opencv26.devpost.com/) · [opencv.org competition page](https://opencv.org/opencv-ai-competition-2026/)
- OpenCV 5:
  - [OpenCV 5.0.0 release](https://opencv.org/release/opencv-5-0-0/)
  - [What's new in OpenCV 5](https://opencv.org/opencv-5/)
  - [opencv-python on PyPI](https://pypi.org/project/opencv-python/)
- COOL:
  - [COOL page](https://opencv.org/cool/)
  - [COOL for Graviton4 (Marketplace)](https://aws.amazon.com/marketplace/pp/prodview-fdvbfiewzuehs)
  - [COOL for Graviton3 (Marketplace)](https://aws.amazon.com/marketplace/pp/prodview-5b2boxpyztidw)
- Server price: [c8g.large pricing](https://www.economize.cloud/resources/aws/pricing/ec2/c8g.large/)
- Published work:
  - [GhostVision repo](https://github.com/PINGEcosystem/GhostVision)
  - [GhostVision paper (JMSE, May 2026)](https://www.mdpi.com/2077-1312/14/10/951)
  - [Crab-pot dataset (Hugging Face)](https://huggingface.co/datasets/PINGEcosystem/sss-crab-pot-detection-ds)
  - [Marinedetect (shadow height)](https://github.com/Vedjamkar/Marinedetect)
- Geotagging tool: [PINGMapper](https://cameronbodine.github.io/PINGMapper/) · [PINGMapper test data](https://cameronbodine.github.io/PINGMapper/docs/gettingstarted/Testing.html)
- Datasets:
  - [UATD paper + licence](https://www.nature.com/articles/s41597-022-01854-w)
  - [Marine PULSE](https://zenodo.org/records/7922705)
  - [AI4Shipwrecks](https://deepblue.lib.umich.edu/data/concern/data_sets/8623hz41x)
  - [TrashCan licence](https://datasetninja.com/trash-can)
- Other SIH 26057 teams:
  - [SonarSentinel](https://github.com/Kalyan14s/sonarsentinel)
  - [SONARIS](https://github.com/maheshepili/SONARIS)
  - [SIH26057-Marine-debris-Detection](https://github.com/aditisingh1010/SIH26057-Marine-debris-Detection)
  - [SIH-2026-SSS-project](https://github.com/NithinPranav-007/SIH-2026-SSS-project)
- Past winners: [OpenCV AI Competition 2023 winners](https://www.seeedstudio.com/blog/2024/01/16/announcing-the-champions-of-the-opencv-ai-competition-2023/)
- Your repo: [madhesh60/depth](https://github.com/madhesh60/depth)

---

**In one line:** fix the data and the claims, run the full pipeline on Graviton with COOL, and add the See → Prove → Decide → Act loop. That turns a common hackathon idea into a trustworthy tool that acts, which is exactly what this competition rewards.
