"""DEPTH on AWS — the architecture diagram, drawn by code in the AWS reference-architecture style.

Every element is placed by this script (no auto-layout), with the official AWS architecture icons that ship
with the `diagrams` package. It draws the deployment exactly as built (infra/cfn/depth.yaml, infra/README.md,
the live server): grey and dashed = not live yet. Nothing is drawn that DEPTH does not run.

    pip install diagrams                          # only for its AWS icon set
    python docs/diagram/depth_architecture.py     # -> docs/diagram/depth_architecture.svg (+ .png via Chrome)
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import textwrap
from html import escape
from pathlib import Path

import diagrams

HERE = Path(__file__).resolve().parent
OUT_SVG, OUT_PNG = HERE / "depth_architecture.svg", HERE / "depth_architecture.png"
ICONS = Path(diagrams.__file__).resolve().parent.parent / "resources"
W, H = 1920, 1180
FONT = "'Amazon Ember', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
INK, LINE, GROUP, PEND = "#232f3e", "#545b64", "#5a6b86", "#a7b0b8"
out: list[str] = []


# ------------------------------------------------------------------------------------------- primitives
def icon(name: str, cx: float, cy: float, size: int = 48, faded: bool = False) -> None:
    data = base64.b64encode((ICONS / name).read_bytes()).decode()
    op = ' opacity="0.35"' if faded else ""
    out.append(f'<image x="{cx - size / 2}" y="{cy - size / 2}" width="{size}" height="{size}"{op} '
               f'href="data:image/png;base64,{data}"/>')


def text(x, y, lines, size=13.0, colour=INK, anchor="middle", weight="400", gap=1.3) -> None:
    for i, ln in enumerate([lines] if isinstance(lines, str) else lines):
        out.append(f'<text x="{x}" y="{y + i * size * gap:.1f}" font-size="{size}" font-weight="{weight}" '
                   f'fill="{colour}" text-anchor="{anchor}">{escape(ln)}</text>')


def svc(name: str, cx: float, cy: float, title: str, sub: str = "", size: int = 48, faded: bool = False) -> None:
    """An AWS-style service: icon, then a two-line label (service · resource)."""
    icon(name, cx, cy, size, faded)
    y = cy + size / 2 + 17
    text(cx, y, title, 13, PEND if faded else INK)
    if sub:
        text(cx, y + 16, sub, 12, PEND if faded else "#545b64")


def group(x, y, w, h, title, dashed=True, stroke=GROUP, fill="none", icon_name=None, title_colour=None) -> None:
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" '
               f'stroke-width="1.4"{dash}/>')
    tx = x + 12
    if icon_name:
        icon(icon_name, x + 16, y + 16, 26)
        tx = x + 36
    text(tx, y + 21, title, 14, title_colour or stroke, "start")


def badge(cx, cy, n) -> None:
    out.append(f'<circle cx="{cx}" cy="{cy}" r="13" fill="#000"/>')
    text(cx, cy + 4.6, str(n), 13, "white", weight="700")


def arrow(points, pending=False) -> None:
    colour = PEND if pending else LINE
    dash = ' stroke-dasharray="6 5"' if pending else ""
    pts = " ".join(f"{x},{y}" for x, y in points)
    out.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="1.5"{dash} '
               f'stroke-linejoin="round" marker-end="url(#{"hp" if pending else "h"})"/>')


def line(points) -> None:  # a connector without a head (it joins another arrow)
    pts = " ".join(f"{x},{y}" for x, y in points)
    out.append(f'<polyline points="{pts}" fill="none" stroke="{LINE}" stroke-width="1.5" stroke-linejoin="round"/>')


# ------------------------------------------------------------------------------------------- canvas
out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="{FONT}">')
out.append("<defs>" + "".join(
    f'<marker id="{m}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
    f'orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" fill="{c}"/></marker>'
    for m, c in (("h", LINE), ("hp", PEND))) + "</defs>")
out.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# AWS Cloud frame, with the badge
out.append(f'<rect x="150" y="20" width="1330" height="1060" fill="none" stroke="{INK}" stroke-width="1.6"/>')
out.append(f'<rect x="150" y="20" width="40" height="34" fill="{INK}"/>')
text(170, 40, "aws", 15, "white", weight="700")
out.append('<path d="M158,45 Q170,51 182,45" fill="none" stroke="#ff9900" stroke-width="2.2" stroke-linecap="round"/>'
           '<path d="M179,43 L183,45 L179,47" fill="none" stroke="#ff9900" stroke-width="1.8"/>')
text(202, 43, "AWS Cloud · us-east-1", 16, INK, "start")

# ------------------------------------------------------------------------------------------- outside: people
svc("aws/general/users.png", 72, 500, "Users", "studio")
svc("aws/general/client.png", 72, 640, "AI agents", "MCP · ask only")
svc("aws/general/client.png", 72, 770, "GIS tools", "OGC API")
svc("aws/general/user.png", 72, 930, "Approver", "today: the owner")
svc("onprem/vcs/github.png", 72, 140, "GitHub", "Release exp003-v1", size=44)

# ------------------------------------------------------------------------------------------- edge (pending)
group(245, 70, 180, 180, "Edge component")
svc("aws/network/cloudfront.png", 335, 145, "Amazon CloudFront", "pending verification", faded=True)

# ------------------------------------------------------------------------------------------- cost
group(450, 70, 600, 180, "Cost component")
svc("aws/cost/budgets.png", 560, 145, "AWS Budgets", "$40 / month")
svc("aws/cost/cost-explorer.png", 750, 145, "Cost Anomaly", "Detection")
svc("aws/integration/eventbridge-scheduler.png", 940, 145, "EventBridge Scheduler", "stop on 10 Nov 2026")

# ------------------------------------------------------------------------------------------- VPC / subnet / EC2
group(175, 290, 875, 470, "VPC", dashed=False, stroke="#248814", icon_name="aws/network/vpc.png")
group(195, 335, 835, 405, "Public subnet", dashed=False, stroke="#7aa116", fill="#f2f6e8",
      icon_name="aws/network/public-subnet.png")
svc("aws/compute/ec2-elastic-ip-address.png", 335, 500, "Elastic IP", "HTTPS link")
group(380, 385, 630, 335, "Amazon EC2 · c8g.large · Graviton4", dashed=False, stroke="#ed7100", fill="white",
      icon_name="aws/compute/ec2-instance.png")
text(392, 430, "stock OpenCV 5 today (rehearsal) · built for the COOL AMI, pending", 12, "#8a4b00", "start")
svc("onprem/network/caddy.png", 460, 500, "Caddy", "HTTPS :443", size=50)
svc("programming/framework/fastapi.png", 620, 500, "DEPTH server", "studio · API · MCP")
svc("programming/language/python.png", 790, 500, "DEPTH agent", "OpenCV 5 core loop")

# the core loop, as step pills
pills = ["check promise", "choose look", "perceive", "decide", "plan & ask"]
px0, pw, pg, py = 404, 106, 14, 616
for i, p in enumerate(pills):
    x = px0 + i * (pw + pg)
    out.append(f'<rect x="{x}" y="{py}" width="{pw}" height="30" rx="15" fill="#fff4e8" stroke="#ed7100" '
               f'stroke-width="1.2"/>')
    text(x + pw / 2, py + 19.5, p, 12.5, "#8a4b00", weight="600")
    if i < len(pills) - 1:
        arrow([(x + pw, py + 15), (x + pw + pg, py + 15)])
text(695, 676, "Stage 1 seabed · YOLO11 on cv2.dnn · shadow evidence · one card per object", 12, "#545b64")
text(695, 694, "every step logged · nothing auto-confirmed", 12, "#545b64")
arrow([(790, 590), (790, 601), (457, 601), (457, 614)])

# ------------------------------------------------------------------------------------------- right column
group(1080, 70, 385, 250, "Monitoring component")
svc("aws/network/route-53.png", 1140, 135, "Amazon Route 53", "health check")
svc("aws/management/cloudwatch-alarm.png", 1272, 135, "CloudWatch", "alarms")
svc("aws/integration/simple-notification-service-sns.png", 1402, 135, "Amazon SNS", "alerts → owner")
svc("aws/management/cloudwatch-logs.png", 1140, 250, "CloudWatch Logs", "per-stage metrics")
svc("aws/integration/eventbridge-rule.png", 1272, 250, "EventBridge", "rules")
arrow([(1164, 135), (1246, 135)])
arrow([(1296, 135), (1376, 135)])
arrow([(1296, 250), (1450, 250), (1450, 145), (1428, 145)])

group(1080, 340, 385, 250, "Data component")
svc("aws/storage/simple-storage-service-s3.png", 1145, 405, "Amazon S3", "DataBucket")
svc("aws/management/systems-manager-parameter-store.png", 1272, 405, "SSM Parameter", "Store · secrets")
svc("aws/management/systems-manager-run-command.png", 1400, 405, "Systems Manager", "Run Command")
svc("aws/storage/elastic-block-store-ebs.png", 1145, 520, "Amazon EBS", "daily snapshots")
svc("aws/security/identity-and-access-management-iam-role.png", 1272, 520, "AWS IAM", "scoped roles")

group(1080, 610, 385, 150, "Notification component")
svc("aws/integration/simple-notification-service-sns.png", 1170, 670, "Amazon SNS", "approval emails")

# ------------------------------------------------------------------------------------------- bottom row
group(175, 790, 525, 250, "Benchmark component")
svc("aws/compute/ec2.png", 265, 880, "Amazon EC2", "c8i.xlarge · x86")
svc("aws/compute/ec2.png", 430, 880, "Amazon EC2", "c8g.xlarge · Graviton4")
svc("aws/general/marketplace.png", 600, 880, "COOL AMI", "pending", faded=True)
text(437, 1000, "same code, frames and model · stock OpenCV 5: 157 ms vs 197 ms per frame", 12, "#545b64")
text(437, 1018, "self-terminating machines · results → S3", 12, "#545b64")

group(730, 790, 320, 250, "Deployment component")
svc("aws/management/cloudformation.png", 900, 880, "AWS CloudFormation", "depth stack")
text(890, 1000, "dry-run by default · removed by teardown.sh", 12, "#545b64")

group(1080, 790, 385, 250, "Optional component")
svc("aws/ml/bedrock.png", 1360, 880, "Amazon Bedrock", "not deployed", faded=True)

# ------------------------------------------------------------------------------------------- the flow
arrow([(98, 500), (309, 500)])
badge(200, 478, 1)
line([(98, 640), (122, 640), (122, 500)])
line([(98, 770), (122, 770), (122, 640)])
arrow([(359, 500), (432, 500)])
badge(396, 478, 2)
arrow([(487, 500), (594, 500)])
badge(540, 478, 3)
arrow([(646, 500), (764, 500)])
badge(705, 478, 4)
arrow([(816, 480), (1068, 480), (1068, 405), (1119, 405)])
badge(1068, 442, 5)
arrow([(816, 522), (1062, 522), (1062, 670), (1144, 670)])
badge(1062, 600, 6)
arrow([(1170, 742), (1170, 775), (1065, 775), (1065, 1100), (120, 1100), (120, 930), (98, 930)])
badge(620, 1100, 7)
arrow([(46, 930), (26, 930), (26, 500), (46, 500)])
badge(26, 715, 8)
badge(1205, 112, 9)
arrow([(940, 222), (940, 383)])
badge(940, 300, 10)
arrow([(900, 854), (900, 722)])
badge(923, 773, 11)
arrow([(98, 140), (162, 140), (162, 405), (378, 405)])
badge(162, 270, 12)
badge(680, 812, 13)
arrow([(335, 216), (335, 474)], pending=True)

# ------------------------------------------------------------------------------------------- how it works
X0 = 1510
text(X0, 52, "DEPTH on AWS", 26, INK, "start", weight="700")
text(X0, 78, "side-scan sonar in, a cleanup plan a person approves out", 13.5, "#545b64", "start")
text(X0, 112, "How it works", 16, INK, "start", weight="700")
steps = [
    "Users, AI agents (MCP) and GIS tools (OGC API) reach https://13-216-140-101.sslip.io on an Elastic IP.",
    "Caddy on the instance terminates TLS (Let's Encrypt) and forwards to the DEPTH server.",
    "The DEPTH server (FastAPI) serves the studio, REST, MCP and OGC, with rate limits and an approver PIN.",
    "The agent runs its core loop with OpenCV 5: check the promise applies, choose how hard to look, "
    "perceive, decide, plan.",
    "Survey results go to Amazon S3. Secrets come from SSM Parameter Store; updates arrive by Run Command, "
    "never SSH.",
    "Each action the agent wants becomes one approval request on an SNS topic.",
    "SNS emails the request to the approver (today the owner's address; APPROVER_EMAIL names someone else).",
    "The approver decides in the studio with a PIN (locked after wrong PINs). The agent re-plans.",
    "Alarms (instance status, CPU, the Route 53 health check of the link) and EventBridge rules alert the "
    "owner through SNS.",
    "EventBridge Scheduler stops the instance on 10 Nov 2026. Budgets and Cost Anomaly Detection watch "
    "spend; EBS snapshots run daily.",
    "One CloudFormation stack creates all of it except the account budget, with scoped IAM roles.",
    "At boot the server downloads the detector from the GitHub Release (SHA-256 checked).",
    "The benchmark fleet runs the same workload on x86 and Graviton4 with stock OpenCV 5.",
]
y = 142
for n, s in enumerate(steps, 1):
    badge(X0 + 11, y + 1, n)
    wrapped = textwrap.wrap(s, 48)
    text(X0 + 32, y + 5, wrapped, 12.5, INK, "start", gap=1.32)
    y += 16.5 * len(wrapped) + 14
y += 6
out.append(f'<rect x="{X0}" y="{y}" width="390" height="104" fill="#f6f8fa" stroke="{PEND}" '
           f'stroke-dasharray="6 4"/>')
text(X0 + 14, y + 22, "Not live yet (drawn grey)", 13, INK, "start", weight="700")
text(X0 + 14, y + 44, ["CloudFront: waiting on AWS account verification",
                       "COOL AMI: waiting on a Marketplace payment method",
                       "Bedrock brief writer: not deployed (template served)"], 12.5, "#545b64", "start", gap=1.36)

text(150, 1168, "Drawn from infra/cfn/depth.yaml, infra/README.md and the live server on 25 Oct 2026 · "
     "source: docs/diagram/depth_architecture.py", 12, "#8c959f", "start")
out.append("</svg>")
OUT_SVG.write_text("\n".join(out), encoding="utf-8")
print("wrote", OUT_SVG)

# ------------------------------------------------------------------------------------------- PNG via headless Chrome
chrome = next((p for p in (os.environ.get("CHROME", ""), r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                           r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                           "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                           shutil.which("google-chrome") or "", shutil.which("chromium") or "") if p and Path(p).exists()),
              None)
if chrome:
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--window-size={W},{H}", f"--screenshot={OUT_PNG}", OUT_SVG.as_uri()],
                   check=False, capture_output=True, timeout=120)
    print("wrote", OUT_PNG) if OUT_PNG.exists() else print("PNG export failed; the SVG is the master copy")
else:
    print("no Chrome/Edge found: the SVG is the master copy")
