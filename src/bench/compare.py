"""
compare.py — turn the per-machine benchmark JSONs into the COOL evidence report (+ chart).

Reads ``benchmarks/<label>__<workload>.json`` (``product_bench.py``), ``<label>__parity.json`` and
``<label>__receipt.json``, and writes ``docs/reports/cool_benchmark.md``:

* one row per run — machine (architecture · instance type), OpenCV provenance (COOL only when ``cv2`` was
  loaded from ``/opt/cool``), threads, p50/p95, throughput across repeats, CPU use, peak memory, cost;
* **where the time goes** — the product frame split into OpenCV image ops vs the ``cv2.dnn`` forward;
* **a real survey, measured** — a raw recording end to end: real-time factor and $ per survey-hour;
* the comparisons the award asks for — **(B) → (C)** Graviton stock → Graviton COOL on identical
  hardware (COOL's own contribution) and **(A) → (C)** x86 → Graviton + COOL (latency and cost);
* **same answers?** — output parity between machines (class, IoU ≥ 0.9, |Δscore| ≤ 0.01, same tier);
* **machines** — instance type, AMI, Marketplace product code and the COOL build, as read from each
  machine, and whether its identity is AWS-signed (``verify_cool`` checks the receipt);
* the cheapest Graviton + COOL size that meets the product target (< 300 ms p95 per frame, ≥ 5 fps).

    python -m src.bench.compare [--dir benchmarks] [--chart docs/img/cool_benchmark.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORDER = ("baseline_x86", "graviton_stock", "graviton_cool")
WORKLOADS = ("product", "cv", "detect", "survey")
OPS = ("decode", "stage1", "dnn_pre", "dnn_forward", "dnn_post", "evidence", "render")
TARGET_P95_MS, TARGET_FPS = 300.0, 5.0                     # the proposal's product targets
PARITY = {"iou": 0.9, "dscore": 0.01}


def _order(label: str) -> int:
    for i, o in enumerate(ORDER):
        if label == o or label.startswith(o + "_"):
            return i
    return 9


def load(d: Path) -> list[dict]:
    runs = []
    for p in sorted(d.glob("*__*.json")):
        if p.stem.endswith(("__parity", "__receipt")):
            continue
        r = json.loads(p.read_text())
        if r.get("workload") in WORKLOADS:
            runs.append(r)
    return sorted(runs, key=lambda r: (WORKLOADS.index(r["workload"]), _order(r["label"]), r["label"]))


def load_parity(d: Path) -> dict[str, dict]:
    return {p.stem[:-len("__parity")]: json.loads(p.read_text()) for p in sorted(d.glob("*__parity.json"))}


def _machine(r: dict) -> str:
    fp = r["fingerprint"]
    ec2 = fp.get("ec2") or {}
    where = ec2.get("instance-type") or fp["host"].get("cpu_model", "")[:28]
    return f"{fp['host']['machine']} · {where}"


def _ocv(r: dict) -> str:
    o = r["fingerprint"]["opencv"]
    return f"{o['version']} {'**COOL**' if o['is_cool_path'] else 'stock'}"


def _row(r: dict) -> str:
    st, c = r["stages_ms"], r.get("cost") or {}
    sh = c.get("survey_hour") or {}
    usd1k = f"${c['usd_per_1k_frames']:.4f}" if c.get("usd_per_1k_frames") is not None else "—"
    ush = f"${sh['usd']:.5f}" if sh.get("usd") is not None else "—"
    rng = r.get("throughput_fps_range")
    fps = f"{r['throughput_fps']}" + (f" ({rng[0]}–{rng[1]})" if rng and rng[0] != rng[1] else "")
    cpu = f"{r['cpu_util']:.0%}" if r.get("cpu_util") is not None else "—"
    rss = f"{r['peak_rss_mb']:.0f} MB" if r.get("peak_rss_mb") is not None else "—"
    reps = (r.get("config") or {}).get("repeats", 1)
    return (f"| `{r['label']}` | {r['workload']} | {_machine(r)} | {_ocv(r)} | {r['runtime']['threads']} "
            f"| **{st['total']['p50']:.1f}** / {st['total']['p95']:.1f} | {fps} | {reps} | {cpu} | {rss} "
            f"| {usd1k} | {sh.get('compute_s', 0):.1f} s · {ush} |")


def _ops_row(r: dict) -> str:
    ops = r.get("ops_ms") or {}
    tot = r["stages_ms"]["total"]["mean"]
    cells = " | ".join(f"{ops[k]['mean']:.1f}" if k in ops else "—" for k in OPS)
    share = f"{ops['dnn_forward']['mean'] / tot:.0%}" if "dnn_forward" in ops and tot else "—"
    return f"| `{r['label']}` | {_machine(r)} | {_ocv(r)} | {cells} | **{tot:.1f}** | {share} |"


def _survey_row(r: dict) -> str:
    s, cfg = r["survey"], r.get("config") or {}
    usd = f"${s['usd_per_survey_hour']:.5f}" if s.get("usd_per_survey_hour") is not None else "—"
    lo, hi = s.get("real_time_factor_range", [None, None])
    return (f"| `{r['label']}` | {_machine(r)} | {_ocv(r)} | `{Path(cfg.get('recording', '?')).name}` "
            f"| {s['sonar_duration_s']:.0f} s · {s['frames']} frames | **{s['real_time_factor_median']:.4f}** "
            f"({lo:.4f}–{hi:.4f}) | {s['compute_s_per_survey_hour']:.0f} s | {usd} |")


def _ratio(a: dict, b: dict, key: str = "p50") -> str:
    x, y = a["stages_ms"]["total"][key], b["stages_ms"]["total"][key]
    return f"{x / y:.2f}×" if y else "—"


def _rep_values(r: dict) -> list[float]:
    """Per-repeat headline value: frame p50 (ms) for frame workloads, real-time factor for the survey."""
    reps = r.get("repeats") or []
    if r["workload"] == "survey":
        return [x["real_time_factor"] for x in reps if "real_time_factor" in x]
    return [x["total_ms"]["p50"] for x in reps if "total_ms" in x]


def separation(slow: dict, fast: dict) -> str:
    """The registered claim rule (STUDY-17): a speed-up is claimed only when the repeat ranges do not
    overlap (the fast run's worst repeat beats the slow run's best); otherwise the ratio is reported
    as within noise."""
    a, b = _rep_values(slow), _rep_values(fast)
    if len(a) < 2 or len(b) < 2:
        return "single repeat - no claim"
    if max(b) < min(a):
        return "repeat ranges separate"
    if max(a) < min(b):
        return "repeat ranges separate (slower)"
    return "repeat ranges overlap - within noise, no claim"


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def parity(ref: dict, other: dict, iou_min: float = PARITY["iou"], dscore_max: float = PARITY["dscore"]) -> dict:
    """Greedy same-class matching per frame; a match needs IoU ≥ iou_min, |Δscore| ≤ dscore_max, same tier."""
    n = ok = 0
    max_d, extra, missing, worst = 0.0, 0, 0, None
    same_frames = ref.get("input_manifest_sha256") == other.get("input_manifest_sha256")
    for fid, dets in ref["frames"].items():
        pool = list(other["frames"].get(fid, []))
        for d in dets:
            n += 1
            best, bi = None, 0.0
            for j, o in enumerate(pool):
                if o["cls"] == d["cls"]:
                    v = _iou(d["bbox"], o["bbox"])
                    if v > bi:
                        best, bi = j, v
            if best is None or bi < iou_min:
                missing += 1
                worst = worst or f"{fid}: {d['cls']} {d['bbox']} has no counterpart"
                continue
            o = pool.pop(best)
            dc = abs(o["conf"] - d["conf"])
            max_d = max(max_d, dc)
            if dc <= dscore_max and o["tier"] == d["tier"]:
                ok += 1
            else:
                worst = worst or f"{fid}: {d['cls']} score {d['conf']} vs {o['conf']}, tier {d['tier']} vs {o['tier']}"
        extra += len(pool)
    return {"detections": n, "match": ok, "missing": missing, "extra": extra, "max_dscore": round(max_d, 5),
            "same_frames": same_frames, "first_problem": worst}


def cool_effect(runs: list[dict]) -> list[str]:
    """Where COOL acts (the audit's strict note): Graviton stock (B) → Graviton COOL (C) per workload and per
    OpenCV op, the share of the stock product frame that is the ``cv2.dnn`` forward, and a headline chosen by
    the registered claim rule — the workload with the largest speed-up whose repeat ranges separate. COOL can
    only move the image ops unless its build also speeds up the forward; the section says which happened."""
    by = {(r["label"], r["workload"]): r for r in runs}
    rows, best = [], None
    for wl in ("cv", "product", "detect", "survey"):
        b, c = by.get(("graviton_stock", wl)), by.get(("graviton_cool", wl))
        if not (b and c):
            continue
        if wl == "survey":
            x, y = b["survey"]["real_time_factor_median"], c["survey"]["real_time_factor_median"]
        else:
            x, y = b["stages_ms"]["total"]["p50"], c["stages_ms"]["total"]["p50"]
        ratio, sep = (x / y if y else None), separation(b, c)
        rows.append(f"| `{wl}` | {x:.4g} | {y:.4g} | {ratio:.2f}× | {sep} |" if ratio else f"| `{wl}` | {x} | {y} | — | {sep} |")
        if ratio and sep == "repeat ranges separate" and (best is None or ratio > best[1]):
            best = (wl, ratio)
    if not rows:
        return []
    out = ["", "## Where COOL acts — Graviton stock (B) → Graviton COOL (C), identical hardware", "",
           "| workload | stock (ms p50, or real-time factor) | COOL | speed-up | claim rule |", "|---|--:|--:|--:|---|"] + rows
    bp, cp = by.get(("graviton_stock", "product")), by.get(("graviton_cool", "product"))
    if bp and cp and bp.get("ops_ms") and cp.get("ops_ms"):
        ob, oc = bp["ops_ms"], cp["ops_ms"]
        out += ["", "Per OpenCV op of the product frame (mean ms):", "", "| op | stock | COOL | speed-up |", "|---|--:|--:|--:|"]
        for op in OPS:
            if op in ob and op in oc and oc[op]["mean"]:
                out.append(f"| `{op}` | {ob[op]['mean']:.1f} | {oc[op]['mean']:.1f} | {ob[op]['mean'] / oc[op]['mean']:.2f}× |")
        tot = bp["stages_ms"]["total"]["mean"]
        fwd_b, fwd_c = ob.get("dnn_forward", {}).get("mean"), oc.get("dnn_forward", {}).get("mean")
        if fwd_b and fwd_c and tot:
            share, fr = fwd_b / tot, fwd_b / fwd_c
            out += ["", f"The `cv2.dnn` forward is **{share:.0%}** of the stock product frame; COOL changed it "
                        f"**{fr:.2f}×**. " + ("COOL's build speeds up the network too, so the product frame moves with it."
                                              if fr >= 1.05 else
                                              f"The forward is unchanged, so on the product frame COOL can move at most the "
                                              f"other {1 - share:.0%} — the `cv` workload (image ops alone) is where its "
                                              "effect shows.")]
    out += ["", ("**Lead with `" + best[0] + f"`**: COOL runs it **{best[1]:.2f}×** faster on identical Graviton, and "
                 "the repeat ranges separate (the registered claim rule)." if best else
                 "**No workload shows a separated COOL speed-up** — reported as within noise (the registered claim "
                 "rule), not as a gain.")]
    return out


def _parity_rows(par: dict[str, dict]) -> list[str]:
    rows = []
    for ref, others in (("graviton_stock", ("graviton_cool",)), ("baseline_x86", ("graviton_stock", "graviton_cool"))):
        for o in others:
            if ref in par and o in par:
                p = parity(par[ref], par[o])
                verdict = "**same answers**" if p["match"] == p["detections"] and not p["extra"] else "differs"
                rows.append(f"| `{ref}` → `{o}` | {p['detections']} | {p['match']} / {p['detections']} · {verdict} "
                            f"| {p['max_dscore']:.5f} | {p['missing']} / {p['extra']} "
                            f"| {'yes' if p['same_frames'] else '**NO**'} | {p['first_problem'] or '—'} |")
    return rows


def _machines(runs: list[dict], d: Path) -> list[str]:
    rows, seen = [], set()
    for r in runs:
        if r["label"] in seen:
            continue
        seen.add(r["label"])
        fp = r["fingerprint"]
        att = (fp.get("cool") or {}).get("attestation") or {}
        build = (fp.get("cool") or {}).get("build") or {}
        ver = "; ".join(f"{Path(k).name}: {v[:60]}" for k, v in (build.get("version_files") or {}).items()) \
            or "; ".join(build.get("dpkg") or []) or "; ".join(x for x in (build.get("python_dists") or []) if "opencv" in x.lower())
        receipt = d / f"{r['label']}__receipt.json"
        rows.append(f"| `{r['label']}` | {att.get('instance_type') or (fp.get('ec2') or {}).get('instance-type') or '—'} "
                    f"| {att.get('architecture') or fp['host']['machine']} | {att.get('image_id') or (fp.get('ec2') or {}).get('ami-id') or '—'} "
                    f"| {', '.join(att.get('marketplace_product_codes') or []) or '—'} "
                    f"| {'yes (`' + receipt.name + '`)' if receipt.exists() else 'no'} | {ver or '—'} |")
    return rows


def report(runs: list[dict], par: dict[str, dict] | None = None, d: Path | None = None) -> str:
    par, d = par or {}, d or REPO / "benchmarks"
    L = ["# COOL benchmark — the DEPTH product workload, per stage",
         "",
         "_Generated by `python -m src.bench.compare` from `benchmarks/*.json` (`python -m src.bench.product_bench`)._",
         "",
         "Workload `product` = what `/api/analyze` runs per frame in the standard look (1 forward pass): JPEG "
         "decode → Stage 1 (canonicalise + bottom track) → YOLO11 via `cv2.dnn` → evidence + calibrated tiers → "
         "overlay + JPEG encode; rows labelled `…_lh` are the look-harder mode (+ the 2×2 tile pass, 5 forward "
         "passes), which an idle server's analyze picks when its 2-s budget allows. `cv` = the product path's "
         "OpenCV image ops without the network. `survey` = a raw sonar recording end to end (read → sonogram → "
         "seabed cross-check → frame agent → plan). A run counts as **COOL** "
         "only if `cv2.__file__` is under `/opt/cool` (provenance, not \"KleidiCV in the build\", which the "
         "stock Arm wheel also reports).",
         "",
         "| run | workload | machine | OpenCV | threads | total ms p50 / p95 | fps (range over repeats) | repeats | CPU | peak RSS | $ / 1k frames | modelled survey-hour: compute · $ |",
         "|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|---|"]
    L += [_row(r) for r in runs if r["workload"] != "survey"]

    prod = [r for r in runs if r["workload"] == "product" and r.get("ops_ms")]
    if prod:
        L += ["", "## Where the time goes — product frame, mean ms per op", "",
              "`dnn_pre` = letterbox + `blobFromImage`, `dnn_forward` = `cv2.dnn` forward (all passes incl. "
              "re-looks), `dnn_post` = decode + `NMSBoxesBatched`, `evidence` = the rest of Prove/Decide. "
              "COOL can speed up both the image ops and the forward; this split shows which one decides the frame time.",
              "", "| run | machine | OpenCV | " + " | ".join(OPS) + " | total | network share |",
              "|---|---|---|" + "--:|" * (len(OPS) + 2)]
        L += [_ops_row(r) for r in prod]

    surv = [r for r in runs if r["workload"] == "survey" and r.get("survey")]
    if surv:
        L += ["", "## A real survey, measured — raw recording end to end", "",
              "Real-time factor = compute seconds ÷ seconds of sonar (< 1 ⇒ keeps up with the boat). "
              "$ per survey-hour = real-time factor × the machine's $/h. Measured on a real Humminbird "
              "recording (PINGMapper sample data), not modelled.",
              "", "| run | machine | OpenCV | recording | sonar | real-time factor (median, range) | compute per survey-hour | $ per survey-hour |",
              "|---|---|---|---|---|--:|--:|--:|"]
        L += [_survey_row(r) for r in surv]

    by = {(r["label"], r["workload"]): r for r in runs}
    cmp = []
    for wl in WORKLOADS:
        a, b, c = by.get(("baseline_x86", wl)), by.get(("graviton_stock", wl)), by.get(("graviton_cool", wl))
        if wl == "survey":
            if b and c:
                x, y = b["survey"]["real_time_factor_median"], c["survey"]["real_time_factor_median"]
                cmp.append(f"| survey | **COOL on identical Graviton** (B → C) | {x / y:.2f}× faster per survey-hour "
                           f"| {separation(b, c)} |")
            if a and c:
                x, y = a["survey"]["real_time_factor_median"], c["survey"]["real_time_factor_median"]
                ua, uc = a["survey"].get("usd_per_survey_hour"), c["survey"].get("usd_per_survey_hour")
                cost = f" · {ua / uc:.2f}× cheaper per survey-hour" if ua and uc else ""
                cmp.append(f"| survey | **x86 → Graviton + COOL** (A → C) | {x / y:.2f}× faster{cost} | {separation(a, c)} |")
            continue
        if b and c:
            cmp.append(f"| {wl} | **COOL on identical Graviton** (B → C) | {_ratio(b, c)} faster p50 · {_ratio(b, c, 'p95')} p95 "
                       f"| {separation(b, c)} |")
        if a and c:
            ca, cc = (a["cost"].get("survey_hour") or {}).get("usd"), (c["cost"].get("survey_hour") or {}).get("usd")
            cost = f" · {ca / cc:.2f}× cheaper per survey-hour" if ca and cc else ""
            cmp.append(f"| {wl} | **x86 → Graviton + COOL** (A → C) | {_ratio(a, c)} faster p50{cost} | {separation(a, c)} |")
    L += cool_effect(runs)
    if cmp:
        L += ["", "## Comparisons", "",
              "A ratio > 1 means the second machine is faster. Registered claim rule (STUDY-17): a speed-up is "
              "claimed only when the per-repeat values do not overlap.", "",
              "| workload | comparison | result | claim rule |", "|---|---|---|---|"] + cmp
    else:
        L += ["", "_A/B/C comparison appears once the EC2 runs (`infra/bench_fleet.sh`) are in `benchmarks/`._"]

    prow = _parity_rows(par)
    if prow:
        L += ["", "## Same answers? — output parity", "",
              f"Every detection of the reference run matched by class, IoU ≥ {PARITY['iou']}, |Δscore| ≤ "
              f"{PARITY['dscore']} and the same tier (never bit-identical: float order differs across CPUs).",
              "", "| reference → run | detections | match | max abs Δscore | missing / extra | same input frames | first difference |",
              "|---|--:|---|--:|--:|---|---|"] + prow

    mrows = _machines(runs, d)
    if any(r["fingerprint"].get("ec2") for r in runs):
        L += ["", "## Machines — read from each machine", "",
              "Instance type, architecture, AMI and Marketplace product code come from the AWS-signed instance identity "
              "document; check any receipt with `python -m src.bench.verify_cool benchmarks/<label>__receipt.json`.",
              "", "| run | instance type | arch | AMI | Marketplace product code | AWS-signed receipt | COOL build (read from the machine) |",
              "|---|---|---|---|---|---|---|"] + mrows

    cool_prod = [r for r in runs if r["workload"] == "product" and r["fingerprint"]["opencv"]["is_cool_path"]
                 and not (r.get("config") or {}).get("threads_requested")]          # sizes, not the 1-thread runs
    meets = [r for r in cool_prod if r["stages_ms"]["total"]["p95"] < TARGET_P95_MS and r["throughput_fps"] >= TARGET_FPS]
    if cool_prod:
        priced = sorted(meets, key=lambda r: r["cost"].get("price_hour_usd") or 1e9)
        L += ["", f"**Cheapest Graviton + COOL size meeting the target** (< {TARGET_P95_MS:.0f} ms p95 per frame, "
                  f"≥ {TARGET_FPS:.0f} fps): " + (f"`{priced[0]['label']}` ({_machine(priced[0])}, "
                                                  f"${priced[0]['cost'].get('price_hour_usd')}/h)" if priced
                                                  else "none of the measured sizes — see the table.")]
    L += ["", "**Modelled survey-hour** (per-frame rows): frames per survey-hour = ping rate × 3600 / pings per frame × "
          "channels (assumptions recorded in each JSON; default 15 Hz, 640 pings/frame, port + starboard). The "
          "`survey` rows measure it on a real recording instead.",
          "", "---", "_Numbers in this file are regenerated by the script; do not edit by hand._", ""]
    return "\n".join(L)


def chart(runs: list[dict], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - chart skipped")
        return
    # the chart compares machines on the standard look: look-harder rows (5 passes) and rows without the
    # per-op split stay in the report's tables, not here, so they cannot squash the bars being compared
    prod = [r for r in runs if r["workload"] == "product" and r.get("ops_ms")
            and (r.get("config") or {}).get("perception_mode") != "look_harder"]         or [r for r in runs if r["workload"] != "survey"]
    if not prod:
        return
    keys = [k for k in OPS if k in (prod[0].get("ops_ms") or {})] or \
        [k for k in prod[0]["stages_ms"] if k != "total"]
    src = "ops_ms" if prod[0].get("ops_ms") else "stages_ms"
    # validated categorical slots (dataviz reference palette, light mode; the table in the report is
    # the relief for the sub-3:1 slots). Stacked MEANS — means add up, p50s do not.
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7d5bd6", "#3ab0c8"]
    ink, ink2, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(7.5, 1.1 + 0.55 * len(prod)), facecolor=surface)
    ax.set_facecolor(surface)
    for i, r in enumerate(prod):
        left = 0.0
        for j, k in enumerate(keys):
            v = (r.get(src) or {}).get(k, {}).get("mean", 0.0)
            ax.barh(i, v, left=left, color=colors[j % len(colors)], label=k if i == 0 else None, height=0.56,
                    edgecolor=surface, linewidth=2)
            left += v
        ax.text(left + 4, i, f"{r['stages_ms']['total']['mean']:.0f} ms", va="center", fontsize=8, color=ink)
    ax.set_yticks(range(len(prod)), [f"{r['label']} · {_machine(r).split(' · ')[-1]}" for r in prod], fontsize=8, color=ink)
    ax.invert_yaxis()
    ax.set_xlabel("mean latency per frame (ms), stacked by op", color=ink2, fontsize=8)
    ax.tick_params(colors=ink2, labelsize=8)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    leg = ax.legend(ncol=min(len(keys), 4), fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    for t in leg.get_texts():
        t.set_color(ink2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=surface)
    plt.close(fig)
    print(f"chart -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dir", default=str(REPO / "benchmarks"))
    ap.add_argument("--out", default=str(REPO / "docs" / "reports" / "cool_benchmark.md"))
    ap.add_argument("--chart", default=str(REPO / "docs" / "img" / "cool_benchmark.png"))
    a = ap.parse_args()
    d = Path(a.dir)
    runs = load(d)
    if not runs:
        raise SystemExit(f"no benchmark JSONs in {a.dir}")
    Path(a.out).write_text(report(runs, load_parity(d), d), encoding="utf-8")
    print(f"report -> {a.out} ({len(runs)} runs)")
    if a.chart:
        chart(runs, Path(a.chart))


if __name__ == "__main__":
    main()
