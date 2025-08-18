#!/usr/bin/env python3
import json, os, glob

RESULTS_DIR = "results"
OUT = "LEADERBOARD.md"
rows = []
for path in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
    with open(path, "r", encoding="utf-8") as f:
        r = json.load(f)
    dataset = r.get("dataset", "?")
    task = r.get("task") or ("Motor Imagery" if dataset=="physionet_mi" else "—")
    model = r.get("model", "?")
    if "mean_acc" in r:
        metric, score = "Acc (LOSO)", r["mean_acc"]
    elif "mean_auroc" in r:
        metric, score = "AUROC (LOPO)", r["mean_auroc"]
    else:
        metric, score = r.get("metric","?"), r.get("score")
    rows.append((dataset, task, model, metric, score))

rows.sort(key=lambda x: (x[0], x[2]))
lines = ["# 🏆 EEGVERSE Leaderboard", "", "| Dataset | Task | Model | Metric | Score |", "|---------|------|-------|--------|-------|"]
for d,t,m,met,sc in rows:
    val = "—" if sc is None else (f"{sc:.3f}" if isinstance(sc,(float,int)) else str(sc))
    lines.append(f"| {d} | {t} | {m} | {met} | {val} |")
lines.append("")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Updated {OUT} with {len(rows)} entries.")
