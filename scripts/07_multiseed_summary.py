"""Aggregate the 500-item multi-seed runs.

Reads out/eval/eval_<tag>.json and the per-item JSONL files, then reports:
  * per-seed accuracy, Wilson interval, wrong/unparseable counts;
  * seed mean ± SD per ratio config;
  * a paired McNemar test against the base model on the same 500 questions;
  * accuracy on the original 100-item subset (ids e0..e99) for continuity.

Usage: .venv/Scripts/python.exe scripts/07_multiseed_summary.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
EVAL_DIR = OUT / "eval"
CONFIGS = ["r100_0", "r70_30", "r30_70", "r0_100"]
SEEDS = [20261014, 20261015, 20261016]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_items(path: Path):
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.endswith("}"):
            rows.append(json.loads(line))
    return rows


def wilson(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return round(min(1.0, 2 * tail), 6)


def main() -> int:
    base = load_json(EVAL_DIR / "eval_base_500.json")
    base_items = load_items(EVAL_DIR / "eval_base_500_items.jsonl")
    if base is None or base_items is None:
        raise SystemExit("base_500 eval missing; run scripts/run_multiseed_500.sh")
    base_ok = {r["id"] for r in base_items if r["outcome"] == "correct"}

    rows = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            tag = f"{cfg}_s{seed}_500"
            summary = load_json(EVAL_DIR / f"eval_{tag}.json")
            items = load_items(EVAL_DIR / f"eval_{tag}_items.jsonl")
            if summary is None or items is None:
                print(f"[warn] missing {tag}")
                continue
            ok = {r["id"] for r in items if r["outcome"] == "correct"}
            first100 = sum(1 for i in range(100) if f"e{i}" in ok)
            rows.append({
                "config": cfg, "seed": seed, "tag": tag,
                "n": summary["n_items"], "correct": summary["correct"],
                "accuracy": summary["accuracy"], "wilson_95": summary["wilson_ci95"],
                "wrong": summary["wrong"], "unparseable": summary["unparseable"],
                "mean_new_tokens": summary["mean_new_tokens"],
                "s_per_item": summary["s_per_item"],
                "first100_correct": first100,
                "first100_accuracy": round(first100 / 100, 4),
                "base_accuracy": base["accuracy"],
                "delta_vs_base": round(summary["accuracy"] - base["accuracy"], 4),
                "paired_base_only": len(base_ok - ok),
                "paired_arm_only": len(ok - base_ok),
                "mcnemar_p_vs_base": mcnemar(len(ok - base_ok), len(base_ok - ok)),
            })

    summary_rows = []
    for cfg in CONFIGS:
        sub = [r for r in rows if r["config"] == cfg]
        if not sub:
            continue
        accs = [r["accuracy"] for r in sub]
        summary_rows.append({
            "config": cfg,
            "n_seeds": len(sub),
            "mean_accuracy": round(statistics.mean(accs), 4),
            "sd_accuracy": round(statistics.stdev(accs), 4) if len(accs) > 1 else None,
            "min_accuracy": min(accs), "max_accuracy": max(accs),
            "mean_delta_vs_base": round(statistics.mean(r["delta_vs_base"] for r in sub), 4),
            "mean_first100": round(statistics.mean(r["first100_accuracy"] for r in sub), 4),
            "min_mcnemar_p": min(r["mcnemar_p_vs_base"] for r in sub),
            "max_mcnemar_p": max(r["mcnemar_p_vs_base"] for r in sub),
        })

    (OUT / "multiseed_500_summary.json").write_text(
        json.dumps({"base": base, "runs": rows, "by_config": summary_rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    with (OUT / "multiseed_500_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["config"])
        w.writeheader()
        w.writerows(rows)

    L = ["# Multi-seed 500-item evaluation", "",
         f"Base: {base['correct']}/{base['n_items']} = {100*base['accuracy']:.1f}% "
         f"(Wilson {100*base['wilson_ci95'][0]:.1f}–{100*base['wilson_ci95'][1]:.1f}%)", "",
         "## Per-seed", "",
         "| config | seed | correct | accuracy | Δ vs base | paired p | unparseable | first-100 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['config']} | {r['seed']} | {r['correct']}/{r['n']} | "
                 f"{100*r['accuracy']:.1f}% | {100*r['delta_vs_base']:+.1f} pt | "
                 f"{r['mcnemar_p_vs_base']:.3f} | {r['unparseable']} | "
                 f"{r['first100_correct']}/100 |")
    L += ["", "## By config (mean ± SD across seeds)", "",
          "| config | mean | SD | range | mean Δ | first-100 |",
          "|---|---|---|---|---|---|"]
    for r in summary_rows:
        sd = "—" if r["sd_accuracy"] is None else f"{100*r['sd_accuracy']:.1f} pt"
        L.append(f"| {r['config']} | {100*r['mean_accuracy']:.1f}% | {sd} | "
                 f"{100*r['min_accuracy']:.1f}–{100*r['max_accuracy']:.1f}% | "
                 f"{100*r['mean_delta_vs_base']:+.1f} pt | {100*r['mean_first100']:.1f}% |")
    L += ["", "Paired p is the exact McNemar test against base on the same 500 questions; "
          "a p above 0.05 is not evidence of a gain."]
    (OUT / "multiseed_500_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
