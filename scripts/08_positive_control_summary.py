"""Summarize the leaky positive control and the shuffled-label control.

Reads out/eval/eval_{base,poscontrol_evaltrain,poscontrol_shuffled}_500.json and
the per-item files, then reports accuracy, Wilson interval, unparseable count and
an exact McNemar test against base. The positive control is deliberately leaky;
this script prints that warning into the artifact.
"""
from __future__ import annotations

import json
import math
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "out" / "eval"
OUT = ROOT / "out"
RUNS = [
    ("base", "base_500", "No SFT"),
    ("poscontrol_evaltrain", "poscontrol_evaltrain_500", "Trained on the 500 eval questions (leaky by design)"),
    ("poscontrol_shuffled", "poscontrol_shuffled_500", "Same questions, deterministic wrong label"),
]


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_items(p: Path):
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.endswith("}")]


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)
    return max(p, 1e-300)


def main() -> int:
    base = load_json(EVAL_DIR / "eval_base_500.json")
    base_items = load_items(EVAL_DIR / "eval_base_500_items.jsonl")
    if base is None or base_items is None:
        raise SystemExit("base_500 eval missing")
    base_ok = {r["id"] for r in base_items if r["outcome"] == "correct"}

    rows = []
    for adapter, tag, note in RUNS:
        s = load_json(EVAL_DIR / f"eval_{tag}.json")
        items = load_items(EVAL_DIR / f"eval_{tag}_items.jsonl")
        if s is None or items is None:
            print(f"[warn] missing {tag}")
            continue
        ok = {r["id"] for r in items if r["outcome"] == "correct"}
        rows.append({
            "adapter": adapter, "tag": tag, "note": note,
            "n_items": s["n_items"], "correct": s["correct"], "accuracy": s["accuracy"],
            "wilson_95": s["wilson_ci95"], "unparseable": s["unparseable"],
            "delta_vs_base": round(s["accuracy"] - base["accuracy"], 4),
            "base_only": len(base_ok - ok), "arm_only": len(ok - base_ok),
            "mcnemar_p_vs_base": mcnemar(len(ok - base_ok), len(base_ok - ok)),
        })

    (OUT / "positive_control_summary.json").write_text(
        json.dumps({"leaky_by_design": True, "runs": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    L = ["# Positive control (leaky by design)", "",
         "The positive-control adapter is trained on the same 500 questions it is evaluated on. "
         "Its accuracy is a harness check, not a capability result.", "",
         "| adapter | correct | accuracy | Wilson 95% | Δ vs base | unparseable | McNemar p |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        lo, hi = r["wilson_95"]
        L.append(f"| {r['adapter']} | {r['correct']}/{r['n_items']} | {100*r['accuracy']:.1f}% | "
                 f"{100*lo:.1f}–{100*hi:.1f}% | {100*r['delta_vs_base']:+.1f} pt | "
                 f"{r['unparseable']} | " + ("<1e-300" if r['mcnemar_p_vs_base'] <= 1e-300 else f"{r['mcnemar_p_vs_base']:.3g}") + " |")
    L += ["", "Interpretation: a near-ceiling positive control plus a near-chance shuffled control "
          "means the pipeline can detect a label-specific gain; it does not mean the model learned medicine."]
    (OUT / "positive_control_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
