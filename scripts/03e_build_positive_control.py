"""Build the positive control and the shuffled-label control.

Positive control: train on the same 500 questions used for evaluation, with the
gold letter as the assistant answer. This is leakage *by design*. Its only job is
to show that the training + evaluation pipeline can detect a gain when the labels
are learnable; it is not a capability result and must never be reported as one.

Shuffled control: identical prompts, but the assistant gets a deterministic wrong
letter. If the positive control rises while this one stays near chance, the gain
is label-specific rather than a generic "answer with one letter" format effect.

Usage: .venv/Scripts/python.exe scripts/03e_build_positive_control.py
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "clean" / "mcq_eval_500.jsonl"
RATIOS = ROOT / "data" / "ratios"
STATS = ROOT / "data" / "stats"
SYSTEM = "你是一名严谨的医学助手。请用中文回答。"
SEED = 20261103


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.endswith("}")]


def main() -> int:
    rows = load(EVAL)
    if len(rows) != 500:
        raise SystemExit(f"expected 500 eval rows, got {len(rows)}")

    rng = random.Random(SEED)
    pos, shuf = [], []
    for i, r in enumerate(rows):
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": r["prompt"]},
        ]
        pos.append({"id": f"pc{i}", "source": "poscontrol",
                    "messages": msgs + [{"role": "assistant", "content": r["gold"]}],
                    "uid": f"poscontrol#{i}"})
        wrong = [x for x in r["letters"] if x != r["gold"]]
        picked = rng.choice(wrong)
        shuf.append({"id": f"ps{i}", "source": "poscontrol_shuffled",
                     "messages": msgs + [{"role": "assistant", "content": picked}],
                     "uid": f"poscontrol_shuffled#{i}"})

    RATIOS.mkdir(parents=True, exist_ok=True)
    for name, data in [("poscontrol_evaltrain", pos), ("poscontrol_shuffled", shuf)]:
        (RATIOS / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in data), encoding="utf-8")

    report = {
        "leaky_by_design": True,
        "purpose": "positive control: can the training/eval pipeline detect a learnable gain?",
        "eval_set": str(EVAL.relative_to(ROOT)),
        "n_rows": len(rows),
        "seed": SEED,
        "positive_label_distribution": dict(Counter(r["messages"][2]["content"] for r in pos)),
        "shuffled_label_distribution": dict(Counter(r["messages"][2]["content"] for r in shuf)),
        "warning": "Do not report positive-control accuracy as a model capability; it is trained on the test questions.",
    }
    STATS.mkdir(parents=True, exist_ok=True)
    (STATS / "positive_control_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
