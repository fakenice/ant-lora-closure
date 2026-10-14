"""Build the four data-ratio configs used by the ablation.

Fixed budget N per config so that 'effect of ratio' is not confounded with
'effect of sample count'. Sources:
  D = cleaned Chinese open-ended medical dialog QA  (data/clean/dialog_zh.jsonl)
  M = cleaned CMExam MCQ (question+options -> gold letter) (data/clean/mcq_train_pool.jsonl)

Configs (D:M): 100:0, 70:30, 30:70, 0:100. Sampling is seeded and eval questions
were already removed from the MCQ pool by 02_clean_dedup.py.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "data" / "ratios"

SEED = 20261014
N_PER_CONFIG = 1200
RATIOS = [("r100_0", 1.00), ("r70_30", 0.70), ("r30_70", 0.30), ("r0_100", 0.00)]

SYSTEM = "你是一名严谨的医学助手。请用中文回答。"


def load(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.endswith("}"):
                rows.append(json.loads(line))
    return rows


def to_sample_dialog(r):
    return {
        "id": r["id"],
        "source": "dialog",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["response"]},
        ],
    }


def to_sample_mcq(r):
    return {
        "id": r["id"],
        "source": "mcq",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["gold"]},
        ],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dialog = load(CLEAN / "dialog_zh.jsonl")
    mcq = load(CLEAN / "mcq_train_pool.jsonl")
    print(f"pool: dialog={len(dialog)} mcq={len(mcq)}")

    meta = {"seed": SEED, "n_per_config": N_PER_CONFIG, "pool_sizes": {"dialog": len(dialog), "mcq": len(mcq)},
            "configs": []}
    for name, d_frac in RATIOS:
        rng = random.Random(f"{SEED}-{name}")
        n_d = int(round(N_PER_CONFIG * d_frac))
        n_m = N_PER_CONFIG - n_d
        if n_d > len(dialog) or n_m > len(mcq):
            raise SystemExit(f"[FAIL] config {name} needs {n_d} dialog + {n_m} mcq, "
                             f"pools are {len(dialog)}/{len(mcq)}")
        pick_d = rng.sample(dialog, n_d)
        pick_m = rng.sample(mcq, n_m)
        rows = [to_sample_dialog(r) for r in pick_d] + [to_sample_mcq(r) for r in pick_m]
        rng.shuffle(rows)
        # unique ids so nothing is double-counted across configs
        for i, r in enumerate(rows):
            r["uid"] = f"{name}#{i}"
        (OUT / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        meta["configs"].append({"name": name, "dialog_frac": d_frac, "n_dialog": n_d, "n_mcq": n_m,
                                "written": len(rows)})
        print(f"  {name}: dialog={n_d} mcq={n_m}")

    (OUT / "ratios_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nconfigs -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
