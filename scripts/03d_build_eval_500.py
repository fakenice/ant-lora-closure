"""Build a 500-item eval set without reintroducing train/eval leakage.

The original 100-item set removed 5 duplicates from the training pool, but the
remaining 46,421 training rows were never checked against questions outside those
100. Simply sampling 400 more test items would therefore reintroduce leakage.
This script keeps the original 100 as-is and adds 400 valid test questions whose
normalized-question hash is absent from both the original eval set and the existing
training pool.

Usage: .venv/Scripts/python.exe scripts/03d_build_eval_500.py
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
STATS = ROOT / "data" / "stats"
SEED_500 = 20261103
N_TOTAL = 500

ZH = re.compile(r"[一-鿿]")
LAT = re.compile(r"[A-Za-z]")


def zh_ratio(s: str) -> float:
    if not s:
        return 0.0
    n_zh = len(ZH.findall(s))
    n_lat = len(LAT.findall(s))
    denom = n_zh + n_lat
    return n_zh / denom if denom else 0.0


def qhash(q: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", "", q).encode("utf-8")).hexdigest()[:16]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.endswith("}"):
                rows.append(json.loads(line))
    return rows


def valid(r: dict) -> dict | None:
    """Same validation as scripts/02_clean_dedup.py:process_mcq."""
    if not isinstance(r, dict):
        return None
    q = (r.get("Question") or "").strip()
    opts = r.get("Options") or []
    ans = (r.get("Answer") or "").strip().upper()
    if not (q and isinstance(opts, list) and len(opts) >= 2):
        return None
    letters, lines = [], []
    for o in opts:
        k = str(o.get("key", "")).strip().upper()
        v = str(o.get("value", "")).strip()
        if not k or not v:
            return None
        letters.append(k)
        lines.append(f"{k}. {v}")
    if ans not in letters:
        return None
    if len(q) < 10 or zh_ratio(q) < 0.4:
        return None
    return {
        "id": None, "kind": "mcq", "question": q, "letters": letters,
        "gold": ans, "options": lines, "explanation": (r.get("Explanation") or "").strip(),
        "prompt": "以下是一道中国执业医师资格考试单项选择题，请只回答正确选项的字母。\n"
                  + q + "\n" + "\n".join(lines),
    }


def main() -> int:
    original = load_jsonl(CLEAN / "mcq_eval_100.jsonl")
    te_ok = []
    for r in load_jsonl(RAW / "cmexam_test.jsonl"):
        v = valid(r)
        if v is not None:
            te_ok.append(v)

    train_hashes = {qhash(r["question"]) for r in load_jsonl(CLEAN / "mcq_train_pool.jsonl")}
    original_hashes = {qhash(r["question"]) for r in original}

    candidates, seen = [], set()
    n_train_overlap = 0
    for r in te_ok:
        h = qhash(r["question"])
        if h in original_hashes or h in seen:
            continue
        if h in train_hashes:
            n_train_overlap += 1
            continue
        seen.add(h)
        candidates.append(r)

    need = N_TOTAL - len(original)
    if len(candidates) < need:
        raise SystemExit(f"only {len(candidates)} leak-free candidates, need {need}")
    rng = random.Random(SEED_500)
    added = rng.sample(candidates, need)

    rows = []
    for i, r in enumerate(original):
        row = dict(r)
        row["id"] = f"e{i}"
        rows.append(row)
    for j, r in enumerate(added, start=len(original)):
        row = dict(r)
        row["id"] = f"e{j}"
        rows.append(row)

    if len(rows) != N_TOTAL:
        raise SystemExit(f"built {len(rows)} rows, expected {N_TOTAL}")
    if len({qhash(r["question"]) for r in rows}) != N_TOTAL:
        raise SystemExit("duplicate question in the built eval set")
    leak = [r["id"] for r in rows if qhash(r["question"]) in train_hashes]
    if leak:
        raise SystemExit(f"train/eval leakage after build: {leak[:5]}")

    CLEAN.mkdir(parents=True, exist_ok=True)
    (CLEAN / "mcq_eval_500.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    report = {
        "seed": SEED_500,
        "raw_test": len(load_jsonl(RAW / "cmexam_test.jsonl")),
        "valid_test": len(te_ok),
        "original_100_preserved": len(original),
        "new_candidates_after_leak_filter": len(candidates),
        "test_questions_excluded_by_train_overlap": n_train_overlap,
        "added": len(added),
        "total": len(rows),
        "train_overlap_after_build": 0,
        "answer_letter_distribution": dict(Counter(r["gold"] for r in rows)),
        "n_options_distribution": dict(Counter(len(r["letters"]) for r in rows)),
        "generated_by": "scripts/03d_build_eval_500.py",
    }
    STATS.mkdir(parents=True, exist_ok=True)
    (STATS / "eval500_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
