"""Clean, language/length-filter, MinHash-dedup, and build the deterministic eval set.

Every filter stage is counted so the README can show an honest funnel instead of a
single 'we cleaned the data' sentence. Leakage between the MCQ train pool and the
eval set is checked by normalized-question hash and reported.
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
STATS = ROOT / "data" / "stats"

SEED = 20261014
EVAL_N = 100
LSH_THRESHOLD = 0.8
NUM_PERM = 128

ZH = re.compile(r"[一-鿿]")
LAT = re.compile(r"[A-Za-z]")


def zh_ratio(s: str) -> float:
    if not s:
        return 0.0
    n_zh = len(ZH.findall(s))
    n_lat = len(LAT.findall(s))
    denom = n_zh + n_lat
    return n_zh / denom if denom else 0.0


def shingles(text: str, k: int = 3):
    text = re.sub(r"\s+", "", text)
    if len(text) < k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for sh in shingles(text):
        m.update(sh.encode("utf-8"))
    return m


def qhash(q: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", "", q).encode("utf-8")).hexdigest()[:16]


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.endswith("}"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def dedup(records, text_fn):
    """MinHash LSH dedup preserving first occurrence; returns (kept, dup_count)."""
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    kept, dup = [], 0
    for i, r in enumerate(records):
        m = minhash(text_fn(r))
        try:
            matches = lsh.query(m)
        except ValueError:
            matches = []
        if matches:
            dup += 1
            continue
        try:
            lsh.insert(str(i), m)
        except ValueError:
            dup += 1
            continue
        kept.append(r)
    return kept, dup


def process_dialog(report):
    src = RAW / "medical_dialog_zh_train_head.jsonl"
    rows = load_jsonl(src)
    stage = Counter()
    stage["raw_records"] = len(rows)

    norm = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        instr = (r.get("instruction") or "").strip()
        inp = (r.get("input") or "").strip()
        out = (r.get("output") or "").strip()
        if not (instr and out):
            stage["drop_missing_field"] += 1
            continue
        prompt = instr if not inp else f"{instr}\n{inp}"
        if zh_ratio(out) < 0.5:
            stage["drop_low_zh_ratio"] += 1
            continue
        if not (20 <= len(out) <= 900):
            stage["drop_response_length"] += 1
            continue
        if len(prompt) > 1200:
            stage["drop_prompt_too_long"] += 1
            continue
        norm.append({"id": f"d{len(norm)}", "kind": "dialog", "prompt": prompt, "response": out})
    stage["after_filters"] = len(norm)
    kept, dup = dedup(norm, lambda x: x["response"])
    stage["minhash_duplicates"] = dup
    stage["final_kept"] = len(kept)
    report["dialog"] = {"source": src.name, "stages": dict(stage), "lsh_threshold": LSH_THRESHOLD,
                        "num_perm": NUM_PERM, "zh_ratio_floor": 0.5,
                        "response_len_range": [20, 900]}
    (CLEAN / "dialog_zh.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in kept), encoding="utf-8")
    return kept


def process_mcq(report):
    train_src = RAW / "cmexam_train.jsonl"
    test_src = RAW / "cmexam_test.jsonl"
    tr = load_jsonl(train_src)
    te = load_jsonl(test_src)
    report["mcq"] = {"train_raw": len(tr), "test_raw": len(te)}
    print(f"  cmexam train={len(tr)} test={len(te)}")

    def valid(r):
        if not isinstance(r, dict):
            return None
        q = (r.get("Question") or "").strip()
        opts = r.get("Options") or []
        ans = (r.get("Answer") or "").strip().upper()
        if not (q and isinstance(opts, list) and len(opts) >= 2):
            return None
        letters = []
        lines = []
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

    tr_ok, te_ok = [], []
    drop = Counter()
    for r in tr:
        v = valid(r)
        if v is None:
            drop["train_invalid"] += 1
        else:
            tr_ok.append(v)
    for r in te:
        v = valid(r)
        if v is None:
            drop["test_invalid"] += 1
        else:
            te_ok.append(v)
    report["mcq"]["dropped_invalid"] = dict(drop)
    report["mcq"]["train_valid"] = len(tr_ok)
    report["mcq"]["test_valid"] = len(te_ok)

    # Eval set: deterministic seeded sample from the official test split.
    import random

    rng = random.Random(SEED)
    eval_rows = rng.sample(te_ok, min(EVAL_N, len(te_ok)))
    for i, r in enumerate(eval_rows):
        r["id"] = f"e{i}"
    eval_hashes = {qhash(r["question"]) for r in eval_rows}
    train_pool = []
    leak = 0
    for i, r in enumerate(tr_ok):
        if qhash(r["question"]) in eval_hashes:
            leak += 1
            continue
        r["id"] = f"m{i}"
        train_pool.append(r)
    report["mcq"]["leakage_removed_from_train"] = leak
    report["mcq"]["eval_n"] = len(eval_rows)
    report["mcq"]["train_pool_after_leak_check"] = len(train_pool)
    report["mcq"]["answer_letter_distribution_eval"] = dict(Counter(r["gold"] for r in eval_rows))
    report["mcq"]["n_options_distribution_eval"] = dict(Counter(len(r["letters"]) for r in eval_rows))

    def dump(path, rows):
        path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")

    dump(CLEAN / "mcq_eval_100.jsonl", eval_rows)
    dump(CLEAN / "mcq_train_pool.jsonl", train_pool)
    return train_pool, eval_rows


def main() -> int:
    CLEAN.mkdir(parents=True, exist_ok=True)
    STATS.mkdir(parents=True, exist_ok=True)
    report = {"seed": SEED, "generated_by": "scripts/02_clean_dedup.py"}
    d = process_dialog(report)
    mp, ev = process_mcq(report)
    (STATS / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n== funnel ==")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:1800])
    print(f"\nclean outputs: dialog={len(d)} mcq_pool={len(mp)} eval={len(ev)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
