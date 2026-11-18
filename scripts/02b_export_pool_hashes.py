"""Export the training-pool question hashes so the no-leakage test runs without the pool.

data/clean/mcq_train_pool.jsonl (46,421 rows) is not redistributed with the repo,
but tests/test_eval500.py must still prove that the 500-item eval set has zero
question-hash overlap with it. This script writes the 16-hex-char normalized
question hashes (sorted, unique, one per line, LF) as a committed artifact; the
test treats that file as the source of truth and, when the pool is present
locally, re-derives the file from it to prove the two cannot drift apart.

Usage: python scripts/02b_export_pool_hashes.py
Writes data/clean/mcq_train_pool_qhashes.txt
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "clean" / "mcq_train_pool.jsonl"
OUT = ROOT / "data" / "clean" / "mcq_train_pool_qhashes.txt"


def qhash(q: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", "", q).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    hashes = set()
    n = 0
    for line in POOL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.endswith("}"):
            n += 1
            hashes.add(qhash(json.loads(line)["question"]))
    OUT.write_text("\n".join(sorted(hashes)) + "\n", encoding="utf-8", newline="\n")
    print(f"{n} pool rows -> {len(hashes)} unique hashes -> {OUT}")


if __name__ == "__main__":
    main()
