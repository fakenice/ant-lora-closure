"""Data-integrity checks for the expanded 500-item MCQ eval set.

The original 100-item set removed 5 train/eval duplicates from the training pool.
Adding 400 new questions therefore requires checking the *existing* training pool
for overlap; otherwise the larger evaluation would silently reintroduce leakage.
"""
from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_100 = ROOT / "data" / "clean" / "mcq_eval_100.jsonl"
EVAL_500 = ROOT / "data" / "clean" / "mcq_eval_500.jsonl"
TRAIN_POOL = ROOT / "data" / "clean" / "mcq_train_pool.jsonl"


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.endswith("}"):
            rows.append(json.loads(line))
    return rows


def qhash(q: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", "", q).encode("utf-8")).hexdigest()[:16]


class Eval500Integrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load(EVAL_500)
        cls.original = load(EVAL_100)
        cls.train_hashes = {qhash(r["question"]) for r in load(TRAIN_POOL)}

    def test_has_500_rows(self):
        self.assertEqual(len(self.rows), 500)

    def test_original_100_are_preserved_as_the_first_100(self):
        self.assertEqual(len(self.original), 100)
        for i, row in enumerate(self.original):
            self.assertEqual(self.rows[i]["question"], row["question"], f"e{i}")
            self.assertEqual(self.rows[i]["gold"], row["gold"], f"e{i}")

    def test_no_duplicate_questions(self):
        hashes = [qhash(r["question"]) for r in self.rows]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_no_overlap_with_the_existing_training_pool(self):
        overlap = [r["id"] for r in self.rows if qhash(r["question"]) in self.train_hashes]
        self.assertEqual(overlap, [])

    def test_every_row_is_a_valid_mcq(self):
        for row in self.rows:
            self.assertIn(row["gold"], row["letters"], row["id"])
            self.assertGreaterEqual(len(row["letters"]), 2, row["id"])
            self.assertIn("prompt", row)
            self.assertTrue(row["prompt"].strip(), row["id"])


if __name__ == "__main__":
    unittest.main()
