"""Checks for the deliberately leaky positive control and the shuffled-label control.

The positive control trains on the same 500 questions it is evaluated on. That is
leakage by design: it answers "can this harness and training recipe detect a gain
at all?" The shuffled-label arm answers "is the gain label-specific, or just the
output format?" Neither is a scientific capability result.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "clean" / "mcq_eval_500.jsonl"
POS = ROOT / "data" / "ratios" / "poscontrol_evaltrain.jsonl"
SHUF = ROOT / "data" / "ratios" / "poscontrol_shuffled.jsonl"
REPORT = ROOT / "data" / "stats" / "positive_control_report.json"


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.endswith("}")]


class PositiveControlData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eval = load(EVAL)
        cls.pos = load(POS)
        cls.shuf = load(SHUF)

    def test_both_controls_have_500_rows(self):
        self.assertEqual(len(self.pos), 500)
        self.assertEqual(len(self.shuf), 500)

    def test_user_prompts_match_the_eval_set(self):
        for i in range(500):
            self.assertEqual(self.pos[i]["messages"][1]["content"], self.eval[i]["prompt"], i)
            self.assertEqual(self.shuf[i]["messages"][1]["content"], self.eval[i]["prompt"], i)

    def test_positive_control_uses_the_gold_answer(self):
        for i, row in enumerate(self.pos):
            self.assertEqual(row["messages"][2]["content"], self.eval[i]["gold"], i)

    def test_shuffled_control_never_uses_the_gold_answer(self):
        for i, row in enumerate(self.shuf):
            picked = row["messages"][2]["content"]
            self.assertIn(picked, self.eval[i]["letters"], i)
            self.assertNotEqual(picked, self.eval[i]["gold"], i)

    def test_report_marks_the_control_as_leaky_by_design(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertTrue(report["leaky_by_design"])
        self.assertEqual(report["n_rows"], 500)


if __name__ == "__main__":
    unittest.main()
