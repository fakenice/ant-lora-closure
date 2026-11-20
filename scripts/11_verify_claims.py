"""README 主要结论的机器校验：把「文档里的每句话都对得上产物」变成可执行的检查。

用法：`python scripts/11_verify_claims.py`（纯 stdlib，只读 `out/` 与 `data/` 里已提交的产物，
无 GPU、无 torch，约 2 秒）。断言全部从产物现算，不在这里复述结论数字；任何一条不成立就非零退出，
并指出是哪个文件哪个字段变了。校验对象是 README §5 的结论 1、2、3、6、8 与 §6 的限制第 1 条。

与既有校验的分工：
- CI 的 drift gate（`scripts/06`／`07`／`08`／`10`）证明产物可逐字节再生；
- `tests/test_scale_check.py` 锁规模复核管线的三个风险与对照表的头条判读；
- 本脚本再往前一步：证明再生出来的产物**仍然支持 README 的说法**——
  数据或口径改动之后，「区间重叠」「配比无稳定提升」「格式效应」这些话会不会悄悄失效。

依据：docs/DECISIONS.md 第 7、18–21 条；README.md §5、§6、§7c。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import base_model  # noqa: E402

OUT = ROOT / "out"
EVAL = OUT / "eval"
CLEAN = ROOT / "data" / "clean"
CONFIGS = ("r100_0", "r70_30", "r30_70", "r0_100")
SEEDS = (20261014, 20261015, 20261016)
SUMMARY_BY_SCALE = {"1.5B": "multiseed_500_summary.json",
                    "0.5B": "multiseed_500_q05b_summary.json"}


def jload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip().endswith("}")]


def items(tag: str) -> list[dict]:
    return jsonl(EVAL / f"eval_{tag}_items.jsonl")


def qhash(question: str) -> str:
    """同 tests/test_eval500.py 与 scripts/02b_export_pool_hashes.py 的归一化口径。"""
    return hashlib.sha256(re.sub(r"\s+", "", question).encode("utf-8")).hexdigest()[:16]


class Artefacts(unittest.TestCase):
    """规模对照表 + 两份发布汇总 + 单 seed 消融表，一次加载。"""

    @classmethod
    def setUpClass(cls):
        cls.sc = jload(OUT / "scale_compare_500.json")
        cls.scales = {s["label"]: s for s in cls.sc["scales"]}
        cls.ms = {label: jload(OUT / name) for label, name in SUMMARY_BY_SCALE.items()}
        cls.abl = jload(OUT / "summary.json")

    def arm(self, label: str, config: str) -> dict:
        return next(a for a in self.scales[label]["arms"] if a["config"] == config)

    def runs(self, label: str, config: str) -> list[dict]:
        return [r for r in self.ms[label]["runs"] if r["config"] == config]


class TestLoadedArtefactsAreThePublishedOnes(Artefacts):
    def test_both_scales_are_the_published_ones(self):
        self.assertEqual(set(self.scales), {"1.5B", "0.5B"})
        self.assertEqual(self.scales["1.5B"]["tag_prefix"], "")
        self.assertEqual(self.scales["0.5B"]["tag_prefix"], "q05b_")
        for label, ms in self.ms.items():
            self.assertEqual(ms.get("tag_prefix", ""), self.scales[label]["tag_prefix"], label)
            self.assertEqual(ms["base"]["correct"], self.scales[label]["base_correct"], label)
            self.assertEqual(ms["base"]["accuracy"], self.scales[label]["base_acc"], label)


class TestConclusion1And2IntervalsOverlap(Artefacts):
    """结论 1（四档彼此不可区分）与结论 2（微调没抬高准确率）：n=100 表的区间口径。"""

    def test_base_and_lora_intervals_contain_each_other_point_estimates(self):
        base = next(r for r in self.abl["runs"] if r["kind"] == "baseline")
        for r in self.abl["runs"]:
            if r["kind"] != "lora":
                continue
            self.assertLessEqual(r["ci95_low"], base["accuracy"], r["run"])
            self.assertGreaterEqual(r["ci95_high"], base["accuracy"], r["run"])
            self.assertLessEqual(base["ci95_low"], r["accuracy"], r["run"])
            self.assertGreaterEqual(base["ci95_high"], r["accuracy"], r["run"])

    def test_lora_pairs_overlap_each_other(self):
        runs = [r for r in self.abl["runs"] if r["kind"] == "lora"]
        self.assertEqual(len(runs), 4)
        for i, a in enumerate(runs):
            for b in runs[i + 1:]:
                self.assertTrue(a["ci95_low"] <= b["ci95_high"] and b["ci95_low"] <= a["ci95_high"],
                                f"{a['run']} vs {b['run']} 区间不再重叠，结论 1 要重写")

    def test_spread_is_within_one_interval_half_width(self):
        """README 结论 1 的原话是「相差 5 个百分点，而半宽约 ±8–9 个百分点」。"""
        lora = [r for r in self.abl["runs"] if r["kind"] == "lora"]
        spread = (max(r["accuracy"] for r in lora) - min(r["accuracy"] for r in lora)) * 100
        half = max(r["ci95_high"] - r["accuracy"] for r in lora) * 100
        self.assertLess(spread, half, f"档差 {spread:.1f} pt 已不小于半宽 {half:.1f} pt")


class TestConclusion3FormatEffect(Artefacts):
    """结论 3：配比改变的是输出格式与判分可解析性，方向在两个基座上一致。"""

    def test_mcq_mixed_arms_collapse_to_one_letter(self):
        for label, s in self.scales.items():
            for arm in s["arms"]:
                if arm["config"] == "r100_0":
                    continue
                self.assertEqual(arm["answer_marker_share_mean"], 0.0, f"{label}/{arm['config']}")
                self.assertLess(arm["mean_new_tokens"], 2.2, f"{label}/{arm['config']}")

    def test_dialogue_only_arm_keeps_long_answers_everywhere(self):
        for label, s in self.scales.items():
            arm = self.arm(label, "r100_0")
            self.assertGreater(arm["answer_marker_share_mean"], 0.0, label)
            self.assertGreater(arm["mean_new_tokens"], 3.0, label)
            others = [a["unparseable_total"] for a in s["arms"] if a["config"] != "r100_0"]
            self.assertGreater(arm["unparseable_total"], max(others), label)

    def test_n100_table_agrees_on_the_direction(self):
        """旧 100 题表里也只有纯对话档出过不可解析（README 结论 3 的第一手证据）。"""
        unp = {r["run"]: r["unparseable"] for r in self.abl["runs"] if r["kind"] == "lora"}
        self.assertEqual(unp["r100_0"], 7)
        self.assertEqual(sum(v for k, v in unp.items() if k != "r100_0"), 0)


class TestConclusion6MultiseedNegative(Artefacts):
    def test_no_arm_is_significant_on_all_three_seeds(self):
        for label, s in self.scales.items():
            for arm in s["arms"]:
                if arm["delta_vs_base"] <= 0:
                    continue
                sig = [p for p in arm["mcnemar_p_vs_base"] if p < 0.05]
                self.assertLess(len(sig), len(arm["mcnemar_p_vs_base"]),
                                f"{label}/{arm['config']}: 三个 seed 全显著，结论 6 要重写")

    def test_non_dialogue_pairs_never_separate(self):
        for label, s in self.scales.items():
            for p in s["pairs"]:
                if "r100_0" in (p["a"], p["b"]):
                    continue
                self.assertFalse(p["excludes_0"], f"{label}: {p['a']} vs {p['b']} 分离了")

    def test_every_config_ran_three_seeds_on_five_hundred_items(self):
        for label in self.ms:
            for cfg in CONFIGS:
                runs = self.runs(label, cfg)
                self.assertEqual(len(runs), 3, f"{label}/{cfg}")
                self.assertEqual({r["seed"] for r in runs}, set(SEEDS), f"{label}/{cfg}")
                for r in runs:
                    rows = items(r["tag"])
                    self.assertEqual(len(rows), r["n"], r["tag"])
                    self.assertEqual(sum(1 for x in rows if x["outcome"] == "correct"),
                                     r["correct"], r["tag"])
                    self.assertEqual(sum(1 for x in rows if x["outcome"] == "unparseable"),
                                     r["unparseable"], r["tag"])


class TestConclusion8ScaleCheck(Artefacts):
    def test_nothing_separates_at_05b(self):
        small = self.scales["0.5B"]
        self.assertFalse(any(a["delta_ci_excludes_0"] for a in small["arms"]),
                         "0.5B 有一档与 base 分开：结论 8 的第一句不再成立")
        self.assertEqual(len(small["pairs"]), 6)
        self.assertFalse(any(p["excludes_0"] for p in small["pairs"]),
                         "0.5B 有两档分开：「配比无稳定提升」在 0.5B 不再成立")

    def test_15b_separation_involves_only_r100_0_and_is_grading_not_knowledge(self):
        big = self.scales["1.5B"]
        separated = [p for p in big["pairs"] if p["excludes_0"]]
        self.assertTrue(separated, "1.5B 连一档差异都测不出了，README 结论 3 的措辞需重写")
        self.assertTrue(all("r100_0" in (p["a"], p["b"]) for p in separated))
        arm = self.arm("1.5B", "r100_0")
        self.assertLess(arm["delta_vs_base"], 0.0, "分离方向变成了「更好」，判读要重写")
        self.assertLess(abs(arm["delta_vs_base_parseable"]), 0.02,
                        "仅可解析口径超出 ±2 pt：那是知识差异，不是判分口径")
        self.assertGreater(arm["unparseable_capped_total"], arm["unparseable_total"] * 0.3,
                           "撞满 token 上限的占比过低，「12-token 截断」的解释不成立")

    def test_lora_effect_itself_stays_unresolved_at_both_scales(self):
        for label, s in self.scales.items():
            lo, hi = s["mean_delta"]["ci95"]
            self.assertLessEqual(lo, 0.0, label)
            self.assertGreaterEqual(hi, 0.0, label)

    def test_selected_statistic_is_labelled_as_such(self):
        for label, s in self.scales.items():
            self.assertIn("挑过的统计量", s["spread"]["note"], label)

    def test_interval_widths_support_the_five_point_claim(self):
        """§7c 与 scale_compare §5 说「能排除 ≥5 pt 的稳定提升」，依据是区间半宽都 <5 pt。"""
        widths = [100 * (a["delta_ci95"][1] - a["delta_ci95"][0]) / 2
                  for s in self.scales.values() for a in s["arms"]]
        widths += [100 * (p["ci95"][1] - p["ci95"][0]) / 2
                   for s in self.scales.values() for p in s["pairs"]]
        self.assertLess(max(widths), 5.0,
                        "有区间半宽 ≥5 pt，「≥5 pt 的稳定提升可以排除」这句要改写")
        self.assertGreater(max(widths), 2.0, "区间窄到 2 pt 以内，可排除门槛应下调")
        line = next(v for v in self.sc["verdict"] if v.startswith("这次复核能排除什么"))
        stated = int(re.search(r"≥(\d+) pt", line).group(1))
        self.assertEqual(stated, math.ceil(max(widths)),
                         "verdict 里写的门槛和区间半宽不再一致")

    def test_bootstrap_is_declared_item_level_with_the_training_randomness_caveat(self):
        b = self.sc["boot"]
        self.assertEqual((b["B"], b["seed"]), (2000, 42))
        self.assertIn("item", b["unit"])
        self.assertIn("training randomness", b["caveat"])

    def test_verdict_numbers_are_still_recomputable_from_the_tables(self):
        """`verdict` 是 §5 判读与 README §7c 的同源文本，其中的效应量必须还能算出来。"""
        nums = set()

        def add(x):
            nums.add(f"{x * 100:.1f}")

        def add_half(ci):
            nums.add(f"{(ci[1] - ci[0]) * 50:.1f}")

        for s in self.scales.values():
            for arm in s["arms"]:
                add(arm["delta_vs_base"])
                add(arm["delta_vs_base_parseable"])
                for bound in arm["delta_ci95"]:
                    add(bound)
                add_half(arm["delta_ci95"])
            for pair in s["pairs"]:
                add(pair["delta"])
                for bound in pair["ci95"]:
                    add(bound)
                add_half(pair["ci95"])
            add(s["spread"]["value"])
            for bound in s["spread"]["ci95"]:
                add(bound)
            add(s["mean_delta"]["value"])
            for bound in s["mean_delta"]["ci95"]:
                add(bound)
        add(abs(self.scales["1.5B"]["base_acc"] - self.scales["0.5B"]["base_acc"]))
        for line in self.sc["verdict"]:
            for token in re.findall(r"(-?\d+\.\d) pt", line):
                self.assertIn(token, nums, f"verdict 里的 {token} pt 已不能从表里算出")


class TestLimitationsAndLeakage(Artefacts):
    """§6 限制第 1 条（前缀样本）＋评测集与训练池零重叠（DECISIONS 第 8、11 条）。"""

    def test_dialogue_corpus_is_a_head_prefix_not_a_random_sample(self):
        rec = next(f for f in jload(ROOT / "data" / "raw_manifest.json")["files"]
                   if f["slug"] == "medical_dialog_zh_train_head")
        total = int(rec["range_returned"].rsplit("/", 1)[1])
        self.assertLess(rec["range_requested"], total,
                        "取数范围不再小于全量文件，「前缀样本」这条限制要重写")
        self.assertEqual(rec["range_requested"], rec["bytes"])
        stages = jload(ROOT / "data" / "stats" / "quality_report.json")["dialog"]["stages"]
        self.assertEqual(stages["final_kept"],
                         stages["after_filters"] - stages["minhash_duplicates"],
                         "清洗漏斗的 final_kept 不再等于过滤后计数减去 MinHash 重复数")

    @unittest.skipUnless((CLEAN / "dialog_zh.jsonl").exists(),
                         "dialog_zh.jsonl 不入库（.gitignore），CI 无此文件")
    def test_final_kept_matches_the_deduped_corpus_on_disk(self):
        """本机有清洗产物时才校验「58,787 条」这句话；CI 走上一条的漏斗内部一致性。"""
        stages = jload(ROOT / "data" / "stats" / "quality_report.json")["dialog"]["stages"]
        self.assertEqual(stages["final_kept"], len(jsonl(CLEAN / "dialog_zh.jsonl")))

    def test_eval_sets_do_not_overlap_the_train_pool(self):
        hashes = set((CLEAN / "mcq_train_pool_qhashes.txt").read_text(encoding="utf-8").split())
        for name in ("mcq_eval_100.jsonl", "mcq_eval_500.jsonl"):
            rows = jsonl(CLEAN / name)
            overlap = [r["id"] for r in rows if qhash(r["question"]) in hashes]
            self.assertEqual(overlap, [], name)

    def test_both_scales_scored_the_same_items_in_the_same_order(self):
        order = [r["id"] for r in jsonl(CLEAN / "mcq_eval_500.jsonl")]
        self.assertEqual(len(order), 500)
        for cfg in CONFIGS:
            for seed in SEEDS:
                self.assertEqual([r["id"] for r in items(f"{cfg}_s{seed}_500")], order, cfg)
                self.assertEqual([r["id"] for r in items(f"q05b_{cfg}_s{seed}_500")], order, cfg)

    def test_the_same_recipe_ran_on_both_bases(self):
        big, small = self.scales["1.5B"], self.scales["0.5B"]
        self.assertEqual([(a["config"], a["trainable_rows"], a["dropped_too_long"])
                          for a in big["arms"]],
                         [(a["config"], a["trainable_rows"], a["dropped_too_long"])
                          for a in small["arms"]])
        self.assertEqual(big["max_new_tokens"], small["max_new_tokens"])
        self.assertEqual(big["base_n"], small["base_n"])


class TestWeightsAndStepBudget(Artefacts):
    """「唯一被有意改变的变量是基座」需要两条支撑：权重来历与训练预算。"""

    def test_both_bases_have_a_registered_digest(self):
        for name in (base_model.DEFAULT_BASE, "Qwen2.5-0.5B-Instruct"):
            recorded = base_model.upstream_sha(name)
            self.assertTrue(recorded, f"{name}: 没有可复核的权重摘要")
            if name in base_model.PUBLISHED_WEIGHTS_SHA:
                self.assertEqual(recorded, base_model.PUBLISHED_WEIGHTS_SHA[name], name)
            reg = jload(base_model.registry_path(name))
            weights = [f for f in reg["files"] if f.get("path") == "model.safetensors"]
            self.assertEqual(len(weights), 1, name)
            self.assertIn(recorded, (weights[0].get("sha256_upstream"), weights[0].get("sha256")),
                          name)
        self.assertEqual(self.scales["1.5B"]["base_model"], base_model.DEFAULT_BASE)

    def test_all_24_runs_logged_150_steps(self):
        paths = sorted(OUT.glob("train_*_s2026*.jsonl"))
        self.assertEqual(len(paths), 24, "训练日志份数不是 4 档 × 3 seed × 2 基座")
        for path in paths:
            done = [e for e in jsonl(path) if e.get("event") == "done"]
            self.assertEqual(len(done), 1, path.name)
            self.assertEqual(done[0].get("steps"), 150, path.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
