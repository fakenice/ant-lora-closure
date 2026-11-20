"""Guards for the 2026-11-20 scale check (same four ratios on a 0.5B base).

The scale check runs the published sweep script with two extra environment variables,
so the plumbing risks are: training on one base while scoring on another, writing 0.5B
artifacts over the published 1.5B ones, or silently training on weights that are not the
published files. These tests lock those three properties without touching the GPU, and
then lock what the comparison artifact (scripts/10_scale_compare.py) says - so that a
future edit of the data cannot quietly flip 结论 8 while the README still asserts it.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(module_name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(module_name, str(SCRIPTS / filename))
    spec = importlib.util.spec_from_loader(module_name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class BaseResolution(unittest.TestCase):
    """scripts/base_model.py is the single source for which weights are used."""

    @classmethod
    def setUpClass(cls):
        cls.bm = load_script("bm", "base_model.py")

    def setUp(self):
        self._saved = os.environ.pop("BASE_MODEL_DIR", None)

    def tearDown(self):
        os.environ.pop("BASE_MODEL_DIR", None)
        if self._saved is not None:
            os.environ["BASE_MODEL_DIR"] = self._saved

    def test_default_is_the_published_base(self):
        self.assertEqual(self.bm.base_name(), self.bm.DEFAULT_BASE)
        self.assertEqual(self.bm.base_name(), "Qwen2.5-1.5B-Instruct")

    def test_published_digest_is_resolvable(self):
        sha = self.bm.upstream_sha("Qwen2.5-1.5B-Instruct")
        self.assertEqual(sha, self.bm.PUBLISHED_WEIGHTS_SHA["Qwen2.5-1.5B-Instruct"])

    def test_unrecorded_base_has_no_digest_to_enforce(self):
        self.assertIsNone(self.bm.upstream_sha("NoSuchBase-Instruct"))

    def test_base_env_moves_both_path_and_registry(self):
        os.environ["BASE_MODEL_DIR"] = "Qwen2.5-0.5B-Instruct"
        self.assertEqual(self.bm.base_dir().name, "Qwen2.5-0.5B-Instruct")
        self.assertEqual(self.bm.weights_path().name, "model.safetensors")
        # a scale check must not read or write the published registry
        self.assertNotEqual(self.bm.registry_path(), self.bm.OUT / "model_download.json")

    def test_missing_weights_abort(self):
        """Run against a scratch models dir so the abort is deterministic whether or not
        the real 0.5B weights happen to sit in models/ on the machine running the test."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            models = Path(td)
            (models / "Qwen2.5-0.5B-Instruct").mkdir()
            saved_models, saved_out = self.bm.MODELS, self.bm.OUT
            self.bm.MODELS, self.bm.OUT = models, models / "out"
            os.environ["BASE_MODEL_DIR"] = "Qwen2.5-0.5B-Instruct"
            try:
                with self.assertRaises(SystemExit) as ctx:
                    self.bm.resolve()
                self.assertIn("model.safetensors missing", str(ctx.exception))
            finally:
                self.bm.MODELS, self.bm.OUT = saved_models, saved_out

    def test_digest_mismatch_aborts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            models = Path(td)
            (models / "fakebase").mkdir()
            (models / "fakebase" / "model.safetensors").write_bytes(b"not the weights")
            saved_models, saved_out = self.bm.MODELS, self.bm.OUT
            self.bm.MODELS, self.bm.OUT = models, models / "out"
            os.environ["BASE_MODEL_DIR"] = "fakebase"
            try:
                self.assertIsNone(self.bm.upstream_sha("fakebase"))
                # with no recorded digest there is nothing to enforce
                self.assertEqual(self.bm.resolve()["weights_sha256"],
                                 "35245741e72e29edc737d79889c146307c8ca445834edf0e084d7c16f91da3c4")
                # once a digest is recorded, a different file must abort
                self.bm.PUBLISHED_WEIGHTS_SHA["fakebase"] = "0" * 64
                with self.assertRaises(SystemExit) as ctx:
                    self.bm.resolve()
                self.assertIn("refusing to run on weights", str(ctx.exception))
            finally:
                self.bm.PUBLISHED_WEIGHTS_SHA.pop("fakebase", None)
                self.bm.MODELS, self.bm.OUT = saved_models, saved_out


class ArtifactNamespacing(unittest.TestCase):
    """A prefixed sweep cannot reach a published artifact path."""

    PUBLISHED_TAGS = [f"{cfg}_s{seed}" for cfg in ("r100_0", "r70_30", "r30_70", "r0_100")
                      for seed in (20261014, 20261015, 20261016)]

    def _summary_module(self, prefix: str):
        os.environ["TAG_PREFIX"] = prefix
        try:
            return load_script(f"ms7_{prefix or 'none'}", "07_multiseed_summary.py")
        finally:
            os.environ.pop("TAG_PREFIX", None)

    def test_unprefixed_summary_keeps_published_names(self):
        m = self._summary_module("")
        self.assertEqual(m.TAG_PREFIX, "")
        self.assertEqual(m.SUF, "")

    def test_prefixed_summary_never_uses_published_names(self):
        m = self._summary_module("q05b_")
        self.assertEqual(m.SUF, "_q05b")
        published = {"multiseed_500_summary.json", "multiseed_500_summary.csv",
                     "multiseed_500_summary.md"}
        produced = {f"multiseed_500{m.SUF}_summary.{ext}" for ext in ("json", "csv", "md")}
        self.assertFalse(produced & published)
        base_tag = f"{m.TAG_PREFIX}base_500"
        run_tags = {f"{m.TAG_PREFIX}{t}_500" for t in self.PUBLISHED_TAGS}
        self.assertFalse(({base_tag} | run_tags) & set(self.PUBLISHED_TAGS))

    def test_published_eval_artifacts_are_untagged(self):
        """The published 500-item evals the committed summary reads still exist."""
        listing = json.loads((ROOT / "out" / "multiseed_500_summary.json")
                             .read_text(encoding="utf-8"))
        self.assertNotIn("tag_prefix", listing)
        for run in listing["runs"]:
            self.assertFalse(run["tag"].startswith("q05b_"))
            self.assertTrue((ROOT / "out" / "eval" / f"eval_{run['tag']}.json").exists(),
                            run["tag"])


class SweepScriptContract(unittest.TestCase):
    def setUp(self):
        self.sh = (SCRIPTS / "run_multiseed_500.sh").read_text(encoding="utf-8")

    def test_base_and_prefix_are_env_driven_with_published_defaults(self):
        self.assertIn('BASE_MODEL_DIR=${BASE_MODEL_DIR:-Qwen2.5-1.5B-Instruct}', self.sh)
        self.assertIn('TAG_PREFIX=${TAG_PREFIX:-}', self.sh)
        self.assertIn("export BASE_MODEL_DIR TAG_PREFIX", self.sh)

    def test_tag_carries_the_prefix_everywhere(self):
        self.assertIn('tag="${TAG_PREFIX}${cfg}_s${seed}"', self.sh)
        self.assertIn('eval_one base "${TAG_PREFIX}base_500"', self.sh)

    def test_sweep_still_refuses_a_busy_gpu(self):
        self.assertIn("free_mib > 8000", self.sh)
        self.assertIn("ollama ps", self.sh)


class ScaleCompareArtifact(unittest.TestCase):
    """out/scale_compare_500.* must stay consistent with the two published summaries.

    This is the regression lock on the answer itself: the scale check concluded that at
    0.5B no ratio arm separates from base and no config pair separates, while at 1.5B the
    only separation involves r100_0 and is mostly the unparseable-answers penalty. If any
    of that changes because the artifacts change, these asserts fail instead of the README.
    """

    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((ROOT / "out" / "scale_compare_500.json")
                             .read_text(encoding="utf-8"))
        cls.scales = {s["label"]: s for s in cls.doc["scales"]}

    def test_both_scales_present_with_the_expected_prefixes(self):
        self.assertEqual(set(self.scales), {"1.5B", "0.5B"})
        self.assertEqual(self.scales["1.5B"]["tag_prefix"], "")
        self.assertEqual(self.scales["0.5B"]["tag_prefix"], "q05b_")
        self.assertEqual(self.scales["1.5B"]["base_model"], "Qwen2.5-1.5B-Instruct")
        self.assertEqual(self.scales["0.5B"]["base_model"], "Qwen2.5-0.5B-Instruct")

    def test_base_numbers_match_the_published_multiseed_summaries(self):
        for label, prefix in (("1.5B", ""), ("0.5B", "q05b_")):
            name = f"multiseed_500{'_' + prefix.strip('_') if prefix else ''}_summary.json"
            pub = json.loads((ROOT / "out" / name).read_text(encoding="utf-8"))
            s = self.scales[label]
            self.assertEqual(s["base_acc"], pub["base"]["accuracy"])
            self.assertEqual(s["base_correct"], pub["base"]["correct"])
            for arm in s["arms"]:
                sub = [r for r in pub["runs"] if r["config"] == arm["config"]]
                self.assertEqual(sum(r["unparseable"] for r in sub), arm["unparseable_total"])
                self.assertEqual([r["mcnemar_p_vs_base"] for r in sub],
                                 arm["mcnemar_p_vs_base"])

    def test_headline_verdict_of_the_scale_check(self):
        small, big = self.scales["0.5B"], self.scales["1.5B"]
        self.assertFalse(any(a["delta_ci_excludes_0"] for a in small["arms"]),
                         "a 0.5B arm separated from base: the 结论 8 wording is now wrong")
        self.assertFalse(any(p["excludes_0"] for p in small["pairs"]),
                         "two 0.5B ratios separated: 配比无稳定提升 no longer holds at 0.5B")
        self.assertEqual(len(small["pairs"]), 6)
        separated = [p for p in big["pairs"] if p["excludes_0"]]
        self.assertTrue(all("r100_0" in (p["a"], p["b"]) for p in separated),
                        "a non-r100_0 pair separated at 1.5B: revisit 结论 3/8")
        r100_big = next(a for a in big["arms"] if a["config"] == "r100_0")
        self.assertLess(r100_big["delta_vs_base"], 0)
        self.assertLess(abs(r100_big["delta_vs_base_parseable"]), 2.0)

    def test_arm_ordering_and_item_counts_are_stable(self):
        for s in self.scales.values():
            self.assertEqual([a["config"] for a in s["arms"]],
                             ["r100_0", "r70_30", "r30_70", "r0_100"])
            self.assertEqual(s["base_n"], 500)
            for a in s["arms"]:
                self.assertLessEqual(a["unparseable_capped_total"], a["unparseable_total"])
        # the training data per arm is identical across scales by construction
        big, small = self.scales["1.5B"], self.scales["0.5B"]
        self.assertEqual([(a["trainable_rows"], a["dropped_too_long"]) for a in big["arms"]],
                         [(a["trainable_rows"], a["dropped_too_long"]) for a in small["arms"]])

    def test_bootstrap_is_declared_and_finite(self):
        b = self.doc["boot"]
        self.assertEqual((b["B"], b["seed"]), (2000, 42))
        self.assertIn("item", b["unit"])
        self.assertIn("training randomness", b["caveat"])
        for s in self.scales.values():
            for a in s["arms"]:
                lo, hi = a["delta_ci95"]
                self.assertLessEqual(lo, a["delta_vs_base"])
                self.assertLessEqual(a["delta_vs_base"], hi)


if __name__ == "__main__":
    unittest.main()
