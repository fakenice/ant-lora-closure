"""Guards for the 2026-11-20 scale check (same four ratios on a 0.5B base).

The scale check runs the published sweep script with two extra environment variables,
so the risk is not the science but the plumbing: training on one base while scoring on
another, writing 0.5B artifacts over the published 1.5B ones, or silently training on
weights that are not the published files. These tests lock those three properties
without touching the GPU.
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
        os.environ["BASE_MODEL_DIR"] = "Qwen2.5-0.5B-Instruct"
        with self.assertRaises(SystemExit) as ctx:
            self.bm.resolve()
        self.assertIn("model.safetensors missing", str(ctx.exception))

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


if __name__ == "__main__":
    unittest.main()
