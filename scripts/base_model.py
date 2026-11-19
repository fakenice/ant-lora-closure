"""Base-model resolution and verification, shared by training and evaluation.

Stdlib only on purpose: scripts/04 (torch) and scripts/05 (torch) must agree on which
weights they load, and the check has to be importable from a test without pulling in
the ML stack.

The published runs use models/Qwen2.5-1.5B-Instruct. The 2026-11-20 scale check sets
BASE_MODEL_DIR=Qwen2.5-0.5B-Instruct for both the training and the evaluation of every
arm, so a ratio comparison across scales differs in exactly one variable. The default
is unchanged, so re-running the published sweep without the variable reproduces it.

Artifacts are namespaced separately by --tag (see scripts/run_multiseed_500.sh): a
scale check writes lora_q05b_*, train_q05b_*.jsonl and eval_q05b_*, never a published
path.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT = ROOT / "out"
DEFAULT_BASE = "Qwen2.5-1.5B-Instruct"
WEIGHTS_FILE = "model.safetensors"

# sha256 of model.safetensors for bases whose digest predates the download registry.
# The registry written by scripts/03c is the first source; this is the fallback.
PUBLISHED_WEIGHTS_SHA = {
    "Qwen2.5-1.5B-Instruct": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
}


def base_name() -> str:
    return os.environ.get("BASE_MODEL_DIR", DEFAULT_BASE)


def base_dir() -> Path:
    return MODELS / base_name()


def weights_path() -> Path:
    return base_dir() / WEIGHTS_FILE


def registry_path(name: str | None = None) -> Path:
    """The download registry scripts/03c writes for this base folder."""
    name = name or base_name()
    if name == DEFAULT_BASE:
        return OUT / "model_download.json"
    return OUT / f"model_download_{name}.json"


def upstream_sha(name: str | None = None) -> str | None:
    """Digest the weights of this base are expected to have, or None if unrecorded.

    The registry carries Hugging Face's own LFS sha256 (and the digest computed at
    download time), so a new base is covered without hardcoding its hash here.
    """
    name = name or base_name()
    reg = registry_path(name)
    if reg.exists():
        try:
            data = json.loads(reg.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            data = {}
        for rec in data.get("files", []):
            if rec.get("path") == WEIGHTS_FILE:
                sha = rec.get("sha256_upstream") or rec.get("sha256")
                if sha:
                    return sha
    return PUBLISHED_WEIGHTS_SHA.get(name)


def file_sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def resolve(verify: bool = True) -> dict:
    """Return {base, dir, weights_sha256, expected_sha256} for logging.

    Raises SystemExit when the weights are absent, or when a recorded expected digest
    does not match the bytes on disk — training and evaluation on the wrong weights is
    the one failure mode a scale check cannot detect from its own numbers.
    """
    d = base_dir()
    wp = weights_path()
    if not wp.exists():
        raise SystemExit(f"{wp} missing; fetch it with "
                         f"BASE_REPO=Qwen/<name> .venv/Scripts/python.exe scripts/03c_fetch_weights.py")
    rel = str(d)
    try:
        rel = str(d.relative_to(ROOT))
    except ValueError:
        pass  # a base outside the repo (tests, symlinked model stores) still logs
    info = {"base": d.name, "path": str(d), "dir": rel,
            "weights_sha256": None, "expected_sha256": upstream_sha(d.name)}
    if verify:
        info["weights_sha256"] = file_sha256(wp)
        if info["expected_sha256"] and info["weights_sha256"] != info["expected_sha256"]:
            raise SystemExit(
                f"{wp}: sha256 {info['weights_sha256']} != recorded "
                f"{info['expected_sha256']}; refusing to run on weights that differ "
                "from the registered download")
    return info
