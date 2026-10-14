"""Download the base model through the HF mirror and register what actually landed.

Only inference/training files are fetched (safetensors + tokenizer + config); no
original/duplicate format weights, so the on-disk size is honest.
"""
import json
import os
import time
from pathlib import Path

# The Xet CAS backend (cas-server.xethub.hf.co) is not reachable through the
# mirror and returns 401, which aborts snapshot_download. Plain HTTP works.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = "Qwen/Qwen2.5-1.5B-Instruct"
DEST = ROOT / "models" / "Qwen2.5-1.5B-Instruct"

PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.py", "*.jinja", "merges.txt", "vocab.json"]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    path = snapshot_download(
        repo_id=REPO, local_dir=str(DEST), allow_patterns=PATTERNS,
        max_workers=8, endpoint="https://hf-mirror.com",
    )
    files = sorted(p for p in DEST.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    manifest = {
        "repo": REPO, "endpoint": "https://hf-mirror.com", "local_dir": str(path),
        "elapsed_s": round(time.time() - t0, 1), "n_files": len(files),
        "total_bytes": total, "total_gb": round(total / 2**30, 2),
        "files": [{"path": str(p.relative_to(DEST)), "bytes": p.stat().st_size} for p in files],
    }
    (ROOT / "out" / "model_download.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"downloaded {len(files)} files, {manifest['total_gb']} GB in {manifest['elapsed_s']}s")
    for f in manifest["files"][:20]:
        print(f"   {f['path']}  {f['bytes']/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
