"""Inspect candidate public datasets on the HF mirror before downloading anything.

Reports: file list, sizes, and (for JSON files) the real record schema of the first
record. Nothing is written except a JSON manifest under data/.
"""
import json
import sys
import urllib.request
from pathlib import Path

MIRROR = "https://hf-mirror.com"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    ("dataset", "shibing624/medical"),
    ("dataset", "FreedomIntelligence/medical-o1-reasoning-SFT"),
    ("model", "Qwen/Qwen2.5-1.5B-Instruct"),
]


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def api(kind: str, repo: str) -> dict:
    return json.loads(get(f"{MIRROR}/api/{kind}s/{repo}").decode("utf-8"))


def main() -> None:
    manifest = {}
    for kind, repo in CANDIDATES:
        try:
            meta = api(kind, repo)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {kind} {repo}: {exc}")
            continue
        files = [s.get("rfilename") for s in meta.get("siblings", []) if s.get("rfilename")]
        manifest[f"{kind}:{repo}"] = {"n_files": len(files), "files": files}
        print(f"\n=== {kind} {repo} :: {len(files)} files ===")
        for f in files:
            print("   ", f)
        # Peek at JSON payloads to learn the true schema.
        for f in files:
            if f.endswith(".json") and ("train" in f or "test" in f or "valid" in f):
                url = f"{MIRROR}/datasets/{repo}/resolve/main/{f}"
                try:
                    raw = get(url, timeout=120)
                except Exception as exc:  # noqa: BLE001
                    print(f"    [peek FAIL] {f}: {exc}")
                    continue
                head = raw[:1200].decode("utf-8", "replace")
                print(f"    [peek {len(raw)/1e6:.2f} MB] {f} -> {head[:220]!r}")
    (OUT / "candidates_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nmanifest written:", OUT / "candidates_manifest.json")


if __name__ == "__main__":
    sys.exit(main())
