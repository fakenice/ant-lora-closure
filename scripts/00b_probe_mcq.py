"""Find a Chinese/English medical MCQ source with gold answers on the HF mirror.

Deterministic scoring requires (question, options, gold letter). This script only
queries the API and peeks the first bytes of the smallest JSON/parquet file so we
can confirm the actual field names before writing the loader.
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

MIRROR = "https://hf-mirror.com"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"

QUERIES = ["CMExam", "medqa", "medmcqa", "chinese medical exam", "disc-medllm", "huatuo"]


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main() -> int:
    found = {}
    for q in QUERIES:
        url = f"{MIRROR}/api/datasets?search={urllib.parse.quote(q)}&limit=8"
        try:
            rows = json.loads(get(url).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] search {q!r}: {exc}")
            continue
        print(f"\n=== search {q!r} ===")
        for r in rows:
            rid = r["id"]
            print(f"   {rid}  dl={r.get('downloads')}")
            found.setdefault(rid, None)
    # For the most exam-like ids, list files.
    interesting = [k for k in found if any(
        t in k.lower() for t in ("exam", "medqa", "mcqa", "disc", "huatuo", "medical"))]
    for rid in interesting[:8]:
        try:
            meta = json.loads(get(f"{MIRROR}/api/datasets/{rid}").decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] meta {rid}: {exc}")
            continue
        files = [s["rfilename"] for s in meta.get("siblings", [])][:20]
        print(f"\n--- {rid} ({meta.get('downloads', 0)} dl) ---")
        for f in files:
            print("   ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
