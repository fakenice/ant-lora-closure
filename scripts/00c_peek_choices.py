"""Peek CMExam record schema and the base model's shard layout/size (Range-only, no big downloads)."""
import json
import urllib.request

MIRROR = "https://hf-mirror.com"


def get(url, timeout=90, limit=None):
    headers = {"User-Agent": "curl/8"}
    if limit:
        headers["Range"] = f"bytes=0-{limit - 1}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers


print("== CMExam test.json (ranged 4 KB) ==")
body, hdr = get("https://hf-mirror.com/datasets/fzkuji/CMExam/resolve/main/test.json", limit=4096)
print("status:", hdr.get_status_untyped() if hasattr(hdr, "get_status_untyped") else "", )
print("content-range:", hdr.get("Content-Range"), "| bytes got:", len(body))
print(body[:900].decode("utf-8", "replace"))

print("\n== CMExam train.json (ranged 2 KB) ==")
body2, hdr2 = get("https://hf-mirror.com/datasets/fzkuji/CMExam/resolve/main/train.json", limit=2048)
print("content-range:", hdr2.get("Content-Range"), "| bytes got:", len(body2))
print(body2[:600].decode("utf-8", "replace"))

print("\n== Qwen2.5-1.5B-Instruct files ==")
meta = json.loads(get(f"{MIRROR}/api/models/Qwen/Qwen2.5-1.5B-Instruct")[0].decode("utf-8"))
for s in meta.get("siblings", []):
    print("   ", s["rfilename"])
