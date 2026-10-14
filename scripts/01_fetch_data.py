"""Fetch and register the public corpora for the LoRA data-pipeline closure.

Sources (schemas verified by scripts/00_probe_sources.py / 00c_peek_choices.py):
  S1 shibing624/medical        finetune/train_zh_0.json  JSONL {instruction,input,output}  (~1.34 GB)
                               -> streamed with an HTTP Range, only the head is used, and that
                                  truncation is recorded as a sampling-bias note in the manifest.
  S2 fzkuji/CMExam             train.json  JSONL {Question,Options[{key,value}],Answer,Explanation} (~48 MB)
                               test.json   JSONL, same schema -> deterministic MCQ eval (gold letters)

Outputs: data/raw/<slug> files + data/raw_manifest.json (sha256, bytes, record counts,
key union of the first 200 records, and the exact HTTP range requested).
"""
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MIRROR = "https://hf-mirror.com"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

HEAD_BYTES = 60 * 1024 * 1024  # stream cap for the 1.34 GB dialog file

JOBS = [
    # (slug, kind, repo, path in repo, max_bytes or None for whole file)
    ("medical_dialog_zh_train_head", "datasets", "shibing624/medical", "finetune/train_zh_0.json", HEAD_BYTES),
    ("cmexam_train", "datasets", "fzkuji/CMExam", "train.json", None),
    ("cmexam_test", "datasets", "fzkuji/CMExam", "test.json", None),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(slug: str, kind: str, repo: str, rel: str, max_bytes):
    dest = RAW / f"{slug}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MIRROR}/{kind}/{repo}/resolve/main/{rel}"
    headers = {"User-Agent": "curl/8"}
    if max_bytes:
        headers["Range"] = f"bytes=0-{max_bytes - 1}"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    cached {slug} ({dest.stat().st_size/1e6:.2f} MB)")
        return dest, url, headers.get("Range")
    # The mirror rate-limits (HTTP 429) after bursts of requests: retry with backoff
    # and pause between jobs rather than hammering it.
    last_exc = None
    for attempt in range(6):
        if attempt:
            wait = 20 * attempt
            print(f"    retry {attempt}/5 for {slug} after {wait}s backoff")
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=600) as r:
                content_range = r.headers.get("Content-Range")
                data = r.read()
            dest.write_bytes(data)
            print(f"    got {slug}: {len(data)/1e6:.2f} MB (Content-Range={content_range})")
            return dest, url, content_range
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"    attempt {attempt} failed: {type(exc).__name__}: {exc}")
    raise last_exc


def count_jsonl(path: Path):
    """Count complete JSONL records; drop a trailing partial line from a ranged read."""
    kept, dropped = 0, 0
    keys = set()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        last = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not line.endswith("}"):
                dropped += 1
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            kept += 1
            if kept <= 200 and isinstance(rec, dict):
                keys.update(rec.keys())
            last = rec
    return kept, dropped, sorted(keys), (json.dumps(last, ensure_ascii=False)[:300] if last else None)


def main() -> int:
    manifest = {"fetched_at": datetime.now(timezone.utc).isoformat(), "mirror": MIRROR, "files": []}
    attempted = False
    for slug, kind, repo, rel, cap in JOBS:
        print(f"\n[{slug}] {repo} :: {rel}")
        if attempted:
            time.sleep(25)  # mirror rate limit guard between jobs
        attempted = True
        try:
            dest, url, range_info = fetch(slug, kind, repo, rel, cap)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {exc}")
            manifest["files"].append({"slug": slug, "repo": repo, "path": rel, "status": "failed", "error": str(exc)})
            continue
        n, dropped, keys, sample = count_jsonl(dest)
        info = {
            "slug": slug, "repo": repo, "path": rel, "url": url,
            "status": "ok", "local": str(dest.relative_to(ROOT)),
            "bytes": dest.stat().st_size, "sha256": sha256(dest),
            "range_requested": cap, "range_returned": range_info,
            "records_complete": n, "records_partial_dropped": dropped,
            "keys_first200": keys, "last_record_sample": sample,
        }
        if cap:
            info["sampling_bias_note"] = (
                "仅取该文件前 %d 字节（约 %.1f MB，Ranged GET），不是全量随机抽样；"
                "配比结论只能适用于“按文件顺序的前缀样本”。" % (cap, cap / 1e6)
            )
        manifest["files"].append(info)
        print(f"    records={n} partial_dropped={dropped} keys={keys}")
    out = ROOT / "data" / "raw_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
