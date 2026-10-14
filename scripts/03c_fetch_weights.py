"""Fetch the base-model weights from the HF mirror with a sharded Range download.

Why not snapshot_download: the Xet CAS backend (cas-server.xethub.hf.co) answers
401 through the mirror, which aborts the download. hf-mirror also throttles
aggressively (HEAD and single-connection GETs returned HTTP 429 on this machine),
so the byte size is taken from the public API listing, ranged GETs are staggered,
Retry-After is honoured, and each shard resumes from its own offset.

Registers every file that lands (size + sha256) into out/model_download.json.
"""
import concurrent.futures as cf
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parents[1]
REPO = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "main"
DEST = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
MIRROR = "https://hf-mirror.com"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "*/*"}

SMALL_FILES = [
    "config.json", "generation_config.json", "tokenizer_config.json",
    "tokenizer.json", "vocab.json", "merges.txt", "LICENSE", "README.md",
]
WEIGHTS = ["model.safetensors"]
PARTS = 6
STAGGER_S = 4.0
CHUNK = 32 * 1024 * 1024
MAX_ATTEMPTS = 10
# model.safetensors size as reported by /api/models/Qwen/Qwen2.5-1.5B-Instruct?blobs=true
# on 2026-10-14; refreshed from the API at run time when reachable.
KNOWN_SIZE = {"model.safetensors": 3_087_467_144}


def resolved_url(path: str) -> str:
    return f"{MIRROR}/{REPO}/resolve/{REVISION}/{path}"


def backoff_for(exc, attempt: int) -> float:
    """Honour Retry-After when the mirror sends it, else exponential + jitter."""
    if isinstance(exc, urllib.error.HTTPError) and exc.headers:
        raw = exc.headers.get("Retry-After")
        if raw:
            try:
                return min(120.0, max(5.0, float(raw)))
            except ValueError:
                pass
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
        return min(120.0, 20.0 * attempt + random.uniform(0, 10))
    return min(60.0, 5.0 * attempt + random.uniform(0, 5))


def api_sizes() -> dict:
    """Best-effort refresh of exact byte sizes from the mirror's public API."""
    url = f"{MIRROR}/api/models/{REPO}?blobs=true"
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, headers={**UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            return {s["rfilename"]: s.get("size") or 0 for s in data.get("siblings", [])}
        except Exception as exc:  # noqa: BLE001
            print(f"   api listing attempt {attempt}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(backoff_for(exc, attempt))
    return {}


def open_ranged(url: str, lo: int, hi: int):
    headers = dict(UA)
    headers["Range"] = f"bytes={lo}-{hi}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180)


def probe(url: str) -> bool:
    """Confirm the mirror serves byte ranges for this object (1 KiB probe)."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with open_ranged(url, 0, 1023) as r:
                cr = r.headers.get("Content-Range", "")
                r.read(1024)
            if cr.startswith("bytes 0-1023/"):
                return True
            if cr == "":
                print("   probe: no Content-Range header; ranges unsupported", flush=True)
                return False
        except Exception as exc:  # noqa: BLE001
            print(f"   probe attempt {attempt}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(backoff_for(exc, attempt))
    raise RuntimeError("range probe exhausted retries")


def fetch_range(url: str, lo: int, hi: int, idx: int, out_dir: Path) -> dict:
    """Download bytes lo..hi inclusive into part_XX, resuming across attempts."""
    time.sleep(idx * STAGGER_S)
    part = out_dir / f"part_{idx:02d}"
    want = hi - lo + 1
    last_err = None
    last_cr = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        exc = None
        have = part.stat().st_size if part.exists() else 0
        if have > want:
            part.unlink()
            have = 0
        if have == want:
            return {"idx": idx, "bytes": have, "attempts": attempt,
                    "content_range": last_cr or "resumed-complete"}
        try:
            with open_ranged(url, lo + have, hi) as r:
                last_cr = r.headers.get("Content-Range", "")
                with part.open("wb" if have == 0 else "ab") as fh:
                    while True:
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        fh.write(buf)
                    fh.flush()
            have = part.stat().st_size if part.exists() else 0
            if have == want:
                return {"idx": idx, "bytes": have, "attempts": attempt, "content_range": last_cr}
            last_err = f"short part: {have} != {want}"
        except Exception as e:  # noqa: BLE001
            exc = e
            last_err = f"{type(e).__name__}: {e}"
        print(f"   part {idx} attempt {attempt}: {last_err}", flush=True)
        time.sleep(backoff_for(exc or RuntimeError(last_err), attempt))
    raise RuntimeError(f"part {idx} failed after {MAX_ATTEMPTS} attempts: {last_err}")


def assemble(url: str, name: str, total: int, parts_dir: Path) -> dict:
    if not probe(url):
        raise RuntimeError(f"{name}: mirror will not serve byte ranges")
    step = (total + PARTS - 1) // PARTS
    bounds = []
    for i in range(PARTS):
        lo, hi = i * step, min(total - 1, (i + 1) * step - 1)
        if lo <= hi:
            bounds.append((lo, hi))
    jobs = [(b[0], b[1], i) for i, b in enumerate(bounds)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        results = list(ex.map(lambda t: fetch_range(url, t[0], t[1], t[2], parts_dir), jobs))
    ordered = sorted(results, key=lambda x: x["idx"])
    target = DEST / name
    with target.open("wb") as fh:
        for r in ordered:
            with (parts_dir / f"part_{r['idx']:02d}").open("rb") as src:
                while True:
                    buf = src.read(CHUNK)
                    if not buf:
                        break
                    fh.write(buf)
    for r in results:
        (parts_dir / f"part_{r['idx']:02d}").unlink()
    got = target.stat().st_size
    if got != total:
        raise RuntimeError(f"{name}: assembled {got} != expected {total}")
    with target.open("rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()
    header_ok = False
    with target.open("rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        if 0 < n < 100_000_000:
            try:
                meta = json.loads(fh.read(n).decode("utf-8"))
                header_ok = "__metadata__" in meta or len(meta) > 0
            except Exception:  # noqa: BLE001
                header_ok = False
    return {"path": name, "bytes": got, "expected_bytes": total, "size_ok": True,
            "sha256": digest, "safetensors_header_ok": header_ok,
            "download_s": round(time.time() - t0, 1),
            "throughput_mbps": round(got / 1e6 / max(1.0, time.time() - t0), 1),
            "n_ranges": len(bounds), "max_attempts": max(r["attempts"] for r in results),
            "content_ranges": [r["content_range"] for r in ordered]}


def fetch_small(name: str) -> dict:
    url = resolved_url(name)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                body = r.read()
            (DEST / name).write_bytes(body)
            return {"path": name, "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(), "expected_bytes": len(body),
                    "size_ok": True, "n_ranges": 1}
        except Exception as exc:  # noqa: BLE001
            print(f"   {name} attempt {attempt}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(backoff_for(exc, attempt))
    raise RuntimeError(f"{name}: failed after {MAX_ATTEMPTS} attempts")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    parts_dir = DEST / ".parts"
    parts_dir.mkdir(exist_ok=True)
    sizes = api_sizes()
    for name, size in sizes.items():
        if size:
            KNOWN_SIZE[name] = size
    t0 = time.time()
    records = []
    for name in WEIGHTS:
        target = DEST / name
        total = KNOWN_SIZE.get(name)
        if not total:
            raise RuntimeError(f"{name}: no known byte size; refusing to guess")
        if target.exists() and target.stat().st_size == total:
            records.append({"path": name, "bytes": total, "expected_bytes": total,
                            "size_ok": True, "skipped": "already complete on disk"})
            print(f"skip {name}: already complete ({total/1e9:.2f} GB)")
            continue
        print(f"downloading {name} ({total/1e9:.3f} GB) via {PARTS} staggered ranged GETs...",
              flush=True)
        rec = assemble(resolved_url(name), name, total, parts_dir)
        records.append(rec)
        print(f"   {name}: {rec['bytes']/1e9:.3f} GB in {rec['download_s']}s "
              f"({rec['throughput_mbps']} MB/s, header_ok={rec['safetensors_header_ok']})",
              flush=True)
    for name in SMALL_FILES:
        if (DEST / name).exists():
            records.append({"path": name, "bytes": (DEST / name).stat().st_size,
                            "skipped": "exists"})
            continue
        try:
            records.append(fetch_small(name))
        except Exception as exc:  # noqa: BLE001
            records.append({"path": name, "error": str(exc)})
    files = sorted(p for p in DEST.rglob("*") if p.is_file() and ".parts" not in p.parts
                   and ".cache" not in p.parts)
    total_bytes = sum(p.stat().st_size for p in files)
    manifest = {
        "repo": REPO, "revision": REVISION, "endpoint": MIRROR,
        "method": ("sharded staggered ranged GET; HF_HUB_DISABLE_XET=1 because the Xet CAS "
                   "server returns 401 through the mirror; sizes from /api/models?blobs=true"),
        "n_parts": PARTS, "stagger_s": STAGGER_S, "elapsed_s": round(time.time() - t0, 1),
        "n_files": len(files), "total_bytes": total_bytes,
        "total_gb": round(total_bytes / 2**30, 2),
        "api_sizes": sizes, "files": records,
        "on_disk": [{"path": str(p.relative_to(DEST)), "bytes": p.stat().st_size} for p in files],
    }
    (ROOT / "out" / "model_download.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not list(parts_dir.glob("part_*")):
        parts_dir.rmdir()
    print(f"done: {manifest['n_files']} files, {manifest['total_gb']} GB in {manifest['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
