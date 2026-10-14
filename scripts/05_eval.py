"""Deterministic MCQ evaluation: base model vs each LoRA adapter.

Scoring is exact letter matching, no LLM judge. Three outcomes are separated so the
number cannot hide a format failure:
  correct      parsed letter == gold
  wrong        parsed letter != gold but is a valid option
  unparseable  no single valid option letter emitted (format compliance failure)

Usage: .venv/Scripts/python.exe scripts/05_eval.py --adapter base
       .venv/Scripts/python.exe scripts/05_eval.py --adapter models/lora_r70_30
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
EVAL_SET = ROOT / "data" / "clean" / "mcq_eval_100.jsonl"
OUT = ROOT / "out" / "eval"
MAX_NEW = 12
LETTER = re.compile(r"([A-E])")

SYSTEM = "你是一名严谨的医学助手。请用中文回答。"


def load_eval():
    rows = []
    with EVAL_SET.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.endswith("}"):
                rows.append(json.loads(line))
    return rows


def parse_letter(text, valid_letters):
    """Take the first standalone option letter; prefer '答案: X' style prefixes."""
    m = re.search(r"答案\s*[:：]?\s*\(?([A-E])\)?", text)
    if m and m.group(1) in valid_letters:
        return m.group(1), "answer-marker"
    for m in LETTER.finditer(text):
        if m.group(1) in valid_letters:
            return m.group(1), "first-letter"
    return None, "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="base", help="'base' or a path under models/")
    ap.add_argument("--limit", type=int, default=0, help="optional subset for a smoke test")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = "base" if a.adapter == "base" else Path(a.adapter).name
    rows = load_eval()
    if a.limit:
        rows = rows[: a.limit]
    print(f"eval items={len(rows)} adapter={a.adapter}")

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="cuda:0")
    if a.adapter != "base":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(ROOT / a.adapter))
        model = model.merge_and_unload()
        print(f"  merged adapter {a.adapter}")
    model.eval()

    results = []
    t0 = time.time()
    for i, r in enumerate(rows):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["prompt"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        x = tok(text, return_tensors="pt", truncation=True, max_length=900).to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**x, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        out_ids = gen[0][x["input_ids"].shape[1]:]
        raw = tok.decode(out_ids, skip_special_tokens=True)
        pred, how = parse_letter(raw.strip(), r["letters"])
        if pred is None:
            outcome = "unparseable"
        elif pred == r["gold"]:
            outcome = "correct"
        else:
            outcome = "wrong"
        results.append({"id": r["id"], "gold": r["gold"], "pred": pred, "parse_path": how,
                        "outcome": outcome, "raw": raw.strip()[:80],
                        "n_new_tokens": int(out_ids.shape[0])})
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(rows)} correct={sum(1 for z in results if z['outcome']=='correct')} "
                  f"({el:.0f}s, {el/(i+1):.2f}s/item)")

    n = len(results)
    c = sum(1 for z in results if z["outcome"] == "correct")
    w = sum(1 for z in results if z["outcome"] == "wrong")
    u = sum(1 for z in results if z["outcome"] == "unparseable")
    z = 1.96
    phat = c / n
    den = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / den
    half = z * (phat * (1 - phat) / n + z**2 / (4 * n**2)) ** 0.5 / den
    summary = {
        "tag": tag, "adapter": a.adapter, "n_items": n, "correct": c, "wrong": w, "unparseable": u,
        "accuracy": round(c / n, 4), "wrong_rate": round(w / n, 4), "unparseable_rate": round(u / n, 4),
        "wilson_ci95": [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)],
        "chance_note": "五选项单选，随机基线约 20%；评测集正确项字母最多的一档（D）占 26%",
        "mean_new_tokens": round(sum(x["n_new_tokens"] for x in results) / n, 2),
        "s_per_item": round((time.time() - t0) / n, 2),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "parse_path_counts": {k: sum(1 for z2 in results if z2["parse_path"] == k)
                              for k in {"answer-marker", "first-letter", "none"}},
        "max_new_tokens": MAX_NEW, "decoding": "greedy",
        "binomial_note": "n=%d 的准确率 95%% 半宽约 %.1f 个百分点（正态近似）"
                         % (n, 1.96 * (0.25 / n) ** 0.5 * 100),
    }
    (OUT / f"eval_{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / f"eval_{tag}_items.jsonl").write_text(
        "\n".join(json.dumps(z, ensure_ascii=False) for z in results), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
