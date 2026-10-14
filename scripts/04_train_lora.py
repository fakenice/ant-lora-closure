"""LoRA SFT on one ratio config, with prefix-masking and the loss path verified.

Small-scale by design: Qwen2.5-1.5B-Instruct, fp16, LoRA r16, micro-batch 1 with
gradient accumulation, gradient checkpointing, and a fixed optimizer-step budget so all
four configs see exactly the same number of updates.

Two things are measured rather than claimed:
  * prefix masking is checked against the tokenised prompt (mask_check event);
  * the hand-rolled response-only cross-entropy is compared against HF's own
    `out.loss` on the same batch before training starts (loss_check event), because
    the entire memory saving comes from *not* materialising fp32 logits over the
    151,936-token vocabulary at prompt positions.

VRAM accounting matters here: the first attempt at MAX_LEN=640 drove this 2080 Ti's
11 GB down to 116 MiB free, WDDM began spilling to shared system memory, and step
time went from 4.6 s to over 6 min. So rows longer than --max_len are dropped whole
(counted in the encode event) instead of trained on a silently truncated answer, the
loss is computed only on supervised positions, the allocator cache is emptied every
optimizer step, and the run aborts after two consecutive steps slower than
--slow_step_abort rather than burning an hour thrashing.

Usage: .venv/Scripts/python.exe scripts/04_train_lora.py --config r70_30 [--steps 150]
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RATIOS = ROOT / "data" / "ratios"
MODELS = ROOT / "models"
OUT = ROOT / "out"

# Local copy (fetched via scripts/03c_fetch_weights.py). The hub id is not usable
# here because hf-mirror aborts snapshot_download through the Xet CAS backend.
BASE = str(MODELS / "Qwen2.5-1.5B-Instruct")
if not (MODELS / "Qwen2.5-1.5B-Instruct" / "model.safetensors").exists():
    raise SystemExit("model.safetensors missing in models/Qwen2.5-1.5B-Instruct; "
                     "run scripts/03c_fetch_weights.py first")
IGNORE = -100


def build_example(tokenizer, messages, max_len):
    """Return (ids, labels, masked, n_full, n_prompt, too_long).

    Prompt tokens are masked to IGNORE only when the tokenised prompt really is a
    prefix of the tokenised full text; otherwise the mismatch is reported. Rows over
    max_len tokens are flagged so the caller can drop them rather than learn a
    cut-off answer.
    """
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    ids_full = tokenizer(full)["input_ids"]
    too_long = len(ids_full) > max_len
    ids = ids_full[:max_len]
    p_ids = tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"]
    labels = list(ids)
    masked = False
    if ids[: len(p_ids)] == p_ids:
        labels[: len(p_ids)] = [IGNORE] * len(p_ids)
        masked = True
    return ids, labels, masked, len(ids_full), len(p_ids), too_long


def response_ce(model, ids, labels):
    """Next-token cross-entropy over supervised positions only, before the lm_head.

    logits[t] predicts labels[t+1]. The supervised span is a suffix (the prompt is
    masked), so hidden states are sliced to [lo, hi) and the lm_head runs only on
    those rows: on an MCQ row (~110 tokens, ~2 supervised) that turns a
    (110 x 151,936) fp16 logits tensor into a (2 x 151,936) one, which is what buys
    the VRAM headroom on this 11 GB card. The result is checked against HF's own
    loss in the loss_check event before training starts.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=ids).last_hidden_state          # (1, T, H)
    tgt = labels[:, 1:]
    sel = tgt != IGNORE
    idx = sel[0].nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return None, 0
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    logits = base.lm_head(hidden[:, lo:hi])                        # (1, hi-lo, V) fp16
    sub_sel = sel[:, lo:hi]
    loss = F.cross_entropy(logits[sub_sel].float(), tgt[:, lo:hi][sub_sel], reduction="mean")
    return loss, int(sel.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--slow_step_abort", type=float, default=25.0,
                    help="abort after two optimizer steps slower than this (VRAM spill guard)")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    run_dir = MODELS / f"lora_{a.config}"
    log_f = (OUT / f"train_{a.config}.jsonl").open("w", encoding="utf-8")

    def log(event: str, **kw) -> None:
        log_f.write(json.dumps({"event": event, **kw}, ensure_ascii=False) + "\n")
        log_f.flush()

    rows = []
    with (RATIOS / f"{a.config}.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.endswith("}"):
                rows.append(json.loads(line))
    print(f"config={a.config} samples={len(rows)} max_len={a.max_len}")

    torch.manual_seed(20261014)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="cuda:0")
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # KV cache is dead weight during training

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable/1e6:.2f} M / {total/1e6:.2f} M ({100*trainable/total:.2f}%)")
    log("setup", config=a.config, samples=len(rows), trainable_params=trainable,
        total_params=total, max_len=a.max_len, steps=a.steps, accum=a.accum, lr=a.lr,
        base=BASE, lora={"r": 16, "alpha": 32, "dropout": 0.05},
        weights_sha256="dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
        torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        gpu_total_mib=round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1),
        loss_fn="response-position CE (verified against HF loss, see loss_check)")

    # --- masking verification on the first 200 records ------------------------------
    probe = [build_example(tok, r["messages"], a.max_len) for r in rows[:200]]
    mask_ok = sum(1 for p in probe if p[2])
    print(f"prefix-mask verified on {mask_ok}/{len(probe)} probe records")
    log("mask_check", ok=mask_ok, n=len(probe))
    if mask_ok < len(probe):
        print("  [warn] some prompts are not exact token prefixes; those rows train on full text")

    # --- loss-path verification: hand-rolled CE vs HF's own loss -------------------
    ids0, labels0 = None, None
    for r in rows[:20]:
        cand = build_example(tok, r["messages"], a.max_len)
        if cand[2] and not cand[5] and any(v != IGNORE for v in cand[1]):
            ids0, labels0 = cand[0], cand[1]
            break
    if ids0 is None:
        raise SystemExit("no usable row for the loss check")
    x0 = torch.tensor([ids0], device="cuda:0")
    y0 = torch.tensor([labels0], device="cuda:0")
    with torch.no_grad():
        ref = model(input_ids=x0, labels=y0).loss.item()
        with torch.autocast("cuda", dtype=torch.float16):
            mine_t, n_sup = response_ce(model, x0, y0)
        mine = float(mine_t.item())
    diff = abs(ref - mine)
    log("loss_check", n_tokens=len(ids0), supervised_tokens=n_sup, hf_loss=round(ref, 6),
        response_ce=round(mine, 6), abs_diff=round(diff, 6),
        rel_diff=round(diff / max(1e-9, abs(ref)), 8))
    print(f"loss check: HF={ref:.6f} response-CE={mine:.6f} |diff|={diff:.2e}")
    if diff > 0.02:
        raise SystemExit(f"loss implementations disagree by {diff:.4f}; refusing to train")

    # --- encode every row, dropping over-length rows --------------------------------
    enc, dropped_long, dropped_empty, unmasked = [], 0, 0, 0
    for r in rows:
        ids, labels, masked, _nf, _np, too_long = build_example(tok, r["messages"], a.max_len)
        if too_long:
            dropped_long += 1
            continue
        if not masked:
            unmasked += 1
        if all(v == IGNORE for v in labels):
            dropped_empty += 1
            continue
        enc.append((ids, labels))
    print(f"encoded {len(enc)} rows (dropped {dropped_long} over {a.max_len} tok, "
          f"{dropped_empty} with no trainable token; {unmasked} unmasked)")
    log("encode", kept=len(enc), dropped_too_long=dropped_long,
        dropped_no_trainable=dropped_empty, unmasked_rows=unmasked,
        train_tokens=sum(len(i) for i, _ in enc))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    dev = torch.device("cuda:0")
    micro, all_step_losses = [], []
    t_all = t_step = time.time()
    prev_marginal = 0.0
    peak_alloc = peak_reserved = 0.0
    gstep = 0
    rng = torch.Generator().manual_seed(20261014)
    order = torch.randperm(len(enc), generator=rng).tolist()
    ptr = 0
    opt.zero_grad(set_to_none=True)

    while gstep < a.steps:
        if ptr >= len(enc):
            order = torch.randperm(len(enc), generator=rng).tolist()
            ptr = 0
        ids, labels = enc[order[ptr]]
        ptr += 1
        x = torch.tensor([ids], device=dev)
        y = torch.tensor([labels], device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            loss, _n_sup = response_ce(model, x, y)
        if loss is None:
            continue
        (loss / a.accum).backward()
        micro.append(float(loss.detach().item()))
        if len(micro) < a.accum:
            continue

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        gstep += 1
        if gstep <= a.warmup:
            for gp in opt.param_groups:
                gp["lr"] = a.lr * gstep / a.warmup
        now = time.time()
        marginal = now - t_step
        t_step = now
        peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated(dev) / 2**20)
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(dev) / 2**20)
        torch.cuda.empty_cache()
        avg = sum(micro) / len(micro)
        all_step_losses.append(avg)
        micro = []
        log("step", gstep=gstep, loss_avg=round(avg, 4), marginal_s=round(marginal, 2),
            s_per_step_mean=round((now - t_all) / gstep, 2),
            peak_alloc_mib=round(peak_alloc, 1), peak_reserved_mib=round(peak_reserved, 1),
            lr=round(opt.param_groups[0]["lr"], 8), seq_len=len(ids))
        if gstep % 10 == 0 or gstep == a.steps:
            print(f"  step {gstep}/{a.steps} loss={avg:.3f} marginal={marginal:.1f}s "
                  f"peakAlloc={peak_alloc:.0f}MiB peakResv={peak_reserved:.0f}MiB")
        if gstep >= 3 and marginal > a.slow_step_abort and prev_marginal > a.slow_step_abort:
            log("abort", reason="slow_step", marginal_s=round(marginal, 1),
                threshold=a.slow_step_abort,
                hint="GPU memory saturated (WDDM spill): lower --max_len or free other GPU users")
            print(f"  [abort] two consecutive steps > {a.slow_step_abort:.0f}s "
                  f"(last {marginal:.0f}s); stopping instead of thrashing")
            raise SystemExit(3)
        prev_marginal = marginal

    model.save_pretrained(run_dir)
    tok.save_pretrained(run_dir)
    summary = {"config": a.config, "steps": gstep,
               "first_loss": round(sum(all_step_losses[:10]) / len(all_step_losses[:10]), 4),
               "final_loss": round(sum(all_step_losses[-10:]) / len(all_step_losses[-10:]), 4),
               "min_loss": round(min(all_step_losses), 4),
               "s_per_step_overall": round((time.time() - t_all) / max(1, gstep), 2),
               "peak_alloc_mib": round(peak_alloc, 1), "peak_reserved_mib": round(peak_reserved, 1),
               "trainable_rows": len(enc), "dropped_too_long": dropped_long,
               "adapter_dir": str(run_dir.relative_to(ROOT))}
    log("done", **summary)
    print(json.dumps(summary, ensure_ascii=False))
    log_f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
