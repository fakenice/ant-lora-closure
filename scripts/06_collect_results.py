"""Collect stage-4 training + evaluation logs into one ablation table.

Reads only files that already exist (never re-runs the model), and writes:
  out/summary.json        machine-readable: per-config data mix, training cost, eval result
  out/ablation.csv        one row per run (base + 4 ratio configs)
  out/ablation.md         the markdown table for the README
  out/resources.md        compute/VRAM/throughput accounting

Accuracy CIs are Wilson intervals at n items; the 95% half-width at n=100 is about
10 percentage points, so config-vs-config gaps smaller than that are reported as
"不可区分" rather than as an effect.
"""
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
CONFIGS = ["r100_0", "r70_30", "r30_70", "r0_100"]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.endswith("}"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def parse_json_blob(text: str) -> dict:
    """The eval log prints a JSON summary; recover it if the json file is missing."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def train_stats(cfg: str) -> dict:
    """Read the event log written by 04_train_lora.py.

    Field names here are the ones that script actually emits (event=setup / mask_check /
    loss_check / encode / step / done). A run whose log has no `done` event is reported as
    incomplete instead of being silently averaged - that is how the r70_30 numbers got lost
    once (a relaunched run truncated the events file at step 12 and died before flushing stdout).
    """
    events = read_jsonl(OUT / f"train_{cfg}.jsonl")
    setup = next((e for e in events if e.get("event") == "setup"), {})
    mask = next((e for e in events if e.get("event") == "mask_check"), {})
    loss_chk = next((e for e in events if e.get("event") == "loss_check"), {})
    enc = next((e for e in events if e.get("event") == "encode"), {})
    done = next((e for e in events if e.get("event") == "done"), {})
    steps = [e for e in events if e.get("event") == "step"]
    losses = [s["loss_avg"] for s in steps if "loss_avg" in s]
    wall = round(sum(s.get("marginal_s", 0.0) for s in steps), 1)
    want = setup.get("steps")
    return {
        "samples": setup.get("samples"),
        "trainable_params": setup.get("trainable_params"),
        "total_params": setup.get("total_params"),
        "max_len": setup.get("max_len"),
        "micro_batch": setup.get("accum"),
        "optimizer_steps": setup.get("steps"),
        "lr": setup.get("lr"),
        "train_log_complete": bool(done) and (want is None or len(steps) == want),
        "steps_logged": len(steps),
        "mask_probe_ok": f"{mask.get('ok')}/{mask.get('n')}",
        "loss_fn_abs_diff": loss_chk.get("abs_diff"),
        "rows_with_trainable_tokens": enc.get("kept"),
        "rows_dropped_too_long": enc.get("dropped_too_long"),
        "rows_dropped_no_trainable": enc.get("dropped_no_trainable"),
        "loss_first10": round(sum(losses[:10]) / len(losses[:10]), 4) if losses else None,
        "loss_last10": round(sum(losses[-10:]) / len(losses[-10:]), 4) if losses else None,
        "loss_min": round(min(losses), 4) if losses else None,
        "loss_first": done.get("first_loss"), "loss_final": done.get("final_loss"),
        "s_per_step": (done.get("s_per_step_overall")
                       if done.get("s_per_step_overall") is not None
                       else (steps[-1].get("s_per_step_mean") if steps else None)),
        "peak_alloc_mib": done.get("peak_alloc_mib") or (steps[-1].get("peak_alloc_mib") if steps else None),
        "peak_reserved_mib": (done.get("peak_reserved_mib")
                              or (steps[-1].get("peak_reserved_mib") if steps else None)),
        "train_wall_s": wall or None,
    }


def eval_stats(tag: str) -> dict:
    path = OUT / "eval" / f"eval_{tag}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    log = ROOT / "logs" / f"eval_{tag}.log"
    return parse_json_blob(log.read_text(encoding="utf-8")) if log.exists() else {}


def main() -> int:
    ratios = json.loads((ROOT / "data" / "ratios" / "ratios_manifest.json")
                        .read_text(encoding="utf-8"))
    mix = {c["name"]: c for c in ratios["configs"]}

    rows = []
    base = eval_stats("base")
    if base:
        rows.append({"run": "base（无 SFT）", "kind": "baseline", "mix_dialog": None,
                     "mix_mcq": None, "eval": base})
    for cfg in CONFIGS:
        ev = eval_stats(f"lora_{cfg}")
        m = mix.get(cfg) or {}
        dialog, mcq = m.get("n_dialog"), m.get("n_mcq")
        rows.append({"run": cfg, "kind": "lora", "mix_dialog": dialog, "mix_mcq": mcq,
                     "train": train_stats(cfg), "eval": ev})

    summary = {
        "eval_set": {"n": base.get("n_items") if base else None,
                     "source": "CMExam (fzkuji/CMExam) test split, deterministic exact-letter scoring",
                     "decoding": base.get("decoding"), "max_new_tokens": base.get("max_new_tokens")},
        "ci_note": "Wilson 95% 区间；n=100 时半宽约 10 个百分点，区间重叠的差值不可作为结论",
        "runs": [],
    }
    for r in rows:
        ev = r["eval"]
        rec = {"run": r["run"], "kind": r["kind"], "dialog_rows": r["mix_dialog"],
               "mcq_rows": r["mix_mcq"]}
        rec.update(r.get("train") or {})
        if ev:
            n = ev["n_items"]
            lo, hi = wilson(ev["correct"], n)
            rec.update({
                "n_items": n, "correct": ev["correct"], "wrong": ev["wrong"],
                "unparseable": ev["unparseable"], "accuracy": ev["accuracy"],
                "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                "mean_new_tokens": ev.get("mean_new_tokens"),
                "s_per_item": ev.get("s_per_item"),
                "eval_peak_vram_mib": ev.get("peak_vram_mib"),
                "parse_paths": ev.get("parse_path_counts"),
            })
        else:
            rec["n_items"] = None
        summary["runs"].append(rec)

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    fields = ["run", "kind", "dialog_rows", "mcq_rows", "optimizer_steps", "micro_batch",
              "train_log_complete", "steps_logged", "s_per_step", "train_wall_s",
              "peak_alloc_mib", "peak_reserved_mib",
              "loss_first", "loss_final", "loss_min", "loss_first10", "loss_last10",
              "rows_with_trainable_tokens", "rows_dropped_too_long", "mask_probe_ok", "loss_fn_abs_diff",
              "n_items", "correct", "wrong", "unparseable", "accuracy", "ci95_low", "ci95_high",
              "mean_new_tokens", "s_per_item", "eval_peak_vram_mib"]
    with (OUT / "ablation.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for rec in summary["runs"]:
            w.writerow(rec)

    def pct(x):
        return "-" if x is None else f"{100*x:.1f}%"

    md = ["| 运行 | 对话:考题 | 步数(日志/计划) | s/步 | 峰值 alloc/reserved MiB | 首/末 loss | 后10步 loss | 准确率 | 95% CI | 错答 | 不可解析 | 平均生成 token |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rec in summary["runs"]:
        mix = (f"{rec['dialog_rows']}:{rec['mcq_rows']}"
               if rec["dialog_rows"] is not None and rec["mcq_rows"] is not None else "-")
        ci = ("-" if rec.get("accuracy") is None
              else f"[{pct(rec.get('ci95_low'))}, {pct(rec.get('ci95_high'))}]")
        steps_col = ("%s/%s%s" % (rec.get("steps_logged") or "-", rec.get("optimizer_steps") or "-",
                                  "" if rec.get("train_log_complete") in (True, None) else " !"))
        vram = ("%s / %s" % (rec.get("peak_alloc_mib") or "-", rec.get("peak_reserved_mib") or "-")
                if rec.get("peak_alloc_mib") else "-")
        loss_rng = ("%s → %s" % (rec.get("loss_first") or "-", rec.get("loss_final") or "-")
                    if rec.get("loss_first") is not None else "-")
        md.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            rec["run"], mix, steps_col,
            rec.get("s_per_step") or "-", vram, loss_rng,
            rec.get("loss_last10") or "-",
            pct(rec.get("accuracy")), ci, rec.get("wrong", "-"),
            rec.get("unparseable", "-"), rec.get("mean_new_tokens") or "-"))
    (OUT / "ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    res = ["# 算力与资源账（本机 RTX 2080 Ti 11 GB，fp16 + LoRA r16）", "",
           "| 运行 | 可训练参数 | 样本数 | 优化步 | 梯度累积 | s/步 | 训练墙钟 s | 峰值 alloc MiB | 峰值 reserved MiB | 评测 s/题 | 评测峰值显存 MiB |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for rec in summary["runs"]:
        tp = rec.get("trainable_params")
        res.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            rec["run"], f"{tp/1e6:.2f} M" if tp else "-", rec.get("samples") or "-",
            rec.get("optimizer_steps") or "-", rec.get("micro_batch") or "-",
            rec.get("s_per_step") or "-", rec.get("train_wall_s") or "-",
            rec.get("peak_alloc_mib") or "-", rec.get("peak_reserved_mib") or "-",
            rec.get("s_per_item") or "-", rec.get("eval_peak_vram_mib") or "-"))
    res += ["", "评测口径：100 题 CMExam 单选题，贪心解码，exact-letter 计分，无 LLM judge。",
            f"基座：Qwen2.5-1.5B-Instruct（fp16 权重 3,087,467,144 B，本地 `models/`）。"]
    (OUT / "resources.md").write_text("\n".join(res) + "\n", encoding="utf-8")

    print(f"wrote out/summary.json, ablation.csv/md, resources.md for {len(summary['runs'])} runs")
    for rec in summary["runs"]:
        print(f"  {rec['run']}: acc={rec.get('accuracy')} ci={[rec.get('ci95_low'), rec.get('ci95_high')]} "
              f"unparseable={rec.get('unparseable')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
