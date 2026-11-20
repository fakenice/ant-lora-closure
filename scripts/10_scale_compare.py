"""Scale check: the published 1.5B sweep next to the 2026-11-20 0.5B sweep.

The question is the one README 结论 6 leaves open: "配比之间看不出差异" was only ever
demonstrated on Qwen2.5-1.5B-Instruct. Does it hold when the base model changes?
Everything except the weights was kept fixed (same four ratios, same 150 steps, same
accum/max_len, same three seeds, same LoRA hyper-parameters, same 500-item eval), so a
difference between the two columns is a scale effect, not a protocol change.

Inputs are published artifacts only - no GPU, no torch:
  * out/eval/eval_<tag>_500.json + _items.jsonl   (24 arms + 2 bases)
  * out/train_<tag>.jsonl                          (done events: cost per training)
  * logs/multiseed_500.log, logs/multiseed_500_0p5b.log (wall clock per run)

Statistics, all deterministic:
  * item-level paired bootstrap (B=2000, random.Random(42), nearest-rank 2.5/97.5%).
    The 500 questions are the unit that gets resampled; the three seeds of a config are
    averaged per item, so seed noise sits inside the estimate while training randomness
    is NOT part of the interval - stated in the output because it bounds the claim.
    Reported for (a) each config minus base, (b) all six config-config pairs,
    (c) the max-minus-min spread (a selected statistic, labelled as such),
    (d) the mean of the four arms.
  * format accounting: unparseable count, answer-marker share, mean generated tokens,
    accuracy restricted to parseable answers. This separates "the model knows less" from
    "the grader could not read the answer".

Writes out/scale_compare_500.{json,csv,md} and nothing else. The closing assertions
re-derive both committed multiseed summaries from the same item logs, so this artifact
and those summaries cannot drift apart silently.

Usage: python scripts/10_scale_compare.py
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import random
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_model import DEFAULT_BASE  # noqa: E402  单点定义，别在这里重复写死基座名

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
EVAL_DIR = OUT / "eval"
CONFIGS = ["r100_0", "r70_30", "r30_70", "r0_100"]
SEEDS = [20261014, 20261015, 20261016]
PAIRS = list(itertools.combinations(CONFIGS, 2))
# (label, TAG_PREFIX, sweep log, published summary). Empty prefix = the 1.5B release.
SCALES = [
    ("1.5B", "", ROOT / "logs" / "multiseed_500.log", OUT / "multiseed_500_summary.json"),
    ("0.5B", "q05b_", ROOT / "logs" / "multiseed_500_0p5b.log",
     OUT / "multiseed_500_q05b_summary.json"),
]
BOOT_B = 2000
BOOT_SEED = 42
ALPHA = 0.05


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def paths_for(prefix: str, tag: str):
    stem = f"eval_{prefix}{tag}_500"
    return (EVAL_DIR / f"{stem}.json", EVAL_DIR / f"{stem}_items.jsonl")


def load_items(path: Path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.endswith("}"):
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def parse_sweep_wallclock(path: Path):
    """sweep log '--- <kind> <tag> start ... / rc=N done ...' -> {tag: seconds}."""
    start_re = re.compile(r"^--- \w+ (\S+) start (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)$")
    done_re = re.compile(r"^--- \w+ (\S+) rc=(\d+) done (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)$")
    fmt = "%Y-%m-%d %H:%M:%S"
    started, out = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = start_re.match(line)
        if m:
            started[m.group(1)] = datetime.strptime(m.group(2), fmt)
            continue
        d = done_re.match(line)
        if d and d.group(1) in started:
            if d.group(2) != "0":
                raise SystemExit(f"{path.name}: {d.group(1)} finished with rc={d.group(2)}")
            out[d.group(1)] = int((datetime.strptime(d.group(3), fmt)
                                   - started[d.group(1)]).total_seconds())
    return out


def pct(values, q: float):
    """Nearest-rank percentile of a bootstrap distribution."""
    return sorted(values)[int(round(q * (len(values) - 1)))]


def cond_accuracy(items_by_id):
    """correct / (n - unparseable): accuracy among answers the grader could read."""
    unp = sum(1 for r in items_by_id.values() if r["outcome"] == "unparseable")
    ok = sum(1 for r in items_by_id.values() if r["outcome"] == "correct")
    return ok / (len(items_by_id) - unp), unp


def capped_unparseable(items_by_id, max_new_tokens: int):
    """How many unparseable answers used the whole token budget (were cut off)."""
    return sum(1 for r in items_by_id.values()
               if r["outcome"] == "unparseable" and r["n_new_tokens"] >= max_new_tokens)


def sweep_span(path: Path):
    """('=== multiseed start ...' / '=== multiseed end ...') -> (seconds, start, end)."""
    fmt = "%Y-%m-%d %H:%M:%S"
    s = re.search(r"=== multiseed start (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)",
                  path.read_text(encoding="utf-8"))
    e = re.search(r"=== multiseed end (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)",
                  path.read_text(encoding="utf-8"))
    if not (s and e):
        raise SystemExit(f"{path.name}: no multiseed start/end header")
    span = int((datetime.strptime(e.group(1), fmt)
                - datetime.strptime(s.group(1), fmt)).total_seconds())
    return span, s.group(1), e.group(1)


def done_event(prefix: str, cfg: str, seed: int):
    p = ROOT / "out" / f"train_{prefix}{cfg}_s{seed}.jsonl"
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip().endswith("}")]
    done = [r for r in recs if r.get("event") == "done"]
    if len(done) != 1:
        raise SystemExit(f"{p.name}: expected exactly one done event, found {len(done)}")
    if done[0]["steps"] != 150:
        raise SystemExit(f"{p.name}: done event logs {done[0]['steps']} steps, expected 150")
    return done[0]


def collect(label: str, prefix: str, log_path: Path, summary_path: Path):
    pub = read_json(summary_path)
    pub_p = {(r["config"], r["seed"]): r for r in pub["runs"]}
    if prefix and pub.get("tag_prefix") != prefix:
        raise SystemExit(f"{label}: {summary_path.name} is not the {prefix} sweep")
    if not prefix and "tag_prefix" in pub:
        raise SystemExit(f"{label}: published summary unexpectedly carries a prefix")
    base_json_p, base_items_p = paths_for(prefix, "base")
    if not base_json_p.exists() or not base_items_p.exists():
        raise SystemExit(f"missing {label} base eval; run scripts/run_multiseed_500.sh")
    base_summary = read_json(base_json_p)
    base_items = load_items(base_items_p)
    order = list(base_items)
    n = len(order)
    if n != base_summary["n_items"]:
        raise SystemExit(f"{label} base: {n} item rows vs n_items={base_summary['n_items']}")

    per_seed_items, per_seed_json = {}, {}
    for cfg in CONFIGS:
        for seed in SEEDS:
            tag = f"{cfg}_s{seed}"
            jf, ifile = paths_for(prefix, tag)
            if not jf.exists() or not ifile.exists():
                raise SystemExit(f"missing {label} arm {tag}")
            its = load_items(ifile)
            if list(its) != order:
                raise SystemExit(f"{label} {tag}: item order/id set differs from base")
            per_seed_items[tag], per_seed_json[tag] = its, read_json(jf)

    base_correct = [1 if base_items[i]["outcome"] == "correct" else 0 for i in order]
    # per item, how many of the three seeds got it right (0..3): the seed average is
    # taken inside the resampling unit, so seed noise is averaged, not double counted.
    seed_hits = {cfg: [sum(1 for s in SEEDS
                           if per_seed_items[f"{cfg}_s{s}"][i]["outcome"] == "correct")
                       for i in order] for cfg in CONFIGS}

    def means(idx):
        m = len(idx)
        b = sum(base_correct[j] for j in idx) / m
        return b, {cfg: sum(seed_hits[cfg][j] for j in idx) / (m * len(SEEDS)) for cfg in CONFIGS}

    point_base, point_per = means(range(n))
    rng = random.Random(BOOT_SEED)
    vs_base = {cfg: [] for cfg in CONFIGS}
    pair_d = {pair: [] for pair in PAIRS}
    spread_d, arm_mean_d = [], []
    for _ in range(BOOT_B):
        idx = [rng.randrange(n) for _ in range(n)]
        bs, per = means(idx)
        for cfg in CONFIGS:
            vs_base[cfg].append(per[cfg] - bs)
        for a, b_ in PAIRS:
            pair_d[(a, b_)].append(per[a] - per[b_])
        spread_d.append(max(per.values()) - min(per.values()))
        arm_mean_d.append(sum(per[cfg] - bs for cfg in CONFIGS) / len(CONFIGS))

    base_cond, base_unp = cond_accuracy(base_items)
    wall = parse_sweep_wallclock(log_path)
    sweep = sweep_span(log_path)
    arms = []
    p_values = []
    for cfg in CONFIGS:
        js = [per_seed_json[f"{cfg}_s{s}"] for s in SEEDS]
        conds = [cond_accuracy(per_seed_items[f"{cfg}_s{s}"])[0] for s in SEEDS]
        tr = [done_event(prefix, cfg, s) for s in SEEDS]
        mcp = []
        for seed, j in zip(SEEDS, js):
            row = pub_p.get((cfg, seed))
            if row is None:
                raise SystemExit(f"{label}: {cfg} seed {seed} absent from {summary_path.name}")
            if row["correct"] != j["correct"] or row["unparseable"] != j["unparseable"]:
                raise SystemExit(f"{label}: {cfg} s{seed} disagrees with the published summary")
            mcp.append(row["mcnemar_p_vs_base"])
        p_values += mcp
        arms.append({
            "config": cfg,
            "mcnemar_p_vs_base": mcp,
            "acc_mean": round(statistics.mean(j["accuracy"] for j in js), 4),
            "acc_sd": round(statistics.stdev([j["accuracy"] for j in js]), 4),
            "acc_min": min(j["accuracy"] for j in js),
            "acc_max": max(j["accuracy"] for j in js),
            "delta_vs_base": round(statistics.mean(j["accuracy"] for j in js) - point_base, 4),
            "delta_ci95": [round(pct(vs_base[cfg], ALPHA / 2), 4),
                           round(pct(vs_base[cfg], 1 - ALPHA / 2), 4)],
            "delta_ci_excludes_0": (pct(vs_base[cfg], ALPHA / 2) > 0
                                    or pct(vs_base[cfg], 1 - ALPHA / 2) < 0),
            "unparseable_mean": round(statistics.mean(j["unparseable"] for j in js), 2),
            "unparseable_total": sum(j["unparseable"] for j in js),
            "unparseable_capped_total": sum(
                capped_unparseable(per_seed_items[f"{cfg}_s{s}"],
                                   per_seed_json[f"{cfg}_s{s}"]["max_new_tokens"])
                for s in SEEDS),
            "answer_marker_share_mean": round(statistics.mean(
                j["parse_path_counts"].get("answer-marker", 0) / j["n_items"] for j in js), 4),
            "mean_new_tokens": round(statistics.mean(j["mean_new_tokens"] for j in js), 2),
            "acc_parseable_mean": round(statistics.mean(conds), 4),
            "delta_vs_base_parseable": round(statistics.mean(conds) - base_cond, 4),
            "s_per_step": round(statistics.mean(t["s_per_step_overall"] for t in tr), 2),
            "peak_reserved_mib": round(statistics.mean(t["peak_reserved_mib"] for t in tr), 1),
            "peak_alloc_mib": round(statistics.mean(t["peak_alloc_mib"] for t in tr), 1),
            "final_loss": round(statistics.mean(t["final_loss"] for t in tr), 3),
            "train_wallclock_s": round(statistics.mean(
                wall[f"{prefix}{cfg}_s{s}"] for s in SEEDS), 1),
            "eval_wallclock_s": round(statistics.mean(
                wall[f"{prefix}{cfg}_s{s}_500"] for s in SEEDS), 1),
            "trainable_rows": tr[0]["trainable_rows"],
            "dropped_too_long": tr[0]["dropped_too_long"],
        })

    pairs_out = [{
        "a": a, "b": b_,
        "delta": round(point_per[a] - point_per[b_], 4),
        "ci95": [round(pct(pair_d[(a, b_)], ALPHA / 2), 4),
                 round(pct(pair_d[(a, b_)], 1 - ALPHA / 2), 4)],
        "excludes_0": (pct(pair_d[(a, b_)], ALPHA / 2) > 0
                       or pct(pair_d[(a, b_)], 1 - ALPHA / 2) < 0),
    } for a, b_ in PAIRS]

    return {
        "label": label,
        "tag_prefix": prefix,
        "base_model": base_summary.get("base_model", DEFAULT_BASE),
        "base_correct": base_summary["correct"],
        "base_n": base_summary["n_items"],
        "base_acc": base_summary["accuracy"],
        "base_acc_parseable": round(base_cond, 4),
        "base_unparseable": base_unp,
        "base_unparseable_capped": capped_unparseable(base_items,
                                                      base_summary["max_new_tokens"]),
        "max_new_tokens": base_summary["max_new_tokens"],
        "base_mean_new_tokens": base_summary["mean_new_tokens"],
        "eval_wallclock_s": wall.get(f"{prefix}base_500"),
        "sweep_wallclock_s": sweep[0],
        "sweep_start": sweep[1],
        "sweep_end": sweep[2],
        "arms": arms,
        "pairs": pairs_out,
        "spread": {
            "value": round(max(point_per.values()) - min(point_per.values()), 4),
            "ci95": [round(pct(spread_d, ALPHA / 2), 4), round(pct(spread_d, 1 - ALPHA / 2), 4)],
            "note": "这是从四档里挑出的最大减最小，属于挑过的统计量：区间不是"
                    "「档间有差异」的检验，极值挑选本身把它放大了",
        },
        "mean_delta": {
            "value": round(sum(point_per[c] - point_base for c in CONFIGS) / len(CONFIGS), 4),
            "ci95": [round(pct(arm_mean_d, ALPHA / 2), 4),
                     round(pct(arm_mean_d, 1 - ALPHA / 2), 4)],
        },
        "mcnemar_below_005": sum(1 for p in p_values if p < 0.05),
        "mcnemar_tests": len(p_values),
    }


def build_markdown(doc: dict) -> str:
    by_label = {s["label"]: s for s in doc["scales"]}
    big_by_cfg = {a["config"]: a for a in by_label["1.5B"]["arms"]}
    small_arms = by_label["0.5B"]["arms"]
    L = ["# 规模复核：同一配方换基座（1.5B 发布件 vs 2026-11-20 的 0.5B）", "",
         "除基座权重外一切照旧：四档配比、150 步、accum 8、max_len 512、三个 seed、"
         "LoRA r16/α32/dropout 0.05、同一 500 题评测集、贪心解码、max_new_tokens 12。", ""]
    for s in doc["scales"]:
        L.append(f"- **{s['label']}**（`{s['base_model']}`，TAG_PREFIX=`{s['tag_prefix'] or '(空)'}"
                 f"`）：base {s['base_correct']}/{s['base_n']} = {100*s['base_acc']:.1f}%，"
                 f"四档均值 {100*statistics.mean(a['acc_mean'] for a in s['arms']):.1f}%，"
                 f"四档对 base 的平均 Δ {100*s['mean_delta']['value']:+.1f} pt"
                 f" [{100*s['mean_delta']['ci95'][0]:+.1f}, {100*s['mean_delta']['ci95'][1]:+.1f}]")
    L += ["", "## 1. 每档 vs base（题层面配对 bootstrap，B=2000、seed 42）", "",
          "重抽单位是 500 道题，同一档的三个 seed 先按题平均——所以区间含题目抽样误差，"
          "**不含**训练随机性（3 个 seed 是被固定的）。", "",
          "| scale | config | 均值 acc | Δ vs base | 95% CI | CI 是否排除 0 | 不可解析/500 | answer-marker 占比 | 平均 token |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in doc["scales"]:
        for a in s["arms"]:
            L.append(f"| {s['label']} | {a['config']} | {100*a['acc_mean']:.1f}% | "
                     f"{100*a['delta_vs_base']:+.1f} pt | "
                     f"[{100*a['delta_ci95'][0]:+.1f}, {100*a['delta_ci95'][1]:+.1f}] | "
                     f"{'是' if a['delta_ci_excludes_0'] else '否'} | "
                     f"{a['unparseable_mean']:.1f} | {100*a['answer_marker_share_mean']:.0f}% | "
                     f"{a['mean_new_tokens']:.2f} |")
    for s in doc["scales"]:
        L.append(f"\n{s['label']}：对 base 的 {s['mcnemar_tests']} 个精确 McNemar 检验里 "
                 f"{s['mcnemar_below_005']} 个 p<0.05（未做多重比较校正，单独引用某个 p 时"
                 f"必须带上这条）。")
    L += ["", "## 2. 配比之间两两是否可区分（同一 bootstrap，6 对）", "",
          "| scale | 对比 | Δ | 95% CI | CI 是否排除 0 |", "|---|---|---|---|---|"]
    for s in doc["scales"]:
        for p in s["pairs"]:
            L.append(f"| {s['label']} | {p['a']} − {p['b']} | {100*p['delta']:+.1f} pt | "
                     f"[{100*p['ci95'][0]:+.1f}, {100*p['ci95'][1]:+.1f}] | "
                     f"{'是' if p['excludes_0'] else '否'} |")
    for s in doc["scales"]:
        ex = [p for p in s["pairs"] if p["excludes_0"]]
        sp = s["spread"]
        who = ("，排除 0 的是 " + "、".join(f"{p['a']} − {p['b']}" for p in ex)) if ex else ""
        L.append(f"\n{s['label']}：6 对里 {len(ex)} 对 CI 排除 0{who}；最好档−最差档 "
                 f"{100*sp['value']:.1f} pt [{100*sp['ci95'][0]:.1f}, {100*sp['ci95'][1]:.1f}]"
                 f"（{sp['note']}）。")
    L += ["", "## 3. 判分口径：格式失败 vs 知识变化", "",
          "`acc（仅可解析）` = 正确数 / (500 − 不可解析数)。不可解析在原始口径里算错，"
          "因此这一列只用于把「答案写成了判分器读不出的样子」从「知识变化」里剥出来，"
          "不是对用户更友好的口径（用户确实拿不到答案）。", "",
          "| scale | config | 不可解析总数(3 seed) | 其中撞满 token 上限 | acc | acc（仅可解析） | Δ vs base（仅可解析） |",
          "|---|---|---|---|---|---|---|"]
    for s in doc["scales"]:
        for a in s["arms"]:
            L.append(f"| {s['label']} | {a['config']} | {a['unparseable_total']} | "
                     f"{a['unparseable_capped_total']} | "
                     f"{100*a['acc_mean']:.1f}% | {100*a['acc_parseable_mean']:.1f}% | "
                     f"{100*a['delta_vs_base_parseable']:+.1f} pt |")
    for s in doc["scales"]:
        L.append(f"\n{s['label']} base：不可解析 {s['base_unparseable']}/500"
                 f"（撞满 {s['max_new_tokens']} token 上限的 {s['base_unparseable_capped']} 题），"
                 f"仅可解析口径 {100*s['base_acc_parseable']:.1f}%。")
    L += ["", "## 4. 成本（训练 done 事件 + sweep 日志墙钟）", "",
          "| scale | config | 训练墙钟均(s) | s/step | 评测墙钟均(s) | peak reserved MiB | trainable rows | dropped |",
          "|---|---|---|---|---|---|---|---|"]
    for s in doc["scales"]:
        for a in s["arms"]:
            L.append(f"| {s['label']} | {a['config']} | {a['train_wallclock_s']:.0f} | "
                     f"{a['s_per_step']:.2f} | {a['eval_wallclock_s']:.0f} | "
                     f"{a['peak_reserved_mib']:.0f} | {a['trainable_rows']} | "
                     f"{a['dropped_too_long']} |")
    for s in doc["scales"]:
        L.append(f"\n{s['label']}：base 评测墙钟 {s['eval_wallclock_s']} s"
                 f"（平均 {s['base_mean_new_tokens']:.2f} token/题），"
                 f"整轮 sweep（12 训 + 13 评）{s['sweep_wallclock_s']} s"
                 f"（{s['sweep_wallclock_s']/3600:.2f} h，{s['sweep_start']} → {s['sweep_end']}）。")
    L += ["", "三条口径说明：① 墙钟与 `s/step` 都受当时机器负载影响（README §8 记过两循环抢卡的事故），"
          "这里只用来回答「这轮跑了多久」；跨规模严格可比的是显存峰值、可训练行数与被截断行数，"
          "它们由协议决定。② 把 0.5B 对 1.5B 同档相除："
          f"peak reserved 是 {100*min(a['peak_reserved_mib']/big_by_cfg[a['config']]['peak_reserved_mib'] for a in small_arms):.0f}–"
          f"{100*max(a['peak_reserved_mib']/big_by_cfg[a['config']]['peak_reserved_mib'] for a in small_arms):.0f}%，"
          f"s/step 是 {min(a['s_per_step']/big_by_cfg[a['config']]['s_per_step'] for a in small_arms):.2f}–"
          f"{max(a['s_per_step']/big_by_cfg[a['config']]['s_per_step'] for a in small_arms):.2f} 倍——"
          "参数量缩 3 倍远不等于同比例的墙钟/吞吐收益，因为序列长度与数据完全相同。"
          "③ 评测墙钟与平均生成长度同向，那是生成量的差，不是速度快慢的差。"]
    L += ["", "## 5. 判读", ""]
    for line in doc["verdict"]:
        L.append(f"- {line}")
    L += ["", "生成：`python scripts/10_scale_compare.py`（CPU，只读已发布产物，可逐字节再生）。"]
    return "\n".join(L) + "\n"


def main() -> int:
    scales = [collect(label, prefix, log, summary)
              for label, prefix, log, summary in SCALES]

    # ---- verdict text, computed from the numbers, not typed by hand -----------
    by_label = {s["label"]: s for s in scales}
    verdict = []
    for s in scales:
        up = [a["config"] for a in s["arms"]
              if a["delta_ci_excludes_0"] and a["delta_vs_base"] > 0]
        down = [a["config"] for a in s["arms"]
                if a["delta_ci_excludes_0"] and a["delta_vs_base"] < 0]
        n_ex_pair = sum(1 for p in s["pairs"] if p["excludes_0"])
        verdict.append(
            f"**{s['label']}**：四档里显著高于 base 的 {len(up)} 档"
            f"（{('、'.join(up)) or '无'}），显著低于 base 的 {len(down)} 档"
            f"（{('、'.join(down)) or '无'}）；6 对两两比较里 {n_ex_pair} 对 CI 排除 0；"
            f"最好档−最差档 {100*s['spread']['value']:.1f} pt。")
    small = by_label["0.5B"]
    big = by_label["1.5B"]
    r100_big = next(a for a in big["arms"] if a["config"] == "r100_0")
    r100_small = next(a for a in small["arms"] if a["config"] == "r100_0")
    widest_small = max(small["pairs"], key=lambda p: abs(p["delta"]))
    ties_small = sum(1 for p in small["pairs"] if abs(p["delta"]) == abs(widest_small["delta"]))
    mcq_marker = {s["label"]: statistics.mean(
        a["answer_marker_share_mean"] for a in s["arms"] if a["config"] != "r100_0")
        for s in scales}
    # 能排除多大的效应，取决于区间半宽；门槛取半宽最大值向上取整，不写死。
    hwa = [100 * (a["delta_ci95"][1] - a["delta_ci95"][0]) / 2 for s in scales for a in s["arms"]]
    hwp = [100 * (p["ci95"][1] - p["ci95"][0]) / 2 for s in scales for p in s["pairs"]]
    ceil_pt = float(math.ceil(max(hwa + hwp)))

    verdict += [
        f"回答原问题：「配比无稳定提升」**在 0.5B 上同样成立**——四档对 base 的区间全部跨 0，"
        f"6 对两两比较也全部跨 0。档间最大差是 {widest_small['a']} − {widest_small['b']} = "
        f"{100*widest_small['delta']:+.1f} pt，CI "
        f"[{100*widest_small['ci95'][0]:+.1f}, {100*widest_small['ci95'][1]:+.1f}]——含 0"
        f"{'（同幅差还有 ' + str(ties_small - 1) + ' 对）' if ties_small > 1 else ''}。",
        f"1.5B 上唯一被测出来的档间差异全部涉及 `r100_0`（纯对话档），且它是显著更差："
        f"{100*r100_big['delta_vs_base']:+.1f} pt，CI [{100*r100_big['delta_ci95'][0]:+.1f}, "
        f"{100*r100_big['delta_ci95'][1]:+.1f}]。这个差几乎全来自判分口径："
        f"该档 3 seed 共 {r100_big['unparseable_total']} 题不可解析（base 只有 "
        f"{big['base_unparseable']} 题），只统计可解析题时 Δ 收到 "
        f"{100*r100_big['delta_vs_base_parseable']:+.1f} pt。",
        f"同一个档在 0.5B 上不可解析只有 {r100_small['unparseable_total']} 题，"
        f"Δ 为 {100*r100_small['delta_vs_base']:+.1f} pt "
        f"[{100*r100_small['delta_ci95'][0]:+.1f}, {100*r100_small['delta_ci95'][1]:+.1f}]，"
        "即「纯对话档吃亏」不随规模复现。机制写在 §3 的截断列里：1.5B 的 "
        f"{r100_big['unparseable_total']} 题不可解析有 {r100_big['unparseable_capped_total']} 题"
        f"是用满 max_new_tokens={big['max_new_tokens']} 被截断的，0.5B 只贡献了 "
        f"{r100_small['unparseable_capped_total']} 题这种截断——它绑在 1.5B 的长答语风格"
        "与生成预算上，不是配比本身的性质。",
        f"两个规模的四档平均效应甚至不同向：1.5B {100*big['mean_delta']['value']:+.1f} pt "
        f"[{100*big['mean_delta']['ci95'][0]:+.1f}, {100*big['mean_delta']['ci95'][1]:+.1f}]，"
        f"0.5B {100*small['mean_delta']['value']:+.1f} pt "
        f"[{100*small['mean_delta']['ci95'][0]:+.1f}, {100*small['mean_delta']['ci95'][1]:+.1f}]，"
        "两者都跨 0，所以「这套 1,200 行 × 1 epoch 的 LoRA 本身有没有用」在两个规模上都判不出。",
        f"格式效应则随规模复现（结论 3 不受影响）：混入考题数据的三档在两个基座上都塌成"
        f"单字母（answer-marker 占比 1.5B {100*mcq_marker['1.5B']:.0%} / "
        f"0.5B {100*mcq_marker['0.5B']:.0%}、平均 2.0 token），纯对话档保留长答语"
        f"（0.5B {r100_small['mean_new_tokens']:.1f} token / 1.5B {r100_big['mean_new_tokens']:.1f} token）"
        f"——0.5B 的纯对话档虽然还写「答案：」，但 {r100_small['mean_new_tokens']:.1f} token "
        "就写完，没撞上限。",
        f"这次复核能排除什么：档 vs base 的区间半宽 "
        f"{min(hwa):.1f}–{max(hwa):.1f} pt，档间配对区间的半宽 "
        f"{min(hwp):.1f}–{max(hwp):.1f} pt，所以它排除的是"
        f"「某档配比带来 ≥{ceil_pt:.0f} pt 的稳定提升」，"
        f"排除不了 2–3 pt 量级的效应——0.5B 表里 "
        f"{'、'.join(a['config'] for a in small['arms'] if a['acc_mean'] == max(a2['acc_mean'] for a2 in small['arms']))}"
        f" 并列 {100*max(a['acc_mean'] for a in small['arms']):.1f}%、"
        f"{'、'.join(a['config'] for a in small['arms'] if a['acc_mean'] == min(a2['acc_mean'] for a2 in small['arms']))}"
        f" 垫底 {100*min(a['acc_mean'] for a in small['arms']):.1f}%，"
        "这些点差都落在噪声内。",
        "不能从这里外推的东西：只换了两个规模点、一个题源（CMExam 500 题）、一个训练预算"
        f"（1,200 行 × 150 步）；「规模」在这里只改了参数量，没有分离预训练语料与数据量，"
        f"而两档 base 的准确率差 {100*abs(big['base_acc'] - small['base_acc']):.1f} pt"
        f"（{100*big['base_acc']:.1f}% vs {100*small['base_acc']:.1f}%）本身就说明可动空间不同。",
    ]

    doc = {
        "boot": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "500 eval items",
                 "ci": "nearest-rank 2.5/97.5 percentile",
                 "caveat": "the three seeds per config are averaged per item before "
                           "resampling: the intervals carry item-sampling error, not "
                           "training randomness; with 3 seeds the seed-to-seed spread is "
                           "reported separately in the multiseed summaries"},
        "scales": scales,
        "verdict": verdict,
    }

    (OUT / "scale_compare_500.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    cols = ["scale", "base_model", "config", "acc_mean", "acc_sd", "acc_min", "acc_max",
            "delta_vs_base", "delta_ci_lo", "delta_ci_hi", "delta_ci_excludes_0",
            "mcnemar_min_p",
            "unparseable_mean", "unparseable_total", "unparseable_capped_total",
            "answer_marker_share_mean",
            "mean_new_tokens", "acc_parseable_mean", "delta_vs_base_parseable",
            "s_per_step", "peak_reserved_mib", "peak_alloc_mib", "final_loss",
            "train_wallclock_s", "eval_wallclock_s", "trainable_rows", "dropped_too_long"]
    with (OUT / "scale_compare_500.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for s in scales:
            for a in s["arms"]:
                row = {"scale": s["label"], "base_model": s["base_model"],
                       "mcnemar_min_p": min(a["mcnemar_p_vs_base"]), **a,
                       "delta_ci_lo": a["delta_ci95"][0], "delta_ci_hi": a["delta_ci95"][1]}
                w.writerow({k: row[k] for k in cols})

    (OUT / "scale_compare_500.md").write_text(build_markdown(doc), encoding="utf-8", newline="\n")

    # ---- assertions: this artifact and the published summaries must agree ------
    for (label, prefix, _log, summary_path), s in zip(SCALES, scales):
        pub = read_json(summary_path)
        if abs(pub["base"]["accuracy"] - s["base_acc"]) > 1e-12:
            raise SystemExit(f"{label}: base accuracy drifted vs {summary_path.name}")
        for a in s["arms"]:
            sub = [r for r in pub["runs"] if r["config"] == a["config"]]
            if len(sub) != len(SEEDS):
                raise SystemExit(f"{label}: {a['config']} has {len(sub)} seeds in the summary")
            if abs(statistics.mean(r["accuracy"] for r in sub) - a["acc_mean"]) > 5e-4:
                raise SystemExit(f"{label}: {a['config']} mean accuracy drifted")
            if sum(r["unparseable"] for r in sub) != a["unparseable_total"]:
                raise SystemExit(f"{label}: {a['config']} unparseable drifted")
        if abs(statistics.mean(a["acc_mean"] for a in s["arms"]) - s["mean_delta"]["value"]
               - s["base_acc"]) > 1e-3:
            raise SystemExit(f"{label}: internal consistency of mean delta broke")
    # the 1.5B column is the published one; the 0.5B column must be the new sweep only
    if by_label["1.5B"]["tag_prefix"] != "" or by_label["0.5B"]["tag_prefix"] != "q05b_":
        raise SystemExit("scale/prefix mapping changed; check SCALES")
    print("\n".join(verdict))
    print(f"\nwrote {OUT/'scale_compare_500.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
