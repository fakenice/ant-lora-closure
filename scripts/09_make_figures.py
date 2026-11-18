"""Figure step: README figures for the multi-seed 500-item result and the controls.

Two panels, both recomputed from the per-item eval logs (out/eval/*_items.jsonl)
and asserted against the published summaries (out/multiseed_500_summary.json,
out/positive_control_summary.json) before anything is drawn. If a summary ever
drifts from the item logs, this script refuses to plot.

fig1_multiseed_500.png  - per-seed accuracy (12 runs) with config means and the
                          base Wilson band, plus the unparseable counts that
                          show what the ratios actually moved: output format.
fig2_positive_control.png - the two diagnostic arms against base with Wilson
                          intervals and the 20% five-choice chance line.

Usage: python scripts/09_make_figures.py   (matplotlib only; no GPU, no torch)
Writes out/figures/*.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "out" / "eval"
FIG = ROOT / "out" / "figures"
MS = json.loads((ROOT / "out" / "multiseed_500_summary.json").read_text(encoding="utf-8"))
PC = json.loads((ROOT / "out" / "positive_control_summary.json").read_text(encoding="utf-8"))

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

C_BASE, C_CFG, C_BAD, C_GOOD = "#7f8c8d", "#1f6fb2", "#c0392b", "#27ae60"


def load_items(tag: str) -> list[dict]:
    path = EVAL / f"eval_{tag}_items.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip().endswith("}")]


def recount(tag: str) -> tuple[int, int]:
    """(correct, unparseable) straight from the item log."""
    rows = load_items(tag)
    correct = sum(1 for r in rows if r["outcome"] == "correct")
    unp = sum(1 for r in rows if r["outcome"] == "unparseable")
    return correct, unp


def assert_ms_run(run: dict) -> tuple[int, int]:
    c, u = recount(run["tag"])
    assert c == run["correct"], f"{run['tag']}: items say {c}, summary says {run['correct']}"
    assert u == run["unparseable"], f"{run['tag']}: unparseable {u} vs {run['unparseable']}"
    assert abs(c / run["n"] - run["accuracy"]) < 5e-4, f"{run['tag']}: accuracy mismatch"
    return c, u


def fig1_multiseed() -> None:
    base_c, _ = recount(MS["base"]["tag"])
    assert base_c == MS["base"]["correct"], "base item log vs summary mismatch"
    accs = {}
    for run in MS["runs"]:
        accs.setdefault(run["config"], []).append(run)
    configs = [c["config"] for c in MS["by_config"]]
    assert sorted(configs) == sorted(accs), "by_config and runs disagree on configs"

    x = range(len(configs))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.1),
                                   constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    lo, hi = MS["base"]["wilson_ci95"]
    ax1.axhspan(lo * 100, hi * 100, color=C_BASE, alpha=0.18,
                label=f"base {base_c}/500 = {MS['base']['accuracy'] * 100:.1f}% "
                      f"[Wilson {lo * 100:.1f}, {hi * 100:.1f}]")
    ax1.axhline(MS["base"]["accuracy"] * 100, color=C_BASE, lw=1.2)
    for i, cfg in enumerate(configs):
        runs = accs[cfg]
        pts = [r["accuracy"] * 100 for r in runs]
        for p, star in zip(pts, [r["mcnemar_p_vs_base"] < 0.05 for r in runs]):
            ax1.plot(i, p, "o", ms=6, color=C_CFG, mec="k", mew=0.4)
            if star:
                ax1.plot(i, p, "*", ms=11, color="none", mec=C_BAD, mew=1.1)
        mean = next(c["mean_accuracy"] for c in MS["by_config"] if c["config"] == cfg)
        ax1.hlines(mean * 100, i - 0.18, i + 0.18, color=C_BAD, lw=2.0)
    ax1.set(xticks=list(x), ylabel="accuracy on 500 items (%)",
            title="4 ratios x 3 seeds (greedy, 12 tokens)\n"
                  "dots: per seed, ring = McNemar p<0.05; bar: mean",
            xticklabels=list(configs))
    ax1.legend(fontsize=7, loc="lower right")
    ax1.set_xlim(-0.5, len(configs) - 0.5)

    for i, cfg in enumerate(configs):
        ups = [assert_ms_run(r)[1] for r in accs[cfg]]
        ax2.bar([i - 0.22, i, i + 0.22], ups, width=0.18,
                color=[C_BAD if u > 10 else C_CFG for u in ups])
    ax2.set(xticks=list(x), xlabel="unparseable answers per run (of 500)",
            xticklabels=configs, ylabel="count")
    ax2.set_title("the ratios moved output format,\nnot accuracy (dialogue-only keeps long answers)",
                  fontsize=8.5)
    fig.savefig(FIG / "fig1_multiseed_500.png")
    plt.close(fig)


def fig2_positive_control() -> None:
    arms = PC["runs"]
    labels, accs, los, his, ps = [], [], [], [], []
    for a in arms:
        c, _ = recount(a["tag"])
        assert c == a["correct"], f"{a['tag']}: items {c} vs summary {a['correct']}"
        lo, hi = a["wilson_95"]
        labels.append(a["adapter"])
        accs.append(a["accuracy"] * 100)
        los.append((a["accuracy"] - lo) * 100)
        his.append((hi - a["accuracy"]) * 100)
        ps.append(a["mcnemar_p_vs_base"])
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    cols = [C_BASE, C_GOOD, C_BAD]
    ax.bar(range(len(labels)), accs, yerr=[los, his], capsize=3, color=cols,
           error_kw={"lw": 0.9})
    ax.axhline(20, color="k", ls=":", lw=1.0)
    ax.text(-0.42, 21.5, "five-choice chance ~20%", fontsize=7, ha="left")
    for i, (a, h, p) in enumerate(zip(accs, his, ps)):
        if labels[i] == "base":
            continue  # McNemar p of base against itself is 1.0 by construction
        ax.text(i, a + h + 2, f"{a:.1f}%  p={p:.1e}", ha="center", fontsize=7.5)
    ax.text(0, accs[0] + his[0] + 2, f"{accs[0]:.1f}%", ha="center", fontsize=7.5)
    ax.set(xticks=range(len(labels)), ylabel="accuracy on 500 items (%)",
           ylim=(0, 115),
           xticklabels=["base\n(no SFT)", "positive control\n(eval labels,\nleak by design)",
                        "shuffled labels\n(same questions,\nwrong letters)"],
           title="Harness sensitivity check (Wilson 95% CI)\n"
                 "the pipeline CAN detect label-specific gains")
    fig.savefig(FIG / "fig2_positive_control.png")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig1_multiseed()
    fig2_positive_control()
    print(f"wrote 2 figures to {FIG}")


if __name__ == "__main__":
    main()
