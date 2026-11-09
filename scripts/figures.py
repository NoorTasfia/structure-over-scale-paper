"""Paper figures from analysis.json + taxonomy.json (no raw runs needed).

Fig 1 (hero): file_hit5 by model/precision in capability order, one line per
toolset, 95% bootstrap CIs. Shows RQ1 (B above A everywhere) and the C crossover.
Fig 2: episode-outcome composition per cell (hit / wrong files / forced-wrong /
no answer) - the RQ3 failure story, incl. 12B's no-answer collapse.

Usage: python scripts/figures.py [analysis.json] [taxonomy.json] [outdir]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["12b-q4", "e4b-q4", "e2b-bf16", "e2b-q8", "e2b-q4"]
PRETTY = {"12b-q4": "12B\nQ4_0", "e4b-q4": "E4B\nQ4_0", "e2b-bf16": "E2B\nBF16",
          "e2b-q8": "E2B\nQ8_0", "e2b-q4": "E2B\nQ4_0"}
TS_NAME = {"A": "A: grep/read", "B": "B: + graph tools", "C": "C: + repo map"}
TS_COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}  # validated all-pairs (light)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.dpi": 300,
})


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_hero(cells, outdir):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = range(len(LABELS))
    for ts in ("A", "B", "C"):
        means = [cells[f"{l}/{ts}"]["mean"] for l in LABELS]
        los = [cells[f"{l}/{ts}"]["lo"] for l in LABELS]
        his = [cells[f"{l}/{ts}"]["hi"] for l in LABELS]
        c = TS_COLOR[ts]
        ax.errorbar(x, means,
                    yerr=[[m - lo for m, lo in zip(means, los)],
                          [hi - m for m, hi in zip(means, his)]],
                    color=c, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2,
                    capsize=2.5, elinewidth=1, ecolor=c, alpha=None)
        ax.annotate(TS_NAME[ts], (x[-1], means[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(list(x), [PRETTY[l] for l in LABELS])
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("file_hit5 (gold file in top 5)")
    ax.set_xlabel("model / precision")
    ax.annotate("protocol collapse:\nthinking overruns the\nreply budget (Fig. 2)",
                (0.06, cells["12b-q4/C"]["mean"] - 0.06), xytext=(0.075, 0.02),
                textcoords="axes fraction", fontsize=7.5, color=MUTED, va="bottom",
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.7))
    style(ax)
    fig.subplots_adjust(right=0.78, left=0.11, bottom=0.17, top=0.96)
    for ext in ("pdf", "png"):
        fig.savefig(f"{outdir}/fig1_hero.{ext}")
    plt.close(fig)


def fig_failures(tax, outdir):
    cats = [("hit", "gold in top 5", "#2a78d6"),
            ("wrong_files", "submitted, missed", "#eb6834"),
            ("forced_wrong", "forced at limit, missed", "#1baf7a"),
            ("no_answer", "no answer", "#c3c2b7")]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    xs, ticks = [], []
    pos = 0.0
    for label in LABELS:
        for ts in ("A", "B", "C"):
            xs.append((pos, label, ts))
            pos += 1
        ticks.append((pos - 2, PRETTY[label].replace("\n", " ")))
        pos += 0.7  # gap between label groups
    for (p, label, ts) in xs:
        row = tax[f"{label}/{ts}"]
        n = row["n"]
        bottom = 0.0
        for key, _, color in cats:
            v = row[key] / n
            ax.bar(p, v, bottom=bottom, width=0.82, color=color,
                   edgecolor=SURFACE, linewidth=1.2)
            bottom += v
        ax.annotate(ts, (p, -0.045), ha="center", va="top", fontsize=7.5,
                    color=MUTED, annotation_clip=False)
    ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=8.5)
    ax.tick_params(axis="x", pad=14)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction of 60 episodes")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in cats]
    ax.legend(handles, [nm for _, nm, _ in cats], ncol=4, frameon=False,
              fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.14),
              handlelength=1.1, columnspacing=1.2)
    style(ax)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.88)
    for ext in ("pdf", "png"):
        fig.savefig(f"{outdir}/fig2_failures.{ext}")
    plt.close(fig)


def main():
    analysis = sys.argv[1] if len(sys.argv) > 1 else "analysis.json"
    taxonomy = sys.argv[2] if len(sys.argv) > 2 else "taxonomy.json"
    outdir = sys.argv[3] if len(sys.argv) > 3 else "figures"
    import os
    os.makedirs(outdir, exist_ok=True)
    cells = json.load(open(analysis, encoding="utf-8"))["cells"]
    tax = json.load(open(taxonomy, encoding="utf-8"))
    fig_hero(cells, outdir)
    fig_failures(tax, outdir)
    print(f"wrote {outdir}/fig1_hero.* and {outdir}/fig2_failures.*")


if __name__ == "__main__":
    main()
