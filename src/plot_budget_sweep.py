"""Render a shareable figure using the already-installed offline Matplotlib."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")
os.environ["MPLBACKEND"] = "Agg"

import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from budget_sweep_outputs import load_saved_run


def render(directory):
    directory = Path(directory).resolve()
    plan, pool, records, saved, derived, budget = load_saved_run(directory)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 15,
        "axes.labelsize": 12, "svg.fonttype": "none", "savefig.facecolor": "#FCFAF5",
    })
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 6.3), sharey=True)
    figure.set_facecolor("#FCFAF5")
    colors = {"thinking_on": "#145D73", "thinking_off": "#BC6C25"}
    panels = (("accuracy", "accuracy_wilson95", "Strict Correct Delivery"),
              ("natural_completion_rate", "completion_wilson95", "Natural Completion"))
    for axis, (metric, bounds, title) in zip(axes, panels):
        axis.set_facecolor("#FCFAF5")
        for condition, color in colors.items():
            rows = [row for row in derived["tables"][condition] if row["n"]]
            if not rows:
                continue
            x = [row["cap"] for row in rows]
            axis.plot(x, [row[metric] for row in rows], color=color, marker="o", linewidth=2.5,
                      label=condition.replace("_", " ").capitalize(), markersize=6)
            axis.fill_between(x, [row[bounds][0] for row in rows], [row[bounds][1] for row in rows],
                              color=color, alpha=0.14, linewidth=0)
        axis.set_xscale("log", base=2)
        axis.set_xticks([512, 1024, 2048, 4096, 8192], ["512", "1,024", "2,048", "4,096", "8,192"])
        axis.set_xlabel("Token budget (log2 scale)", labelpad=10)
        axis.set_title(title, loc="left", fontweight="bold", pad=16)
        axis.set_ylim(0, 1)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", color="#DAD4C8", linewidth=0.7)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("Proportion of development questions")
    counts = {condition: derived["tables"][condition][0]["n"] for condition in colors}
    label = "n=30 per condition" if derived["complete_data"] else f"PARTIAL: n(on)={counts['thinking_on']}, n(off)={counts['thinking_off']}"
    figure.suptitle("Accuracy And Completion Vs. Token Budget", x=0.07, y=0.97,
                     ha="left", fontsize=21, fontweight="bold", color="#213D41")
    figure.text(0.07, 0.87, f"Unedited Qwen3.5-0.8B  |  GSM8K TRAIN dev sample  |  {label}", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.10), ncol=2, frameon=False)
    figure.text(0.07, 0.055, "95% pointwise Wilson bands. One long trajectory per question/condition; one paired seed per question.",
                fontsize=9, color="#555B58")
    figure.text(0.07, 0.025, "GSM8K TRAIN may be present in pretraining; absolute accuracy is not a capability or benchmark claim.",
                fontsize=9, color="#555B58")
    figure.subplots_adjust(top=0.76, bottom=0.25, left=0.08, right=0.97, wspace=0.16)
    for extension in ("png", "svg"):
        figure.savefig(directory / ("accuracy_vs_token_budget." + extension), dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    render(parser.parse_args().run)
