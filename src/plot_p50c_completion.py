"""Data-only rendering with the preinstalled system Matplotlib; no new packages."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")
os.environ["MPLBACKEND"] = "Agg"

import argparse
import importlib.metadata
import json
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUTCOMES = ("natural_completion", "strict_correct_delivery", "thinking_closed")
LABELS = {
    "natural_completion": "Natural Completion (Primary)",
    "strict_correct_delivery": "Strict Correct Delivery (Secondary)",
    "thinking_closed": "Thinking Closed (Diagnostic)",
}
STEM = "p50c_vs_original_token_budget"


def validate_analysis(analysis):
    if not isinstance(analysis, dict) or analysis.get("complete_data") is not True or analysis.get("edited_main_responses") != 30:
        raise ValueError("Plot requires a complete, recomputed 30-question analysis.")
    for outcome in OUTCOMES:
        for variant in ("original", "edited"):
            rows = analysis["tables"][outcome][variant]
            if not isinstance(rows, list) or [row["cap"] for row in rows] != [512, 1024, 2048, 4096, 8192]:
                raise ValueError("Invalid plot caps.")
            for row in rows:
                interval = row.get("wilson95")
                if (
                    type(row.get("n")) is not int or row["n"] != 30
                    or type(row.get("successes")) is not int or not 0 <= row["successes"] <= 30
                    or type(row.get("rate")) not in (int, float) or not math.isfinite(row["rate"])
                    or abs(row["rate"] - row["successes"] / 30) > 1e-12
                    or not isinstance(interval, (list, tuple)) or len(interval) != 2
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in interval)
                    or not 0 <= interval[0] <= row["rate"] <= interval[1] <= 1
                ):
                    raise ValueError("Invalid plotted binomial rates or confidence bounds.")


def plot_comparison(analysis):
    validate_analysis(analysis)
    colors = {"original": "#245C78", "edited": "#C0602D"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
                         "axes.labelsize": 11, "figure.facecolor": "#FAFBF9", "axes.facecolor": "#FAFBF9",
                         "svg.fonttype": "none"}):
        figure, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharey=True)
        for axis, outcome in zip(axes, OUTCOMES):
            for variant, label in (("original", "Original"), ("edited", "Published p50c")):
                rows = analysis["tables"][outcome][variant]
                caps = [row["cap"] for row in rows]
                axis.plot(caps, [row["rate"] for row in rows], marker="o", linewidth=2.3,
                          color=colors[variant], label=f"{label} (n={rows[0]['n']})")
                axis.fill_between(caps, [row["wilson95"][0] for row in rows],
                                  [row["wilson95"][1] for row in rows], color=colors[variant], alpha=0.14)
            axis.set_xscale("log", base=2)
            axis.set_xticks([512, 1024, 2048, 4096, 8192], ["512", "1,024", "2,048", "4,096", "8,192"])
            axis.set_ylim(0, 1)
            axis.set_title(LABELS[outcome], pad=12)
            axis.set_xlabel("Generated-Token Cap (log2 scale)")
            axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            axis.grid(axis="y", alpha=0.22)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(loc="upper left", fontsize=9.5, frameon=False)
        axes[0].set_ylabel("Response Rate")
        figure.suptitle("Published p50c Edit vs Original - Qwen3.5-0.8B, Thinking On", fontsize=17, fontweight="bold", y=0.98)
        figure.text(0.5, 0.885, "Fixed GSM8K TRAIN development sample; n=30 per variant; shaded bands: 95% pointwise Wilson",
                    ha="center", fontsize=11)
        figure.text(0.5, 0.065 if "text_bound_violations" in analysis else 0.035,
                    "Natural EOS = token 248044. Thinking closure = literal </think> inside the generated prefix. These are distinct events.",
                    ha="center", fontsize=10)
        if "text_bound_violations" in analysis:
            figure.text(0.5, 0.022,
                        "Text-bound amendment declared after 11 saved responses, before the remaining 19. GSM8K TRAIN is likely in pretraining; no capability claim.",
                        ha="center", fontsize=9.2)
        figure.subplots_adjust(left=0.055, right=0.99, bottom=0.18, top=0.79, wspace=0.12)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    path = args.analysis.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Expected a bounded project-local analysis file.")
    with path.open("rb") as handle:
        raw = handle.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Plot analysis exceeds its bounded size.")
    analysis = json.loads(raw)
    figure = plot_comparison(analysis)
    for extension in ("png", "svg"):
        target = path.parent / (STEM + "." + extension)
        if target.exists():
            raise FileExistsError("Refusing to replace an existing figure.")
        figure.savefig(target, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    (path.parent / "plot_renderer.json").write_text(json.dumps({
        "renderer": "Preinstalled system Python / Matplotlib; data-only, no model runtime",
        "matplotlib": importlib.metadata.version("matplotlib"),
        "model_inference": False, "package_installations": 0, "downloads": 0,
    }, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
