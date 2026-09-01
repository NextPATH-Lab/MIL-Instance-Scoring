"""Generate the 3 feature-fraction sweep figures from
experiment-feature-fraction-sweep.py's results CSV, styled to match Figure 1D
in the manuscript (line + circle marker + std-dev error bars per point,
matplotlib default color order, x-axis swapped from tau to feature fraction):

  1. Instance classification ROC AUC (attention / logit / clustering) vs. feature fraction
  2. Pearson + Spearman correlation (instance logit vs. attention) vs. feature fraction
  3. Bag classification performance (ROC AUC / balanced accuracy / average precision) vs. feature fraction

Each is saved as its own PNG (not one combined 3-panel figure) so they can be
dropped into the manuscript independently. Item 1 from the request (the
per-fraction za-space scatter plots) needs no separate script -- those are
already saved by the sweep script itself, reusing
experiment-uni2-camelyon16-abmil.py's existing filename convention.
"""
import os
import argparse
from pathlib import Path

import dotenv; dotenv.load_dotenv(override = True)

import pandas as pd
import matplotlib.pyplot as plt

OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR")) / "camelyon16"

def _plot_metrics(ax, df: pd.DataFrame, metrics: list[tuple[str, str]], ylabel: str, ylim: tuple[float, float]):
    """metrics: list of (column_name, display_label), plotted in matplotlib's default color order."""
    fractions = sorted(df["fraction"].unique())
    for col, label in metrics:
        means = [df.loc[df["fraction"] == f, col].mean() for f in fractions]
        stds = [df.loc[df["fraction"] == f, col].std() for f in fractions]
        ax.errorbar(fractions, means, yerr = stds, fmt = "-o", label = label, capsize = 3, markersize = 5)
    ax.set_xlabel("Feature Fraction")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.legend(loc = "best")
    ax.grid(True, linestyle = ":", alpha = 0.4)

def main():
    cli = argparse.ArgumentParser(description = "Generate feature-fraction sweep figures (Fig 1D style).")
    cli.add_argument("--csv", type = Path, default = OUTPUTS_DIR / "feature_fraction_sweep.csv")
    cli.add_argument("--out-dir", type = Path, default = OUTPUTS_DIR / "figures")
    args = cli.parse_args()

    df = pd.read_csv(args.csv)
    args.out_dir.mkdir(parents = True, exist_ok = True)

    n_reps = df.groupby("fraction").size()
    print("Replicates per fraction:\n", n_reps)

    # 1. Instance classification performance
    fig, ax = plt.subplots(figsize = (6, 5))
    _plot_metrics(
        ax, df,
        [("attn_roc_auc", "Attention"), ("logit_roc_auc", "Logit"), ("cluster_roc_auc", "Clustering")],
        ylabel = "ROC AUC", ylim = (0.0, 1.0),
    )
    ax.set_title("Instance Classification Performance")
    fig.savefig(args.out_dir / "feature_fraction_instance_roc_auc.png", dpi = 300, bbox_inches = "tight")
    plt.close(fig)

    # 2. Instance logit-attention correlation
    fig, ax = plt.subplots(figsize = (6, 5))
    _plot_metrics(
        ax, df,
        [("pearson_r", "Pearson"), ("spearman_r", "Spearman")],
        ylabel = "Correlation", ylim = (-1.0, 1.0),
    )
    ax.set_title("Instance Logit-Attention Correlation")
    fig.savefig(args.out_dir / "feature_fraction_correlation.png", dpi = 300, bbox_inches = "tight")
    plt.close(fig)

    # 3. Bag prediction performance
    fig, ax = plt.subplots(figsize = (6, 5))
    _plot_metrics(
        ax, df,
        [("bag_balanced_accuracy", "Bag Balanced Accuracy"), ("bag_roc_auc", "Bag ROC Score"), ("bag_average_precision", "Bag Average Precision")],
        ylabel = "Bag Prediction Performance", ylim = (0.0, 1.0),
    )
    ax.set_title("Bag Classification Performance")
    fig.savefig(args.out_dir / "feature_fraction_bag_performance.png", dpi = 300, bbox_inches = "tight")
    plt.close(fig)

    print(f"Saved 3 figures to {args.out_dir}")

if __name__ == "__main__":
    main()
