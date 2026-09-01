"""Sweep experiment-uni2-camelyon16-abmil.py's approach-2 (fixed random
feature subset) across multiple feature fractions, with a nested replicate
design for error bars -- reproduces the structure of Figure 1D in the
manuscript (ROC AUC / correlation / bag performance vs. a swept parameter,
mean +/- std across replicates), swapping tau for feature_fraction.

Replicate design mirrors the Proportion MNIST Bags experiment exactly (see
Methods, subsubsection 2.2.1, and Fig 1's caption): for each fraction, draw
--feature-versions independent random feature subsets ("the dataset
generated N times with different seeds"), and train --model-versions
independently-seeded models on *each* subset ("models trained on each
dataset N times with different seeds"), for feature_versions * model_versions
replicates. Same feature subset is reused across a subset's model-version
block; a fresh model init + train/val fold split is drawn per model version.
At fraction=1.0 the feature-subset axis is degenerate (an all-dims
permutation carries no real variability for a linear readout), so only one
feature version is used there -- matching the paper's own precedent of
excluding a redundant dataset seed at tau=0.9.

Saves one long-format CSV (one row per fraction x feature_version x
model_version) with every metric needed for the 3 sweep figures (see
figures-feature-fraction-sweep.py), plus the by-request za-space scatter
figure (item 1) for one designated replicate per fraction, using the exact
naming experiment-uni2-camelyon16-abmil.py already produces
(camelyon16-uni2-za-plot_{embeds_dir}_frac{fraction}.png).

Resumable: skips any (fraction, feature_version, model_version) combination
already present in the output CSV, so a crashed or interrupted sweep can
just be rerun.
"""
import os
import csv
import argparse
import logging
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime

import dotenv; dotenv.load_dotenv(override = True)
from tqdm import tqdm

import numpy as np
import torch as th

from src.camelyon16_training import run_one_trial

LOG_DIR = Path(os.getenv("GLOBAL_LOG_DIR")) / "camelyon16"
LOG_DIR.mkdir(exist_ok = True, parents = True)
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR")) / "camelyon16"
(OUTPUTS_DIR / "figures").mkdir(exist_ok = True, parents = True)
CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "./experiments/config")) / "camelyon16"

log = logging.getLogger(__name__)
now = datetime.now().strftime("%Y-%m-%d %Hh%Mm")
logging.basicConfig(
    level = logging.INFO,
    filename = LOG_DIR / f"{now} {Path(__file__).stem}.log",
    format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)

CSV_FIELDS = [
    "fraction", "feature_version", "model_version", "epochs", "lr", "feature_seed", "trial_seed", "n_params",
    "bag_roc_auc", "bag_balanced_accuracy", "bag_average_precision",
    "attn_roc_auc", "logit_roc_auc", "cluster_roc_auc",
    "pearson_r", "spearman_r",
]

def load_completed(csv_path: Path) -> set[tuple[float, int, int, int, float]]:
    """Keyed on (fraction, feature_version, model_version, epochs, lr) --
    including the hyperparameters, not just the replicate indices, so a
    rerun with different epochs/lr can't silently skip as "already done"
    against results from a different hyperparameter setting."""
    if not csv_path.exists():
        return set()
    with open(csv_path, "r", newline = "") as f:
        return {
            (float(row["fraction"]), int(row["feature_version"]), int(row["model_version"]),
             int(row["epochs"]), float(row["lr"]))
            for row in csv.DictReader(f)
        }

def append_row(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def main():
    cli = argparse.ArgumentParser(description = "Sweep CAMELYON16 ABMILite across feature fractions (approach 2), nested feature x model replicate seeds for error bars.")
    cli.add_argument("--embeds-dir", type = Path, default = None, help = "Directory of per-slide embeddings. Defaults to CAMELYON16_EMBEDS_DIR.")
    cli.add_argument("--embedding-dim", type = int, default = 1536, help = "Full per-instance feature dimensionality before subsetting.")
    cli.add_argument("--fractions", type = float, nargs = "+", default = [1.0, 0.5, 0.25, 0.125], help = "Feature fractions to sweep.")
    cli.add_argument("--feature-versions", type = int, default = 5, help = "Independent feature-subset draws per fraction (ignored, forced to 1, at fraction=1.0).")
    cli.add_argument("--model-versions", type = int, default = 5, help = "Independently-seeded models trained on each feature-subset version.")
    cli.add_argument("--base-seed", type = int, default = 2380, help = "Base seed; feature_seed = base_seed + 100*feature_version, trial_seed = feature_seed + model_version.")
    cli.add_argument("--epochs", type = int, default = 10, help = "Epochs per trial. Default of 10 (down from the paper's 40) plus --lr 5e-4 (up from 1e-4) was spot-checked at fractions 0.125 and 0.5 to converge as well or better than 40 epochs at the paper's original 1e-4 -- see experiments/logs/camelyon16 for the comparison runs.")
    cli.add_argument("--lr", type = float, default = 5e-4, help = "AdamW learning rate. See --epochs for why this isn't the paper's original 1e-4.")
    cli.add_argument("--za-plot-feature-version", type = int, default = 0, help = "Which feature_version's za-space scatter gets saved as the representative figure per fraction (item 1).")
    cli.add_argument("--za-plot-model-version", type = int, default = 0, help = "Which model_version (within --za-plot-feature-version) gets the representative za-space scatter.")
    cli.add_argument("--max-bag-size", type = int, default = None, help = "Cap on instances per training bag (stratified subsample). None = no cap. See CAMELYON16UNI2Embeddings._stratified_sample.")
    cli.add_argument("--min-tumor-fraction", type = float, default = 0.01, help = "Floor on tumor-instance fraction kept when subsampling via --max-bag-size; ignored if --max-bag-size is None.")
    cli.add_argument("--device", type = str, default = "cuda" if th.cuda.is_available() else "cpu")
    args = cli.parse_args()

    embeds_dir = args.embeds_dir or Path(os.getenv("CAMELYON16_EMBEDS_DIR"))
    csv_path = OUTPUTS_DIR / "feature_fraction_sweep.csv"
    checkpoint_dir = OUTPUTS_DIR / "sweep_checkpoints"
    checkpoint_dir.mkdir(exist_ok = True, parents = True)

    os.environ["PYTHONHASHSEED"] = str(args.base_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    data_dir = list(embeds_dir.glob("*.pt"))
    if not data_dir:
        raise ValueError(f"No .pt files found in {embeds_dir}")
    train_files = np.array([x for x in data_dir if not x.stem.startswith("test")])
    test_files = np.array([x for x in data_dir if x.stem.startswith("test_")])

    completed = load_completed(csv_path)

    # Build the full (fraction, feature_version, model_version) trial list.
    # fraction=1.0 gets only 1 feature version (the subset axis is
    # degenerate there), every other fraction gets --feature-versions.
    trials = []
    for frac in args.fractions:
        n_feature_versions = 1 if frac == 1.0 else args.feature_versions
        for fv in range(n_feature_versions):
            for mv in range(args.model_versions):
                if (frac, fv, mv, args.epochs, args.lr) not in completed:
                    trials.append((frac, fv, mv))

    log.info(
        "Sweeping fractions=%s (%d feature versions x %d model versions, 1 feature version at frac=1.0) -- "
        "%d trials queued, %d already done. embeds_dir=%s, embedding_dim=%d, epochs=%d",
        args.fractions, args.feature_versions, args.model_versions,
        len(trials), len(completed), embeds_dir, args.embedding_dim, args.epochs
    )

    for frac, fv, mv in tqdm(trials, "Feature-fraction sweep"):
        # feature_seed only matters the first time this (fraction, fv)'s
        # subset file is created; trial_seed is unique per (fv, mv) and
        # drives model init + the train/val fold split.
        feature_seed = args.base_seed + 100 * fv
        trial_seed = args.base_seed + 100 * fv + mv
        log.info("Running fraction=%s feature_version=%d model_version=%d (feature_seed=%d, trial_seed=%d)...",
                  frac, fv, mv, feature_seed, trial_seed)

        feature_subset_path = CONFIG_DIR / f"sweep_feature_subset_{embeds_dir.name}_frac{frac}_fv{fv}.json"
        za_plot_path = (
            OUTPUTS_DIR / "figures" / f"camelyon16-uni2-za-plot_{embeds_dir.name}_frac{frac}.png"
            if fv == args.za_plot_feature_version and mv == args.za_plot_model_version else None
        )
        checkpoint_path = checkpoint_dir / f"abmilite_{embeds_dir.name}_frac{frac}_fv{fv}_mv{mv}.pt"

        metrics = run_one_trial(
            train_files, test_files,
            embedding_dim = round(args.embedding_dim * frac),
            seed = trial_seed, epochs = args.epochs, lr = args.lr,
            feature_fraction = frac, feature_subset_path = feature_subset_path,
            feature_subset_seed = feature_seed,
            max_bag_size = args.max_bag_size, min_tumor_fraction = args.min_tumor_fraction,
            device = args.device, za_plot_path = za_plot_path, checkpoint_path = checkpoint_path,
        )

        row = {
            "fraction": frac, "feature_version": fv, "model_version": mv,
            "epochs": args.epochs, "lr": args.lr,
            "feature_seed": feature_seed, "trial_seed": trial_seed, **metrics,
        }
        append_row(csv_path, row)
        log.info("fraction=%s feature_version=%d model_version=%d done: %s", frac, fv, mv, metrics)

    log.info("Sweep complete. Results at %s", csv_path)

if __name__ == "__main__":
    main()
