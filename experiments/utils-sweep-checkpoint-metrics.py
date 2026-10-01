"""Recompute an arbitrary instance-level metric (attention / instance logit /
clustering) from experiment-feature-fraction-sweep.py's saved checkpoints,
WITHOUT retraining -- reruns just the instance-inference + clustering step
(same logic src/camelyon16_training.py's run_one_trial does after training),
scores it, and merges the result back into the sweep CSV as new columns
(attn_<metric>, logit_<metric>, cluster_<metric>).

This exists because run_one_trial only computes the metrics it was written to
compute at the time (ROC AUC, average precision, Pearson/Spearman) -- if a
different metric turns out to matter later, there's no need to redo the
~16 min/replicate training run, just the instance-inference pass against the
checkpoint that's already on disk.

Batched by (fraction, feature_version): every replicate sharing a
(fraction, feature_version) pair also shares the same underlying instance
data (feature subset included) -- only the model weights differ across
model_version. So each slide's .pt file is loaded and feature-subset once per
group and reused across all of that group's model_version checkpoints,
instead of re-reading it from disk once per checkpoint (a 5x cut in disk I/O
for the default 5-model_versions-per-group sweep design, since data loading,
not the model's forward pass, is what dominates the per-checkpoint cost).

Metric resolution: --metric takes any name sklearn.metrics.get_scorer()
recognizes (e.g. "average_precision", "roc_auc", "f1", "balanced_accuracy",
"precision", "recall", "accuracy") -- the exact same resolution mechanism
MILTrainer.validate() already uses elsewhere in this codebase
(get_scorer(name)._score_func), so metric names stay consistent across both.

Finalized vs. raw scores: some sklearn metrics (roc_auc, average_precision)
score a continuous ranking directly; others (f1, accuracy, balanced_accuracy,
precision, recall) need a binary decision. MILTrainer handles this for the
classifier logit alone via a fixed `logit > 0` threshold, but this script
also scores attention weights and cluster probabilities, which don't share
logit's zero-centered scale -- so --finalized here thresholds at each score's
own median across the test set instead, giving a consistent top-half/
bottom-half split regardless of the score's native range. Pass --finalized
for the second kind of metric; leave it off (default) for ranking metrics.

Checkpoint selection: defaults to every row in --csv ("all checkpoints").
Pass any combination of --fraction/--feature-version/--model-version to
restrict to specific replicates instead -- only matching rows are touched;
every other row and column in the CSV is left exactly as it was.
"""
import os
import json
import argparse
import logging
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime

import dotenv; dotenv.load_dotenv(override = True)

import numpy as np
import pandas as pd
import torch as th

from sklearn.metrics import get_scorer
from sklearn.mixture import BayesianGaussianMixture as BGM

from src.model import ABMILite
from src.utils import cluster_get_pos_class

LOG_DIR = Path(os.getenv("GLOBAL_LOG_DIR")) / "camelyon16"
LOG_DIR.mkdir(exist_ok = True, parents = True)
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR")) / "camelyon16"
CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "./experiments/config")) / "camelyon16"

log = logging.getLogger(__name__)
now = datetime.now().strftime("%Y-%m-%d %Hh%Mm")
logging.basicConfig(
    level = logging.INFO,
    filename = LOG_DIR / f"{now} {Path(__file__).stem}.log",
    format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)

def score(score_func, scores: np.ndarray, labels: np.ndarray, finalized: bool) -> float:
    if finalized:
        scores = (scores > np.median(scores)).astype(int)
    return float(score_func(labels, scores))

def load_instance_data(files: np.ndarray, feature_indices: th.Tensor) -> list[tuple[th.Tensor, th.Tensor]]:
    """Load every slide's (embeddings, labels) once, feature subset already
    applied, kept on CPU -- shared across every model_version in a
    (fraction, feature_version) group so their forward passes don't each
    re-read the same files from disk."""
    data = []
    for f in files:
        d = th.load(f, weights_only = True, map_location = "cpu")
        embeddings = d["features"].index_select(1, feature_indices).float()
        labels = d["labels"].unsqueeze(1).float()
        data.append((embeddings, labels))
    return data

def run_instance_inference(abmilite, data: list[tuple[th.Tensor, th.Tensor]], device: str):
    za_space, ins_labels, bag_labels = [], [], []
    for x, y in data:
        with th.no_grad():
            z_hat = abmilite(x.to(device), mode = "instance").cpu()
            a_hat = abmilite.attention_module(x.to(device)).cpu()
        za_space.append(th.cat([z_hat, a_hat], dim = 1))
        ins_labels.append(y)
        bag_labels.append(th.ones_like(y) if y.max() > 0 else y)
    return (
        th.cat(za_space, dim = 0).numpy(force = True),
        th.cat(ins_labels, dim = 0).numpy(force = True),
        th.cat(bag_labels, dim = 0).numpy(force = True),
    )

def main():
    cli = argparse.ArgumentParser(
        description = "Recompute an instance-level metric from saved sweep checkpoints, no retraining required.")
    cli.add_argument("--csv", type = Path, default = OUTPUTS_DIR / "feature_fraction_sweep.csv",
                      help = "Sweep results CSV to read the replicate list from and merge results back into.")
    cli.add_argument("--checkpoint-dir", type = Path, default = OUTPUTS_DIR / "sweep_checkpoints")
    cli.add_argument("--embeds-dir", type = Path, default = None,
                      help = "Directory of per-slide embeddings. Defaults to CAMELYON16_EMBEDS_DIR.")
    cli.add_argument("--embedding-dim", type = int, default = 1536,
                      help = "Full per-instance feature dimensionality before subsetting (1536 for UNI2-H).")
    cli.add_argument("--metric", type = str, required = True,
                      help = "Any metric name sklearn.metrics.get_scorer() recognizes, e.g. "
                             "'average_precision', 'roc_auc', 'f1', 'balanced_accuracy', 'precision', 'recall', 'accuracy'.")
    cli.add_argument("--finalized", action = "store_true",
                      help = "Threshold each score at its own median before scoring (for metrics that need a binary "
                             "decision, e.g. f1/accuracy/precision/recall). Leave off for ranking metrics like roc_auc/average_precision.")
    cli.add_argument("--fraction", type = float, nargs = "+", default = None,
                      help = "Restrict to these feature fractions. Default: every fraction in --csv.")
    cli.add_argument("--feature-version", type = int, nargs = "+", default = None,
                      help = "Restrict to these feature_version values. Default: all.")
    cli.add_argument("--model-version", type = int, nargs = "+", default = None,
                      help = "Restrict to these model_version values. Default: all.")
    cli.add_argument("--device", type = str, default = "cuda" if th.cuda.is_available() else "cpu")
    args = cli.parse_args()

    embeds_dir = args.embeds_dir or Path(os.getenv("CAMELYON16_EMBEDS_DIR"))
    score_func = get_scorer(args.metric)._score_func

    data_dir = list(embeds_dir.glob("*.pt"))
    train_files = np.array([x for x in data_dir if not x.stem.startswith("test")])
    test_files = np.array([x for x in data_dir if x.stem.startswith("test_")])

    df = pd.read_csv(args.csv)
    mask = pd.Series(True, index = df.index)
    if args.fraction is not None:
        mask &= df["fraction"].isin(args.fraction)
    if args.feature_version is not None:
        mask &= df["feature_version"].isin(args.feature_version)
    if args.model_version is not None:
        mask &= df["model_version"].isin(args.model_version)
    selected = df[mask]

    log.info(
        "Scoring %d of %d checkpoints with metric=%s (finalized=%s), writing attn/logit/cluster_%s",
        len(selected), len(df), args.metric, args.finalized, args.metric
    )

    for col in [f"attn_{args.metric}", f"logit_{args.metric}", f"cluster_{args.metric}"]:
        if col not in df.columns:
            df[col] = np.nan

    groups = selected.groupby(["fraction", "feature_version"])
    n_done = 0
    for (frac, fv), group_rows in groups:
        fv = int(fv)
        feature_subset_path = CONFIG_DIR / f"sweep_feature_subset_{embeds_dir.name}_frac{frac}_fv{fv}.json"
        with open(feature_subset_path, "r") as f:
            feature_indices = th.tensor(json.load(f), dtype = th.long)
        embedding_dim = feature_indices.shape[0]

        log.info(
            "Loading shared instance data for fraction=%s feature_version=%d (%d model_versions in group)...",
            frac, fv, len(group_rows)
        )
        train_data = load_instance_data(train_files, feature_indices)
        test_data = load_instance_data(test_files, feature_indices)

        for idx, row in group_rows.iterrows():
            mv = int(row["model_version"])
            seed = int(row["trial_seed"])

            ckpt_path = args.checkpoint_dir / f"abmilite_{embeds_dir.name}_frac{frac}_fv{fv}_mv{mv}.pt"
            model_init = {
                "in_channels": embedding_dim, "hidden_size": 384,
                "use_feature_extractor": False, "attention_mode": "simple",
                "attention_branches": 3,
            }
            abmilite = ABMILite(**model_init)
            abmilite.load_state_dict(th.load(ckpt_path, map_location = "cpu", weights_only = True))
            abmilite = abmilite.to(args.device)
            abmilite.eval()

            train_za, train_labels, _ = run_instance_inference(abmilite, train_data, args.device)
            test_za, test_labels, _ = run_instance_inference(abmilite, test_data, args.device)

            rng = np.random.default_rng(seed)
            fit_mask = rng.choice(train_za.shape[0], train_za.shape[0] // 10, replace = False)
            bgm = BGM(n_components = 3, warm_start = True, random_state = seed, n_init = 3)
            bgm.fit(train_za[fit_mask, :])
            cluster_scores = cluster_get_pos_class(test_za, bgm)

            df.loc[idx, f"attn_{args.metric}"] = score(score_func, test_za[:, 1], test_labels, args.finalized)
            df.loc[idx, f"logit_{args.metric}"] = score(score_func, test_za[:, 0], test_labels, args.finalized)
            df.loc[idx, f"cluster_{args.metric}"] = score(score_func, cluster_scores, test_labels, args.finalized)

            n_done += 1
            log.info("[%d/%d] fraction=%s fv=%d mv=%d done", n_done, len(selected), frac, fv, mv)

    df.to_csv(args.csv, index = False)
    log.info("Done. %d rows scored, results merged into %s", len(selected), args.csv)

if __name__ == "__main__":
    main()
