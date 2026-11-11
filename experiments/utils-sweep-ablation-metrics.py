"""Ablations of the instance scoring for JMI-260453 (reviewer 2, major
comments 3, 4 and 5B), from the saved checkpoints of
experiment-feature-fraction-sweep.py. Inference only, no retraining.

For each checkpoint, the test instances get every score of
src/instance_ablation.py: logit, attention, the 2-D (z, a) mixture, a 1-D
mixture on the logit alone, Additive MIL alpha*z, gradient x input and
integrated gradients. Both mixtures use the settings of run_one_trial
(src/camelyon16_training.py): 3 components, fit on a 10% subsample of the
training instances, scored on the test instances.

Output: one row for each (checkpoint, method) in
$OUTPUTS_DIR/camelyon16/sweep_ablation_metrics.csv. The check of this
script: the attn and logit rows must reproduce the attn_*/logit_* columns of
feature_fraction_sweep.csv exactly. The cluster_za rows are close, not
exact: the 10% subsample depends on the order of the training files, which
is sorted here (the order of Windows/NTFS) and was the glob order before.

The embeddings are about 105 GB, so the slides are read one at a time:
each slide is read once for each (fraction, feature_version) group and
scored by every model_version checkpoint of that group while it is loaded.
Only the per-instance scores stay in memory.
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

from joblib import Parallel, delayed
from sklearn.mixture import BayesianGaussianMixture as BGM

from src.model import ABMILite
from src.instance_ablation import bag_scores, instance_metrics, IG_STEPS

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

def iter_slides(files: np.ndarray, feature_indices: th.Tensor, device: str):
    """(embeddings, instance labels) of one slide at a time, feature subset applied."""
    for f in files:
        d = th.load(f, weights_only = True, map_location = "cpu")
        embeddings = d["features"].index_select(1, feature_indices).float().to(device)
        yield embeddings, d["labels"].numpy().astype(int)

def fit_mixture(za: np.ndarray, seed: int) -> BGM:
    """The mixture of run_one_trial: 3 components on 10% of the training instances."""
    rng = np.random.default_rng(seed)
    fit_mask = rng.choice(za.shape[0], za.shape[0] // 10, replace = False)
    return BGM(n_components = 3, warm_start = True, random_state = seed, n_init = 3).fit(za[fit_mask, :])

def main():
    cli = argparse.ArgumentParser(description = "Ablations of the instance scoring from saved sweep checkpoints.")
    cli.add_argument("--csv", type = Path, default = OUTPUTS_DIR / "feature_fraction_sweep.csv",
                      help = "Sweep results CSV: the list of checkpoints and their trial_seed.")
    cli.add_argument("--checkpoint-dir", type = Path, default = OUTPUTS_DIR / "sweep_checkpoints")
    cli.add_argument("--embeds-dir", type = Path, default = None,
                      help = "Directory of per-slide embeddings. Defaults to CAMELYON16_EMBEDS_DIR.")
    cli.add_argument("--out", type = Path, default = OUTPUTS_DIR / "sweep_ablation_metrics.csv")
    cli.add_argument("--ig-steps", type = int, default = IG_STEPS)
    cli.add_argument("--fraction", type = float, nargs = "+", default = None,
                      help = "Restrict to these feature fractions. Default: every fraction in --csv.")
    cli.add_argument("--device", type = str, default = "cuda" if th.cuda.is_available() else "cpu")
    cli.add_argument("--n-jobs", type = int, default = max(1, os.cpu_count() // 2),
                      help = "Parallel mixture fits. Default: half the cores.")
    args = cli.parse_args()

    embeds_dir = args.embeds_dir or Path(os.getenv("CAMELYON16_EMBEDS_DIR"))
    data_dir = sorted(embeds_dir.glob("*.pt"))
    train_files = np.array([x for x in data_dir if not x.stem.startswith("test")])
    test_files = np.array([x for x in data_dir if x.stem.startswith("test_")])

    df = pd.read_csv(args.csv)
    if args.fraction is not None:
        df = df[df["fraction"].isin(args.fraction)]
    log.info("Scoring %d checkpoints on %s, ig_steps=%d", len(df), args.device, args.ig_steps)

    rows, n_done = [], 0
    for (frac, fv), group_rows in df.groupby(["fraction", "feature_version"]):
        fv = int(fv)
        with open(CONFIG_DIR / f"sweep_feature_subset_{embeds_dir.name}_frac{frac}_fv{fv}.json") as f:
            feature_indices = th.tensor(json.load(f), dtype = th.long)

        models = {}
        for _, row in group_rows.iterrows():
            mv = int(row["model_version"])
            abmilite = ABMILite(
                in_channels = feature_indices.shape[0], hidden_size = 384,
                use_feature_extractor = False, attention_mode = "simple",
                attention_branches = 3,
            )
            ckpt_path = args.checkpoint_dir / f"abmilite_{embeds_dir.name}_frac{frac}_fv{fv}_mv{mv}.pt"
            abmilite.load_state_dict(th.load(ckpt_path, map_location = "cpu", weights_only = True))
            models[mv] = (abmilite.to(args.device).eval(), int(row["trial_seed"]))

        # >> Training instances: only (z, a), for the mixture fits
        train_za = {mv: [] for mv in models}
        with th.no_grad():
            for x, _ in iter_slides(train_files, feature_indices, args.device):
                for mv, (abmilite, _) in models.items():
                    za = th.cat([abmilite.classifier(x), abmilite.attention_module(x)], dim = 1)
                    train_za[mv].append(za.cpu().numpy())

        # >> Test instances: every score, one bag at a time
        test_bags = {mv: [] for mv in models}
        labels, negative_bag = [], []
        for x, y in iter_slides(test_files, feature_indices, args.device):
            labels.append(y)
            negative_bag.append(np.full(len(y), y.max() == 0))
            for mv, (abmilite, _) in models.items():
                test_bags[mv].append(bag_scores(abmilite, x, args.ig_steps))
        labels, negative_bag = np.concatenate(labels), np.concatenate(negative_bag)

        # >> The mixture fits are single-threaded and slow: run the 2 fits of each model in parallel
        fit_args = []
        for mv, (_, seed) in models.items():
            za = np.concatenate(train_za[mv])
            fit_args += [(za, seed), (za[:, :1], seed)]
        fits = iter(Parallel(n_jobs = args.n_jobs)(delayed(fit_mixture)(*a) for a in fit_args))

        for mv, (_, seed) in models.items():
            bgm_za, bgm_z = next(fits), next(fits)
            scores = {k: np.concatenate([b[k] for b in test_bags[mv]]) for k in test_bags[mv][0]}
            for metrics in instance_metrics(scores, labels, negative_bag, bgm_za, bgm_z):
                rows.append({"fraction": frac, "feature_version": fv, "model_version": mv, **metrics})
            n_done += 1
            log.info("[%d/%d] fraction=%s fv=%d mv=%d done", n_done, len(df), frac, fv, mv)

        pd.DataFrame(rows).to_csv(args.out, index = False) # After each group, so a stop keeps the finished groups

    log.info("Done. %d rows written to %s", len(rows), args.out)

if __name__ == "__main__":
    main()
