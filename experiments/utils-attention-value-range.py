"""Compute a shared attention/instance-logit color range across multiple
sweep checkpoints (and every test slide each one covers), for use with
figures-camelyon16-attention-overlay.py's --attn-vmin/--attn-vmax/
--logit-abs-max -- so attention rendered for different WSIs (and different
feature-fraction checkpoints) uses the SAME colorbar scale instead of each
slide auto-scaling to its own local min/max, which would visually hide real
differences between them.

One caveat worth keeping in mind: each feature-fraction checkpoint is a
SEPARATELY TRAINED model, so raw (pre-softmax) attention/logit magnitudes
aren't guaranteed to be on an inherently comparable scale across fractions
the way, say, ROC AUC is -- this script just reports the actual min/max
across whatever checkpoints you pass in, which is the right input for "make
every rendering use the same colorbar," but a wide range spanning very
different-scaled checkpoints will make any one of them look flatter/more
washed out than its own single-checkpoint range would.

Selects checkpoints the same way utils-sweep-checkpoint-metrics.py does --
via --fraction/--feature-version/--model-version filters against --csv
(default: every checkpoint in the sweep). Reports both the combined range
across everything selected AND a per-fraction breakdown, so you can decide
whether one shared range across all fractions or a per-fraction range makes
more sense for what you're showing.
"""
import os
import json
import argparse
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import dotenv; dotenv.load_dotenv(override = True)

import numpy as np
import pandas as pd
import torch as th

from src.model import ABMILite

OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR")) / "camelyon16"
CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "./experiments/config")) / "camelyon16"

def main():
    cli = argparse.ArgumentParser(
        description = "Compute a shared attention/logit colorbar range across sweep checkpoints, for figures-camelyon16-attention-overlay.py.")
    cli.add_argument("--csv", type = Path, default = OUTPUTS_DIR / "feature_fraction_sweep.csv")
    cli.add_argument("--checkpoint-dir", type = Path, default = OUTPUTS_DIR / "sweep_checkpoints")
    cli.add_argument("--embeds-dir", type = Path, default = None, help = "Directory of per-slide embeddings. Defaults to CAMELYON16_EMBEDS_DIR.")
    cli.add_argument("--embedding-dim", type = int, default = 1536, help = "Full per-instance feature dimensionality before subsetting (1536 for UNI2-H).")
    cli.add_argument("--fraction", type = float, nargs = "+", default = None, help = "Restrict to these feature fractions. Default: every fraction in --csv.")
    cli.add_argument("--feature-version", type = int, nargs = "+", default = None)
    cli.add_argument("--model-version", type = int, nargs = "+", default = None)
    cli.add_argument("--slides", type = str, default = "test", choices = ["test", "train", "all"], help = "Which slides to scan. 'test' matches what the overlay script is usually pointed at.")
    cli.add_argument("--device", type = str, default = "cuda" if th.cuda.is_available() else "cpu")
    args = cli.parse_args()

    embeds_dir = args.embeds_dir or Path(os.getenv("CAMELYON16_EMBEDS_DIR"))
    data_dir = list(embeds_dir.glob("*.pt"))
    if args.slides == "test":
        files = [x for x in data_dir if x.stem.startswith("test_")]
    elif args.slides == "train":
        files = [x for x in data_dir if not x.stem.startswith("test")]
    else:
        files = data_dir

    df = pd.read_csv(args.csv)
    mask = pd.Series(True, index = df.index)
    if args.fraction is not None:
        mask &= df["fraction"].isin(args.fraction)
    if args.feature_version is not None:
        mask &= df["feature_version"].isin(args.feature_version)
    if args.model_version is not None:
        mask &= df["model_version"].isin(args.model_version)
    selected = df[mask]

    print(f"Scanning {len(selected)} checkpoints over {len(files)} '{args.slides}' slides...")

    per_fraction_attn: dict[float, list[float]] = {}
    per_fraction_logit_abs: dict[float, list[float]] = {}

    for (frac, fv), group_rows in selected.groupby(["fraction", "feature_version"]):
        fv = int(fv)
        feature_subset_path = CONFIG_DIR / f"sweep_feature_subset_{embeds_dir.name}_frac{frac}_fv{fv}.json"
        with open(feature_subset_path, "r") as f:
            feature_indices = th.tensor(json.load(f), dtype = th.long)
        embedding_dim = feature_indices.shape[0]

        loaded = [
            th.load(f, weights_only = True, map_location = "cpu")["features"].index_select(1, feature_indices).float()
            for f in files
        ]

        for _, row in group_rows.iterrows():
            mv = int(row["model_version"])
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

            for x in loaded:
                with th.no_grad():
                    attn = abmilite.attention_module(x.to(args.device), softmax = False, transpose = False)
                    logit = abmilite(x.to(args.device), mode = "instance")
                per_fraction_attn.setdefault(frac, []).extend(attn.cpu().numpy().ravel().tolist())
                per_fraction_logit_abs.setdefault(frac, []).extend(np.abs(logit.cpu().numpy()).ravel().tolist())

            print(f"  fraction={frac} fv={fv} mv={mv} done")

    print("\nPer-fraction range:")
    for frac in sorted(per_fraction_attn):
        a = np.array(per_fraction_attn[frac])
        l = np.array(per_fraction_logit_abs[frac])
        print(f"  fraction={frac}: attn [{a.min():.4f}, {a.max():.4f}]  logit_abs_max={l.max():.4f}")

    all_attn = np.concatenate(list(per_fraction_attn.values()))
    all_logit_abs = np.concatenate(list(per_fraction_logit_abs.values()))
    print(
        f"\nCombined range across everything selected:\n"
        f"  --attn-vmin {all_attn.min():.4f} --attn-vmax {all_attn.max():.4f} --logit-abs-max {all_logit_abs.max():.4f}"
    )

if __name__ == "__main__":
    main()
