"""Script to create the figures from manuscript.
"""
import os, logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(override=True)

import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

from src.jointrow import joint_row

DEVICE = os.getenv("ACCELERATION_DEVICE")
# ==== Logging Set Up ==== #
LOG_DIR = Path(os.getenv("GLOBAL_LOG_DIR")) / "ProMNIST"
LOG_DIR.mkdir(exist_ok = True, parents = True)
OUTPUT_DIR = Path(os.getenv("OUTPUTS_DIR")) / "ProMNIST"
(OUTPUT_DIR / "figures").mkdir(exist_ok = True, parents = True)

log = logging.getLogger(__name__)
now = datetime.now().strftime("%Y-%m-%d %Hh%Mm")
logging.basicConfig(
    level = logging.INFO,
    filename = LOG_DIR / f"{now} {Path(__file__).stem}.log", 
    format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)

promnist_results = pl.read_csv(OUTPUT_DIR / "promnist.csv")
# Compute stats
promnist_avg_results = (
    promnist_results
        .group_by('tau')
        .agg(pl.all().mean())
        .sort('tau')
)
promnist_sd_results = (
    promnist_results
        .group_by('tau')
        .agg(pl.all().std())
        .sort('tau')
)

# ================
# First Figure
# ================
f,a = plt.subplots(ncols = 3, figsize = (25, 5.25), dpi = 300)
plt.style.use('default')
sns.set_theme(context="talk", style = 'white') 

yaxis_labels = [
    'ROC AUC',
    'Correlation',
    'Bag Prediction Performance'
]
score_labels = [
    (0, 'attn_roc', "Attention"),
    (0, 'logit_roc', "Logit"),
    (0, 'cluster_roc', "Clustering"),
    (1, 'pearson_stat', 'Pearson'),
    (1, 'spearman_stat', 'Spearman'),
    (2, 'balanced_accuracy', 'Bag Balanced Accuracy'),
    (2, 'roc_auc', "Bag ROC Score"),
    (2, 'average_precision', "Bag Average Precision")
]
for _idx, _col, _label in score_labels:
    a[_idx].errorbar(
        promnist_avg_results.get_column("tau"),
        promnist_avg_results.get_column(_col),
        promnist_sd_results.get_column(_col),
        label = _label,
        fmt = "o-"
    )
    a[_idx].legend(loc = 'lower left')
    a[_idx].set_ylabel(yaxis_labels[_idx])
    a[_idx].set_xlabel("Proportion Threshold (tau)")

a[2].set_ylim(0, 1.)
f.savefig(
    OUTPUT_DIR / "figures/Res-Figure1D.png",
    dpi = 300,
    bbox_inches = 'tight'
)

# =============
# Second figure
# =============
za_df = pl.read_csv(OUTPUT_DIR / "promnist_za_dist.csv")
datasets = []
for name, data in za_df.sort('t').group_by("t"):
    _dataset = {
        'x' : data.get_column("z").to_numpy(),
        'y' : data.get_column('a').to_numpy(),
        'hue' : data.get_column('y').to_numpy(),
    }
    datasets.append(_dataset)
f_joint, a_joint = joint_row(
    datasets,
    xlabel = "Instance Logit",
    ylabel = "Attention Weight")
f_joint.tight_layout()
f_joint.savefig(
    OUTPUT_DIR / "figures/Res-Figure1C.png",
    dpi = 300,
    bbox_inches = 'tight'
)