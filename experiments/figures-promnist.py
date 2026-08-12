"""Script to create the figures from manuscript.
"""

import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

from src.jointrow import joint_row

promnist_results = pl.read_csv("../outputs/promnist.csv")
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
f.savefig("../outputs/Res-Figure1D.png", dpi = 300, bbox_inches = 'tight')

# =============
# Second figure
# =============
za_df = pl.read_csv("../outputs/promnist_za_dist.csv")
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
f_joint.savefig("../outputs/Res-Figure1C.png", dpi = 300, bbox_inches = 'tight')