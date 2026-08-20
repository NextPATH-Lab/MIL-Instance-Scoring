"""
Bootstraps at the *slide* level (resampling `test_files` with replacement, not 
individual patches, since patches within a slide are correlated). Attention and 
instance logit are bag-invariant per-patch quantities (no cross-instance 
interaction in `ABMILite`), so they're precomputed once per slide; only the 
`BayesianGaussianMixture` clustering step is refit every bootstrap iteration on 
that iteration's pooled `[logit, attention]` patches, since its stability under 
resampling is exactly what we're trying to measure. `warm_start=True` on a 
single persistent `BGM` instance means each iteration continues from the 
previous iteration's converged parameters rather than reinitializing -- much 
faster over 500 iterations, but consecutive replicates aren't fully independent 
fits (more likely to stay near the same local optimum than a fresh `n_init=3` 
restart every time). Worth keeping in mind when reading the width of the 
resulting CIs.
"""
import os, logging, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

import dotenv; dotenv.load_dotenv(override = True)
from tqdm import tqdm

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("poster")

import torch as th
from torch.utils.data import DataLoader

from sklearn import metrics as skm
from sklearn.mixture import BayesianGaussianMixture as BGM

from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from src.model import ABMILite
from src.datasets import CAMELYON16UNI2Embeddings

### === Environment === ###
LOG_DIR = Path(os.getenv("GLOBAL_LOG_DIR")) / "camelyon16"
LOG_DIR.mkdir(exist_ok = True, parents = True)
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR")) / "camelyon16"
(OUTPUTS_DIR / "figures").mkdir(exist_ok = True, parents = True)

# ==== HYPERPARAMTERS FOR SCRIPT ==== #
N_BOOT = 250
N_COMPONENTS = 3
FIT_SUBSAMPLE_CAP = 1_000_000

SHOW_TEST_PERFORMANCE = True
ACCELERATION_DEVICE = "cuda"
SEED = 2380

EMBEDDING_DIM = 1536
HIDDEN_DIM = EMBEDDING_DIM // 4
# ==== END HYPERPARAMTERS DEFS ==== #

log = logging.getLogger(__name__)
now = datetime.now().strftime("%Y-%m-%d %Hh%Mm")
logging.basicConfig(
    level = logging.INFO,
    filename = LOG_DIR / f"{now} {Path(__file__).stem}.log", 
    format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

th.manual_seed(SEED)
th.cuda.manual_seed(SEED)
th.use_deterministic_algorithms(True)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True

rng = np.random.default_rng(int(SEED))

###
data_dir = list(Path(os.getenv("CAMELYON16_EMBEDS_DIR")).glob("*.pt"))

# Get the test files
test_files = (
    np.array(
        list(filter(
            lambda x: x.stem.startswith("test_"),
            data_dir
        ))
    )
)

model_init = {
    "in_channels" : EMBEDDING_DIM,
    "hidden_size" : HIDDEN_DIM,
    'use_feature_extractor' : False,
    'attention_mode' : "simple",
    'attention_branches' : 3,
}

abmilite = ABMILite(**model_init)
abmilite.load_state_dict(
    th.load(
        OUTPUTS_DIR / "abmilite_camelyon16_uni2h.pt",
        map_location = "cpu",
        weights_only = True
    )
)

test_instance_loader = (
    DataLoader(
        CAMELYON16UNI2Embeddings(test_files, labels_by_instance = True),
        batch_size = 1
    )
)

test_per_slide = []
for x, y in tqdm(test_instance_loader):
    x = x.squeeze(0)
    y = y.squeeze(0)

    with th.no_grad():
        z_hat = abmilite(x.to(ACCELERATION_DEVICE), mode = "instance").cpu()
        a_hat = abmilite.attention_module(x.to(ACCELERATION_DEVICE)).cpu()

    test_per_slide.append(
        {
            "labels" : y.squeeze(-1).numpy(force = True),
            "logit" : z_hat.squeeze(-1).numpy(force = True),
            "attention" : a_hat.squeeze(-1).numpy(force = True)
        }
    )

n_test_slides = len(test_per_slide)

log.info(
    f"Precomputed scores for {n_test_slides} test slides, "
    f"{sum(len(s['labels']) for s in test_per_slide):,} total patches."
)

boot_roc_attention = np.empty(N_BOOT)
boot_roc_logit = np.empty(N_BOOT)
boot_roc_cluster = np.empty(N_BOOT)

# Single persistent BGM instance -- warm_start=True only has an effect because
# .fit() is called on the SAME object across iterations (a fresh instance per
# iteration would always do a full n_init=3 restart, making warm_start a no-op).
bgm = BGM(
    n_components = N_COMPONENTS,
    n_init = 3,
    warm_start = True,
    random_state = int(SEED)
)

for b in tqdm(range(N_BOOT)):
    boot_idx = rng.integers(0, n_test_slides, size = n_test_slides)

    labels_pool = np.concatenate([test_per_slide[i]["labels"] for i in boot_idx])
    logit_pool = np.concatenate([test_per_slide[i]["logit"] for i in boot_idx])
    attn_pool = np.concatenate([test_per_slide[i]["attention"] for i in boot_idx])

    X = np.stack([logit_pool, attn_pool], axis = 1)  # column 0 = logit, column 1 = attention

    # Fit on a random subsample (fitting is the expensive part -- millions of
    # points otherwise); score the FULL pool below so the ROC-AUC estimates
    # still use all available data.
    fit_n = min(FIT_SUBSAMPLE_CAP, X.shape[0])
    fit_idx = rng.choice(X.shape[0], size = fit_n, replace = False)
    bgm.fit(X[fit_idx])

    tumor_cluster = np.argmax(bgm.means_[:, 0])
    cluster_score = bgm.predict_proba(X)[:, tumor_cluster]

    boot_roc_attention[b] = skm.roc_auc_score(labels_pool, attn_pool)
    boot_roc_logit[b] = skm.roc_auc_score(labels_pool, logit_pool)
    boot_roc_cluster[b] = skm.roc_auc_score(labels_pool, cluster_score)

def summarize(name, arr):
    lo, hi = np.percentile(arr, [2.5, 97.5])
    log.info(f"{name:<18s} mean={arr.mean():.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
    return lo, hi

log.info(f"Patch-level ROC-AUC over {N_BOOT} test-slide bootstraps:")
summarize("Attention", boot_roc_attention)
summarize("Instance logit", boot_roc_logit)
summarize("GMM cluster score", boot_roc_cluster)

log.info(
    "\nPaired difference vs. attention (CI excluding 0 => significant at 95%):"
)
named_arr = [
    ("Instance logit", boot_roc_logit),
    ("GMM cluster score", boot_roc_cluster)
]
for name, arr in named_arr:
    diff = arr - boot_roc_attention
    lo, hi = np.percentile(diff, [2.5, 97.5])
    significant = lo > 0 or hi < 0
    log.info(
        f"{name:<18s} - Attention: mean={diff.mean():+.4f}  "
        f"95% CI=[{lo:+.4f}, {hi:+.4f}]  significant={significant}"
    )

### === Make Plots to show CI around ROCs === ###
# Okabe-Ito colorblind-safe categorical triple, fixed order (not matplotlib's default cycle).
_METHOD_COLORS = ["#0072B2", "#E69F00", "#009E73"]  # Attention, Instance logit, GMM cluster
_METHOD_NAMES = ["Attention", "Instance logit", "GMM cluster"]
_data = [boot_roc_attention, boot_roc_logit, boot_roc_cluster]
positions = np.arange(1, len(_data) + 1)

fig, ax = plt.subplots(figsize=(10, 10))

# Violin for the full distribution shape
violin = ax.violinplot(_data, positions=positions, showmedians=False, showextrema=False, widths=0.7)
for body, color in zip(violin["bodies"], _METHOD_COLORS):
    body.set_facecolor(color)
    body.set_alpha(0.35)
    body.set_edgecolor(color)

# Narrow boxplot on top for quartiles/median (fliers hidden -- violin already shows the tails)
box = ax.boxplot(
    _data,
    positions=positions,
    tick_labels=_METHOD_NAMES,
    patch_artist=True,
    widths=0.15,
    showfliers=False,
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
)
for patch, color in zip(box["boxes"], _METHOD_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.9)
for median in box["medians"]:
    median.set_color("black")

ax.set_ylabel("Patch-level ROC-AUC")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
fig.tight_layout()
# >> Save Figure and Data
_data = pl.DataFrame(_data, schema = _METHOD_NAMES)
_data.write_csv(OUTPUTS_DIR / "bootstrap-CI-camelyon16.csv")
fig.savefig(
    OUTPUTS_DIR / "figures/Bootstrap-ROC-CI.png",
    dpi=300,
    bbox_inches="tight"
)

### === Run Statistical Tests === ###
long_df = (
    pl.DataFrame(
        {
            "iteration" : np.tile(np.arange(N_BOOT), 3),
            "method" : np.repeat(["attention", "logit", "cluster"], N_BOOT),
            "roc_auc" : np.concatenate(
                [boot_roc_attention, boot_roc_logit, boot_roc_cluster]
            )
        }
    )
)
# >> ANOVA
rm_anova = (
    AnovaRM(
        long_df.to_pandas(),
        depvar = "roc_auc",
        subject = "iteration",
        within = ['method']
    ).fit()
)
log.info(
    f"ANOVA Results:\n{rm_anova}"
)
# >> Post-hoc TUKEY HSD
tukey = (
    pairwise_tukeyhsd(
        endog = long_df['roc_auc'].to_numpy(),
        groups = long_df['method'].to_numpy(),
        alpha = 0.05
    )
)
log.info(
    f"TUKEYHSD Result:\n{tukey}"
)