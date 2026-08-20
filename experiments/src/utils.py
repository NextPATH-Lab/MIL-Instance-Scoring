import os, random, logging, time
from dotenv import load_dotenv
load_dotenv(override = True)

DEVICE = os.getenv("ACCELERATION_DEVICE")

from typing import Optional

import numpy as np
import matplotlib as mpl

import torch as th
from torch.utils.data import DataLoader

import sklearn.metrics as skm
from sklearn.mixture import BayesianGaussianMixture as BGM

from scipy import stats

from .model import ABMILite

log = logging.getLogger(__name__)

def make_reproducible(seed: int = 42):
    """To set deterministic/reproducible workflows.
    Thanks Gemini!

    Args:
        seed (int, optional): _description_. Defaults to 42.
    """
    # 1. Standard Python & Library Seeding
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    
    # 2. PyTorch Core & GPU Engine Seeding
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed) # Dynamic multi-GPU stability
    
    # 3. Force Deterministic NVIDIA cuDNN Backends
    # Forces cuDNN to select deterministic algorithms (sacrifices slight speed)
    th.backends.cudnn.deterministic = True
    # Disables autotuner benchmarking which dynamically changes algorithms
    th.backends.cudnn.benchmark = False
    
    # 4. Strict Error Handlers for Non-Deterministic Operations
    # Forces PyTorch to raise an error if an operation cannot be run deterministically
    th.use_deterministic_algorithms(True)
    
    # 5. Environment variable override for specific CUDA atomic operations
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # or ":16:8"

def generate_discrete_colors(
    n_colors: int,
    colormap: str = "gist_rainbow"
) -> np.ndarray:
    """Generates a number of RGB values from a matplotlib colormap.

    Args:
        n_colors (int): Number of RGB values to generate along the colormap
        colormap (str, optional): The matplotlib colormap to generate
            colors from. Defaults to "gist_rainbow".

    Returns:
        np.ndarray: n_colors x 3 (RGB) color values to generate
    """
    colormap = mpl.colormaps[colormap]

    max_n = max(colormap.N, n_colors)

    cmap_rgba = [colormap(i) for i in range(max_n)]
    rgb_array = np.asarray(cmap_rgba)[:,:3]
    color_indices = [int(i) for i in np.linspace(0, max_n - 1, n_colors)]

    return np.uint8(rgb_array[color_indices] * 255)

# ======== Type ALIAS Defs ======== # 
type za_vec = np.ndarray[tuple[int, int], np.dtype[np.float32]]
type vector_f32 = np.ndarray[tuple[int,], np.dtype[np.float32]]
# ======== End Type ALIAS ======== #

def cluster_get_pos_class(
        xy_data: za_vec,
        bgm: Optional[BGM] = None,
) -> vector_f32:
    """
    Fit a variational Bayesian Gaussian Mixture model, and get the 'positive'
    instance class (max instance logit component)

    Args:
        xy_data (np.ndarray): The logit-attention vector for each instance.

    Returns:
        Array of cluster labels for each instance.
    """
    log.info("Starting clustering now.")
    start = time.time()
    if xy_data.shape[0] > 100_000:
        # Limit number of training samples
        mask = np.random.choice(xy_data.shape[0], 100_000, replace=False)
        _xy_data = xy_data[mask]
    else:
        _xy_data = xy_data

    # Initialize Bayesian Gaussian Mixture
    if bgm is None or not isinstance(bgm, BGM):
        bgm = (
            BGM(
                n_components = 3,
                n_init = 3,
                warm_start = True,
                random_state = 2380
            )
        )
        bgm.fit(_xy_data)

    # Identify cluster ID with highest logit component
    cluster_id = np.argmax(bgm.means_[:,0])
    cluster_scores = bgm.predict_proba(xy_data)[:,cluster_id]

    end = time.time()

    log.info("Clustering done in %.4f seconds.", end - start)

    return cluster_scores

def get_za_vectors_roc_correlation(
        model: ABMILite,
        data_loader: DataLoader
) -> dict[str,float]:
    """
    """
    data_loader.dataset.mode = "instance"
    za_mx = []
    instance_labels = []
    for x, y in data_loader:
        x = x.to(DEVICE)
        with th.no_grad():
            za = model.za_transform(x)
        instance_labels.append(y.squeeze())
        za_mx.append(za)
    za_mx = th.cat(za_mx, dim = 0).cpu()

    instance_labels = th.cat(instance_labels, dim = 0).squeeze()
    instance_labels = (instance_labels == 9).float()

    cluster_scores = cluster_get_pos_class(za_mx.numpy())

    cluster_roc = skm.roc_auc_score(instance_labels, cluster_scores)
    attn_roc = skm.roc_auc_score(instance_labels, za_mx[:,1])
    logit_roc = skm.roc_auc_score(instance_labels, za_mx[:,0])
    pearson = stats.pearsonr(za_mx[:,0], za_mx[:,1])
    spearman = stats.spearmanr(za_mx[:,0], za_mx[:,1])

    results = {
        'attn_roc' : float(attn_roc),
        'logit_roc' : float(logit_roc),
        'cluster_roc' : float(cluster_roc),
        'pearson_stat' : float(pearson.statistic),
        'pearson_pval' : float(pearson.pvalue),
        'spearman_stat' : float(spearman.statistic),
        'spearman_pval' : float(spearman.pvalue)
    }

    return results