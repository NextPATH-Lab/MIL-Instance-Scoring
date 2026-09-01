"""Single-trial ABMILite train+evaluate logic for CAMELYON16 experiments,
factored out of experiment-uni2-camelyon16-abmil.py so sweep scripts (e.g.
experiment-feature-fraction-sweep.py) don't duplicate ~150 lines of
training/instance-inference/clustering logic. Mirrors that script's single-
fold training, losses, and clustering setup exactly, so results stay
directly comparable between a one-off run and a sweep replicate.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch as th
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

from sklearn import metrics as skm
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.model_selection import StratifiedKFold as SKfold

from src.model import ABMILite
from src.model_trainer import MILTrainer
from src.datasets import CAMELYON16UNI2Embeddings
from src.jointrow import joint_row
from src.utils import cluster_get_pos_class

def _run_instance_inference(abmilite, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    za_space, ins_labels, bag_labels = [], [], []
    for x, y in loader:
        x, y = x.squeeze(0), y.squeeze(0)
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

def run_one_trial(
        train_files: np.ndarray,
        test_files: np.ndarray,
        embedding_dim: int,
        seed: int,
        epochs: int,
        lr: float = 1e-4,
        feature_fraction: Optional[float] = None,
        feature_subset_path: Optional[Path] = None,
        feature_subset_seed: Optional[int] = None,
        max_bag_size: Optional[int] = None,
        min_tumor_fraction: float = 0.01,
        device: str = "cuda",
        za_plot_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
) -> dict:
    """Train ABMILite once (single StratifiedKFold fold, seeded by `seed`)
    and evaluate it, returning a flat dict of bag- and instance-level
    metrics. Optionally saves the za-space scatter figure and/or the
    trained checkpoint.

    `seed` drives model init and the train/val fold split. `feature_subset_seed`
    (defaults to `seed` if not given) drives the feature-subset draw --
    decoupled so a sweep can share one feature-subset "version" across
    several model-seed replicates (same `feature_subset_path`, different
    `seed`): the subset is only actually drawn the first time that path is
    used, so `feature_subset_seed` is simply ignored on every later reuse.
    """
    if feature_subset_seed is None:
        feature_subset_seed = seed

    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

    # Fixed (not scaled with embedding_dim) so that model capacity doesn't
    # confound the feature-fraction sweep: fraction should be the only
    # variable affecting attention reliability, not a shrinking hidden layer.
    # Matches the hidden_size used at fraction=1.0 (1536 // 4 = 384) so that
    # anchor point is unchanged.
    model_init = {
        "in_channels": embedding_dim,
        "hidden_size": 384,
        "use_feature_extractor": False,
        "attention_mode": "simple",
        "attention_branches": 3,
    }

    train_labels = np.array([int(x.stem.startswith("tumor")) for x in train_files])
    splitter = SKfold(random_state = seed, shuffle = True)
    train_idx, val_idx = next(splitter.split(train_files, train_labels))

    trainer = MILTrainer()
    abmilite = ABMILite(**model_init).to(device)
    n_params = sum(x.nelement() for x in abmilite.parameters())
    adam = optim.AdamW(abmilite.parameters(), lr = lr)

    ds_kwargs = dict(
        feature_fraction = feature_fraction, feature_subset_seed = feature_subset_seed,
        feature_subset_path = feature_subset_path,
    )
    train_ds = CAMELYON16UNI2Embeddings(
        train_files[train_idx], device = device,
        max_bag_size = max_bag_size, min_tumor_fraction = min_tumor_fraction,
        **ds_kwargs,
    )
    val_ds = CAMELYON16UNI2Embeddings(train_files[val_idx], device = device, **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size = 1, shuffle = True, prefetch_factor = None, num_workers = 0)
    val_loader = DataLoader(val_ds, batch_size = 1, prefetch_factor = None, num_workers = 0)

    trainer.fit(
        abmilite, train_loader, adam, binary_cross_entropy_with_logits, None,
        val_loader, num_epochs = epochs, on_device = device, batch_size = 2,
        val_metric = ["roc_auc", "balanced_accuracy", "average_precision"],
        model_selection_method = "average_precision",
        binarize = [False, True, False],
        l1_strength = 1e-4, l1_classifier_only = True, l1_attention_in_layer_only = True,
    )

    if checkpoint_path is not None:
        th.save(abmilite.cpu().state_dict(), checkpoint_path)
        abmilite = abmilite.to(device)

    # >> Bag-level test metrics
    test_ds = CAMELYON16UNI2Embeddings(test_files, device = device, **ds_kwargs)
    test_loader = DataLoader(test_ds, batch_size = 1)
    bag_metrics = trainer.validate(
        abmilite, test_loader,
        ["roc_auc", "balanced_accuracy", "average_precision"],
        [False, True, False], on_device = device,
    )

    # >> Instance-level inference (train: fits clustering; test: scored)
    train_instance_ds = CAMELYON16UNI2Embeddings(train_files, labels_by_instance = True, **ds_kwargs)
    test_instance_ds = CAMELYON16UNI2Embeddings(test_files, labels_by_instance = True, **ds_kwargs)
    train_instance_loader = DataLoader(train_instance_ds, batch_size = 1)
    test_instance_loader = DataLoader(test_instance_ds, batch_size = 1)

    train_za_space, train_ins_labels, _ = _run_instance_inference(abmilite, train_instance_loader, device)
    test_za_space, test_ins_labels, test_bag_labels = _run_instance_inference(abmilite, test_instance_loader, device)

    # >> Clustering: fit BGM on a 10% subsample of train instances, score test
    rng = np.random.default_rng(seed)
    fit_mask = rng.choice(train_za_space.shape[0], train_za_space.shape[0] // 10, replace = False)
    bgm = BGM(n_components = 3, warm_start = True, random_state = int(seed), n_init = 3)
    bgm.fit(train_za_space[fit_mask, :])
    cluster_scores = cluster_get_pos_class(test_za_space, bgm)

    def _auc(scores):
        fpr, tpr, _ = skm.roc_curve(test_ins_labels, scores)
        return float(skm.auc(fpr, tpr))

    attn_roc_auc = _auc(test_za_space[:, 1])
    logit_roc_auc = _auc(test_za_space[:, 0])
    cluster_roc_auc = _auc(cluster_scores)

    pearson_r, _ = pearsonr(test_za_space[:, 0], test_za_space[:, 1])
    spearman_r, _ = spearmanr(test_za_space[:, 0], test_za_space[:, 1])

    if za_plot_path is not None:
        za_sample_mask = rng.choice(test_ins_labels.shape[0], test_ins_labels.shape[0] // 10, replace = False)
        datasets = [
            {
                "x": test_za_space[za_sample_mask, 0],
                "y": test_za_space[za_sample_mask, 1],
                "hue": label_set[za_sample_mask, 0],
                "title": label_str,
                "ylabel": "Attention Weight",
            }
            for label_str, label_set in zip(
                ["By Patch Label", "By Slide Label"], [test_ins_labels, test_bag_labels])
        ]
        fig_za, _ = joint_row(
            datasets, xlabel = "Instance Classifier-head Logit",
            scatter_kws = {"s": 1, "alpha": 0.25}, maintype = "scatter",
        )
        za_plot_path.parent.mkdir(parents = True, exist_ok = True)
        fig_za.savefig(za_plot_path, dpi = 300, bbox_inches = "tight")
        import matplotlib.pyplot as plt
        plt.close(fig_za)

    return {
        "n_params": n_params,
        "bag_roc_auc": bag_metrics["roc_auc"],
        "bag_balanced_accuracy": bag_metrics["balanced_accuracy"],
        "bag_average_precision": bag_metrics["average_precision"],
        "attn_roc_auc": attn_roc_auc,
        "logit_roc_auc": logit_roc_auc,
        "cluster_roc_auc": cluster_roc_auc,
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
    }
