"""Instance scores and metrics for the JMI-260453 ablations (reviewer 2,
major comments 3, 4 and 5B).

Every score is computed from the instance embeddings `h` of one bag, after
the encoder of the model, so the same code serves ABMILite (CAMELYON16) and
ABMIL (MNIST).

Continuous scores (ranking metrics: ROC AUC, average precision):
    logit, attn                 the two axes of the attention-logit space
    alpha_z                     Additive MIL contribution: softmax(a)_i * z_i
    grad_x_input                gradient x input of the bag logit
    integrated_gradients        integrated gradients of the bag logit (zero baseline)
    cluster_za, cluster_z       posterior of the positive mixture component,
                                on (z, a) and on z alone

Label-free decisions (threshold metrics: F1, sensitivity, specificity):
    logit                       z > 0
    cluster_za, cluster_z       the positive component has the highest posterior
"""
import numpy as np
import torch as th
import sklearn.metrics as skm
from scipy import stats
from sklearn.mixture import BayesianGaussianMixture as BGM

IG_STEPS = 32

def bag_logit(model, h: th.Tensor) -> th.Tensor:
    """Bag logit of the model from instance embeddings (same as ABMILite.forward)."""
    attn = model.attention_module(h, softmax = True, transpose = True)
    return th.mm(attn, model.classifier(h)).squeeze()

def bag_scores(model, h: th.Tensor, ig_steps: int = IG_STEPS) -> dict[str, np.ndarray]:
    """Continuous per-instance scores of one bag, before the mixtures."""
    with th.no_grad():
        z = model.classifier(h).squeeze(1)
        a = model.attention_module(h).squeeze(1) # Pre-softmax, mean over branches
        alpha_z = th.softmax(a, dim = 0) * z

    with th.enable_grad():
        h = h.detach().requires_grad_(True)
        grad, = th.autograd.grad(bag_logit(model, h), h)
        grad_x_input = (grad * h).sum(1)

        # >> Midpoint Riemann sum of the path integral from h = 0 to h
        path_grad = th.zeros_like(h)
        for step in (th.arange(ig_steps, device = h.device) + 0.5) / ig_steps:
            h_step = (step * h).detach().requires_grad_(True)
            path_grad += th.autograd.grad(bag_logit(model, h_step), h_step)[0]
        integrated_gradients = (h * path_grad / ig_steps).sum(1)

    scores = {
        "logit": z, "attn": a, "alpha_z": alpha_z,
        "grad_x_input": grad_x_input, "integrated_gradients": integrated_gradients,
    }
    return {k: v.detach().cpu().numpy() for k, v in scores.items()}

def positive_component(bgm: BGM) -> int:
    """The positive component has the highest mean logit (column 0)."""
    return int(np.argmax(bgm.means_[:, 0]))

def instance_metrics(
        scores: dict[str, np.ndarray],
        labels: np.ndarray,
        negative_bag: np.ndarray,
        bgm_za: BGM,
        bgm_z: BGM,
) -> list[dict]:
    """One row for each method. `negative_bag` marks the instances of negative
    bags; the rate of positive decisions there is the effective tau (major 5B)."""
    za = np.stack([scores["logit"], scores["attn"]], axis = 1)
    pos_za, pos_z = positive_component(bgm_za), positive_component(bgm_z)
    continuous = {
        **scores,
        "cluster_za": bgm_za.predict_proba(za)[:, pos_za],
        "cluster_z": bgm_z.predict_proba(za[:, :1])[:, pos_z],
    }
    decisions = {
        "logit": scores["logit"] > 0,
        "cluster_za": bgm_za.predict(za) == pos_za,
        "cluster_z": bgm_z.predict(za[:, :1]) == pos_z,
    }
    pearson_r = float(stats.pearsonr(scores["logit"], scores["attn"]).statistic)

    rows = []
    for method, score in continuous.items():
        row = {
            "method": method,
            "roc_auc": skm.roc_auc_score(labels, score),
            "average_precision": skm.average_precision_score(labels, score),
            "pearson_r": pearson_r,
        }
        if method in decisions:
            decision = decisions[method]
            tn, fp, fn, tp = skm.confusion_matrix(labels, decision, labels = [0, 1]).ravel()
            sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
            row.update(
                f1 = skm.f1_score(labels, decision, zero_division = 0),
                sensitivity = sensitivity,
                specificity = specificity,
                balanced_accuracy = (sensitivity + specificity) / 2,
                negative_bag_positive_rate = float(decision[negative_bag].mean()),
            )
        rows.append(row)
    return rows

if __name__ == "__main__":
    # >> Self-check: python -m src.instance_ablation (from experiments/)
    from .model import ABMILite
    th.manual_seed(0)
    model = ABMILite(in_channels = 8, hidden_size = 4, attention_branches = 3).eval()
    h = th.randn(20, 8)
    s = bag_scores(model, h, ig_steps = 512)
    f_h = bag_logit(model, h).item()
    f_0 = bag_logit(model, th.zeros_like(h)).item()
    assert np.isclose(s["alpha_z"].sum(), f_h, atol = 1e-5), "alpha_z must sum to the bag logit"
    assert np.isclose(s["integrated_gradients"].sum(), f_h - f_0, rtol = 1e-3), "IG completeness"
    assert np.allclose(s["logit"], model(h, mode = "instance").squeeze(1).detach().numpy())

    labels = np.r_[np.zeros(80), np.ones(20)].astype(int)
    z = np.r_[np.random.default_rng(0).normal(-3, 0.5, 80), np.random.default_rng(1).normal(3, 0.5, 20)]
    scores = {k: z for k in ["logit", "attn", "alpha_z", "grad_x_input", "integrated_gradients"]}
    za = np.stack([z, z], axis = 1)
    bgm_za = BGM(n_components = 2, random_state = 0).fit(za)
    bgm_z = BGM(n_components = 2, random_state = 0).fit(za[:, :1])
    rows = {r["method"]: r for r in instance_metrics(scores, labels, labels == 0, bgm_za, bgm_z)}
    for method in ["logit", "cluster_za", "cluster_z"]:
        assert rows[method]["f1"] == 1.0 and rows[method]["negative_bag_positive_rate"] == 0.0, method
    print("instance_ablation self-check passed")
