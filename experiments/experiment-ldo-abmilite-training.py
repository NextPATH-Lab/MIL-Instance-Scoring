"""Script to fit ABMILite model on LDO data.
"""

### === Training Hyperparameters === ###
SHOW_TEST_METRICS = True
K_SPLITS = 5
NUM_EPOCHS = 500
LEARN_RATE = 1e-3
L1_STRENGTH = 1e-3
METRICS = ['balanced_accuracy', 'average_precision', 'roc_auc']
BINARIZE = [True, False, False]
model_init_kwargs = {
    "hidden_size" : 4,
    "attention_mode" : "simple",
    "attention_branches" : 2,
    "attention_branch_reduction" : "sum",
    "use_batch_norm" : False
}

### === Configure Script Environment === ###
import os, logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override = True)
SEED = os.getenv("SEED")
DEVICE = os.getenv("ACCELERATION_DEVICE")
LOG_DIR = Path(os.getenv("GLOBAL_LOG_DIR"))
CONFIG_DIR = Path(os.getenv("CONFIG_DIR"))
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR"))
TRAINING_DATA_DIR = Path(os.getenv("TRAINING_DATA_DIR"))

log = logging.getLogger(__name__)
now = datetime.now().strftime("%Y-%m-%d %Hh%Mm")
logging.basicConfig(
    level = logging.INFO,
    filename = LOG_DIR / f"{now} {Path(__file__).stem}.log", 
    format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)

import json, joblib
from tqdm import tqdm
from functools import partial

import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import BayesianGaussianMixture as BGM

import seaborn as sns
from matplotlib import pyplot as plt

# >> Settings for plots
plt.style.use("default")
sns.set_theme(context = "talk", style = "white")

import numpy as np
import torch as th
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits

from src.model import ABMILite
from src.model_trainer import MILTrainer
from src.datasets import LDODataset
from src.utils import make_reproducible

make_reproducible(int(SEED))

### === Get Patient Information === ###
with open(CONFIG_DIR / "train-test-ids-dw2y.json") as f:
    train_test_ids = json.load(f)

data_dir = TRAINING_DATA_DIR / "filtered_copy"
data_files = list(data_dir.glob("*.csv"))

filter_fn = lambda x, id_str: int(x.stem) in train_test_ids[id_str]
train_files = list(filter(partial(filter_fn, id_str = "train_ids"), data_files))
test_files = list(filter(partial(filter_fn, id_str = "test_ids"), data_files))

# >> Generate stratified K-fold splits
# >>>> Get patient IDs and their clinical outcome
train_patient_files = []
train_patient_y = []

for f in train_files:
    x = pl.read_csv(f)
    y = x.get_column("DW2Y")[[0]]

    train_patient_files.append(f)
    train_patient_y.append(y)


train_patient_files = np.array(train_patient_files)
train_patient_y = np.array(train_patient_y)

# >>>> Generate split
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 23)
skf_split = skf.split(train_patient_files, train_patient_y)

### === Stratified K-Fold Cross Validation === ###
feature_config = CONFIG_DIR / "filtered-feature-config.json"

# >> Training
scores = []
losses = []
trainer = MILTrainer()

for (train_idx, val_idx) in tqdm(skf_split, total = 5, desc = "K Fold Fit"):
    # >> Retrieve the train/val files of split
    train_f = train_patient_files[train_idx].ravel().tolist()
    val_f = train_patient_files[val_idx].ravel().tolist()

    # >> Instantiate Dataset for Fold
    _filter_condn = pl.col("Visit_No") == 1
    _train_ds = (
        LDODataset(
            train_f,
            feature_config,
            use_caching = True,
            filter_condition = _filter_condn
        )
    )
    _val_ds = (
        LDODataset(
            val_f,
            feature_config,
            use_caching = True,
            filter_condition = _filter_condn
        )
    )

    # >> Instantiate Model for Fold
    model_init_kwargs['in_channels'] = (
        model_init_kwargs.get(
            "in_channels",
            _train_ds[0][0].shape[1]
        )
    )

    _abmilite = ABMILite(**model_init_kwargs).to(DEVICE)
    n_params = sum([x.nelement() for x in _abmilite.parameters()])

    log.info("Training model with %s parameters.", f"{n_params:,}")

    # >> Model Optimization
    _optimizer = (
        AdamW(_abmilite.parameters(), lr = LEARN_RATE, weight_decay = 0.1)
    )

    _loss_values, _best_epoch = (
        trainer.fit(
            _abmilite,
            data_loader_train = DataLoader(_train_ds, batch_size = 1),
            optimizer = _optimizer,
            loss_func = binary_cross_entropy_with_logits,
            data_loader_val = DataLoader(_val_ds, batch_size = 1),
            val_metric = METRICS,
            binarize = BINARIZE,
            model_selection_method = ["balanced_accuracy", "average_precision"],
            l1_strength = L1_STRENGTH,
            l1_classifier_only = True,
            l1_attention_in_layer_only = True,
            num_epochs = NUM_EPOCHS,
            on_device = DEVICE
        )
    )
    losses.append(_loss_values)

    _val_scores = (
        trainer.validate(
            _abmilite,
            DataLoader(_val_ds, batch_size = 1),
            metric = METRICS,
            binarize = BINARIZE
        )
    )
    _val_scores['best_epoch'] = _best_epoch
    scores.append(_val_scores)

scores = pl.DataFrame(scores)
scores_mean = scores.mean() # >> TODO: Find appropriate format to store

print(scores)
print(scores_mean)

# >> Full Training Loop
make_reproducible(int(SEED))
train_ds = (
    LDODataset(
        train_files,
        feature_config,
        use_caching = True,
        filter_condition = _filter_condn
    )
)

test_ds = (
    LDODataset(
        test_files,
        feature_config,
        use_caching = True,
        filter_condition = _filter_condn
    )
)

abmilite = ABMILite(**model_init_kwargs)
optimizer = AdamW(abmilite.parameters(), lr = LEARN_RATE, weight_decay = 0.1)
full_losses, best_epoch = (
    trainer.fit(
        abmilite,
        data_loader_train = DataLoader(train_ds, batch_size = 1),
        optimizer = optimizer,
        loss_func = binary_cross_entropy_with_logits,
        # NOTE: validation should also be training - NOT test set
        data_loader_val = DataLoader(train_ds, batch_size = 1),
        val_metric = METRICS,
        binarize = BINARIZE,
        model_selection_method = ["balanced_accuracy", "average_precision"],
        l1_strength = L1_STRENGTH,
        l1_classifier_only = True,
        l1_attention_in_layer_only = True,
        num_epochs = NUM_EPOCHS,
        on_device = "cuda"
    )
)

# >> NOTE: This should be masked when running the script :D
test_performance = (
    trainer.validate(
        abmilite,
        DataLoader(test_ds, batch_size = 1),
        metric = METRICS,
        binarize = BINARIZE,
        on_device = "cuda"
    )
)

if SHOW_TEST_METRICS:
    print(test_performance)

### === Get Results === ###
# >> Loss Plots
losses = np.array(losses)

losses_mean = losses.mean(axis = 0)
losses_min = losses.min(axis = 0)
losses_max = losses.max(axis = 0)
epochs = np.arange(NUM_EPOCHS) + 1

# >>>> Make Figure
f_loss, a_loss = plt.subplots()

a_loss.plot(losses_mean, label = "Mean Loss Across CV")
a_loss.plot(full_losses)
a_loss.fill_between(epochs, losses_min, losses_max, alpha = 0.1)

# >> Feature Importances
with open(feature_config, "r") as f:
    feature_names = json.load(f)['feature_names']

classifier_w = abmilite.classifier.weight.numpy(force = True).ravel()
classifier_w_sort_idx = np.argsort(np.abs(classifier_w))[::-1]
feature_names_sorted = [feature_names[i] for i in classifier_w_sort_idx]

# >>>> Make Figure
fig_ft_impt, ax_ft_impt = plt.subplots(figsize = (25,5))

ax_ft_impt.bar(
    (x_axis := np.arange(len(classifier_w))),
    classifier_w[classifier_w_sort_idx]
)
ax_ft_impt.set_xticks(
    x_axis, feature_names_sorted, rotation = 60, ha = 'right'
)
ax_ft_impt.set_ylabel("Parameter Value (Classifier Head)")

# >> Save All Figures
(OUTPUTS_DIR / "figures").mkdir(exist_ok = True, parents = True)
f_loss.savefig(
    OUTPUTS_DIR / "figures/abmilite_loss.png",
    dpi = 100,
    bbox_inches = 'tight'
)
fig_ft_impt.savefig(
    OUTPUTS_DIR / "figures/feat_impt.png",
    dpi = 300,
    bbox_inches = 'tight'
)

### === Apply Scoring to All Available Data Files === ###
# training_dir already contains all data not just training/testing
# entire_ds = LDODataset(training_dir, feature_configs, use_caching = False)
abmilite.eval()
abmilite = abmilite.to(DEVICE)
target_name = "DW2Y"
score_mapping = {
    "patient_id" : [],
    "patient_ldo_score" : [],
    "biopsy_id" : [],
    "biopsy_ldo_score" : [],
}

for _f in tqdm(data_files, desc = "Inference on all", total = len(data_files)):
    # Load data
    # NOTE: These files are already normalized and feature-selected.
    # For generalized applications, we will need to pull the features from
    # The `filtered_feature_config.json` and use the normalizer
    data = pl.read_csv(_f)
    # We have loaded the predictor names as `feature_names` variable.

    data_pt = (
        data # Patient-level data
            .filter(pl.col("Visit_No") == 1)
            .select(feature_names)
            .to_torch()
            .float()
    )
    data_bx = data.group_by("BiopsyID") # By biopsy as (name, data)

    with th.no_grad():
        patient_score = abmilite(data_pt.to(DEVICE)).item()

    # >> NOTE: All biopsies, regardless of Visit No are analyzed below
    # >> Visit_No == 1 is only for calculating patient level scores
    for f_of_origin, _data_bx in data_bx:
        _data_bx = _data_bx.select(feature_names).to_torch().float()
        with th.no_grad():
            biopsy_score = abmilite(_data_bx.to(DEVICE)).item()

        score_mapping['biopsy_id'] += [f_of_origin[0]]
        score_mapping['biopsy_ldo_score'] += [biopsy_score]
        score_mapping['patient_id'] += [_f.stem]
        score_mapping['patient_ldo_score'] += [patient_score]

score_mapping_df = pl.DataFrame(score_mapping)

# >> SAVE EVERYTHING
score_mapping_df.write_csv(OUTPUTS_DIR / "score_mapping.csv")
th.save(abmilite.cpu().state_dict(), OUTPUTS_DIR / "abmilite-params.pt")
with open(OUTPUTS_DIR / "abmilite_init_config.json", "w") as f:
    json.dump(model_init_kwargs, f, indent = 4)

### === Verify Distribution of Instance Logit/Attention === ###
logit = []
attention = []
labels = []

abmilite.eval()
abmilite = abmilite.to(DEVICE)
for x, y in DataLoader(train_ds, batch_size = 1):
    with th.no_grad():
        y_instances = abmilite(x.to(DEVICE), mode = 'instance').squeeze()
        a_instances = (
            abmilite.attention_module(
                x.to(DEVICE), softmax = False, transpose = False
            )
            .squeeze()
        )
        assert y_instances.shape == a_instances.shape

    labels += [y] * y_instances.shape[0]

    attention.append(a_instances)
    logit.append(y_instances)

attention = th.cat(attention, dim = 0).cpu().numpy().ravel()
logit = th.cat(logit, dim = 0).cpu().numpy().ravel()
labels = th.cat(labels, dim = 0).cpu().numpy().ravel()

za_mx = np.stack([logit, attention], axis = 1)
za_scaler = StandardScaler()
za_mx_scaled = za_scaler.fit_transform(za_mx)

# >> Fit Clustering
sample_idx = np.random.choice(len(labels), len(labels), replace = False)
bgm = BGM(
    n_components=3, tol = 1e-3, n_init = 3, 
    covariance_type='spherical',
    warm_start=True, random_state = 2380,
    max_iter = 500
)
log.info("Fitting Bayesian GMM on %d nuclei", len(sample_idx))
cluster_labels = bgm.fit_predict(za_mx_scaled[sample_idx, :])

# >> Get Boundaries and Plot
X = za_mx_scaled.copy()

# 2. Create a dense grid spanning data limits
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = (
    np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
)

# 3. Predict the cluster assignment for every point on the grid
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = bgm.predict(grid_points)
Z = Z.reshape(xx.shape)

# Draw the colored cluster boundaries/regions
_mask = attention > -np.inf

# 1. Initialize the grid
g = sns.JointGrid(
    x=za_mx_scaled[_mask,0],
    y=za_mx_scaled[_mask,1],
    height = 10
)

# 2. Draw scatter plot in center
g.plot_joint(sns.scatterplot, hue=labels[_mask], s = 10, alpha = 0.5)
g.ax_joint.contour(xx,yy,Z,colors='k', linewidths=1)
xlim = g.ax_joint.get_xlim()
ylim = g.ax_joint.get_ylim()

# 3. Draw ONLY KDE curves on the outside
g.plot_marginals(sns.kdeplot, fill=True, hue=labels[_mask])
g.set_axis_labels(
    "Instance Classifier-head Logit (Z-scaled)",
    "Instance Attention Weight (Z-scaled)",
    fontsize=12
)
g.ax_joint.set_xlim(-4, 5)
g.ax_joint.set_ylim(-3, 4)

g.savefig(
    OUTPUTS_DIR / "figures/ldo_za_plot.png",
    dpi = 300,
    bbox_inches = "tight"
)