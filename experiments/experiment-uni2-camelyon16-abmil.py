"""Train attention-based MIL on CAMELYON16 UNI2-H Embeddings
"""
import os
import logging
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime

import dotenv; dotenv.load_dotenv(override = True)
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt

import torch as th
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader

from torchvision.ops import sigmoid_focal_loss

from sklearn import metrics as skm
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.model_selection import StratifiedKFold as SKfold

from src.model import ABMILite
from src.model_trainer import MILTrainer
from src.datasets import CAMELYON16UNI2Embeddings

from src.jointrow import joint_row
from src.utils import cluster_get_pos_class

### === Environment === ###
LOG_DIR = Path(os.getenv("GLOBAL_LOG_DIR")) / "camelyon16"
LOG_DIR.mkdir(exist_ok = True, parents = True)
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR")) / "camelyon16"
(OUTPUTS_DIR / "figures").mkdir(exist_ok = True, parents = True)

log = logging.getLogger(__name__)
now = datetime.now().strftime("%Y-%m-%d %Hh%Mm")
logging.basicConfig(
    level = logging.INFO,
    filename = LOG_DIR / f"{now} {Path(__file__).stem}.log",
    format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)

# ==== HYPERPARAMTERS FOR SCRIPT ==== #
SHOW_TEST_PERFORMANCE = True
ACCELERATION_DEVICE = "cuda"
SEED = 2380
EPOCHS = 40

EMBEDDING_DIM = 1536
HIDDEN_DIM = EMBEDDING_DIM // 4
# ==== END HYPERPARAMTERS DEFS ==== #

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

th.manual_seed(SEED)
th.cuda.manual_seed(SEED)
th.use_deterministic_algorithms(True)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True

###
data_dir = list(Path(os.getenv("CAMELYON16_EMBEDS_DIR")).glob("*.pt"))

# Get the training files, test_ins_labels, and the test files
train_files = (
    np.array(
        list(filter(
            lambda x: not x.stem.startswith("test"),
            data_dir
        ))
    )
)
train_labels = (
    np.array(
        list(map(
            lambda x: int(x.stem.startswith("tumor")),
            train_files
        ))
    )
)

test_files = (
    np.array(
        list(filter(
            lambda x: x.stem.startswith("test_"),
            data_dir
        ))
    )
)

splitter = SKfold(random_state = 2380, shuffle = True)
### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ###

### ====== START TRAINING LOOP ======= ###
trainer = MILTrainer()
model_init = {
    "in_channels" : EMBEDDING_DIM,
    "hidden_size" : HIDDEN_DIM,
    'use_feature_extractor' : False,
    'attention_mode' : "simple",
    'attention_branches' : 3,
}

for train_idx, val_idx in splitter.split(train_files, train_labels):
    abmilite = ABMILite(**model_init).to("cuda")
    n_params = sum([x.nelement() for x in abmilite.parameters()])
    print(f"Training {abmilite.__class__.__name__} with {n_params= :,}.")
    adam = optim.AdamW(abmilite.parameters(), lr = 1e-4)
    # Create dataset
    train_ds = CAMELYON16UNI2Embeddings(train_files[train_idx], device = "cuda")
    val_ds = CAMELYON16UNI2Embeddings(train_files[val_idx], device = "cuda")

    train_loader = (
        DataLoader(
            train_ds,
            batch_size = 1,
            shuffle = True,
            prefetch_factor = None,
            num_workers = 0,
        )
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size = 1,
            prefetch_factor = None,
            num_workers = 0
        )
    )

    trainer.fit(
        abmilite,
        train_loader,
        adam,
        binary_cross_entropy_with_logits,
        None, # {"alpha" : 0.25, "gamma" : 2., "reduction" : "mean"},
        val_loader,
        num_epochs = EPOCHS,
        on_device = "cuda",
        batch_size = 2,
        val_metric = ["roc_auc", "balanced_accuracy", "average_precision"],
        model_selection_method = "average_precision",
        binarize = [False, True, False],
        l1_strength = 1e-4,
        l1_classifier_only = True,
        l1_attention_in_layer_only = True
    )
    break
th.save(
    abmilite.cpu().state_dict(),
    OUTPUTS_DIR / "abmilite_camelyon16_uni2h.pt"
)

### === Run Trained ABMILite model === ###
# >> Ensure ABMILite remains on correct device
abmilite = abmilite.to(ACCELERATION_DEVICE)

# >> Get Test Metrics First
test_ds = CAMELYON16UNI2Embeddings(test_files, device = ACCELERATION_DEVICE)
test_dataloader = DataLoader(test_ds, batch_size = 1)

test_metrics = (
    trainer.validate(
        abmilite,
        test_dataloader,
        ["roc_auc", "balanced_accuracy", "average_precision"],
        [False, True, False]
    )
)

if SHOW_TEST_PERFORMANCE:
    print(test_metrics)

### === Instance-level Inference === ###
train_instance_ds = (
    CAMELYON16UNI2Embeddings(
        train_files,
        labels_by_instance = True
    )
)
test_instance_ds = (
    CAMELYON16UNI2Embeddings(
        test_files,
        labels_by_instance = True
    )
)

# >> Create data loader instances
train_instance_loader = DataLoader(train_instance_ds, batch_size = 1)
test_instance_loader = DataLoader(test_instance_ds, batch_size = 1)

# >> Instance inference for all data
train_za_space = []
train_bag_labels = []
train_ins_labels = []
# >> >> Inference loop for training set
tqdm_msg = "Instance-level Inference (Train)"
for x, y in tqdm(train_instance_loader, tqdm_msg, len(train_instance_ds)):
    x = x.squeeze(0)
    y = y.squeeze(0)
    with th.no_grad():
        z_hat = abmilite(x.to(ACCELERATION_DEVICE), mode = 'instance').cpu()
        # By default, attention module does NOT apply softmax or transpose
        a_hat = abmilite.attention_module(x.to(ACCELERATION_DEVICE)).cpu()

    za_vec = th.cat([z_hat, a_hat], dim = 1)

    train_za_space.append(za_vec)
    train_ins_labels.append(y)
    train_bag_labels.append(
        th.ones_like(y) if y.max() > 0 else y
    )

train_za_space = th.cat(train_za_space, dim = 0).numpy(force = True)
train_ins_labels = th.cat(train_ins_labels, dim = 0).numpy(force = True)
train_bag_labels = th.cat(train_bag_labels, dim = 0).numpy(force = True)

# >> >> Inference loop for test data
test_za_space = []
test_bag_labels = []
test_ins_labels = []
# >> >> Inference loop for test set
tqdm_msg = "Instance-level Inference (Test)"
for x, y in tqdm(test_instance_loader, tqdm_msg, len(test_instance_ds)):
    x = x.squeeze(0)
    y = y.squeeze(0)
    with th.no_grad():
        z_hat = abmilite(x.to(ACCELERATION_DEVICE), mode = 'instance').cpu()
        # By default, attention module does NOT apply softmax or transpose
        a_hat = abmilite.attention_module(x.to(ACCELERATION_DEVICE)).cpu()

    za_vec = th.cat([z_hat, a_hat], dim = 1)

    test_za_space.append(za_vec)
    test_ins_labels.append(y)
    test_bag_labels.append(
        th.ones_like(y) if y.max() > 0 else y
    )

test_za_space = th.cat(test_za_space, dim = 0).numpy(force = True)
test_ins_labels = th.cat(test_ins_labels, dim = 0).numpy(force = True)
test_bag_labels = th.cat(test_bag_labels, dim = 0).numpy(force = True)

### === Make Figure: ZA Space by Instance Label and Slide Label === ###
sample_mask = (
    np.random.choice(
        _n_ins := test_ins_labels.shape[0], 
        _n_ins // 10,
        replace = False
    )
)

datasets = [
    {
        "x" : test_za_space[sample_mask, 0],
        "y" : test_za_space[sample_mask, 1],
        "hue" : label_set[sample_mask, 0],
        "title" : label_str,
        "ylabel" : "Attention Weight"
    }
    for label_str, label_set in 
    zip(
        ["By Patch Label", "By Slide Label"],
        [test_ins_labels, test_bag_labels]
    )
]

fig_za_scatter, axes_za_scatter = (
    joint_row(
        datasets,
        xlabel = "Instance Classifier-head Logit",
        scatter_kws = {'s' : 1, 'alpha' : 0.25},
        maintype = 'scatter'
    )
)

### === Make Figure: ROC AUC of Instance Attention, Logit, Clustering === ###
# >> Train Bayesian Gaussian Mixture on Training Data
sample_mask = (
    np.random.choice(
        _n_ins := train_za_space.shape[0],
        _n_ins // 10,
        replace = False
    )
)
bgm = (
    BGM( # Bayesian Gaussian Mixture
        n_components = 3,
        warm_start = True,
        random_state = int(SEED),
        n_init = 3
    )
)
bgm.fit(train_za_space[sample_mask, :])

cluster_scores = cluster_get_pos_class(test_za_space, bgm)

# >> Make plot and Get ROC curves for each scoring method
fig_roc, ax_roc = plt.subplots(figsize = (6,5))

# >> >> Attention Only
fpr, tpr, thresholds = skm.roc_curve(test_ins_labels, test_za_space[:,1])
attn_roc_auc = skm.auc(fpr, tpr)
ax_roc.plot(
    fpr, tpr,
    color = 'darkorange',
    lw = 2,
    label = f'Attention Weight (AUC = {attn_roc_auc:.2f})'
)

# >> >> Logit Only
fpr, tpr, thresholds = skm.roc_curve(test_ins_labels, test_za_space[:,0])
logit_roc_auc = skm.auc(fpr, tpr)
ax_roc.plot(
    fpr, tpr,
    color = 'darkblue',
    lw = 2,
    label = f"Instance Logit (AUC = {logit_roc_auc:.2f})"
)

# >> >> Cluster Score
fpr, tpr, threshold = skm.roc_curve(test_ins_labels, cluster_scores)
cluster_roc_auc = skm.auc(fpr, tpr)
ax_roc.plot(
    fpr, tpr,
    color = 'cyan',
    lw = 2,
    label = f"Clustering (AUC = {cluster_roc_auc:.2f})"
)

# >> Plot cosmetics
ax_roc.plot(
    [0,1], [0,1],
    color = 'navy',
    lw = 2,
    linestyle = '--',
    label = 'Random Guess'
)

ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
ax_roc.set_ylabel("True Positive Rate (Sensitivity)")
ax_roc.set_title("Instance Class Estimation Performance")
ax_roc.legend(loc = "lower right")
ax_roc.grid(True, linestyle = ":", alpha = 0.6)

### === Save all plots === ###
fig_za_scatter.savefig(
    OUTPUTS_DIR / "figures/camelyon16-uni2-za-plot.png",
    dpi = 300,
    bbox_inches = 'tight'
)
fig_roc.savefig(
    OUTPUTS_DIR / "figures/instance_roc.png",
    dpi = 300,
    bbox_inches = 'tight'
)
