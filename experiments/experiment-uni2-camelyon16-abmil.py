"""Train attention-based MIL on CAMELYON16 UNI2-H Embeddings
"""
import os, sys
import logging
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import dotenv; dotenv.load_dotenv(override = True)
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt

import torch as th
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader

from torchvision.ops import sigmoid_focal_loss

from sklearn.model_selection import StratifiedKFold as SKfold

from src.model import ABMILite
from src.model_trainer import MILTrainer
from src.datasets import CAMELYON16UNI2Embeddings

logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout)

# ==== HYPERPARAMTERS FOR SCRIPT ==== #
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
data_dir = list(Path(os.getenv("DATA_CAMELYON_UNI2_VEC_PATH")).glob("*.pt"))

# Get the training files, labels, and the test files
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
th.save(abmilite.cpu().state_dict(), "./abmilite_camelyon16_uni2h.pt")

