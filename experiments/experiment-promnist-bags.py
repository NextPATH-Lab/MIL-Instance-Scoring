"""
====== Proportion MNIST Bags Experiment ======

"""
# ==== HYPERPARAMETERS OF EXPERIMENTATION ==== #
from pathlib import Path

# ======== EDIT VARIABLES HERE ======== #
BAG_SIZE = 20
SEEDS_TO_TEST = [2380, 1780, 0, 100, 260115]
TAUS_TO_TEST = list(map(lambda x: x / 100, range(0, 105, 5)))
DATA_SAVE_DIR = Path("./data/ProportionMNISTBags")
DATA_SAVE_DIR.mkdir(exist_ok = True, parents = True)
# ======== END VARIABLE EDITS ========= #

# Short-hand to import Native Python Deps
import re, os, json, time, random, logging, warnings
from tqdm import tqdm
from datetime import datetime
from itertools import product
from dotenv import load_dotenv
load_dotenv(override=True)

import polars as pl
import numpy as np

from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader
from torch import optim
import torch as th

from src.model import ABMIL
from src.model_trainer import MILTrainer
from src.datasets import ProportionMnistBags
from src.utils import get_za_vectors_roc_correlation

warnings.filterwarnings('ignore')
# ==== Logging Set Up ==== #
LOG_DIR = Path(os.getenv("GLOBAL_LOG_DIR")) / "ProMNIST"
LOG_DIR.mkdir(exist_ok = True, parents = True)
OUTPUT_DIR = Path(os.getenv("OUTPUTS_DIR")) / "ProMNIST"
OUTPUT_DIR.mkdir(exist_ok = True, parents = True)

log = logging.getLogger(__name__)
now = datetime.now().strftime("%Y-%m-%d %Hh%Mm")
logging.basicConfig(
    level = logging.INFO,
    filename = LOG_DIR / f"{now} {Path(__file__).stem}.log", 
    format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)
DEVICE = os.getenv("ACCELERATION_DEVICE")

# ============================ #
# ===== Dataset Creation ===== #
# ============================ #
# >>> Dataset creation is given a separate loop for convenience
ds_name = "%s_ds_t%s_seed_%d.pt"
for t in TAUS_TO_TEST:
    tau_str = str(float(t)).replace(".", "")
    pos_mean = min(t + 0.15, 1.0)
    pos_sd = 0.05 if pos_mean < 1 else 0
    null_mean = max(0, t - 0.15)
    null_sd = 0.05 if null_mean > 0 else 0

    dataset = {}
    for ds_seed in SEEDS_TO_TEST:
        for _set in ['train', 'test']:
            dss_path = DATA_SAVE_DIR / (ds_name % (_set, tau_str, ds_seed))
            if dss_path.exists():
                continue

            _dataset = (
                ProportionMnistBags(
                    threshold_prop_for_positivity = t,
                    target_num_prop_mean = pos_mean,
                    target_num_prop_max = 1,
                    target_num_prop_sd = pos_sd,
                    null_prop_mean = null_mean,
                    null_prop_sd = null_sd,
                    mnist_source_is_training = (_set == "train"),
                    with_noisy_boundary = False,
                    bag_size_mean = BAG_SIZE,
                    random_seed = ds_seed
                )
            )

            th.save(_dataset, str(dss_path))

model_trainer = MILTrainer()
dataset_files = list(DATA_SAVE_DIR.glob("*train*.pt"))
taus_with_seeds = list(product(dataset_files, SEEDS_TO_TEST))
metrics_to_compute = ['balanced_accuracy', 'roc_auc', 'average_precision']

za_dist = {}
dataset_level_metrics = {}

for (t, s) in tqdm(taus_with_seeds, total = len(taus_with_seeds)):
    start = time.time()
    iteration_key = f"{t.stem}_modelseed_{s}"
    if (DATA_SAVE_DIR / f"{iteration_key}.json").exists():
        log.info(f"{iteration_key} already done. Skipping...")
        continue

    dataset_seed = t.stem.split("seed_")[-1]
    random.seed(s)
    np.random.seed(s)
    th.manual_seed(s)
    th.cuda.manual_seed(s)
    match = re.search(r"_ds_t(.*?)_seed_", t.stem)

    if match:
        tau_str = match.group(1)
        tau = float(tau_str[0] + "." + tau_str[1:])

    log.info(
        f"Starting iteration with {tau=}, {dataset_seed=}, model seed: {s}"
    )

    train_ds = th.load(t, weights_only = False)
    train_loader = DataLoader(train_ds, batch_size = 1)
    test_ds = (
        th.load(
            t.parent / t.name.replace("train", "test"),
            weights_only = False
        )
    )
    test_loader = DataLoader(test_ds, batch_size = 1)

    abmil = (
        ABMIL(
            conv_layers = [5, 20],
            linear_in_channels = 32,
            hidden_size = 16
        )
    )
    abmil = abmil.to(DEVICE)
    adam = optim.Adam(abmil.parameters(), lr = 1e-3)

    model_trainer.fit(
        abmil,
        train_loader,
        adam,
        binary_cross_entropy_with_logits,
        data_loader_val = train_loader,
        val_metric = metrics_to_compute,
        binarize = [True, False, False],
        model_selection_method = "average_precision",
        num_epochs = 5,
        on_device = DEVICE
    )

    metrics = (
        model_trainer.validate(
            abmil,
            test_loader,
            metric = metrics_to_compute,
            binarize = [True, False, False],
            on_device = DEVICE
        )
    )

    za_roc = get_za_vectors_roc_correlation(abmil, test_loader)

    metrics = metrics | za_roc
    metrics['model_seed'] = s
    metrics['dataset_seed'] = dataset_seed
    metrics['tau'] = tau

    for k,v in metrics.items():
        dataset_level_metrics[k] = dataset_level_metrics.get(k, []) + [v]

    with open(DATA_SAVE_DIR / f"{iteration_key}.json", "w") as f:
        json.dump(metrics, f, indent = 4)

    end = time.time()
    log.info(f"Iteration took {(end - start) / 60 :.4f} minutes")

dataset_level_metrics = pl.DataFrame(dataset_level_metrics)
dataset_level_metrics.write_csv(OUTPUT_DIR / "promnist.csv")
