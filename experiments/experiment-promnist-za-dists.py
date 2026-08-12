"""
Get the distribution of instance logit x attention from
all datasets generated in `experiment-promnist-bags.py`
"""

import re
from pathlib import Path

import polars as pl
from tqdm import tqdm

import torch as th
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader
from torch import optim

from src.model import ABMIL
from src.model_trainer import MILTrainer

DEVICE = "cuda"

def get_za_of_ds(
        model: ABMIL,
        data_loader: DataLoader
) -> th.Tensor:
    data_loader.dataset.mode = 'instance'
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

    return za_mx.numpy(), instance_labels.numpy()

data_save_dir = Path("./data/ProportionMNISTBags")
files_to_plot_za = [
    "test_ds_t00_seed_1780",
    "test_ds_t025_seed_1780",
    "test_ds_t05_seed_1780",
    "test_ds_t075_seed_1780",
]
model_trainer = MILTrainer()
metrics_to_compute = ['balanced_accuracy', 'roc_auc', 'average_precision']
za_dist = {}
for f in tqdm(files_to_plot_za, total = len(files_to_plot_za)):
    trainf = data_save_dir / f"{f.replace('test', 'train')}.pt"
    testf = data_save_dir / f"{f}.pt"

    train_ds = th.load(trainf, weights_only = False)
    train_loader = DataLoader(train_ds, batch_size = 1)
    test_ds = th.load(testf, weights_only = False)
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

    match = re.search(r"_ds_t(.*?)_seed_", trainf.stem)

    if match:
        tau_str = match.group(1)
        tau = float(tau_str[0] + tau_str[1:])

    za_mx, labels = get_za_of_ds(abmil, test_loader)
    _taus = [tau] * za_mx.shape[0]
    za_dist['z'] = za_dist.get('z', []) + za_mx[:,0].tolist()
    za_dist['a'] = za_dist.get('a', []) + za_mx[:,1].tolist()
    za_dist['y'] = za_dist.get('y', []) + labels.tolist()
    za_dist['t'] = za_dist.get('t', []) + _taus

za_dist = pl.DataFrame(za_dist)
za_dist.write_csv("./outputs/promnist_za_dist.csv")
