"""Definition of various datasets used in experiments.

"""
import time
from pathlib import Path
from typing import Union, Literal

import zarr
import numpy as np
import torch as th
from torch.utils.data import Dataset

from torchvision import datasets, transforms


class ProportionMnistBags(Dataset):
    """Class to generate proportion mnist bags dataset."""
    def __init__(
        self,
        binary_target: int = 9,
        with_noisy_boundary: bool = False,
        positive_bag_prop_in_dataset: float = 0.5,
        threshold_prop_for_positivity: float = 0.15,
        target_num_prop_mean: float = 0.225,
        target_num_prop_max: float = 0.5,
        target_num_prop_sd: float = 0.05,
        null_prop_mean: float = 0.1,
        null_prop_sd: float = 0.05,
        bag_size_mean: int = 10,
        bag_size_sd: float = 1.,
        random_seed: int = 2380,
        mnist_source_is_training: bool = True,
        label_mode: Literal['bag', 'instance'] = 'bag'
    ) -> None:
        """Initialize Proportion MNIST dataset with specified distribution.
        """
        # Save input parameters
        for name, value in vars().items():
            if name != 'self':
                setattr(self, name, value)

        # Shorthand alias for convenience
        self.tau = threshold_prop_for_positivity

        # Initialization Logic
        super().__init__()
        self.mnist = (
            datasets.MNIST(
                root = "./data",
                train = mnist_source_is_training,
                download = True,
                transform = transforms.ToTensor()
            )
        )
        self.mode = label_mode

        # Load data in
        self.x, self.y, self.instance_labels, self.propdist = self.load_bags()

    def load_bags(self):
        """
        Create bags whose labels are determined by the proportion of positive
        instances.
        """
        # Load instances from MNIST datset
        mnist_x = self.mnist.data
        mnist_y = self.mnist.targets

        # Initialize Random Number Generator
        self.rng = np.random.RandomState(self.random_seed)

        # Indices of instances in MNIST dataset to go through
        indices = th.arange(len(mnist_y))
        target_idx = indices[mnist_y == self.binary_target]
        non_target_idx = indices[~(mnist_y == self.binary_target)]

        # Pointers to track where in the dataset we are
        pos_ptr = neg_ptr = 0

        # Containers for return
        bags_x = [] # Bags of instance data - list of BxHxW
        bags_y = [] # Labels of bag data - list of 1
        instance_labels = [] # Instance labels
        dist_pos_prop = { # Distribution of positive instance props/bag
            0: [],
            1: []
        }

        while True:
            bag_is_pos = np.random.choice(
                [True, False],
                p = [
                    self.positive_bag_prop_in_dataset, # Prob of being pos
                    1 - self.positive_bag_prop_in_dataset # p of being neg
                ]
            )
            # Get the number of instances in this bag
            n_i = int(self.rng.normal(self.bag_size_mean, self.bag_size_sd))

            # Get the prop of positive instances for this bag
            # bag_is_pos indicates bag label
            # with_noisy_boundary means the bag can have a negative
            # bag label with prop of positive instances slightly
            # above the prop threshold (and vice versa)
            if bag_is_pos and not self.with_noisy_boundary:
                _low = int(np.ceil(self.tau * n_i))
                _up = int(np.floor(self.target_num_prop_max * n_i))
            elif bag_is_pos and self.with_noisy_boundary:
                _low = 0.
                _up = int(np.floor(self.target_num_prop_max * n_i))
            elif not bag_is_pos and not self.with_noisy_boundary:
                _low = 0.
                _up = int(np.floor(self.tau * n_i))
            elif not bag_is_pos and self.with_noisy_boundary:
                _low = 0.
                _up = n_i

            # Get prop of positive instances
            _prop = (
                self.target_num_prop_mean if bag_is_pos
                else self.null_prop_mean
            )
            _sd = self.target_num_prop_sd if bag_is_pos else self.null_prop_sd

            positive_instance_prop = abs(self.rng.normal(_prop, _sd))
            n_positive_instances = round(positive_instance_prop * n_i)
            n_positive_instances = min(max(n_positive_instances, _low), _up)
            n_negative_instances = n_i - n_positive_instances

            # Get indices of pos/neg instances
            pos_idx = target_idx[pos_ptr : pos_ptr + n_positive_instances]
            neg_idx = non_target_idx[neg_ptr : neg_ptr + n_negative_instances]
            bag_idx = th.cat([pos_idx, neg_idx])

            # Double check bag label
            bag_label = th.tensor([[int(bag_is_pos)],], dtype = th.float32)
            bags_x.append(mnist_x[bag_idx])
            bags_y.append(bag_label)
            instance_labels.append(mnist_y[bag_idx])

            # Update pointers
            pos_ptr += n_positive_instances
            neg_ptr += n_negative_instances

            # Update stats
            empirical_prop = len(pos_idx) / (len(pos_idx) + len(neg_idx))
            dist_pos_prop[int(bag_is_pos)] += [round(empirical_prop, 4)]

            if pos_ptr > len(target_idx) or neg_ptr > len(non_target_idx):
                break

        return bags_x, bags_y, instance_labels, dist_pos_prop

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        if self.mode == "instance":
            y = self.instance_labels[idx]
        else:
            y = self.y[idx].squeeze(1)

        return self.x[idx].unsqueeze(1).float(), y.float()

### --- CAMELYON16 DATASET DEFINITION --- ###

class CAMELYON16UNI2Embeddings(Dataset):
    """Dataset of CAMELYON16 whole slide images (WSIs) as bags of patches.

    See ___ for specification of data file schema.

    """
    def __init__(
            self,
            directory: Union[Path | list[Path]],
            labels_by_instance: bool = False,
            with_caching: bool = False,
            device: str = "cpu"
    ) -> None:
        """Load in directory of data as PyTorch dataset.

        Args:
            directory (Path): Folder containing .pt files where each .pt file
                corresponds to a CAMELYON16 WSI whose patches (256x256) are
                embedded by UNI2-H.
            labels_by_instance (bool): Retrieved label will be on the instance
                level instead of bag-level (slide).
            with_caching (bool): Use caching for faster retrieval if memory
                capacity allows.
            device (str): custom arg available on the Dataset level to load
                .pt files onto specific device.
        """
        # If `directory` is NOT a Path object, it is assumed to be a
        # pre-filtered list of files (Path objects)
        self.files = (
            list(directory.glob("*.pt")) if isinstance(directory, Path)
            else directory
        )
        self.labels_by_instance = labels_by_instance
        self.use_caching = with_caching
        self.device = device
        self.cache = {}

    def __len__(self):
        """Return number of bags."""
        return len(self.files)

    def __getitem__(self, idx: int) -> th.Tensor:
        """Get specific slide from files and the UNI2-H embeddings of patches.

        Args:
            idx (int): Index of WSI in self.files.

        """
        data = (
            th.load(
                self.files[idx],
                weights_only = True,
                map_location = self.device
            )
        )

        labels = data['labels'].unsqueeze(1)
        embeddings = data['features']
        label = (
            labels if self.self.labels_by_instance
            else th.max(labels, dim = 0, keepdim = False).values
        )

        return embeddings.float(), label.float()