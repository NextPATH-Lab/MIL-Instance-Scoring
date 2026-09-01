"""Definition of various datasets used in experiments.

"""
import os
import json
import logging
from pathlib import Path
from typing import Union, Literal, Optional

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

import torch as th
from torch.utils.data import Dataset
from torchvision import datasets, transforms

log = logging.getLogger(__name__)

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
            empirical_prop = len(pos_idx) / max(len(pos_idx) + len(neg_idx), 1)
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
            device: str = "cpu",
            max_bag_size: Optional[int] = None,
            min_tumor_fraction: float = 0.01,
            feature_fraction: Optional[float] = None,
            feature_subset_path: Optional[Path] = None,
            feature_subset_seed: Optional[int] = None,
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
            max_bag_size (Optional[int]): If set, subsamples each bag down to
                at most this many instances (re-sampled fresh on every
                __getitem__ call, i.e. every epoch). Stratified to preserve
                the bag's true tumor:benign ratio, except when that would
                keep fewer than `min_tumor_fraction * max_bag_size` tumor
                instances -- e.g. a micro-metastasis slide where tumor
                patches are a tiny fraction of the bag -- in which case every
                tumor instance is kept instead and the remainder is filled
                with randomly sampled benign instances. None (default)
                disables subsampling; leave it None for validation/test bags
                so metrics stay unbiased.
            min_tumor_fraction (float): Minimum fraction of `max_bag_size`
                that should be tumor instances, when any exist, before
                falling back to keeping every tumor instance. Ignored if
                `max_bag_size` is None.
            feature_fraction (Optional[float]): If set (e.g. 0.5), restricts
                every embedding to a *fixed* random subset of that fraction
                of its feature dimensions (e.g. 768 of UNI2-H's 1536) -- a
                deliberately incomplete feature set, chosen once rather than
                per-sample. The subset is generated on first use and saved to
                `feature_subset_path` as JSON; any dataset instance pointed
                at the same path (e.g. train/val/test splits of the same
                experiment) loads the identical subset instead of each
                independently randomizing its own -- required for the
                feature at a given index to mean the same thing across
                splits. None (default) uses the full feature vector.
            feature_subset_path (Optional[Path]): Where the chosen indices
                are saved/loaded. Defaults to
                `CONFIG_DIR/camelyon16/uni2_feature_subset_frac{fraction}.json`
                (`CONFIG_DIR` from the environment). Ignored if
                `feature_fraction` is None.
            feature_subset_seed (Optional[int]): RNG seed used only the first
                time the subset is generated (i.e. when `feature_subset_path`
                doesn't exist yet). Irrelevant on every subsequent load, since
                the saved indices are reused as-is.
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
        self.max_bag_size = max_bag_size
        self.min_tumor_fraction = min_tumor_fraction

        self.feature_indices = None
        if feature_fraction is not None:
            if feature_subset_path is None:
                config_dir = Path(os.getenv("CONFIG_DIR", "./experiments/config")) / "camelyon16"
                feature_subset_path = (
                    config_dir / f"uni2_feature_subset_frac{feature_fraction}.json")
            self.feature_indices = (
                self._load_or_create_feature_subset(
                    Path(feature_subset_path), feature_fraction, feature_subset_seed)
                .to(device)
            )

    def _load_or_create_feature_subset(
            self,
            path: Path,
            fraction: float,
            seed: Optional[int],
    ) -> th.Tensor:
        """Load a previously-saved fixed feature-index subset, or generate
        and save a new one if `path` doesn't exist yet. See `feature_fraction`
        in __init__ for why this must be shared (not re-randomized) across
        dataset splits."""
        if path.exists():
            with open(path, "r") as f:
                indices = json.load(f)
            log.info("Loaded fixed feature subset (%d indices) from %s", len(indices), path)
            return th.tensor(indices, dtype = th.long)

        # Peek the first file to learn the embedding dimension without
        # loading every file up front.
        sample = th.load(self.files[0], weights_only = True, mmap = True, map_location = "cpu")
        total_dim = sample["features"].shape[1]
        n_keep = round(fraction * total_dim)

        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(total_dim, size = n_keep, replace = False).tolist())

        path.parent.mkdir(parents = True, exist_ok = True)
        with open(path, "w") as f:
            json.dump(indices, f)
        log.info(
            "Generated fixed feature subset: %d of %d dims (fraction=%s), saved to %s",
            n_keep, total_dim, fraction, path
        )

        return th.tensor(indices, dtype = th.long)

    def __len__(self):
        """Return number of bags."""
        return len(self.files)

    def _stratified_sample(self, instance_labels: th.Tensor) -> th.Tensor:
        """Indices to subsample a bag down to self.max_bag_size instances.

        Preserves the bag's true tumor:benign ratio via proportional
        stratified sampling, unless that would keep fewer than
        `min_tumor_fraction * max_bag_size` tumor instances -- in that case
        every tumor instance is kept and the remainder filled with randomly
        sampled benign instances, so tumor-sparse (e.g. micro-met) bags don't
        lose their only positive evidence to rounding.
        """
        n_total = instance_labels.shape[0]
        if n_total <= self.max_bag_size:
            return th.arange(n_total, device = instance_labels.device)

        dev = instance_labels.device
        tumor_idx = th.nonzero(instance_labels > 0, as_tuple = True)[0]
        benign_idx = th.nonzero(instance_labels <= 0, as_tuple = True)[0]
        n_tumor = tumor_idx.shape[0]

        tumor_fraction = n_tumor / n_total
        target_tumor = round(tumor_fraction * self.max_bag_size)
        floor_tumor = self.min_tumor_fraction * self.max_bag_size

        if n_tumor > 0 and target_tumor < floor_tumor:
            n_tumor_keep = min(n_tumor, self.max_bag_size)
        else:
            n_tumor_keep = min(target_tumor, n_tumor)

        n_benign_keep = min(self.max_bag_size - n_tumor_keep, benign_idx.shape[0])

        sampled_tumor = tumor_idx[th.randperm(n_tumor, device = dev)[:n_tumor_keep]]
        sampled_benign = (
            benign_idx[th.randperm(benign_idx.shape[0], device = dev)[:n_benign_keep]])

        return th.cat([sampled_tumor, sampled_benign])

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

        instance_labels = data['labels']
        embeddings = data['features']

        if self.feature_indices is not None:
            embeddings = embeddings.index_select(1, self.feature_indices)

        if self.max_bag_size is not None:
            keep_idx = self._stratified_sample(instance_labels)
            embeddings = embeddings[keep_idx]
            instance_labels = instance_labels[keep_idx]

        labels = instance_labels.unsqueeze(1)
        label = (
            labels if self.labels_by_instance
            else th.max(labels, dim = 0, keepdim = False).values
        )

        return embeddings.float(), label.float()

"""
Structure of data:
1.  Initially, each whole slide image (WSI) is an individual .cmg file, 
    containing segmented nuclei and binary masks. Large-scale DNA Organization
    (LDO) features are calculated for each file in .mc0 files.
    For data cleaning purposes, features quantifying cell overlaps are also
    computed, hence requiring files to be kept within each WSI.

2.  Following file cleaning, all .cmg.mc0 files for each patient are merged
    into one file for multiple instance learning. The structure of these files
    are .pt files, and are structured as follows:

    {
        "feature_names" : list[str]
        "tensor" : torch.Tensor (NxD),
    }
    The torch tensor should already contain all clinical information/variables
    of interest in it.

3.  The `LDODataset` instance takes a `config.json` parameter to generate the
    dataset. The config.json should be formatted as the following:

    {
        "feature_names" : list[str]
        "feature_idx" : list[int]
        "target_name" : str
        "target_idx" : int
    }

    feature_names is a list of feature names from the polars dataframe, and 
    feature_idx is the corresponding integer indices of the columns. 
    target_name is the name of the column which we are trying to predict, and
    target_idx is the corresponding integer index of the target column.
"""

class LDODataset(Dataset):
    def __init__(
            self,
            file_directory: str | Path | list[Path | str],
            config_file: Optional[str | Path | dict] = None,
            use_caching: bool = False,
            normalizer: Optional[StandardScaler] = None,
            filter_condition: Optional[pl.Expr] = None
    ) -> None:
        """Load data directory

        Args:
            file_directory (Path): _description_
        """
        self.files = (
            list(file_directory.glob("*.pt")) 
            if isinstance(file_directory, Path)
            else file_directory
        )

        self.cache = {}
        self.use_cache = use_caching
        self.normalizer = (
            normalizer.set_output("polars")
            if isinstance(normalizer, StandardScaler)
            else lambda x: x
        )
        self.filter_condition = filter_condition

        if isinstance(config_file, dict):
            self.config = config_file

        elif config_file is not None:
            with open(config_file, "r") as f:
                self.config = json.load(f)
        else:
            # Currently errors
            self.config = {}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(
            self,
            idx: int,
            filter_condn: Optional[pl.Expr] = None
    ) -> th.Tensor:
        # filter_condn is a per-call override; only results computed with the
        # dataset's fixed self.filter_condition are safe to cache by idx alone.
        cacheable = self.use_cache and filter_condn is None
        if cacheable:
            cached = self.cache.get(idx, None)
            if cached is not None:
                return cached

        _data = pl.read_csv(self.files[idx])

        # Extract feature data
        # Filter condition - prioritize one given as arg
        if filter_condn is not None:
            _data = _data.filter(filter_condn)
        elif self.filter_condition is not None:
            _data = _data.filter(self.filter_condition)

        data = _data.select(self.config['feature_names'])
        data = self.normalizer(data)
        data = data.to_torch().float()

        # Extract label data
        _label = (
            _data.get_column(self.config['target_name'])
                .to_torch().float()[[0]]
        )

        result = (data, _label)

        if cacheable:
            self.cache[idx] = result

        return result
