import os
import random
import numpy as np
import torch

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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Dynamic multi-GPU stability
    
    # 3. Force Deterministic NVIDIA cuDNN Backends
    # Forces cuDNN to select deterministic algorithms (sacrifices slight speed)
    torch.backends.cudnn.deterministic = True
    # Disables autotuner benchmarking which dynamically changes algorithms
    torch.backends.cudnn.benchmark = False
    
    # 4. Strict Error Handlers for Non-Deterministic Operations
    # Forces PyTorch to raise an error if an operation cannot be run deterministically
    torch.use_deterministic_algorithms(True)
    
    # 5. Environment variable override for specific CUDA atomic operations
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # or ":16:8"
