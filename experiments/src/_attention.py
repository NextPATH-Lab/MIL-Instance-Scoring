"""
Definition of attention modules for multiple instance learning (MIL).

`MILAttention` is the primary exposed class, which serves as a wrapper for
different modes of attention.
"""
from itertools import chain
from typing import Literal, Generator

import torch as th
from torch import nn

class GatedAttention(nn.Module):
    """Implementation of Gated Attention as per Ilse et al. 2018
    
    """
    def __init__(
            self,
            in_channels: int = 64,
            hidden_size: int = 64,
            attention_branches: int = 1,
    ) -> None:
        """Initialize torch module components.
        """
        super().__init__()

        self.attention_v = (
            nn.Sequential(
                nn.Linear(in_channels, hidden_size),
                nn.Tanh()
            )
        )

        self.attention_u = (
            nn.Sequential(
                nn.Linear(in_channels, hidden_size),
                nn.Sigmoid()
            )
        )

        self.attention_w = nn.Linear(hidden_size, attention_branches)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Gated Attention forward pass

        Args:
            h (Tensor): Input linear 2D tensor of N x in_channels for N
                instances per bag.

        Returns:
            Tensor: Linear attention projection of N x 1
        """
        a_v = self.attention_v(x)
        a_u = self.attention_u(x)

        return self.attention_w(a_v * a_u)

class SimpleAttention(nn.Module):
    """Implementation of instance attention as per Ilse et al. 2018
    """
    def __init__(
            self,
            in_channels: int = 64,
            hidden_size: int = 64,
            attention_branches: int = 1,
    ) -> None:
        """Initialize torch module components
        """
        super().__init__()
        self.attention_w = (
            nn.Sequential(
                nn.Linear(in_channels, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, attention_branches)
            )
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Instance attention forward pass

        Args:
            h (Tensor): Input linear 2D tensor of N x in_channels for N
                instances per bag.

        Returns:
            Tensor: Linear attention projection of N x 1
        """
        return self.attention_w(x)

class MILAttention(nn.Module):
    """Wrapper for different modes of instance attention.
    """
    attn_modes = {
        "simple" : SimpleAttention,
        "gated" : GatedAttention
    }
    # CAUTION: Reduction assumes input is N x H, where N & H are the number of
    # instances per bag and hidden dimension respectively.
    # This can sometimes cause bugs if the dataset is passed through a
    # DataLoader instance which attaches a batched dimension (B x N x H)
    reduction_funcs = {
        "mean" : lambda x: th.mean(x, dim = 1, keepdim = True),
        "max"  : lambda x: th.max(x, dim = 1, keepdim = True),
        "sum"  : lambda x: th.sum(x, dim = 1, keepdim = True)
    }

    def __init__(
            self,
            mode: Literal['simple', 'gated'],
            in_channels: int,
            hidden_size: int,
            attention_branches: int,
            reduction: Literal['mean', 'max', 'sum'] = 'mean'
    ) -> None:
        """Initialize specified attention mode
        """
        super().__init__()
        attn_kwargs = {
            k : v
            for k, v in locals().items()
            if k not in ['self', 'mode', '__class__', 'reduction']
        }
        # (OOP) Instance attributes
        self.mode = mode
        self.num_branches = attention_branches
        self.reduction_method = reduction.lower()

        self.attention_module = MILAttention.attn_modes[mode](**attn_kwargs)
        self.reduction_func = MILAttention.reduction_funcs[reduction.lower()]

    def forward(
            self,
            x: th.Tensor,
            softmax: bool = False,
            transpose: bool = False
    ) -> th.Tensor:
        """Forward pass of selected attention mode.

        Args:
            x (th.Tensor): Input 2D tensor of shape N x H.
            softmax (bool): Whether to apply a softmax norm on the N dim.
            transpose (bool): Whether to transpose the output attention weights.

        Returns:
            th.Tensor: The output attenttion weights of inputted instance
                embeddings.
        """
        attn: th.Tensor = self.attention_module(x)
        # CHECK: If batch dimension exists
        if attn.ndim == 3:
            # We assume if ndim == 3, the data has been provided thorugh a
            # DataLoader with an extra (empty) batch dim of 1.
            attn = attn.squeeze(0)

        if self.num_branches > 1:
            # Reduce attention branches into dim of size 1
            attn = self.reduction_func(attn)
        if softmax:
            # Apply a softmax norm in the N dimension.
            attn = th.softmax(attn, dim = 0)
        if transpose:
            # Transpose attention for matrix multiplication later
            attn = th.transpose(attn, 1, 0)

        return attn