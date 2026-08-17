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
    """Gated attention module as per Ilse et al. 2018
    """
    def __init__(
            self,
            in_channels: int = 64,
            hidden_size: int = 64,
            attention_branches: int = 1,
    ) -> None:
        super().__init__()
        self.attention_v = (
            nn.Sequential(
                nn.Linear(in_channels, hidden_size, bias = False),
                nn.Tanh()
            )
        )
        self.attention_u = (
            nn.Sequential(
                nn.Linear(in_channels, hidden_size, bias = False),
                nn.Sigmoid()
            )
        )
        self.attention_w = (
            nn.Linear(
                hidden_size,
                attention_branches,
                bias = False
            )
        )

    def forward(self, h):
        """Forward pass of `Attention` instance.

        Args:
            h (Tensor): Input tensor of N x in_channels

        Returns:
            Tensor: Linear attention projection of N x 1
        """
        a_v = self.attention_v(h)
        a_u = self.attention_u(h)
        return self.attention_w(a_v * a_u)

class SimpleAttention(nn.Module):
    """Instance attention as per Ilse et al. 2018
    """
    def __init__(
            self,
            in_channels: int = 64,
            hidden_size: int = 64,
            attention_branches: int = 1,
    ) -> None:
        super().__init__()
        self.attention_w = (
            nn.Sequential(
                nn.Linear(in_channels, hidden_size, bias = False,),
                nn.Tanh(),
                nn.Linear(hidden_size, attention_branches, bias = False)
            )
        )

    def forward(self, h: th.Tensor):
        """Forward pass of `Attention` instance.

        Args:
            h (Tensor): Input tensor of N x in_channels

        Returns:
            Tensor: Linear attention projection of N x 1
        """
        return self.attention_w(h)

class MILAttention(nn.Module):
    attn_modes = {
        'simple' : SimpleAttention,
        'gated' : GatedAttention
    }
    reduction_funcs = {
        "mean" : lambda x: th.mean(x, dim = 1, keepdim = True),
        "max" : lambda x: th.max(x, dim = 1, keepdim = True).values,
        "sum" : lambda x: th.sum(x, dim = 1, keepdim = True)
    }
    def __init__(
            self,
            mode: Literal['simple', 'gated'],
            in_channels: int,
            hidden_size: int,
            attention_branches: int,
            reduction: Literal['mean', 'max', 'sum'] = "mean"
    ):
        super().__init__()
        kwargs = {
            k:v
            for k,v in locals().items()
            if k not in ["self", "mode", "__class__", "reduction"]
        }
        self.mode = mode
        self.attention_module = MILAttention.attn_modes[mode](**kwargs)
        self.num_branches = attention_branches
        self.reduction_method = reduction.lower()
        self.reduction_func = MILAttention.reduction_funcs[reduction.lower()]

    def forward(
        self,
        h: th.Tensor,
        softmax: bool = False,
        transpose: bool = False
    ) -> th.Tensor:
        attn = self.attention_module(h.squeeze(0)) # Remove batch dim
        # If attn has more than one branch, reduce it into one dim.
        if self.num_branches > 1:
            attn = self.reduction_func(attn)

        if softmax and transpose:
            return th.softmax(th.transpose(attn, 1, 0), dim = 1)
        elif softmax:
            return th.softmax(attn, dim = 0)
        elif transpose:
            return th.transpose(attn, 1, 0)
        return attn

    def get_first_layer(self) -> Generator[th.Tensor, None, None]:
        if self.mode == "simple":
            return self.attention_module.attention_w[0].parameters()
        elif self.mode == "gated":
            return chain(
                self.attention_module.attention_v[0].parameters(),
                self.attention_module.attention_u[0].parameters())
        else:
            raise NotImplementedError(f"{self.mode=} not supported.")

if __name__ == "__main__":
    attention = MILAttention("gated", 64,64,1,"mean")
    print(attention.get_first_layer())
