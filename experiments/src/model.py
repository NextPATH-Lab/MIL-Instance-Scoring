"""Definition of Attention-based Deep Multiple Instance Learning Lite

"""
from typing import Literal

import torch as th
from torch import nn

from ._attention import MILAttention

class ABMILite(nn.Module):
    """
    The minimalist architecture for a MIL classifier with attention-based
    instance pooling.

    """

    def __init__(
            self,
            in_channels: int = 64,
            hidden_size: int = 64,
            attention_branches: int = 1,
            attention_mode: Literal['simple', 'gated'] = 'simple',
            attention_branch_reduction: Literal["mean", "max", "sum"] = "mean",
            use_feature_extractor: bool = False,
            feature_extraction_layers: int = 1,
            use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        # >> Attributes
        self.use_feature_extractor = use_feature_extractor
        self.use_bn = use_batch_norm
        # >> NOTE: The implementation of batch norm is INCORRECT here.
        # >> The features will NOT Z-score normalize as expected.

        #########################
        ### LAYER DEFINITIONS ###
        #########################
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(in_channels)
            
        if use_feature_extractor:
            feature_extractor = []
            for _ in range(feature_extraction_layers):
                feature_extractor.extend(
                    [
                        nn.Linear(in_channels, in_channels),
                        nn.LeakyReLU()
                    ]
                )
            self.feature_extractor = nn.Sequential(*feature_extractor)

        self.attention_module = (
            MILAttention(
                attention_mode,
                in_channels,
                hidden_size,
                attention_branches,
                attention_branch_reduction
            )
        )

        self.classifier = nn.Linear(in_channels, 1)

    def encode(self, x: th.Tensor) -> th.Tensor:
        """
        Safely overridden.

        Args:
            x (th.Tensor): _description_

        Returns:
            th.Tensor: _description_
        """
        x = x.squeeze(0)
        if self.use_bn:
            x = self.bn(x)
        if self.use_feature_extractor:
            return self.feature_extractor(x)
        return x

    def attention_transform(self, x: th.Tensor) -> th.Tensor:
        x = x.squeeze(0)
        if self.use_bn:
            x = self.bn(x)
        return self.attention_module(self.encode(x))

    def za_transform(self, x: th.Tensor) -> th.Tensor:
        """
        Also safely overridden?

        Args:
            x (th.Tensor): _description_

        Returns:
            th.Tensor: _description_
        """
        x = self.encode(x)
        z = self.classifier(x)
        a = self.attention_module(x, softmax = False, transpose = False)
        za = th.cat([z, a], dims = 1)
        return za

    def forward(
            self,
            x: th.Tensor,
            mode: Literal['bag', 'instance'] = 'bag'
    ) -> th.Tensor:
        """Forward pass of minimalist ABMIL.

        The forward pass uses the implementation from Cai et al. 
        in order to easily extract classification logits and attention weights
        from each instance in the bag.

        Args:
            x (th.Tensor): Input tensor of N x in_channels x x x W. If 
                instances are feature vectors, x x W should be 1 x 1.
            return_attn_vector (bool, optional): Whether to return the 
                attention vector. Defaults to False.
            mode (Literal['bag', 'instance'], optional): Whether to run 
                forward pass for instances or bag. Defaults to 'bag'.

        Returns:
            tuple[Tensor, Tensor] | Tensor: Returns either the logit of x or
                logit of x and the attention vector if `return_attn_vector`
        """
        # Default to 'bag' mode if `mode` is invalid
        mode = 'bag' if mode not in ['bag', 'instance'] else mode

        x = self.encode(x.squeeze(0))

        ### Linear Projection of Features ###
        z = self.classifier(x) # Instance logits
        if mode == 'instance':
            return z

        ### Attention Module ###
        attn_vector = self.attention_module(x, softmax = True, transpose = True)

        ### Bag Logit ###
        z_bag = th.mm(attn_vector, z)  # ATTENTION_BRANCHESxM

        return z_bag

class ABMIL(ABMILite):
    """ABMIL as per Ilse et al. 2018
    """
    def __init__(
        self,
        image_size: int = 28, # Default MNIST
        conv_layers: list[int] = (20,50),
        conv_in_channels: int = 1,
        linear_in_channels: int = 64,
        hidden_size: int = 64,
        attention_branches: int = 1,
        attention_mode: Literal['simple', 'gated'] = 'simple',
        attention_branch_reduction: Literal["mean", "max", "sum"] = "mean",
    ):
        self.conv_layers = conv_layers

        super().__init__(
            in_channels = linear_in_channels,
            hidden_size = hidden_size,
            attention_mode = attention_mode,
            attention_branches = attention_branches,
            attention_branch_reduction = attention_branch_reduction,
            use_feature_extractor = False
        )

        ### === Construct Convolution Encoder === ###
        inc = conv_in_channels
        dims = image_size
        _conv_layers = []
        for conv_channels in conv_layers:
            # Details of the original ABMIL implementation are kept here.
            _conv_layers.append(nn.Conv2d(inc, conv_channels, kernel_size = 5))
            _conv_layers.append(nn.LeakyReLU())
            _conv_layers.append(nn.MaxPool2d(2, stride=2))
            dims = int((dims - 4) / 2)
            inc = conv_channels

        self.flattened_dims = (dims ** 2) * conv_channels
        self.feature_extractor_part1 = nn.Sequential(*_conv_layers)
        self.feature_extractor_part2 = (
            nn.Sequential(
                nn.Linear(self.flattened_dims, linear_in_channels),
                nn.LeakyReLU()
            )
        )

    def encode(self, x: th.Tensor) -> th.Tensor:
        """Override of ABMILite.encode
        """
        x = x.squeeze(0)
        x = self.feature_extractor_part1(x)
        x = x.reshape(-1, self.flattened_dims)
        x = self.feature_extractor_part2(x)  # KxM
        return x

    def attention_transform(self, x: th.Tensor):
        """Override of ABMILite.attention_transform
        """
        x = x.squeeze(0)
        attn_vector = self.attention_module(self.encode(x))

        return attn_vector

    def za_transform(self, x: th.Tensor) -> th.Tensor:
        """Override of ABMILite.za_transform though I don't think it's needed.
        """
        x = x.squeeze(0)

        x = self.encode(x)

        # Attention
        a = self.attention_module(x, softmax = False, transpose = False)
        z = self.classifier(x)

        return th.cat([z, a], dim = 1)

    def forward(
        self,
        x: th.Tensor,
        mode: Literal['bag', 'instance'] = 'bag'
    ) -> th.Tensor:
        """ABMIL forward pass. Override of ABMILite.forward
        """
        # Default to 'bag' mode if `mode` is invalid
        mode = 'bag' if mode not in ['bag', 'instance'] else mode

        x = self.encode(x.squeeze(0))

        ### Linear Projection of Features ###
        z = self.classifier(x)
        if mode == 'instance':
            return z

        ### Attention Module ###
        attn_vector = self.attention_module(x, softmax = True, transpose = True)

        ### Bag Logit ###
        z_bag = th.mm(attn_vector, z)  # ATTENTION_BRANCHESxM

        return z_bag
