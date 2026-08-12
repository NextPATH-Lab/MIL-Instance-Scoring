"""Definition of Attention-based Deep Multiple Instance Learning Models

Note:
Code modified from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
commit b121d5a.


"""
from typing import Literal

import torch as th
from torch import nn

from _attention import MILAttention

class ABMILite(nn.Module):
    """
    The minimalist architecture for a MIL classifier with attention-based
    instance pooling.

    Originally pulled from 
    https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
    on 2025-07-21, with changes to control attention mechanism 
    (gated vs non-gated) and number of parameters.
    """

    def __init__(
            self,
            in_channels: int = 64,
            hidden_size: int = 64,
            attention_branches: int = 1,
            use_feature_extractor: bool = False,
            feature_extraction_layers: int = 1,
            attention_mode: Literal['simple', 'gated'] = 'simple',
    ) -> None:
        """
        """
        super().__init__()
        #########################
        ### LAYER DEFINITIONS ###
        #########################
        self.use_feature_extractor = use_feature_extractor

        if use_feature_extractor:
            # Add feature extraction layers if specified
            # Not as flexible
            feature_extractor = []
            for _ in range(feature_extraction_layers):
                feature_extractor.extend(
                    [
                        nn.Linear(in_channels, in_channels),
                        nn.LeakyReLU()
                    ]
                )

            self.feature_extractor = (
                nn.Sequential(
                    *feature_extractor
                )
            )

        # Define attention module
        self.attention_module = (
            MILAttention(
                attention_mode,
                in_channels,
                hidden_size,
                attention_branches
            )
        )
        # Classifier layer is a single linear layer by design
        self.classifier = nn.Linear(in_channels, 1)

    def attention_transform(self, x: th.Tensor) -> th.Tensor:
        """Convert input tensor into attention weights (projections).

        Args:
            x (th.Tensor): Input tensor (2D, N x H)

        Returns:
            th.Tensor: Attention projection (pre-softmax)
        """
        x = x.squeeze(0)
        if self.use_feature_extractor:
            x = self.feature_extractor(x)

        return self.attention_module(x)

    def encode(self, x: th.Tensor) -> th.Tensor:
        """Encode input features
        """
        x = x.squeeze(0) # Remove batch dimension
        if self.use_feature_extractor:
            x = self.feature_extractor(x)
        return x

    def forward(
        self,
        x: th.Tensor,
        mode: Literal['bag', 'instance'] = 'bag'
    ) -> th.Tensor:
        """Forward pass of minimalist ABMIL

        Args:
            x (th.Tensor): Input tensor of N x in_channels x x x W. If 
                instances are feature vectors, x x W should be 1 x 1.
            mode (Literal['bag', 'instance'], optional): Whether to run 
                forward pass for instances or bag. Defaults to 'bag'.

        Returns:
            Tensor: Returns either the logit of x or logit of x and the
                attention vector if `return_attn_vector`
        """
        # Default to 'bag' mode if `mode` is invalid
        mode = 'bag' if mode not in ['bag', 'instance'] else mode

        x = self.encode(x)

        ### Linear Projection of Features ###
        z = self.classifier(x) # Instance scores
        if mode == 'instance':
            # Return instance-wise classifier logit
            return z

        ### Attention Module ###
        attn_vector = self.attention_module(x, softmax = True, transpose = True)

        ### Bag Logit ###
        z_bag = th.mm(attn_vector, z)

        return z_bag

class ABMIL(nn.Module):
    """Attention-based MIL implementation by Ilse et al. 2018
    """
    def __init__(
            self,
            conv_layers: list[int] = (20,50),
            conv_in_channels: int = 1,
            linear_in_channels: int = 64,
            hidden_size: int = 64,
            attention_branches: int = 1,
            attention_mode: Literal['simple', 'gated'] = 'simple',
            image_size: int = 28 # Default MNIST
    ) -> None:
        """
        """
        super().__init__()
        self.conv_layers = conv_layers

        ### Set up Convolution encoder ###
        inc = conv_in_channels
        dims = image_size # Alias
        _conv_layers = []
        for conv_channels in conv_layers:
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

        self.attention_module = (
            MILAttention(
                attention_mode,
                linear_in_channels,
                hidden_size,
                attention_branches
            )
        )

        self.classifier = nn.Linear(linear_in_channels, 1)

    def encode(self, x: th.Tensor) -> th.Tensor:
        """Embed input tensor (N x C x H x W) into N x H feature embeddings.
        """
        x = x.squeeze(0)
        x = self.feature_extractor_part1(x)
        x = x.reshape(-1, self.flattened_dims)
        x = self.feature_extractor_part2(x)  # KxM
        return x

    def attention_transform(
            self,
            x: th.Tensor,
            softmax: bool = False,
            transpose: bool = False
    ) -> th.Tensor:
        """Compute attention projections for input tensor.
        """
        x = self.encode(x)

        attn_vec = (
            self.attention_module(x, softmax = softmax, transpose = transpose)
        )
        return attn_vec

    def za_transform(self, x: th.Tensor) -> th.Tensor:
        """Compute the 2D vectors for each instance of logit (z) x attention.
        """
        x = self.encode(x)

        # Attention
        attn_vector = (
            self.attention_module(x, softmax = False, transpose = False)
        )
        z_logit = self.classifier(x)

        return th.cat([z_logit, attn_vector], dim = 1)

    def forward(
            self,
            x: th.Tensor,
            mode: Literal['bag', 'instance'] = 'bag'
    ) -> tuple[th.Tensor, th.Tensor] | th.Tensor:
        # Default to 'bag' mode if `mode` is invalid
        mode = 'bag' if mode not in ['bag', 'instance'] else mode

        x = self.encode(x)

        ### Linear Projection of Features ###
        z = self.classifier(x)
        if mode == 'instance':
            return z

        ### Attention Module ###
        attn_vector = self.attention_module(x, softmax = True, transpose = True)

        ### Bag Logit ###
        z_bag = th.mm(attn_vector, z)  # ATTENTION_BRANCHESxM

        return z_bag

if __name__ == "__main__":
    abmil = ABMIL(
        conv_layers = [5, 20],
        linear_in_channels = 32,
        hidden_size = 16
    )
    print(f"ABMIL with {sum([x.nelement() for x in abmil.parameters()]):,} params")