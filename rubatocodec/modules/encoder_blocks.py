"""Convolutional building blocks for the acoustic encoder."""

import torch.nn as nn
from torch.nn.utils import weight_norm

from . import activations
from .alias_free_torch.act import Activation1d


def weight_norm_conv1d(*args, **kwargs) -> nn.Conv1d:
    """Construct a Conv1d with checkpoint-compatible weight normalization."""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def _make_activation(dim: int, activation: str = "snake_beta") -> nn.Module:
    """Create an activation module based on the activation type string.

    Args:
        dim: Number of channels (used for SnakeBeta's per-channel parameters).
        activation: ``"snake_beta"`` for alias-free SnakeBeta, ``"silu"`` for SiLU/Swish.
    """
    if activation == "silu":
        return nn.SiLU()
    return Activation1d(activation=activations.SnakeBeta(dim, alpha_logscale=True))


class ResidualUnit(nn.Module):
    """Dilated residual convolution preserving the (B, C, T) layout."""

    def __init__(self, dim: int = 16, dilation: int = 1, activation: str = "snake_beta"):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            _make_activation(dim, activation),
            weight_norm_conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            _make_activation(dim, activation),
            weight_norm_conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    """Residual units followed by strided downsampling and channel expansion."""

    def __init__(
        self, dim: int = 16, stride: int = 1, dilations=(1, 3, 9), activation: str = "snake_beta"
    ):
        super().__init__()
        runits = [ResidualUnit(dim // 2, dilation=d, activation=activation) for d in dilations]
        self.block = nn.Sequential(
            *runits,
            _make_activation(dim // 2, activation),
            weight_norm_conv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
            ),
        )

    def forward(self, x):
        return self.block(x)
