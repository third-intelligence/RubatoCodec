# SnakeBeta is adapted from https://github.com/NVIDIA/BigVGAN (MIT), via X-Codec-2.0.
# See incl_licenses/LICENSE_bigvgan and incl_licenses/LICENSE_xcodec2.
# Implementation adapted from https://github.com/EdwardDixon/snake under the MIT license.
#   LICENSE is in incl_licenses directory.

import torch
from torch import nn, pow, sin
from torch.nn import Parameter


class SnakeBeta(nn.Module):
    """Apply x + sin(alpha * x)^2 / beta with per-channel parameters.

    Args:
        in_features: Number of channels C in input tensors shaped (B, C, T).
        alpha: Initial parameter value (used for both alpha and beta).
        alpha_trainable: Whether alpha and beta receive gradients.
        alpha_logscale: Store parameters in log space, initialized to zero.
            Defaults to False; codec encoder and decoder call sites explicitly
            use True. The state dict contains alpha and beta, each shaped (C,).

    Example:
        >>> activation = SnakeBeta(256, alpha_logscale=True)
        >>> output = activation(torch.randn(1, 256, 10))
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform a (B, C, T) tensor, preserving its shape."""
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x
