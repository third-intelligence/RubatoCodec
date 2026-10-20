"""Acoustic encoder: mono waveforms to features on the latent grid."""

import numpy as np
import torch
from torch import nn

from .encoder_blocks import EncoderBlock, _make_activation, weight_norm_conv1d


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class CodecEncoder(nn.Module):
    """Encode (B, 1, samples) audio as (B, frames, hidden_dim) features."""

    def __init__(
        self,
        ngf=48,
        up_ratios=None,
        dilations=(1, 3, 9),
        hidden_dim=1024,
        activation="snake_beta",
    ):
        super().__init__()
        if up_ratios is None:
            up_ratios = [2, 2, 4, 4, 5]
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        d_model = ngf
        self.conv_blocks = [weight_norm_conv1d(1, d_model, kernel_size=7, padding=3)]

        for _i, stride in enumerate(up_ratios):
            d_model *= 2
            self.conv_blocks += [
                EncoderBlock(d_model, stride=stride, dilations=dilations, activation=activation)
            ]

        self.conv_blocks = nn.Sequential(*self.conv_blocks)

        self.conv_final_block = nn.Sequential(
            _make_activation(d_model, activation),
            weight_norm_conv1d(d_model, hidden_dim, kernel_size=3, padding=1),
        )

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode mono audio; the sample axis is downsampled by ``hop_length``."""
        x = self.conv_blocks(x)
        x = self.conv_final_block(x)
        x = x.permute(0, 2, 1)
        return x

    def reset_parameters(self):
        self.apply(init_weights)
