"""Normalization without dependencies across time steps."""

import torch
import torch.nn as nn


class ChannelRMSNorm(nn.Module):
    """Per-timestep RMSNorm on channels.  No temporal dependency.

    Input/output shape: ``(B, C, T)`` – normalizes over ``C`` at each ``t``.
    Matches the RMSNorm used in the Transformer blocks (attention.py).
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> normalize over C at each t
        norm_x = torch.mean(x**2, dim=1, keepdim=True)
        return x * torch.rsqrt(norm_x + self.eps) * self.weight.unsqueeze(1)
