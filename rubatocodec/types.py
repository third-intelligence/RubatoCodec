"""Tensor layouts returned by the public codec API."""

from typing import TypedDict

import torch


class EncodedAudio(TypedDict):
    """Adaptive event codes on the codec device.

    Attributes:
        codes: Int64 tensor (Q, N): quantizer layers by events.
        durs: Positive int64 tensor (N,): duration of each event in grid frames.
            The sum gives the padded utterance length on the latent grid.
    """

    codes: torch.Tensor
    durs: torch.Tensor
