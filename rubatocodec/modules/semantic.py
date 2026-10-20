"""Semantic feature pooling and convolutional projections."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_semantic_features(features: torch.Tensor, codec_hop_length: int) -> torch.Tensor:
    """Average 50-Hz w2v-BERT features onto the codec grid at 16 kHz.

    Windows start at frame zero and do not overlap. A partial final window
    averages only its available frames; no padding values enter the mean.

    Args:
        features: Semantic features shaped (B, C, T).
        codec_hop_length: Codec frame spacing in 16-kHz samples.

    Returns:
        Features shaped (B, C, ceil(T / stride)), or the input when the codec
        hop is at most 320 samples.

    Raises:
        ValueError: A hop greater than 320 is not an integer multiple of 320.
    """
    if codec_hop_length <= 320:
        return features
    if codec_hop_length % 320:
        raise ValueError("Codec hop must be an integer multiple of the 320-sample semantic hop")
    stride = int(codec_hop_length // 320)
    return F.avg_pool1d(features, stride, stride=stride, ceil_mode=True, count_include_pad=False)


class SemanticEncoder(nn.Module):
    """Project semantic teacher features while preserving the temporal grid."""

    def __init__(
        self,
        input_channels: int,
        code_dim: int,
        encode_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()

        self.initial_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=encode_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

        self.residual_blocks = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(
                encode_channels,
                encode_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                encode_channels,
                encode_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias,
            ),
        )

        self.final_conv = nn.Conv1d(
            in_channels=encode_channels,
            out_channels=code_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features without changing their temporal length.

        Args:
            x: Input features shaped (B, input_channels, T).

        Returns:
            Encoded features shaped (B, code_dim, T).
        """
        x = self.initial_conv(x)  # (Batch, Encode_channels, Length)
        x = self.residual_blocks(x) + x  # Residual connection
        x = self.final_conv(x)  # (Batch, Code_dim, Length)
        return x


class SemanticDecoder(nn.Module):
    """Reconstruct teacher features from quantized semantic latents during training."""

    def __init__(
        self,
        code_dim: int,
        output_channels: int,
        decode_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()

        # Initial convolution to map code_dim to decode_channels
        self.initial_conv = nn.Conv1d(
            in_channels=code_dim,
            out_channels=decode_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

        # Residual Blocks
        self.residual_blocks = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(
                decode_channels,
                decode_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                decode_channels,
                decode_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias,
            ),
        )

        # Final convolution to map decode_channels to output_channels
        self.final_conv = nn.Conv1d(
            in_channels=decode_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map (B, code_dim, T) latents to (B, output_channels, T) features."""
        # z: (Batch, Code_dim, Length)
        x = self.initial_conv(z)  # (Batch, Decode_channels, Length)
        x = self.residual_blocks(x) + x  # Residual connection
        x = self.final_conv(x)  # (Batch, Output_channels, Length)
        return x
