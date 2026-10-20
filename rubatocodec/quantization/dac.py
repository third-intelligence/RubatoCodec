"""Vendored VectorQuantize of DAC (Descript Audio Codec).

Adapted from DAC's dac/nn/quantize.py
(https://github.com/descriptinc/descript-audio-codec, MIT license — the LICENSE
is in the incl_licenses directory).

Design (as in the original):
- Per layer, weight-norm 1x1 Conv in_proj (input_dim -> codebook_dim=8) / out_proj
- Nearest-neighbor search L2-normalizes both encodings and codebook first
  (equivalent to cosine similarity; the Improved-VQGAN "factorized +
  l2-normalized codes" tricks); the quantized values themselves are the
  unnormalized codebook embeddings
- The codebook is an nn.Embedding, learned by gradients (not EMA). Two loss terms:
  commitment_loss (pulls the encoder toward the code) and codebook_loss (pulls the
  code toward the encoder output)
- Straight-through gradients

Changes from the original:
- Channel convention: the original forward takes (B, D, T); this repo's
  RubatoResidualQuantizer calls with channel-last (B, N, D), so forward transposes.
- Losses are returned per position (B, N) instead of the original's per-sample
  mean, so the caller can apply the per-sample quantizer-dropout mask.
- Return signature is (z_q, indices, commitment_loss, codebook_loss), and a
  get_output_from_indices helper is added for index-based decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class DacVectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """z: (B, N, D) channel-last. Returns (z_q (B,N,D), indices (B,N) long,
        commitment_loss (B,N), codebook_loss (B,N)).

        Losses are returned per position (only the channel dimension of the original
        implementation's mean([1,2]) is reduced). The time reduction is done by the
        caller (RubatoResidualQuantizer) so the per-sample quantizer-dropout mask can be
        applied."""
        z = z.transpose(1, 2)  # (B, D, N) — the original channel convention
        z_e = self.in_proj(z)  # (B, codebook_dim, N)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean(1)
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean(1)

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q.transpose(1, 2), indices.long(), commitment_loss, codebook_loss

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        b, d, t = latents.shape
        encodings = latents.permute(0, 2, 1).reshape(b * t, d)
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-dist).max(1)[1].reshape(b, t)
        z_q = self.decode_code(indices)
        return z_q, indices

    @torch.no_grad()
    def get_output_from_indices(self, indices):
        """indices (B, N) -> quantized output (B, N, input_dim)."""
        return self.out_proj(self.decode_code(indices)).transpose(1, 2)
