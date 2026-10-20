"""RubatoCodec two-stage residual quantizer (cb1 + residual stack).

Design notes (the quantization follows DAC's conventions):
- cb1 (the semantic-distillation layer) and the residual stack are separate modules.
  Distillation must be applied to cb1's forward output (which carries the STE
  gradient); reconstructing via indices would cut the gradient to the encoder.
- Residual layers are standalone FSQ with per-layer projections, or DAC-style
  VectorQuantize (vendored in quantization/dac.py: factorized projection with
  codebook_dim=8 + L2-normalized lookup + gradient-learned codebook). The
  monolithic ResidualFSQ/ResidualVQ of the vector-quantize-pytorch library are not
  used: they share one projection across all layers, so the output is confined to a
  rank-len(levels) subspace, and ResidualFSQ's per-layer scale levels^-i vanishes
  under fp32 addition (from layer ~9 onward).
- Quantize dropout follows DAC: during training, a quantize_dropout
  fraction of the batch is assigned depth ~ U{1..Q} (the rest use full depth);
  **all layers run every step**, and the contributions and losses of layers beyond
  the depth are masked per sample. Since every layer runs each step, no DDP
  cross-rank synchronization or unused-parameter handling is needed. The default
  here is fraction 1.0 (all samples); the DAC paper recommends 0.5.
- The residual-chain subtraction is not detached, as in DAC (losses of deeper
  layers also backpropagate into the projections of shallower layers).
- Pad slots (mask=False) are packed out at the entry and excluded from the
  quantization chain (so they do not contaminate the commitment/codebook losses).
"""

import math

import torch
from torch import nn
from vector_quantize_pytorch import FSQ

from .dac import DacVectorQuantize


class RubatoResidualQuantizer(nn.Module):
    """Residual quantizer with cb1 + (num_quantizers-1) residual layers.

    Forward input is channel-last (B, N, dim). Returns:
        q_full  : (B, N, dim)  sum of the executed layers (with gradients; dropout
                               is masked per sample)
        indices : (B, N, Q)    long. All layers during training; layers beyond the
                               eval depth and pad slots are -1
        commit  : (Q,)         per-layer VQ loss (commitment*w + codebook*w,
                               averaged after the dropout mask. 0 for FSQ layers)
        q_cb1   : (B, N, dim)  quantized output of cb1 alone (with gradients, for
                               distillation)
    The per-sample depths of the most recent training forward are kept in
    self.last_depths (B,) (for monitoring; no gradients).
    """

    def __init__(
        self,
        dim: int,
        num_quantizers: int = 16,
        codebook_size: int = 4096,
        fsq_levels=None,
        cb1_type: str = "fsq",
        residual_type: str = "fsq",
        quantize_dropout=1.0,
        vq_codebook_dim: int = 8,
        vq_commitment_weight: float = 0.25,
        vq_codebook_weight: float = 1.0,
        cb1_codebook_size=None,
        cb1_fsq_levels=None,
    ):
        super().__init__()
        if fsq_levels is None:
            fsq_levels = [8, 8, 8, 8]
        fsq_levels = [int(v) for v in fsq_levels]
        # cb1 may use a different size (for semantic-vocabulary ablations, e.g.
        # 32768 = levels [8,8,8,8,8]). None shares the residual layers' size.
        cb1_codebook_size = int(cb1_codebook_size or codebook_size)
        cb1_fsq_levels = [int(v) for v in cb1_fsq_levels] if cb1_fsq_levels else fsq_levels

        def _prod(vs):
            p = 1
            for v in vs:
                p *= v
            return p

        assert _prod(fsq_levels) == codebook_size, (
            f"prod(fsq_levels)={_prod(fsq_levels)} != codebook_size={codebook_size}"
        )
        assert _prod(cb1_fsq_levels) == cb1_codebook_size, (
            f"prod(cb1_fsq_levels)={_prod(cb1_fsq_levels)} != cb1_codebook_size={cb1_codebook_size}"
        )
        assert cb1_type in ("fsq", "vq"), cb1_type
        assert residual_type in ("fsq", "vq"), residual_type
        assert num_quantizers >= 1

        self.dim = dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.cb1_codebook_size = cb1_codebook_size
        self.cb1_type = cb1_type
        self.residual_type = residual_type
        # float fraction (bool True/False are treated as 1.0/0.0 for backward compatibility)
        self.quantize_dropout = float(quantize_dropout)
        assert 0.0 <= self.quantize_dropout <= 1.0
        # Per-layer bits can differ between cb1 and the residual layers (bitrate
        # accounting goes through bits_for_depth).
        self.bits_cb1 = math.log2(cb1_codebook_size)
        self.bits_per_quantizer = math.log2(codebook_size)  # bits of a residual layer
        # DAC loss weights (the original setting is commitment 0.25 / codebook 1.0)
        self.vq_commitment_weight = vq_commitment_weight
        self.vq_codebook_weight = vq_codebook_weight
        # inference-time depth override (None = all layers); set by the caller for depth control
        self.eval_depth = None
        # per-sample depths of the most recent training forward (for monitoring)
        self.last_depths = None

        def make_layer(kind: str, size: int, levels) -> nn.Module:
            if kind == "fsq":
                # dim != len(levels), so each layer has its own project_in/out
                return FSQ(levels=list(levels), dim=dim)
            return DacVectorQuantize(
                input_dim=dim,
                codebook_size=size,
                codebook_dim=vq_codebook_dim,
            )

        self.cb1 = make_layer(cb1_type, cb1_codebook_size, cb1_fsq_levels)
        self.residual_layers = nn.ModuleList(
            [
                make_layer(residual_type, codebook_size, fsq_levels)
                for _ in range(num_quantizers - 1)
            ]
        )

    def _run_layer(self, layer: nn.Module, x: torch.Tensor):
        """Absorb the return-value differences between FSQ and DacVectorQuantize.

        Returns (quantized, indices long, per-position loss (B,N) | None)."""
        if isinstance(layer, DacVectorQuantize):
            q, idx, commit, codebook_loss = layer(x)
            loss = commit * self.vq_commitment_weight + codebook_loss * self.vq_codebook_weight
            return q, idx, loss
        q, idx = layer(x)  # FSQ: (quantized, indices int32)
        return q, idx.long(), None

    def _sample_depths(self, B: int, device) -> torch.Tensor:
        """Per-sample depths (B,): a quantize_dropout fraction of the samples gets
        U{1..Q}, the rest full depth (same convention as the forward of DAC's
        quantizer)."""
        Q = self.num_quantizers
        depths = torch.full((B,), Q, device=device, dtype=torch.long)
        n_dropout = int(B * self.quantize_dropout)
        if n_dropout > 0:
            depths[:n_dropout] = torch.randint(1, Q + 1, (n_dropout,), device=device)
        return depths

    def forward(self, x: torch.Tensor, mask=None):
        """Quantize valid event slots through the residual stack.

        Args:
            x: Event features shaped (B, N, dim).
            mask: Optional boolean tensor (B, N); True marks a valid slot.
                Invalid slots are excluded from quantization and loss computation.

        Returns:
            Tuple (quantized, indices, loss, semantic_quantized) with shapes
            (B, N, dim), (B, N, Q), (Q,), and (B, N, dim). Invalid slots have
            zero output and index -1. Quantizer dropout masks contributions
            and losses beyond each sample's selected depth during training.
        """
        assert x.ndim == 3 and x.shape[-1] == self.dim, x.shape
        B, N, _ = x.shape
        Q = self.num_quantizers

        use_dropout = self.training and self.quantize_dropout > 0.0 and Q > 1
        if self.training:
            run_layers = Q
        else:
            d = self.eval_depth if self.eval_depth is not None else Q
            run_layers = max(1, min(int(d), Q))

        # Under bf16-mixed, computing the residual subtraction (x - q) in bf16
        # accumulates cancellation error in deeper layers, so the whole chain runs
        # in fp32 with autocast disabled.
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            if mask is not None:
                assert mask.shape == (B, N) and mask.dtype == torch.bool
                xq_in = x[mask].unsqueeze(0)  # (1, M, dim) valid slots only
            else:
                xq_in = x

            # per-sample depth -> per-slot depth (matching the packed shape)
            depth_slots = None
            if use_dropout:
                depths = self._sample_depths(B, x.device)
                self.last_depths = depths.detach()
                d_bn = depths.unsqueeze(1).expand(B, N)
                if mask is not None:
                    depth_slots = d_bn[mask].unsqueeze(0)  # (1, M)
                else:
                    depth_slots = d_bn  # (B, N)

            null_idx = torch.full(xq_in.shape[:2], -1, device=x.device, dtype=torch.long)
            zero = x.new_zeros(())

            q_full = torch.zeros_like(xq_in)
            residual = xq_in
            q_cb1 = None
            idx_list = []
            loss_list = []

            layers = [self.cb1] + list(self.residual_layers)
            for i, layer in enumerate(layers):
                if not self.training and i >= run_layers:
                    idx_list.append(null_idx)
                    loss_list.append(zero)
                    continue
                qi, ii, li = self._run_layer(layer, residual)
                if i == 0:
                    q_cb1 = qi  # first-layer output returned separately (for distillation)
                if depth_slots is not None:
                    m_i = (depth_slots > i).to(qi.dtype)  # (*, M)
                    q_full = q_full + qi * m_i.unsqueeze(-1)
                    li = (li * m_i).mean() if li is not None else zero
                else:
                    q_full = q_full + qi
                    li = li.mean() if li is not None else zero
                # not detached, as in DAC (deeper-layer losses backpropagate into
                # shallower-layer projections)
                residual = residual - qi
                idx_list.append(ii)
                loss_list.append(li)

            indices = torch.stack(idx_list, dim=-1)  # (*, M, Q)
            commit = torch.stack([li.reshape(()) for li in loss_list])  # (Q,)

            if mask is not None:
                # unpack back into pad rows (output 0 / index -1)
                q_full_o = x.new_zeros(B, N, self.dim)
                q_cb1_o = x.new_zeros(B, N, self.dim)
                idx_o = torch.full((B, N, Q), -1, device=x.device, dtype=torch.long)
                q_full_o[mask] = q_full[0]
                q_cb1_o[mask] = q_cb1[0]
                idx_o[mask] = indices[0]
                q_full, q_cb1, indices = q_full_o, q_cb1_o, idx_o
        return q_full, indices, commit, q_cb1

    def _codes_from_indices(self, layer: nn.Module, kind: str, idx: torch.Tensor):
        if kind == "fsq":
            return layer.indices_to_codes(idx)
        return layer.get_output_from_indices(idx)

    @torch.no_grad()
    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct the quantized output from indices (B, N, Qd)
        (depth control via Qd <= Q).

        Layers with index=-1 contribute 0. cb1 (layer 0) must not be -1.
        """
        assert indices.ndim == 3, indices.shape
        assert indices.shape[-1] <= self.num_quantizers, indices.shape
        i1 = indices[..., 0]
        assert (i1 >= 0).all(), "cb1 (layer 0) indices contain -1"
        out = self._codes_from_indices(self.cb1, self.cb1_type, i1)
        for j in range(1, indices.shape[-1]):
            ij = indices[..., j]
            mask = ij < 0
            cj = self._codes_from_indices(
                self.residual_layers[j - 1], self.residual_type, ij.clamp_min(0)
            )
            cj = cj.masked_fill(mask.unsqueeze(-1), 0.0)
            out = out + cj
        return out

    def bits_for_depth(self, depth=None) -> float:
        d = depth if depth is not None else self.num_quantizers
        # cb1 and residual layers may have different bits (when cb1_codebook_size differs)
        return self.bits_cb1 + self.bits_per_quantizer * (d - 1)
