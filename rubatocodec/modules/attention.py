"""Transformer attention with rotary embeddings and streaming KV caches."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchtune.modules import RotaryPositionalEmbeddings


class MLP(nn.Module):
    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim, expansion * dim, bias=False)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(expansion * dim, dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        rotary_embed: RotaryPositionalEmbeddings,
        dropout_p: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()

        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.rotary_embed = rotary_embed
        self.dropout_p = dropout_p
        # causal=False gives bidirectional attention (for non-causal decoder
        # ablations; the KV cache/streaming paths assume causal and are unsupported)
        self.causal = causal

        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, layer_past=None, use_cache=False, pos_offset=None, attn_ctx=None):
        """Apply rotary attention, optionally updating the streaming KV cache.

        Args:
            x: Input features shaped (B, T, C).
            layer_past: optional tuple of (past_key, past_value), each (B, n_heads, past_T, head_dim)
            use_cache: if True, return (output, (key, value)) for KV caching
            pos_offset: absolute position of x[:, 0] (total frames so far).
                Required for streaming with eval_attn_window set, since cache
                eviction makes cache length != absolute position. If None, the
                position is recovered from the cache length (identical to the
                behavior without eviction).
            attn_ctx: optional dict of tensors precomputed once per stack call
                (they are identical for every layer): "input_pos" =
                arange(pos_offset, pos_offset+T), "mask" = the attention mask of
                the branch taken (or None), "past_len" = KV cache length used to
                build the mask. Values must equal what this function would
                compute itself; passing them avoids rebuilding the same tensors
                in each of the stacked layers on the streaming path.

        Returns:
            Output features shaped (B, T, C), or (output, (key, value)) when
            use_cache=True. Here C = n_heads * head_dim.
        """
        B, T, C = x.size()

        q, k, v = rearrange(self.c_attn(x), "b t (r h d) -> r b h t d", r=3, h=self.n_heads)
        # q, k, v: (b, h, t, d)

        # Determine position offset from past KV cache
        past_len = 0
        if layer_past is not None:
            past_key, past_value = layer_past
            past_len = past_key.shape[2]

        # Inference-time sliding window. On the KV cache path, "keep only the most
        # recent Wa entries (eviction)" + "absolute-position mask" yields exactly the
        # same attended set as the batch windowed attention (0 <= i-j <= Wa).
        Wa = getattr(self, "eval_attn_window", None)
        if self.training:
            Wa = None
        if Wa is not None and layer_past is not None:
            assert pos_offset is not None, (
                "eval_attn_window + KV cache requires pos_offset "
                "(use via VocosBackbone.forward_streaming)"
            )
            assert past_len <= Wa, (past_len, Wa)
        if pos_offset is None:
            pos_offset = past_len

        # Transpose for RoPE: (b, h, t, d) -> (b, t, h, d)
        # torchtune expects [b, s, n_h, h_d] where s is the sequence dimension
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Apply RoPE with absolute positions
        if attn_ctx is not None and attn_ctx.get("input_pos") is not None:
            input_pos = attn_ctx["input_pos"]
        else:
            input_pos = torch.arange(pos_offset, pos_offset + T, device=x.device)
        q = self.rotary_embed(q, input_pos=input_pos)
        k = self.rotary_embed(k, input_pos=input_pos)

        # Transpose back: (b, t, h, d) -> (b, h, t, d)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Concatenate with past KV (post-RoPE, so positions are already correct)
        if layer_past is not None:
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)

        S = k.shape[2]  # total KV length
        present = None
        if use_cache:
            if Wa is not None and S > Wa:
                # The next step's queries (absolute positions pos_offset+T onward)
                # can only attend to the most recent Wa entries, so older KV is dropped.
                present = (k[:, :, S - Wa :], v[:, :, S - Wa :])
            else:
                present = (k, v)

        # Causal attention (bidirectional if causal=False)
        dp = self.dropout_p if self.training else 0.0
        causal = getattr(self, "causal", True)
        if T == 1:
            # Single-token incremental decode: no mask needed.
            # With a window, the cache is already evicted down to <= Wa entries, so
            # attending to "all past + self" is exactly the in-window attended set.
            assert causal, "the KV cache path assumes causal (non-causal unsupported)"
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dp, is_causal=False
            )
        elif T == S:
            # Full-sequence (no KV cache).
            # If eval_attn_window (in frames, eval only) is set, apply sliding-window
            # attention (causal: 0 <= i-j <= W / non-causal: |i-j| <= W).
            if Wa is not None and T > Wa:
                if attn_ctx is not None and attn_ctx.get("mask") is not None:
                    attn_mask = attn_ctx["mask"]
                else:
                    idx = torch.arange(T, device=x.device)
                    rel = idx.view(-1, 1) - idx.view(1, -1)  # i - j
                    if causal:
                        attn_mask = (rel >= 0) & (rel <= Wa)
                    else:
                        attn_mask = rel.abs() <= Wa
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dp)
            else:
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=dp, is_causal=causal
                )
        else:
            # Multi-token chunk with KV cache: absolute-position-based causal
            # (+ window) mask. Query absolute position = pos_offset + i,
            # KV absolute position = pos_offset - past_len + j
            assert causal, "the KV cache path assumes causal (non-causal unsupported)"
            if attn_ctx is not None and attn_ctx.get("mask") is not None:
                # the mask depends only on (pos_offset, T, past_len, Wa), which are
                # identical for every layer of the stack
                assert attn_ctx.get("past_len") == past_len, (attn_ctx.get("past_len"), past_len)
                attn_mask = attn_ctx["mask"]
            else:
                q_pos = torch.arange(pos_offset, pos_offset + T, device=x.device)
                k_pos = torch.arange(pos_offset - past_len, pos_offset + T, device=x.device)
                rel = q_pos.view(-1, 1) - k_pos.view(1, -1)  # i - j
                if Wa is not None:
                    attn_mask = (rel >= 0) & (rel <= Wa)
                else:
                    attn_mask = rel >= 0
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dp)

        y = rearrange(y, "b h t d -> b t (h d)")
        y = self.c_proj(y)

        if use_cache:
            return y, present
        return y


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        rotary_embed: RotaryPositionalEmbeddings,
        mlp_expansion: int = 4,
        attn_dropout: float = 0.0,
        attn_causal: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        self.att_norm = nn.RMSNorm(dim, eps=1e-6)
        self.ffn_norm = nn.RMSNorm(dim, eps=1e-6)
        self.att = Attention(
            dim=dim,
            n_heads=n_heads,
            rotary_embed=rotary_embed,
            dropout_p=attn_dropout,
            causal=attn_causal,
        )
        self.mlp = MLP(dim=dim, expansion=mlp_expansion)

    def forward(self, x, layer_past=None, use_cache=False, pos_offset=None, attn_ctx=None):
        if use_cache:
            attn_out, present = self.att(
                self.att_norm(x),
                layer_past,
                use_cache=True,
                pos_offset=pos_offset,
                attn_ctx=attn_ctx,
            )
            x = x + attn_out
        else:
            x = x + self.att(self.att_norm(x))
            present = None

        x = x + self.mlp(self.ffn_norm(x))

        if use_cache:
            return x, present
        return x
