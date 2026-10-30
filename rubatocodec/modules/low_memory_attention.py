"""Bounded-memory full-context relative-key attention for long inference inputs.

Only the query axis is tiled. Every query still attends to every key, with the
original clipped relative positions and full-key softmax. No waveform/window
splitting, reduced precision, or attention approximation is used.
"""

import math
from types import MethodType

import torch


def _tiled_forward(
    self,
    hidden_states,
    attention_mask=None,
    relative_position_embeddings=None,
    output_attentions=False,
):
    if hidden_states.shape[1] <= self._low_memory_threshold:
        return self._original_attention_forward(
            hidden_states, attention_mask, relative_position_embeddings, output_attentions
        )
    if self.training or output_attentions:
        raise ValueError("Tiled attention requires eval mode and output_attentions=False")
    if self.position_embeddings_type != "relative_key":
        raise ValueError("Only relative_key attention is supported")
    b, length, _ = hidden_states.shape
    h, d = self.num_heads, self.head_size
    q = self.linear_q(hidden_states).view(b, length, h, d).transpose(1, 2)
    k = self.linear_k(hidden_states).view(b, length, h, d).transpose(1, 2)
    v = self.linear_v(hidden_states).view(b, length, h, d).transpose(1, 2)
    result = torch.empty_like(q)
    keys = torch.arange(length, device=q.device)
    scale = math.sqrt(d)
    # There are only left+right+1 distinct relative-position vectors. Project
    # those once per query, then gather; never materialize [T,T,head_size].
    positions = self.distance_embedding.weight.to(q.dtype)
    for start in range(0, length, self._low_memory_chunk):
        end = min(start + self._low_memory_chunk, length)
        query = q[:, :, start:end]
        scores = torch.matmul(query, k.transpose(-2, -1)) / scale
        distances = keys[None, :] - torch.arange(start, end, device=q.device)[:, None]
        distances.clamp_(-self.left_max_position_embeddings, self.right_max_position_embeddings)
        distances += self.left_max_position_embeddings
        relative = torch.einsum("bhld,rd->bhlr", query, positions)
        relative = relative.gather(-1, distances[None, None].expand(b, h, -1, -1))
        scores = scores + relative / scale
        del relative, distances
        if attention_mask is not None:
            mask = (
                attention_mask
                if attention_mask.shape[-2] == 1
                else attention_mask[..., start:end, :]
            )
            scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        result[:, :, start:end] = torch.matmul(probs, v)
        del scores, probs
    result = result.transpose(1, 2).reshape(b, length, h * d)
    return self.linear_out(result), None


def enable_low_memory_attention(model):
    """Install inference-only full-context attention for inputs over 4,096 frames."""
    from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (
        Wav2Vec2BertSelfAttention,
    )

    threshold, chunk = 4096, 128
    modules = [m for m in model.modules() if isinstance(m, Wav2Vec2BertSelfAttention)]
    if not modules:
        raise ValueError("low_memory_encode requires Wav2Vec2-BERT self-attention")
    # Validate every layer before changing any of them.
    for module in modules:
        if module.position_embeddings_type != "relative_key":
            raise ValueError("low_memory_encode requires relative_key position embeddings")
        expected = module.left_max_position_embeddings + module.right_max_position_embeddings + 1
        if tuple(module.distance_embedding.weight.shape) != (expected, module.head_size):
            raise ValueError("Unsupported Wav2Vec2-BERT relative-position embedding shape")
    count = 0
    for module in modules:
        if not hasattr(module, "_original_attention_forward"):
            module._original_attention_forward = module.forward
            module.forward = MethodType(_tiled_forward, module)
        module._low_memory_threshold = threshold
        module._low_memory_chunk = chunk
        count += 1
    if count == 0:
        raise ValueError("No Wav2Vec2Bert attention layers found")
    return count
