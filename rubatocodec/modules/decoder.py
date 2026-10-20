"""Transformer synthesis, convolutional upsampling, and streaming ISTFT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from vector_quantize_pytorch import ResidualFSQ

from ..quantization.residual import RubatoResidualQuantizer
from .activations import SnakeBeta
from .attention import TransformerBlock
from .normalization import ChannelRMSNorm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_pad(total_pad: int, mode: str = "symmetric") -> tuple[int, int]:
    """Compute (left_pad, right_pad) for a given padding mode.

    Args:
        total_pad: Total padding to distribute (= ``(kernel_size - 1) * dilation``).
        mode: ``"symmetric"`` (default, non-causal), ``"causal"`` / ``"causal_full"``
              (all left), or ``"asymmetric"`` (right_pad fixed to 1).

    Returns:
        ``(left_pad, right_pad)`` tuple.
    """
    if mode in ("causal", "causal_full"):
        return total_pad, 0
    elif mode == "asymmetric":
        if total_pad <= 1:
            return total_pad, 0
        return total_pad - 1, 1
    else:  # symmetric
        left = total_pad // 2
        return left, total_pad - left


def _sym_conv_with_buffer(
    conv: nn.Conv1d,
    x: torch.Tensor,
    buf: torch.Tensor | None,
    flush: bool = False,
    padding_mode: str = "symmetric",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run a Conv1d (padding=0) with configurable padding-equivalent buffering.

    Matches the batch-mode ``F.pad(x, (left_pad, right_pad)) → conv(...)`` by
    buffering input across chunk boundaries.

    The conv is run on all available input (buffer + new input).  The last
    ``right_pad`` output frames are **discarded** because they lack right
    context — they will be recomputed on the next call when more input arrives
    and the buffer provides the necessary overlap.

    When ``flush=True`` (end of stream), the right side is zero-padded just
    like batch mode, and all remaining output frames are produced.

    Buffer: stores the last ``total_pad + right_pad`` input samples (= left context
    for recomputing the first discarded frame plus the inputs whose outputs were
    discarded).

    On the first call the left side is zero-padded.

    Args:
        padding_mode: ``"symmetric"`` (default), ``"causal"`` / ``"causal_full"``,
            or ``"asymmetric"``.  Controls how total_pad is split into left/right.

    Returns ``(output, new_buffer)``.  Output length may be 0 if the chunk is
    too short to produce any ready frames.
    """
    k = conv.kernel_size[0]
    d = conv.dilation[0]
    total_pad = (k - 1) * d
    if total_pad == 0:
        if x.shape[2] == 0:
            return torch.zeros(x.shape[0], conv.out_channels, 0, device=x.device), None
        return conv(x), None

    left_pad, right_pad = _compute_pad(total_pad, padding_mode)

    if buf is None:
        # First call: zero-pad on the left
        x_all = F.pad(x, (left_pad, 0))
    else:
        x_all = torch.cat([buf, x], dim=2)

    if flush:
        # End of stream: zero-pad right side (matches batch F.pad(x, (lp, rp)))
        x_all = F.pad(x_all, (0, right_pad))

    # Run conv on everything available
    total_out_len = x_all.shape[2] - total_pad
    if total_out_len <= 0:
        if flush:
            return torch.zeros(x.shape[0], conv.out_channels, 0, device=x.device), None
        # Not enough even for 1 output frame — buffer everything
        new_buf = x_all.detach()
        return torch.zeros(x.shape[0], conv.out_channels, 0, device=x.device), new_buf

    all_out = conv(x_all)  # length = total_out_len

    if flush:
        # Return all output frames, no buffering needed
        return all_out, None

    # The last right_pad output frames don't have right context yet → discard.
    # They will be recomputed next call via the overlap in the buffer.
    ready_len = max(total_out_len - right_pad, 0)

    if ready_len == 0:
        new_buf = x_all.detach()
        return torch.zeros(x.shape[0], conv.out_channels, 0, device=x.device), new_buf

    ready_out = all_out[:, :, :ready_len]

    # Buffer from the input position of the first discarded output onward.
    # This includes: left context (left_pad) for re-computing the discarded
    # frames, plus the right_pad input samples whose outputs were discarded.
    # Total buffer size = total_pad + right_pad (constant).
    new_buf = x_all[:, :, ready_len:].detach()

    return ready_out, new_buf


def _conv_transpose_with_buffer(
    conv_t: nn.Module,
    x: torch.Tensor,
    buf: torch.Tensor | None,
    flush: bool = False,
    padding_mode: str = "symmetric",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run a padded ConvTranspose1d with an input buffer.

    The ConvTranspose1d uses ``padding = sym_pad`` so that
    ``output_length = input_length * stride``.

    The last ``sym_pad`` output frames depend on input beyond the current
    chunk.  These frames are discarded and recomputed on the next call
    (analogous to ``_sym_conv_with_buffer``).

    When ``flush=True``, all remaining output frames are produced.

    Buffer stores input frames for overlap recomputation.

    Note: ``padding_mode`` is accepted for API consistency but has no effect.
    ConvTranspose1d always uses symmetric padding internally.

    Returns ``(output, new_buffer)``.
    """
    # Unwrap weight_norm / parametrizations wrappers to get raw conv
    raw = conv_t
    while hasattr(raw, "module"):
        raw = raw.module
    if not isinstance(raw, nn.ConvTranspose1d):
        raw = conv_t

    k = raw.kernel_size[0]
    u = raw.stride[0]
    sym_pad = raw.padding[0]

    if sym_pad == 0:
        return conv_t(x), None

    # Number of unreliable output frames at the tail.
    # ConvTranspose1d always uses symmetric padding internally, so the
    # discard strategy is always symmetric regardless of padding_mode.
    # (padding_mode only affects Conv1d layers, not ConvTranspose1d.)
    discard = sym_pad

    # Buffer ceil((k-1)/u) input frames for overlap recomputation.
    # The recomputed region starts from the first discarded output of the
    # previous call, which needs input from L - floor((k-1)/u);
    # ceil((k-1)/u) frames are buffered as a safe upper bound.
    ctx = (k - 1 + u - 1) // u  # ceil((k-1)/u)

    if buf is not None:
        x_with_ctx = torch.cat([buf, x], dim=2)
    else:
        x_with_ctx = x

    if x_with_ctx.shape[2] == 0:
        if flush:
            return torch.zeros(x.shape[0], raw.out_channels, 0, device=x.device), None
        return torch.zeros(x.shape[0], raw.out_channels, 0, device=x.device), buf

    out = conv_t(x_with_ctx)

    if buf is not None:
        # Skip the overlap region: the buffered frames produce buf_len*u
        # output frames, but only the last `discard` of those are new
        # (they were discarded by the previous call).
        buf_len = buf.shape[2]
        skip = buf_len * u - discard
        if skip > 0:
            out = out[:, :, skip:]

    if flush:
        return out, None

    # Discard last `discard` output frames (they lack right context)
    if out.shape[2] > discard:
        ready_out = out[:, :, :-discard]
    else:
        # All output unreliable — buffer input and wait
        new_buf = x_with_ctx[:, :, -ctx:].detach()
        return torch.zeros(x.shape[0], raw.out_channels, 0, device=x.device), new_buf

    new_buf = x_with_ctx[:, :, -ctx:].detach()
    return ready_out, new_buf


def _conv_transpose_causal_with_buffer(
    conv_t: nn.Module,
    x: torch.Tensor,
    carry: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run a padding=0 ConvTranspose1d left-aligned (causal_full) in streaming mode.

    The batch version computes ``conv_t(x_full)[:, :, :L*u]``. The chunked version
    emits the first ``L*u`` samples of the raw output of length ``(L-1)*u + k`` and
    keeps the trailing ``k-u`` samples as a carry, added onto the head of the next
    chunk's raw output (global positions line up). The carry already contains the
    bias once, so the bias is subtracted before storing to avoid double counting.
    No flush is needed (the batch version truncates at L_total*u, so the carry is
    discarded at the end).
    """
    raw = conv_t
    while hasattr(raw, "module"):
        raw = raw.module
    if not isinstance(raw, nn.ConvTranspose1d):
        raw = conv_t
    u = raw.stride[0]
    assert raw.padding[0] == 0, "causal_full assumes a ConvTranspose with padding=0"

    L = x.shape[2]
    if L == 0:
        return torch.zeros(x.shape[0], raw.out_channels, 0, device=x.device), carry

    out = conv_t(x)  # (B, C, L*u + (k−u))
    if carry is not None:
        c = carry.shape[2]
        out = torch.cat([out[:, :, :c] + carry, out[:, :, c:]], dim=2)
    emit = out[:, :, : L * u]
    tail = out[:, :, L * u :]
    if raw.bias is not None:
        tail = tail - raw.bias.view(1, -1, 1)
    return emit, tail.detach()


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


# ---------------------------------------------------------------------------
# Upsampler blocks
# ---------------------------------------------------------------------------


class UpsamplerResNetBlock(nn.Module):
    """ResNet block with ChannelRMSNorm for the upsampler.

    Padding mode is configurable: ``"symmetric"`` (default, non-causal),
    ``"causal"`` (all-left padding, zero streaming delay), or
    ``"asymmetric"`` (right_pad=1, minimal delay).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
        activation: str = "silu",
        padding_mode: str = "symmetric",
    ):
        super().__init__()
        self._padding_mode = padding_mode
        total_pad = (kernel_size - 1) * dilation
        self._total_pad = total_pad
        self.norm1 = ChannelRMSNorm(channels)
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=0, dilation=dilation
        )
        self.norm2 = ChannelRMSNorm(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=0, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)
        if activation == "snake_beta":
            self.act1 = SnakeBeta(channels, alpha_logscale=True)
            self.act2 = SnakeBeta(channels, alpha_logscale=True)
        else:
            self.act1 = nn.SiLU()
            self.act2 = nn.SiLU()

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if self._total_pad == 0:
            return x
        lp, rp = _compute_pad(self._total_pad, self._padding_mode)
        return F.pad(x, (lp, rp))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.act1(x)
        x = self._pad(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self._pad(x)
        x = self.conv2(x)
        return x + residual

    def forward_streaming(
        self,
        x: torch.Tensor,
        state: dict | None = None,
        flush: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Streaming forward with conv buffers.

        The conv path may produce fewer frames than the input due to
        buffering.  The residual is buffered to maintain alignment.

        When ``flush=True``, zero-pads the conv buffers to produce all
        remaining output (matching batch-mode padding at sequence end).
        """
        if state is None:
            state = {"conv1_buf": None, "conv2_buf": None, "res_buf": None}
        residual = x
        x = self.norm1(x)
        x = self.act1(x)
        x, state["conv1_buf"] = _sym_conv_with_buffer(
            self.conv1,
            x,
            state["conv1_buf"],
            flush=flush,
            padding_mode=self._padding_mode,
        )
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x, state["conv2_buf"] = _sym_conv_with_buffer(
            self.conv2,
            x,
            state["conv2_buf"],
            flush=flush,
            padding_mode=self._padding_mode,
        )
        out_len = x.shape[2]
        # Buffer residual and extract aligned portion
        if state["res_buf"] is not None:
            residual = torch.cat([state["res_buf"], residual], dim=2)
        if out_len == 0:
            state["res_buf"] = residual.detach()
            return torch.zeros(x.shape[0], x.shape[1], 0, device=x.device), state
        res = residual[:, :, :out_len]
        state["res_buf"] = residual[:, :, out_len:].detach()
        return res + x, state


class UpSamplerBlock(nn.Module):
    """Upsampler using transposed convolutions with SnakeBeta activations and ResNet blocks.

    Adapted from tts/core/codec/upsampler.py of Inworld TTS
    (https://github.com/inworld-ai/tts, MIT license — the LICENSE is in the
    incl_licenses directory): the per-stage weight-norm ConvTranspose1d with channel
    halving, the trailing ResNet blocks, and the final Linear projection back to
    in_channels follow that implementation. Changes here: configurable activations
    (SnakeBeta / silu / silu_no_inter), causal / causal_full padding modes,
    streaming forward, and ChannelRMSNorm in the ResNet blocks
    (UpsamplerResNetBlock above).

    Increases frame rate before ISTFT for higher sample rate output.
    E.g., a [3] factor gives 3x upsampling: 50 Hz -> 150 Hz, matched with an
    ISTFT hop of 294 for 44.1 kHz output (150 Hz x 294 = 44100).

    Padding mode is configurable: ``"symmetric"`` (default), ``"causal"``,
    ``"causal_full"``, or ``"asymmetric"``.  Only ``"causal_full"`` makes the whole
    block causal (zero streaming delay); ``"causal"`` makes only the Conv1d layers
    causal — the ConvTranspose1d keeps symmetric padding, so a lookahead remains.
    """

    def __init__(
        self,
        in_channels: int,
        upsample_factors: list[int],
        kernel_sizes: list[int] = None,
        activation: str = "snake_beta",
        padding_mode: str = "symmetric",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.upsample_factors = list(upsample_factors)
        self.kernel_sizes = list(kernel_sizes or [f * 2 for f in self.upsample_factors])
        self._padding_mode = padding_mode

        if len(self.kernel_sizes) != len(self.upsample_factors):
            raise ValueError("kernel_sizes and upsample_factors must have the same length")

        # "silu_no_inter": SiLU activations with no inter-stage activation
        # (matches inworld-tts upsampler design)
        use_inter = activation != "silu_no_inter"
        resnet_act = "silu" if activation == "silu_no_inter" else activation

        self.upsample_layers = nn.ModuleList()
        self.inter_activations = nn.ModuleList()
        self.resnet_blocks = nn.ModuleList()

        final_channels = self.in_channels // (2 ** len(self.upsample_factors))
        self.out_proj = nn.Linear(final_channels, self.in_channels, bias=True)
        if resnet_act == "silu":
            self.out_act = nn.SiLU()
        else:
            self.out_act = SnakeBeta(self.in_channels, alpha_logscale=True)

        # causal_full: build the ConvTranspose with padding=0 and keep only the first
        # L*stride samples in forward (output p depends only on inputs i <= p//stride,
        # i.e. zero backward leakage). The other modes (symmetric/causal/asymmetric)
        # use a symmetrically padded ConvTranspose.
        self._causal_full = padding_mode == "causal_full"

        for i, (k, u) in enumerate(zip(self.kernel_sizes, self.upsample_factors, strict=False)):
            c_in = self.in_channels // (2**i)
            c_out = self.in_channels // (2 ** (i + 1))
            if self._causal_full:
                # With k < u the raw length (L-1)*u+k < L*u, so the head trim would
                # silently return a shorter output and the mismatch with streaming
                # would go unnoticed.
                assert k >= u, f"causal_full requires kernel >= stride (k={k}, u={u})"
                sym_pad, out_pad = 0, 0
            else:
                # Symmetric padding: output_length = input_length * stride
                # From 2*p = k - u + op, with op = (k-u) % 2:
                out_pad = (k - u) % 2
                sym_pad = (k - u + out_pad) // 2
            self.upsample_layers.append(
                nn.utils.parametrizations.weight_norm(
                    nn.ConvTranspose1d(
                        c_in,
                        c_out,
                        kernel_size=k,
                        stride=u,
                        padding=sym_pad,
                        output_padding=out_pad,
                    )
                )
            )
            if not use_inter:
                self.inter_activations.append(nn.Identity())
            elif resnet_act == "silu":
                self.inter_activations.append(nn.SiLU())
            else:
                self.inter_activations.append(SnakeBeta(c_out, alpha_logscale=True))
            self.resnet_blocks.append(
                UpsamplerResNetBlock(
                    channels=c_out,
                    dropout=0.0,
                    activation=resnet_act,
                    # for causal_full, the Conv1d side reuses the existing causal left padding
                    padding_mode="causal" if self._causal_full else padding_mode,
                )
            )

    @property
    def total_upsample_factor(self) -> int:
        result = 1
        for f in self.upsample_factors:
            result *= f
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample channel-first features and return channel-last features.

        Args:
            x: (B, C, L) where C is in_channels

        Returns:
            (B, L * total_upsample_factor, C) transposed for ISTFTHead compatibility
        """
        for up, act, resblk, u in zip(
            self.upsample_layers,
            self.inter_activations,
            self.resnet_blocks,
            self.upsample_factors,
            strict=False,
        ):
            if self._causal_full:
                # keep the first L*u samples of the padding=0 raw output (length
                # (L-1)*u + k): output p depends only on inputs i <= p//u
                # (zero backward leakage)
                L = x.shape[2]
                x = up(x)[:, :, : L * u]
            else:
                x = up(x)
            x = act(x)
            x = resblk(x)
        x = self.out_proj(x.transpose(1, 2))
        x = self.out_act(x.transpose(1, 2))
        return x.transpose(1, 2)

    def forward_streaming(
        self,
        x: torch.Tensor,
        state: dict | None = None,
        flush: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Streaming forward with ConvTranspose1d and ResNet conv input buffers.

        When ``flush=True``, each stage's ResNetBlock is flushed after processing
        the current input.  The flushed frames are then fed into the next stage's
        ConvTranspose1d so the cascade completes correctly.
        """
        if state is None:
            state = {
                "conv_t_bufs": [None] * len(self.upsample_layers),
                "resnet_states": [None] * len(self.resnet_blocks),
            }

        if self._causal_full:
            # causal_full: the ConvTranspose uses the carry-add scheme (no right
            # trim, zero delay); the ResNet uses buffered convs with causal left padding
            for i, (up, act, resblk) in enumerate(
                zip(self.upsample_layers, self.inter_activations, self.resnet_blocks, strict=False)
            ):
                x, state["conv_t_bufs"][i] = _conv_transpose_causal_with_buffer(
                    up, x, state["conv_t_bufs"][i]
                )
                x = act(x)
                x, state["resnet_states"][i] = resblk.forward_streaming(
                    x, state["resnet_states"][i], flush=flush
                )
            x = self.out_proj(x.transpose(1, 2))
            x = self.out_act(x.transpose(1, 2))
            return x.transpose(1, 2), state

        # Handle empty input (conv buffers consumed all frames)
        if x.shape[2] == 0 and not flush:
            return x.transpose(1, 2), state

        for i, (up, act, resblk) in enumerate(
            zip(
                self.upsample_layers,
                self.inter_activations,
                self.resnet_blocks,
                strict=False,
            )
        ):
            # ConvTranspose: process new input and/or flush remaining buffer
            x, state["conv_t_bufs"][i] = _conv_transpose_with_buffer(
                up,
                x,
                state["conv_t_bufs"][i],
                flush=flush,
                padding_mode=self._padding_mode,
            )
            if x.shape[2] > 0:
                x = act(x)
            elif not flush:
                return x.transpose(1, 2), state

            # ResNetBlock
            x, state["resnet_states"][i] = resblk.forward_streaming(
                x, state["resnet_states"][i], flush=flush
            )

            # If ResNet consumed all frames into buffers, stop early
            if x.shape[2] == 0 and not flush:
                return x.transpose(1, 2), state

        if x.shape[2] == 0:
            return x.transpose(1, 2), state

        x = self.out_proj(x.transpose(1, 2))
        x = self.out_act(x.transpose(1, 2))
        return x.transpose(1, 2), state


# ---------------------------------------------------------------------------
# ISTFT
# ---------------------------------------------------------------------------


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center", "same", or
            "causal" (forward-placed windows, fully causal). Defaults to "same".
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same", "causal"]:
            raise ValueError("Padding must be 'center', 'same', or 'causal'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)
        if padding == "causal":
            # In the degenerate win == hop configuration the correction is empty and
            # the env[0]=0 amplification remains.
            assert win_length > hop_length, (win_length, hop_length)
            # Complete the overlap-add envelope at the waveform start using
            # the squared windows that would precede it. This avoids division
            # by a near-zero envelope and gives the initial samples a fade-in.
            # The correction is independent of the input.
            wsq = window.square()
            corr = torch.zeros(max(win_length - hop_length, 1))
            m = 1
            while m * hop_length < win_length:
                seg = wsq[m * hop_length :]
                n = min(seg.numel(), corr.numel())
                corr[:n] += seg[:n]
                m += 1
            self.register_buffer("causal_env_corr", corr, persistent=False)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            return torch.istft(
                spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        elif self.padding == "causal":
            # Place the window of frame f forward onto output samples
            # [f*hop, f*hop+win): sample s depends only on frames <= floor(s/hop),
            # i.e. zero backward leakage. Output length is T*hop, same as "same"
            # (the trailing window overhang is cut). The first win-hop samples get
            # the leading-window envelope correction (see __init__), turning
            # the partial coverage into the full-coverage constant; the result is a
            # deterministic hann^2 fade-in.
            pad = 0
        else:
            raise ValueError("Padding must be 'center', 'same', or 'causal'.")

        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        end = pad + T * self.hop_length if self.padding == "causal" else -pad
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:end]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:end]

        if self.padding == "causal":
            # leading-window correction (see __init__): turn the partial
            # coverage at the head into full coverage
            n = min(self.causal_env_corr.numel(), window_envelope.numel())
            window_envelope[:n] = window_envelope[:n] + self.causal_env_corr[:n]

        # Normalize
        y = y / window_envelope.clamp(min=1e-11)

        return y


class FourierHead(nn.Module):
    """Base class for inverse fourier modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward method.")


class ISTFTHead(FourierHead):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center", "same", or
            "causal" (forward-placed windows, fully causal). Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H).

        Returns:
            Tuple of (audio (B, 1, T), x_pred (B, n_fft+2, L))
        """
        x_pred = self.out(x)
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        x = torch.cos(p)
        y = torch.sin(p)
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio.unsqueeze(1), x_pred

    def _x_to_spec(self, x: torch.Tensor) -> torch.Tensor:
        """Convert backbone output to complex spectrogram."""
        x_pred = self.out(x)
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        S = mag * (torch.cos(p) + 1j * torch.sin(p))
        return S

    def streaming_forward(
        self, x: torch.Tensor, state: dict[str, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Exact streaming ISTFT using sample-level overlap-add buffer.

        Performs manual overlap-add at the sample level and maintains a buffer
        of partial (un-normalized) audio samples and the corresponding window
        envelope.  This produces output that is bit-exact (up to float
        precision) with the batch ``forward`` regardless of chunk size.

        Args:
            x: (B, L, H) new frames for this chunk
            state: dict with 'audio_buf', 'win_env_buf', 'output_samples', or None

        Returns:
            audio: (B, 1, T_new) new audio samples for this chunk
            new_state: updated state dict
        """
        istft = self.istft
        S = self._x_to_spec(x)  # (B, N, T_chunk)
        B, N, T_chunk = S.shape
        hop = istft.hop_length
        win_len = istft.win_length
        # causal padding has no left padding = initial skip 0 (matches the batch version)
        same_pad = 0 if istft.padding == "causal" else (win_len - hop) // 2

        if state is None:
            state = {
                "audio_buf": None,
                "win_env_buf": None,
                "output_samples": 0,
                "same_pad_skip": same_pad,  # samples still to skip
            }

        # iFFT + windowing
        ifft = torch.fft.irfft(S, istft.n_fft, dim=1, norm="backward")
        ifft = ifft * istft.window[None, :, None]

        # Overlap-add for this chunk
        chunk_len = (T_chunk - 1) * hop + win_len
        y_chunk = F.fold(
            ifft,
            output_size=(1, chunk_len),
            kernel_size=(1, win_len),
            stride=(1, hop),
        )[:, 0, 0, :]  # (B, chunk_len)

        window_sq = istft.window.square().expand(1, T_chunk, -1).transpose(1, 2)
        env_chunk = F.fold(
            window_sq,
            output_size=(1, chunk_len),
            kernel_size=(1, win_len),
            stride=(1, hop),
        ).squeeze()  # (chunk_len,)

        # Merge with previous buffer
        if state["audio_buf"] is None:
            audio_buf = y_chunk
            env_buf = env_chunk
            if istft.padding == "causal":
                # apply the leading-window envelope correction to the first
                # chunk (identical to batch)
                corr = istft.causal_env_corr
                n = min(corr.numel(), env_buf.numel())
                env_buf[:n] = env_buf[:n] + corr[:n]
        else:
            prev_len = state["audio_buf"].shape[-1]
            overlap = min(win_len - hop, prev_len, chunk_len)
            total_len = prev_len + chunk_len - overlap
            # Preserve the input dtype in overlap-add buffers.
            audio_buf = torch.zeros(B, total_len, device=y_chunk.device, dtype=y_chunk.dtype)
            env_buf = torch.zeros(total_len, device=env_chunk.device, dtype=env_chunk.dtype)
            audio_buf[:, :prev_len] = state["audio_buf"]
            env_buf[:prev_len] = state["win_env_buf"]
            s = prev_len - overlap
            audio_buf[:, s : s + chunk_len] += y_chunk
            env_buf[s : s + chunk_len] += env_chunk

        # Output samples whose window envelope is complete
        keep = win_len - hop
        can_output = audio_buf.shape[-1] - keep

        if can_output <= 0:
            new_state = {
                "audio_buf": audio_buf.detach(),
                "win_env_buf": env_buf.detach(),
                "output_samples": state["output_samples"],
                "same_pad_skip": state.get("same_pad_skip", 0),
            }
            return (
                torch.zeros(B, 1, 0, device=y_chunk.device, dtype=y_chunk.dtype),
                new_state,
            )

        out = audio_buf[:, :can_output] / env_buf[:can_output].clamp(min=1e-11)

        # "same" padding: skip initial samples
        skip = state.get("same_pad_skip", 0)
        if skip > 0:
            if out.shape[-1] <= skip:
                # All samples consumed by the skip — emit nothing
                new_state = {
                    "audio_buf": audio_buf[:, can_output:].detach(),
                    "win_env_buf": env_buf[can_output:].detach(),
                    "output_samples": 0,
                    "same_pad_skip": skip - out.shape[-1],
                }
                return (
                    torch.zeros(B, 1, 0, device=y_chunk.device, dtype=y_chunk.dtype),
                    new_state,
                )
            out = out[:, skip:]
            skip = 0

        new_state = {
            "audio_buf": audio_buf[:, can_output:].detach(),
            "win_env_buf": env_buf[can_output:].detach(),
            "output_samples": state["output_samples"] + out.shape[-1],
            "same_pad_skip": 0,
        }
        return out.unsqueeze(1), new_state

    def flush_streaming(self, state: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Flush remaining samples from the streaming buffer.

        Call this after the last chunk to retrieve the final samples.

        Returns:
            audio: (B, 1, T_remaining)
        """
        if state is None or state.get("audio_buf") is None:
            return torch.zeros(1, 1, 0)
        if self.istft.padding == "causal":
            # In causal mode the batch version truncates at T*hop, so the cumulative
            # emits (= F*hop) already cover the full output. The remaining buffer
            # (win-hop samples) is discarded.
            b = state["audio_buf"]
            return torch.zeros(b.shape[0], 1, 0, device=b.device, dtype=b.dtype)
        same_pad = (self.istft.win_length - self.istft.hop_length) // 2
        buf = state["audio_buf"]
        env = state["win_env_buf"]
        if buf.shape[-1] == 0:
            return torch.zeros(buf.shape[0], 1, 0, device=buf.device)
        out = buf / env.clamp(min=1e-11)
        if same_pad > 0 and out.shape[-1] > same_pad:
            out = out[:, :-same_pad]
        return out.unsqueeze(1)


# ---------------------------------------------------------------------------
# ResnetBlock
# ---------------------------------------------------------------------------


class ResnetBlock(nn.Module):
    """ResNet block with configurable Conv1d padding.

    ``padding_mode`` selects symmetric / causal / causal_full / asymmetric padding.
    Streaming inference uses ``_sym_conv_with_buffer``: non-causal modes incur a
    right_pad-frame delay (discard/recompute buffering); causal modes stream with
    zero delay.
    """

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        activation="silu",
        padding_mode="symmetric",
        kernel_size=3,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self._padding_mode = padding_mode
        self._kernel_size = kernel_size

        self.norm1 = ChannelRMSNorm(in_channels)
        # padding=0: padding applied manually in forward(),
        # streaming uses _sym_conv_with_buffer with the same conv.
        self.conv1 = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0
        )
        self.norm2 = ChannelRMSNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0
                )
            else:
                self.nin_shortcut = torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

        if activation == "snake_beta":
            self.act1 = SnakeBeta(in_channels, alpha_logscale=True)
            self.act2 = SnakeBeta(out_channels, alpha_logscale=True)
            self._use_snake = True
        else:
            self._use_snake = False

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Apply padding according to kernel_size and padding_mode."""
        total_pad = self._kernel_size - 1
        if total_pad == 0:
            return x
        lp, rp = _compute_pad(total_pad, self._padding_mode)
        return F.pad(x, (lp, rp))

    def _activate(self, x, idx):
        if self._use_snake:
            return self.act1(x) if idx == 1 else self.act2(x)
        return nonlinearity(x)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self._activate(h, 1)
        h = self._pad(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self._activate(h, 2)
        h = self.dropout(h)
        h = self._pad(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self._pad(x)
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

    def forward_streaming(
        self,
        x: torch.Tensor,
        state: dict | None = None,
        flush: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Streaming forward with conv buffers.

        The conv path (h) may produce fewer frames than the input due to
        buffering.  The residual path buffers input to maintain
        alignment with the conv path.

        When ``flush=True``, zero-pads conv buffers to produce all remaining
        output (matching batch-mode padding at sequence end).
        """
        if state is None:
            state = {
                "conv1_buf": None,
                "conv2_buf": None,
                "shortcut_buf": None,
                "res_buf": None,
                "res_pos": 0,  # cumulative output position
            }

        h = x
        h = self.norm1(h)
        h = self._activate(h, 1)
        h, state["conv1_buf"] = _sym_conv_with_buffer(
            self.conv1,
            h,
            state["conv1_buf"],
            flush=flush,
            padding_mode=self._padding_mode,
        )

        h = self.norm2(h)
        h = self._activate(h, 2)
        h = self.dropout(h)
        h, state["conv2_buf"] = _sym_conv_with_buffer(
            self.conv2,
            h,
            state["conv2_buf"],
            flush=flush,
            padding_mode=self._padding_mode,
        )

        out_len = h.shape[2]

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x, state["shortcut_buf"] = _sym_conv_with_buffer(
                    self.conv_shortcut,
                    x,
                    state["shortcut_buf"],
                    flush=flush,
                    padding_mode=self._padding_mode,
                )
            else:
                x = self.nin_shortcut(x)

        # Buffer residual input and extract the correct aligned portion.
        # The conv output corresponds to batch positions [res_pos, res_pos + out_len).
        if state["res_buf"] is not None:
            x = torch.cat([state["res_buf"], x], dim=2)
        # x now contains all residual input from res_pos onward (possibly more)
        if out_len == 0:
            state["res_buf"] = x.detach()
            return torch.zeros(x.shape[0], self.out_channels, 0, device=x.device), state
        res = x[:, :, :out_len]
        state["res_buf"] = x[:, :, out_len:].detach()
        state["res_pos"] = state["res_pos"] + out_len

        return res + h, state


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class VocosBackbone(nn.Module):
    """
    Vocos backbone with a transformer (causal by default), configurable conv padding,
    and KV cache support.

    Conv1d layers use configurable padding (symmetric/causal/causal_full/asymmetric)
    during training and corresponding buffering during streaming inference.
    The transformer uses a causal attention mask unless ``attn_causal=False``
    (a non-causal variant that does not support KV cache/streaming).

    Args:
        hidden_dim (int): Hidden dimension of the model.
        depth (int): Number of transformer layers.
        heads (int): Number of attention heads.
        pos_emb_dim (int): Dimension for rotary positional embeddings.
        padding_mode (str): ``"symmetric"`` (default), ``"causal"``, ``"causal_full"``,
            or ``"asymmetric"``.
        attn_causal (bool): causal self-attention (default True).
        resnet_kernel_size (int): Kernel size for ResnetBlock convolutions (default: 3).
        embed_kernel_size (int): Kernel size for the embed conv (default: 7).
    """

    def __init__(
        self,
        hidden_dim=1024,
        depth=12,
        heads=16,
        pos_emb_dim=64,
        activation="mixed",
        dropout=0.1,
        mlp_expansion=4,
        attn_dropout=0.0,
        padding_mode="symmetric",
        resnet_kernel_size=3,
        embed_kernel_size=7,
        attn_causal=True,
    ):
        super().__init__()
        self._padding_mode = padding_mode
        self._embed_kernel_size = embed_kernel_size

        # Embed conv: padding=0 (applied manually in forward / buffer in streaming)
        self.embed = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=embed_kernel_size, padding=0)

        block_in = hidden_dim

        # "mixed" uses SiLU in ResnetBlock.
        resnet_act = "silu" if activation == "mixed" else activation

        prior_net: list[nn.Module] = [
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                dropout=dropout,
                activation=resnet_act,
                padding_mode=padding_mode,
                kernel_size=resnet_kernel_size,
            ),
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                dropout=dropout,
                activation=resnet_act,
                padding_mode=padding_mode,
                kernel_size=resnet_kernel_size,
            ),
        ]
        self.prior_net = nn.Sequential(*prior_net)

        # RoPE cache length (no learned parameters, not in the state_dict):
        # bounds the absolute positions a single decode/stream may span.
        # 65536 grid frames = about 22 minutes of audio at 50 Hz.
        time_rotary_embed = RotaryPositionalEmbeddings(dim=pos_emb_dim, max_seq_len=65536)

        transformer_blocks = [
            TransformerBlock(
                dim=hidden_dim,
                n_heads=heads,
                rotary_embed=time_rotary_embed,
                mlp_expansion=mlp_expansion,
                attn_dropout=attn_dropout,
                attn_causal=attn_causal,
            )
            for _ in range(depth)
        ]

        self.transformers = nn.Sequential(*transformer_blocks)
        self.final_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        post_net: list[nn.Module] = [
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                dropout=dropout,
                activation=resnet_act,
                padding_mode=padding_mode,
                kernel_size=resnet_kernel_size,
            ),
            ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                dropout=dropout,
                activation=resnet_act,
                padding_mode=padding_mode,
                kernel_size=resnet_kernel_size,
            ),
        ]
        self.post_net = nn.Sequential(*post_net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform projected decoder features.

        Args:
            x: (B, L, hidden_dim)

        Returns:
            (B, L, hidden_dim). Incremental decoding uses ``forward_streaming``.
        """
        x = x.transpose(1, 2)
        embed_total_pad = self._embed_kernel_size - 1
        if embed_total_pad > 0:
            lp, rp = _compute_pad(embed_total_pad, self._padding_mode)
            x = F.pad(x, (lp, rp))
        x = self.embed(x)
        x = self.prior_net(x)
        x = x.transpose(1, 2)
        # nn.Sequential works because TransformerBlock.forward(x) defaults
        # (layer_past=None, use_cache=False) are compatible
        x = self.transformers(x)

        x = x.transpose(1, 2)
        x = self.post_net(x)
        x = x.transpose(1, 2)
        x = self.final_layer_norm(x)
        return x

    def forward_streaming(
        self,
        x: torch.Tensor,
        state: dict | None = None,
        flush: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Streaming forward with all conv buffers and KV cache.

        Args:
            x: (B, L, hidden_dim) — new input frames.  May be empty when
               ``flush=True`` to drain remaining buffered frames.
            state: dict with conv buffers and KV cache, or None
            flush: if True, zero-pad conv buffers to produce all remaining
                   output (matching batch-mode padding at sequence end)

        Returns:
            (x, new_state)
        """
        if state is None:
            state = {
                "embed_buf": None,
                "prior_states": [None, None],
                "past_key_values": [None] * len(self.transformers),
                "post_states": [None, None],
                # total number of frames fed into the transformer (absolute position).
                # Kept explicitly because the position cannot be recovered from the
                # cache length once eval_attn_window evicts cache entries.
                "pos": 0,
            }

        x = x.transpose(1, 2)
        x, state["embed_buf"] = _sym_conv_with_buffer(
            self.embed,
            x,
            state["embed_buf"],
            flush=flush,
            padding_mode=self._padding_mode,
        )

        x, state["prior_states"][0] = self.prior_net[0].forward_streaming(
            x, state["prior_states"][0], flush=flush
        )
        x, state["prior_states"][1] = self.prior_net[1].forward_streaming(
            x, state["prior_states"][1], flush=flush
        )

        # If convolutions consumed all input into buffers, return empty output
        if x.shape[2] == 0:
            return x.transpose(1, 2), state

        x = x.transpose(1, 2)

        pos0 = state.get("pos", 0)
        # Precompute the attention tensors once per call — they are identical for
        # every transformer layer (same T / pos_offset / cache length / window),
        # so rebuilding them inside each of the stacked layers is pure dispatch
        # overhead on the streaming path.
        T_new = x.shape[1]
        past0 = state["past_key_values"][0]
        past_len = past0[0].shape[2] if past0 is not None else 0
        Wa = getattr(self.transformers[0].att, "eval_attn_window", None)
        input_pos = torch.arange(pos0, pos0 + T_new, device=x.device)
        mask = None
        if T_new > 1 and past_len > 0:
            k_pos = torch.arange(pos0 - past_len, pos0 + T_new, device=x.device)
            rel = input_pos.view(-1, 1) - k_pos.view(1, -1)
            mask = (rel >= 0) & (rel <= Wa) if Wa is not None else (rel >= 0)
        attn_ctx = {"input_pos": input_pos, "mask": mask, "past_len": past_len}

        presents = []
        for i, layer in enumerate(self.transformers):
            x, kv = layer(
                x,
                layer_past=state["past_key_values"][i],
                use_cache=True,
                pos_offset=pos0,
                attn_ctx=attn_ctx,
            )
            presents.append(kv)
        state["past_key_values"] = presents
        state["pos"] = pos0 + x.shape[1]

        x = x.transpose(1, 2)

        x, state["post_states"][0] = self.post_net[0].forward_streaming(
            x, state["post_states"][0], flush=flush
        )
        x, state["post_states"][1] = self.post_net[1].forward_streaming(
            x, state["post_states"][1], flush=flush
        )

        x = x.transpose(1, 2)
        x = self.final_layer_norm(x)

        return x, state


# ---------------------------------------------------------------------------
# Codec Decoder
# ---------------------------------------------------------------------------


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class CodecDecoderVocos(nn.Module):
    """Quantize latent features or synthesize waveforms from projected features."""

    def __init__(
        self,
        hidden_dim=1024,
        depth=12,
        heads=16,
        pos_emb_dim=64,
        hop_length=320,
        vq_dim=2048,
        upsampler_factors=None,
        upsampler_kernel_sizes=None,
        output_n_fft=None,
        output_hop_length=None,
        activation="mixed",
        upsampler_activation=None,
        fsq_levels=None,
        fsq_num_quantizers=1,
        dropout=0.1,
        mlp_expansion=4,
        attn_dropout=0.0,
        padding_mode="symmetric",
        resnet_kernel_size=3,
        embed_kernel_size=7,
        quantizer_cfg=None,
        attn_causal=True,
    ):
        super().__init__()
        self.hop_length = hop_length

        # quantizer config section. If absent or mode=fsq, build the
        # single-stage ResidualFSQ of the original X-Codec-2.0.
        qcfg = quantizer_cfg if quantizer_cfg is not None else {}
        self.quantizer_mode = qcfg.get("mode", "fsq") if hasattr(qcfg, "get") else "fsq"
        if self.quantizer_mode == "fsq":
            if fsq_levels is None:
                fsq_levels = [4, 4, 4, 4, 4, 4, 4, 4]
            self.quantizer = ResidualFSQ(
                dim=vq_dim, levels=fsq_levels, num_quantizers=fsq_num_quantizers
            )
        elif self.quantizer_mode == "rvq":
            self.quantizer = RubatoResidualQuantizer(
                dim=vq_dim,
                num_quantizers=qcfg.get("num_quantizers", 16),
                codebook_size=qcfg.get("codebook_size", 4096),
                fsq_levels=qcfg.get("fsq_levels", None),
                cb1_type=qcfg.get("cb1_type", "fsq"),
                residual_type=qcfg.get("residual_type", "fsq"),
                quantize_dropout=qcfg.get("quantize_dropout", True),
                vq_codebook_dim=qcfg.get("vq_codebook_dim", 8),
                vq_commitment_weight=qcfg.get("vq_commitment_weight", 0.25),
                vq_codebook_weight=qcfg.get("vq_codebook_weight", 1.0),
                cb1_codebook_size=qcfg.get("cb1_codebook_size", None),
                cb1_fsq_levels=qcfg.get("cb1_fsq_levels", None),
            )
        else:
            raise ValueError(f"unknown quantizer mode: {self.quantizer_mode}")

        self.backbone = VocosBackbone(
            hidden_dim=hidden_dim,
            depth=depth,
            heads=heads,
            pos_emb_dim=pos_emb_dim,
            activation=activation,
            dropout=dropout,
            mlp_expansion=mlp_expansion,
            attn_dropout=attn_dropout,
            padding_mode=padding_mode,
            resnet_kernel_size=resnet_kernel_size,
            embed_kernel_size=embed_kernel_size,
            attn_causal=attn_causal,
        )

        # Upsampler for high sample rate output (e.g., 44.1kHz)
        # upsampler_activation overrides if set; otherwise "mixed" -> "snake_beta"
        if upsampler_activation is not None:
            upsampler_act = upsampler_activation
        else:
            upsampler_act = "snake_beta" if activation == "mixed" else activation
        self.upsampler = None
        if upsampler_factors is not None:
            self.upsampler = UpSamplerBlock(
                in_channels=hidden_dim,
                upsample_factors=upsampler_factors,
                kernel_sizes=upsampler_kernel_sizes,
                activation=upsampler_act,
                padding_mode=padding_mode,
            )

        # ISTFT head parameters: use output params if provided, else derive from encoder hop_length.
        # Only padding_mode="causal_full" makes the ISTFT head causal; other modes,
        # including "causal", keep the symmetric ("same") head and its lookahead.
        actual_n_fft = output_n_fft if output_n_fft is not None else (self.hop_length * 4)
        actual_hop_length = output_hop_length if output_hop_length is not None else self.hop_length
        self.head = ISTFTHead(
            dim=hidden_dim,
            n_fft=actual_n_fft,
            hop_length=actual_hop_length,
            padding="causal" if padding_mode == "causal_full" else "same",
        )

        self.reset_parameters()

    def forward(self, x, vq=True, vq_mask=None):
        """Quantize latents or decode projected features into audio.

        Args:
            x: Features shaped (B, vq_dim, N) for quantization, or
                (B, T, hidden_dim) for waveform decoding.
            vq: if True, quantize input; if False, decode from embeddings
            vq_mask: (B, N) bool, valid-slot mask (rvq mode only; excludes pad slots
                from the VQ layers' commitment/codebook losses. Ignored in fsq
                mode since FSQ is stateless)

        Returns:
            If vq=True (fsq): (quantized, vq_code, None)
            If vq=True (rvq): (quantized, vq_code (B,Q,N), aux dict) —
                aux = {"commit_loss": (Q,), "cb1_quantized": (B,D,N)}
            If vq=False: (audio, x_pred). Incremental decoding uses
                ``decode_streaming`` / ``backbone.forward_streaming``.
        """
        if vq is True:
            x = x.permute(0, 2, 1)
            if self.quantizer_mode == "fsq":
                x, q = self.quantizer(x)
                x = x.permute(0, 2, 1)
                q = q.permute(0, 2, 1)
                return x, q, None
            xq, q, commit, q_cb1 = self.quantizer(x, mask=vq_mask)
            aux = {
                "commit_loss": commit,
                "cb1_quantized": q_cb1.permute(0, 2, 1),
            }
            if self.quantizer.training and self.quantizer.last_depths is not None:
                # mean of the per-sample dropout depths (for monitoring)
                aux["mean_depth"] = self.quantizer.last_depths.float().mean()
            return xq.permute(0, 2, 1), q.permute(0, 2, 1), aux

        x = self.backbone(x)

        if self.upsampler is not None:
            x = x.transpose(1, 2)  # (B, hidden_dim, L) for UpSamplerBlock
            x = self.upsampler(x)  # (B, L*upsample_factor, hidden_dim)
        x, x_pred = self.head(x)
        return x, x_pred

    @torch.no_grad()
    def decode_streaming(
        self,
        vq_codes_chunk: torch.Tensor,
        fc_post_a: nn.Linear,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Streaming decode: process VQ codes through backbone → upsampler → ISTFT.

        The symmetric Conv1d padding in the backbone and upsampler is handled by
        ``_sym_conv_with_buffer``, which discards unreliable tail frames and
        recomputes them on the next call.  No explicit look-ahead buffer is
        needed at the decoder level.

        Args:
            vq_codes_chunk: (B, Q, chunk_len) integer VQ code indices
            fc_post_a: Linear(vq_dim, hidden_dim) projection layer
            state: streaming state dict, or None for first call

        Returns:
            audio_chunk: (B, 1, T_new) new audio samples (may be empty if
                         still filling conv buffers)
            new_state: updated state dict
        """
        if state is None:
            state = {
                "backbone_state": None,
                "upsampler_state": None,
                "istft_state": None,
            }

        return self._decode_frames(vq_codes_chunk, fc_post_a, state)

    def _decode_frames(
        self,
        vq_codes: torch.Tensor,
        fc_post_a: nn.Linear,
        state: dict,
        flush: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Decode VQ codes through backbone → upsampler → ISTFT."""
        # 1. Dequantize: indices -> embeddings
        vq_emb = self.quantizer.get_output_from_indices(vq_codes.transpose(1, 2))

        # 2. Project to hidden_dim
        vq_emb = fc_post_a(vq_emb)

        # 3. Backbone with conv buffers + KV cache
        x, state["backbone_state"] = self.backbone.forward_streaming(
            vq_emb, state=state["backbone_state"], flush=flush
        )

        # If backbone consumed all input into buffers, return empty audio
        if x.shape[1] == 0:
            return torch.zeros(vq_codes.shape[0], 1, 0, device=vq_codes.device), state

        # 4. Upsampler with ConvTranspose1d + ResNet conv buffers
        if self.upsampler is not None:
            x = x.transpose(1, 2)
            x, state["upsampler_state"] = self.upsampler.forward_streaming(
                x, state=state["upsampler_state"], flush=flush
            )

        # If upsampler consumed all frames into buffers, return empty audio
        if x.shape[1] == 0:
            return torch.zeros(vq_codes.shape[0], 1, 0, device=vq_codes.device), state

        # 5. Exact streaming ISTFT
        audio_chunk, state["istft_state"] = self.head.streaming_forward(
            x, state=state["istft_state"]
        )

        return audio_chunk, state

    def flush_streaming(
        self,
        state: dict | None = None,
    ) -> torch.Tensor:
        """Flush remaining audio from conv buffers and ISTFT overlap-add buffer.

        Sends a zero-length input with ``flush=True`` through the backbone and
        upsampler to drain their symmetric-padding buffers, then flushes the
        ISTFT buffer.

        Args:
            state: streaming state dict

        Returns:
            audio: (B, 1, T_remaining)
        """
        if state is None:
            return torch.zeros(1, 1, 0)

        audio_parts = []

        # Flush backbone and upsampler conv buffers by sending empty input with flush=True
        device = next(self.parameters()).device
        hidden_dim = self.backbone.embed.out_channels
        # Infer batch size from backbone state buffers (falls back to 1)
        batch_size = 1
        bb_state = state.get("backbone_state")
        if bb_state and bb_state.get("embed_buf") is not None:
            batch_size = bb_state["embed_buf"].shape[0]
        empty_emb = torch.zeros(batch_size, 0, hidden_dim, device=device)

        # Flush backbone
        x, state["backbone_state"] = self.backbone.forward_streaming(
            empty_emb, state=state["backbone_state"], flush=True
        )

        # Flush upsampler (even if backbone produced 0 frames, upsampler may have buffered data)
        if self.upsampler is not None:
            x = (
                x.transpose(1, 2)
                if x.shape[1] > 0
                else torch.zeros(batch_size, hidden_dim, 0, device=device)
            )
            x, state["upsampler_state"] = self.upsampler.forward_streaming(
                x, state=state["upsampler_state"], flush=True
            )

        if x.shape[1] > 0:
            audio_chunk, state["istft_state"] = self.head.streaming_forward(
                x, state=state["istft_state"]
            )
            if audio_chunk.shape[2] > 0:
                audio_parts.append(audio_chunk)

        # Flush ISTFT buffer
        istft_audio = self.head.flush_streaming(state.get("istft_state"))
        if istft_audio.shape[2] > 0:
            audio_parts.append(istft_audio)

        if not audio_parts:
            return torch.zeros(batch_size, 1, 0, device=device)
        return torch.cat(audio_parts, dim=2)

    def reset_parameters(self):
        self.apply(init_weights)
