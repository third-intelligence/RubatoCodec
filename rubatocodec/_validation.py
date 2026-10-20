"""Input contracts shared by offline and streaming inference."""

from numbers import Integral

import torch

_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def validate_positive_integer(value, name: str) -> None:
    """Reject nonintegers, booleans, and nonpositive values."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_waveform(wav: torch.Tensor, sr: int | None) -> None:
    """Require finite, nonempty, floating-point mono audio with batch size one."""
    if not isinstance(wav, torch.Tensor):
        raise TypeError("wav must be a torch.Tensor")
    if wav.ndim not in (1, 2) or (wav.ndim == 2 and wav.shape[0] != 1):
        raise ValueError(
            "wav must have shape (samples,) or (1, samples); encoding requires mono B=1"
        )
    if not wav.numel():
        raise ValueError("wav must not be empty")
    if not wav.is_floating_point():
        raise TypeError("wav must have a floating-point dtype")
    if not bool(torch.isfinite(wav).all()):
        raise ValueError("wav must contain only finite values")
    if sr is not None:
        validate_positive_integer(sr, "sr")


def validate_codes(
    codes: torch.Tensor | None,
    durs: torch.Tensor | None,
    depth: int | None,
    quantizer,
    *,
    adaptive: bool,
    allow_empty: bool = False,
) -> None:
    """Validate code layout, vocabulary bounds, prefix depth, and durations.

    Adaptive codes have shape (Q, N); fixed-rate codes have shape (B, Q, T).
    Empty chunks (or None) are accepted only by the streaming entry point.
    Residual index -1 denotes an omitted layer; layer zero must be present.
    """
    if depth is not None:
        validate_positive_integer(depth, "depth")
        if depth > quantizer.num_quantizers:
            raise ValueError(f"depth must not exceed {quantizer.num_quantizers}")
    if codes is None:
        if not allow_empty:
            raise TypeError("codes must be a torch.Tensor")
        if durs is not None and (not isinstance(durs, torch.Tensor) or durs.numel()):
            raise ValueError("A flush-only call cannot contain durations")
        return
    if not isinstance(codes, torch.Tensor):
        raise TypeError("codes must be a torch.Tensor")
    if codes.ndim != (2 if adaptive else 3):
        raise ValueError(
            "Expected codes of shape (Q, N)" if adaptive else "Expected codes of shape (B, Q, T)"
        )
    if codes.dtype not in _INTEGER_DTYPES:
        raise TypeError("codes must have an integer dtype")
    layers = codes.shape[-2]
    if not 1 <= layers <= quantizer.num_quantizers:
        raise ValueError(f"codes must contain 1 to {quantizer.num_quantizers} layers")
    if depth is not None and depth > layers:
        raise ValueError("depth must not exceed the number of supplied code layers")
    if not adaptive and codes.shape[0] == 0:
        raise ValueError("codes must contain at least one batch item")
    events = codes.shape[-1]
    if not events and not allow_empty:
        raise ValueError("codes must contain at least one event or frame")
    if events:
        # Promote before comparing: vocabulary bounds can overflow int8/uint8.
        indices = codes.long()
        first = indices[..., 0, :]
        residual = indices[..., 1:, :]
        first_size = getattr(quantizer, "cb1_codebook_size", quantizer.codebook_size)
        if bool(((first < 0) | (first >= first_size)).any()):
            raise ValueError(f"Layer 0 indices must be in [0, {first_size})")
        if bool(((residual < -1) | (residual >= quantizer.codebook_size)).any()):
            raise ValueError(f"Residual indices must be -1 or in [0, {quantizer.codebook_size})")
    if not adaptive:
        if durs is not None:
            raise ValueError("This checkpoint does not accept event durations")
        return
    if durs is None and not events and allow_empty:
        return
    if not isinstance(durs, torch.Tensor):
        raise TypeError("Adaptive codes require a durs tensor")
    if durs.dtype not in _INTEGER_DTYPES:
        raise TypeError("durs must have an integer dtype (grid frames)")
    if durs.ndim != 1 or durs.numel() != events:
        raise ValueError("durs must contain one duration per event")
    if bool((durs <= 0).any()):
        raise ValueError("Event durations must be positive")
