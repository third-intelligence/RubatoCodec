"""Checkpoint metadata shared by loading, export, and training initialization."""

from __future__ import annotations

import math
from pathlib import Path

import torch

INFERENCE_PREFIXES = (
    "CodecEnc.",
    "generator.",
    "fc_post_a.",
    "SemanticEncoder_module.",
    "fc_prior.",
    "firing.",
    "firing_agg.",
)

# Grid composition remains necessary for adaptive decoding; aggregation is encoding-only.
DECODER_PREFIXES = ("generator.", "fc_post_a.", "firing.")


def resolve_inference_settings(config: dict) -> dict:
    """Merge saved overrides with the codec's recommended inference defaults.

    Fully causal decoders use six-second windows, a two-second stride, and a
    10-ms crossfade (300 / 100 frames on the released 50-Hz grid). Explicit
    None values are retained. Older noncausal decoders use full-grid decoding.
    """
    decoder = config["codec_decoder"]
    inference = {"attention_window_seconds": 6.0}
    if decoder.get("conv_padding_mode") == "causal_full" and decoder.get("attn_causal", True):
        hop = math.prod(config["codec_encoder"]["up_ratios"])
        window = round(6.0 * config["sr"] / hop)
        inference.update(
            decode_window_frames=window,
            decode_stride_frames=window // 3,
            decode_crossfade_ms=10.0,
        )
    inference.update(
        {
            key: value
            for key, value in (config.get("inference") or {}).items()
            if key
            in (
                "attention_window_seconds",
                "decode_window_frames",
                "decode_stride_frames",
                "decode_crossfade_ms",
            )
        }
    )
    return inference


def extract_checkpoint_config(cfg) -> dict:
    """Extract model and inference settings from a Lightning training config."""
    from omegaconf import OmegaConf

    config = {
        "codec_encoder": OmegaConf.to_container(cfg.model.codec_encoder, resolve=True),
        "codec_decoder": OmegaConf.to_container(cfg.model.codec_decoder, resolve=True),
        "firing": (
            OmegaConf.to_container(cfg.model.firing, resolve=True)
            if cfg.model.get("firing") is not None
            else None
        ),
        "semantic_model": cfg.train.get("semantic_model", "facebook/w2v-bert-2.0"),
        "semantic_layer_index": cfg.train.get("semantic_layer_index", 16),
        "sr": cfg.dataset.audio.sr,
        "output_sr": cfg.dataset.audio.get("output_sr", cfg.dataset.audio.sr),
        "inference": OmegaConf.to_container(
            (cfg.get("inference") or OmegaConf.create({})), resolve=True
        ),
    }
    if config["firing"]:
        config["firing"]["mode"] = (
            "stride" if config["firing"].get("mode") == "stride" else "adaptive"
        )
    config["inference"] = resolve_inference_settings(config)
    return config


def load_lightning_checkpoint(
    path: str | Path, *, decoder_only: bool = False
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Read inference weights and metadata from a trusted Lightning checkpoint.

    PyTorch deserializes the whole file before selecting weights. Prefer
    safetensors when decoder-only loading should also avoid reading encoder tensors.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = extract_checkpoint_config(checkpoint["hyper_parameters"]["cfg"])
    prefixes = DECODER_PREFIXES if decoder_only else INFERENCE_PREFIXES
    weights = {
        key: value for key, value in checkpoint["state_dict"].items() if key.startswith(prefixes)
    }
    return config, weights


def clean_inference_config(config: dict) -> dict:
    """Export settings consumed by the public loader, preserving eval behavior.

    Conditional keys are retained when they select modules or affect inference.
    Keep these allowlists in sync with RubatoCodec._build when adding options.
    """

    def select(source, names):
        return {key: value for key, value in source.items() if key in names.split()}

    result = select(config, "semantic_model semantic_layer_index sr output_sr")
    result["codec_encoder"] = select(
        config["codec_encoder"], "ngf hidden_dim up_ratios dilations activation"
    )
    decoder = select(
        config["codec_decoder"],
        "hidden_dim depth heads pos_emb_dim vq_dim activation mlp_expansion "
        "conv_padding_mode resnet_kernel_size embed_kernel_size attn_causal "
        "upsampler_factors upsampler_kernel_sizes upsampler_activation "
        "output_n_fft output_hop_length fsq_levels fsq_num_quantizers",
    )
    quantizer = config["codec_decoder"].get("quantizer") or {}
    if quantizer.get("mode", "fsq") == "rvq":
        decoder.pop("fsq_levels", None)
        decoder.pop("fsq_num_quantizers", None)
        decoder["quantizer"] = select(
            quantizer,
            "mode num_quantizers codebook_size fsq_levels cb1_type residual_type "
            "vq_codebook_dim cb1_codebook_size cb1_fsq_levels",
        )
        decoder["quantizer"] = {k: v for k, v in decoder["quantizer"].items() if v is not None}
    elif quantizer.get("mode", "fsq") != "fsq":
        raise ValueError(f"Unsupported quantizer mode: {quantizer['mode']}")
    result["codec_decoder"] = decoder
    firing = config.get("firing")
    result["firing"] = None
    if firing and firing.get("enabled", False):
        cleaned = select(
            firing,
            "enabled mode target_ratio decision temp max_gap aggregate "
            "grid_compose head_in_budget hnet_ms_offsets",
        )
        cleaned["mode"] = "stride" if firing.get("mode") == "stride" else "adaptive"
        signals = firing.get("signals")
        if signals is None:
            signals = [] if firing.get("mode", "adaptive") == "stride" else ["hnet_ms", "flux"]
        cleaned["signals"] = list(signals)
        if "hnet_ms" not in signals:
            cleaned.pop("hnet_ms_offsets", None)
        if firing.get("decision", "clock") != "clock":
            cleaned.pop("head_in_budget", None)
        if firing.get("decision", "clock") != "level":
            cleaned.pop("temp", None)
        result["firing"] = cleaned
    result["inference"] = resolve_inference_settings(config)
    return result
