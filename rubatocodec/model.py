"""Public waveform encoding and offline or streaming decoding API."""

from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._validation import validate_codes, validate_positive_integer, validate_waveform
from .checkpoint import (
    DECODER_PREFIXES,
    INFERENCE_PREFIXES,
    load_lightning_checkpoint,
    resolve_inference_settings,
)
from .firing import FiringModule, SegmentAggregator, mel_flux
from .modules.decoder import CodecDecoderVocos
from .modules.encoder import CodecEncoder
from .modules.semantic import SemanticEncoder, pool_semantic_features
from .types import EncodedAudio


class _Default(Enum):
    CONFIG = "config"


_DEFAULT = _Default.CONFIG


class RubatoCodec:
    """RubatoCodec: top-level codec interface for encoding and decoding audio.

    Usage::

        codec = RubatoCodec.from_checkpoint("path/to/model.safetensors", device="cuda")

        # fixed-rate checkpoint (no firing module):
        codes = codec.encode(wav_tensor)   # (B, Q, T_enc)
        audio = codec.decode(codes)        # (B, 1, T_audio)

        # adaptive-firing checkpoint — variable-length (token, dur) codes:
        enc = codec.encode(wav_tensor)     # {"codes", "durs"}
        audio = codec.decode(enc)          # full depth
        audio = codec.decode(enc, depth=8) # residual depth control (first 8 layers)

    Adaptive-firing code convention (dur form):
        codes: (Q, N) long — quantizer indices in firing order (layer 0 is
               the semantic layer)
        durs:  (N,) long — length of segment i = [pos_i, pos_{i+1}) (the last
               segment extends to the end of the grid). pos_0 = 0,
               pos_i = sum_{j<i} dur_j, sum(durs) = total grid frames,
               every dur >= 1.

    Model configuration supplies inference defaults. Explicit decoding and
    attention-window arguments override them.
    """

    def __init__(
        self,
        encoder: CodecEncoder | None,
        decoder: CodecDecoderVocos,
        semantic_model: nn.Module | None,
        semantic_encoder: SemanticEncoder | None,
        fc_prior: nn.Linear | None,
        fc_post_a: nn.Linear,
        feature_extractor,
        semantic_layer_index: int = 16,
        sr: int = 16000,
        output_sr: int = 44100,
        device: torch.device | str = "cpu",
        firing: FiringModule | None = None,
        firing_agg: SegmentAggregator | None = None,
        inference: dict | None = None,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.semantic_model = semantic_model
        self.semantic_encoder = semantic_encoder
        self.fc_prior = fc_prior
        self.fc_post_a = fc_post_a
        self.feature_extractor = feature_extractor
        self.semantic_layer_index = semantic_layer_index
        self.sr = sr
        self.output_sr = output_sr
        self.device = torch.device(device)
        self.firing = firing
        self.firing_agg = firing_agg
        self.inference = dict(inference or {})
        self._streaming_states: dict[str, dict] = {}

    @property
    def decoder_only(self) -> bool:
        """Whether this instance was loaded without the encoding components."""
        return self.encoder is None

    def set_attn_window(self, window: int | None) -> int:
        """Set the window size of the inference-time sliding-window attention.

        Args:
            window: window size in frames. None disables the window (full past context).

        Returns:
            Number of attention layers configured
        """
        if window is not None:
            validate_positive_integer(window, "window")
        n = 0
        for root in (self.encoder, self.decoder):
            if root is None:
                continue
            for m in root.modules():
                if hasattr(m, "rotary_embed"):
                    m.eval_attn_window = window
                    n += 1
        return n

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "cuda",
        attn_window_sec: float | None | _Default = _DEFAULT,
        *,
        decoder_only: bool = False,
        low_memory_encode: bool = False,
    ) -> RubatoCodec:
        """Load RubatoCodec from a safetensors file or a PyTorch Lightning checkpoint.

        Args:
            path: path to ``model.safetensors`` (preferred) or ``.ckpt`` file
            device: target device
            attn_window_sec: inference-time sliding-window attention span in
                seconds. Omitted: model config (6 seconds if absent).
                None disables the window.
            decoder_only: Load only components required for decode and
                decode_streaming. Skips acoustic/semantic encoders and the
                semantic model download. encode is unavailable on this instance.
                Safetensors reads only decoding tensors; .ckpt still deserializes
                the complete file before selecting weights.
            low_memory_encode: Opt into bounded-memory Wav2Vec2-BERT attention
                for inputs over 4,096 semantic frames. Preserves full context;
                does not split audio. Defaults to False. Ignored in decoder-only
                mode. Requires relative_key semantic attention.

        Returns:
            RubatoCodec instance ready for inference
        """
        if not isinstance(low_memory_encode, bool):
            raise TypeError("low_memory_encode must be a bool")
        path = Path(path)

        if path.suffix == ".safetensors":
            config, state_dict = cls._load_safetensors(path, decoder_only=decoder_only)
        elif path.suffix == ".ckpt":
            config, state_dict = cls._load_ckpt(path, decoder_only=decoder_only)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return cls._build(
            config,
            state_dict,
            device,
            attn_window_sec=attn_window_sec,
            decoder_only=decoder_only,
            low_memory_encode=low_memory_encode,
        )

    @staticmethod
    def _load_safetensors(
        path: Path, *, decoder_only: bool = False
    ) -> tuple[dict, dict[str, torch.Tensor]]:
        from safetensors import safe_open
        from safetensors.torch import load_file

        config_path = path.parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        if decoder_only:
            with safe_open(path, framework="pt", device="cpu") as tensors:
                state_dict = {
                    key: tensors.get_tensor(key)
                    for key in tensors.keys()
                    if key.startswith(DECODER_PREFIXES)
                }
        else:
            state_dict = load_file(path, device="cpu")
        return config, state_dict

    @staticmethod
    def _load_ckpt(
        path: Path, *, decoder_only: bool = False
    ) -> tuple[dict, dict[str, torch.Tensor]]:
        return load_lightning_checkpoint(path, decoder_only=decoder_only)

    @classmethod
    def _build(
        cls,
        config: dict,
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device,
        attn_window_sec: float | None | _Default = _DEFAULT,
        *,
        decoder_only: bool = False,
        low_memory_encode: bool = False,
    ) -> RubatoCodec:
        enccfg = config["codec_encoder"]
        deccfg = config["codec_decoder"]
        semantic_model_name = config.get("semantic_model", "facebook/w2v-bert-2.0")
        semantic_layer_index = config.get("semantic_layer_index", 16)
        sr = config["sr"]
        output_sr = config.get("output_sr", sr)

        # --- split state_dict by prefix ---
        prefixes = {prefix: OrderedDict() for prefix in INFERENCE_PREFIXES}
        for key, value in state_dict.items():
            for prefix, sd in prefixes.items():
                if key.startswith(prefix):
                    sd[key[len(prefix) :]] = value
                    break

        # The decoder's grid spacing is metadata, so it does not require an encoder instance.
        hop_length = math.prod(enccfg["up_ratios"])
        encoder = None
        # --- build encoder ---
        if not decoder_only:
            encoder = CodecEncoder(
                ngf=enccfg["ngf"],
                up_ratios=list(enccfg["up_ratios"]),
                dilations=list(enccfg["dilations"]),
                hidden_dim=enccfg["hidden_dim"],
                activation=enccfg.get("activation", "snake_beta"),
            )
            encoder.load_state_dict(prefixes["CodecEnc."])
            encoder.eval().to(device)

        # --- build decoder ---
        decoder = CodecDecoderVocos(
            hidden_dim=deccfg["hidden_dim"],
            depth=deccfg["depth"],
            heads=deccfg["heads"],
            pos_emb_dim=deccfg["pos_emb_dim"],
            hop_length=hop_length,
            vq_dim=deccfg["vq_dim"],
            upsampler_factors=list(deccfg["upsampler_factors"])
            if deccfg.get("upsampler_factors")
            else None,
            upsampler_kernel_sizes=list(deccfg["upsampler_kernel_sizes"])
            if deccfg.get("upsampler_kernel_sizes")
            else None,
            output_n_fft=deccfg.get("output_n_fft"),
            output_hop_length=deccfg.get("output_hop_length"),
            activation=deccfg.get("activation", "mixed"),
            upsampler_activation=deccfg.get("upsampler_activation"),
            fsq_levels=list(deccfg["fsq_levels"]) if deccfg.get("fsq_levels") else None,
            fsq_num_quantizers=deccfg.get("fsq_num_quantizers", 1),
            dropout=deccfg.get("dropout", 0.1),
            mlp_expansion=deccfg.get("mlp_expansion", 4),
            attn_dropout=deccfg.get("attn_dropout", 0.0),
            padding_mode=deccfg.get("conv_padding_mode", "symmetric"),
            resnet_kernel_size=deccfg.get("resnet_kernel_size", 3),
            embed_kernel_size=deccfg.get("embed_kernel_size", 7),
            attn_causal=deccfg.get("attn_causal", True),
            quantizer_cfg=deccfg.get("quantizer"),
        )
        decoder.load_state_dict(prefixes["generator."])
        decoder.eval().to(device)

        # --- build firing (adaptive-firing checkpoints only) ---
        firing = None
        firing_agg = None
        fcfg = config.get("firing")
        if fcfg is not None and fcfg.get("enabled", False):
            firing = FiringModule(
                dim=deccfg["vq_dim"],
                mode=fcfg.get("mode", "adaptive"),
                target_ratio=fcfg.get("target_ratio", 0.25),
                temp=fcfg.get("temp", 1.0),
                decision=fcfg.get("decision", "clock"),
                signals=fcfg.get("signals", None),
                head_in_budget=fcfg.get("head_in_budget", True),
                hnet_ms_offsets=fcfg.get("hnet_ms_offsets", ((1, 0), (1, 1), (2, 1))),
                max_gap=fcfg.get("max_gap", 64),
                grid_compose=fcfg.get("grid_compose", "empty"),
            )
            firing.load_state_dict(prefixes["firing."])
            firing.eval().to(device)
            if not decoder_only and fcfg.get("aggregate", "none") != "none":
                firing_agg = SegmentAggregator(
                    dim=deccfg["vq_dim"],
                    mode=fcfg["aggregate"],
                    max_gap=fcfg.get("max_gap", 64),
                )
                firing_agg.load_state_dict(prefixes["firing_agg."])
                firing_agg.eval().to(device)

        semantic_model = feature_extractor = semantic_enc = fc_prior = None
        if not decoder_only:
            # --- build semantic model ---
            from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

            semantic_model = Wav2Vec2BertModel.from_pretrained(
                semantic_model_name, output_hidden_states=True
            )
            semantic_model.eval().to(device)
            if low_memory_encode:
                from .modules.low_memory_attention import enable_low_memory_attention

                enable_low_memory_attention(semantic_model)

            feature_extractor = AutoFeatureExtractor.from_pretrained(semantic_model_name)

            # --- detect semantic_dim and build projection layers ---
            semantic_dim = semantic_model.config.hidden_size

            semantic_enc = SemanticEncoder(semantic_dim, semantic_dim, semantic_dim)
            semantic_enc.load_state_dict(prefixes["SemanticEncoder_module."])
            semantic_enc.eval().to(device)

            fc_prior = nn.Linear(semantic_dim + enccfg["hidden_dim"], deccfg["vq_dim"])
            fc_prior.load_state_dict(prefixes["fc_prior."])
            fc_prior.eval().to(device)

        fc_post_a = nn.Linear(deccfg["vq_dim"], deccfg["hidden_dim"])
        fc_post_a.load_state_dict(prefixes["fc_post_a."])
        fc_post_a.eval().to(device)

        codec = cls(
            encoder=encoder,
            decoder=decoder,
            semantic_model=semantic_model,
            semantic_encoder=semantic_enc,
            fc_prior=fc_prior,
            fc_post_a=fc_post_a,
            feature_extractor=feature_extractor,
            semantic_layer_index=semantic_layer_index,
            sr=sr,
            output_sr=output_sr,
            device=device,
            firing=firing,
            firing_agg=firing_agg,
            inference=resolve_inference_settings(config),
        )
        if attn_window_sec is _DEFAULT:
            attn_window_sec = codec.inference.get("attention_window_seconds", 6.0)
        if attn_window_sec is not None:
            if not math.isfinite(attn_window_sec) or attn_window_sec <= 0:
                raise ValueError("attention_window_seconds must be positive or None")
            codec.set_attn_window(int(round(attn_window_sec * sr / hop_length)))
        return codec

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        device: str | torch.device = "cuda",
        *,
        attn_window_sec: float | None | _Default = _DEFAULT,
        decoder_only: bool = False,
        low_memory_encode: bool = False,
    ) -> RubatoCodec:
        """Load RubatoCodec from a HuggingFace Hub repository.

        Expects the repo to contain ``model.safetensors`` and ``config.json``.

        Args:
            repo_id: HuggingFace Hub model ID (e.g. ``"third-intelligence/RubatoCodec"``)
            device: Target device.
            attn_window_sec: Attention span in seconds. Omitted uses the model
                config (6 seconds if absent); None enables full past context.
            decoder_only: Load only decoding components, without downloading
                the semantic model. The codec weight file is still downloaded
                in full from the Hub; only decoding tensors are read from it.
            low_memory_encode: Opt into full-context, bounded-memory semantic
                attention for long encoding inputs. Defaults to False; ignored
                with decoder_only=True. See from_checkpoint.

        Returns:
            RubatoCodec instance ready for inference.
        """
        if not isinstance(low_memory_encode, bool):
            raise TypeError("low_memory_encode must be a bool")
        from huggingface_hub import hf_hub_download

        safetensors_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        hf_hub_download(repo_id=repo_id, filename="config.json")
        return cls.from_checkpoint(
            safetensors_path,
            device=device,
            attn_window_sec=attn_window_sec,
            decoder_only=decoder_only,
            low_memory_encode=low_memory_encode,
        )

    @torch.inference_mode()
    def encode(
        self,
        wav: torch.Tensor,
        sr: int | None = None,
        input_features: torch.Tensor | None = None,
    ) -> torch.Tensor | EncodedAudio:
        """Encode a waveform to VQ codes.

        Args:
            wav: Finite floating-point mono audio, shaped ``(T,)`` or
                ``(1, T)``. Encoding processes one utterance (batch size 1) per call.
            sr: sample rate of ``wav``. If it differs from ``self.sr``
                (16 kHz), the waveform is resampled internally. None assumes
                ``self.sr``.
            input_features: precomputed W2V-BERT feature_extractor output
                (for overlapping CPU feature extraction with GPU work in an
                extraction pipeline; requires input at ``self.sr``). If None,
                it is computed internally (identical result). **It must be
                computed as ``F.pad(wav[0].cpu(), (160, 160))`` on the wav
                after hop_length padding**.

        Returns:
            Fixed-rate checkpoint: VQ code indices of shape ``(B, Q, T_enc)``.
            Adaptive-firing checkpoint: dict with
                ``codes`` (Q, N) long / ``durs`` (N,) long
                (sum(durs) = total grid frames). Both tensors are int64 on
                the codec device. Input is padded to a multiple of the hop.

        Raises:
            RuntimeError: The codec was loaded with decoder_only=True.
            TypeError: Audio or features do not have floating-point tensor dtypes.
            ValueError: Input shapes, sample rate, or values are invalid.
        """
        if self.decoder_only:
            raise RuntimeError(
                "encode is unavailable on a decoder-only codec; load with decoder_only=False"
            )
        validate_waveform(wav, sr)
        if input_features is not None:
            if sr is not None and sr != self.sr:
                raise ValueError("precomputed input_features require input already at self.sr")
            if (
                not isinstance(input_features, torch.Tensor)
                or not input_features.is_floating_point()
            ):
                raise TypeError("input_features must be a floating-point tensor")
            if (
                input_features.ndim != 3
                or input_features.shape[0] != 1
                or not input_features.numel()
            ):
                raise ValueError(
                    "input_features must have shape (1, frames, features) and be nonempty"
                )
            if not bool(torch.isfinite(input_features).all()):
                raise ValueError("input_features must contain only finite values")
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        wav = wav.to(self.device)

        if sr is not None and sr != self.sr:
            import torchaudio.functional as F_audio

            wav = F_audio.resample(wav, sr, self.sr)

        # Pad to multiple of encoder hop_length
        hop = self.encoder.hop_length
        pad_len = (hop - (wav.shape[1] % hop)) % hop
        if pad_len > 0:
            wav = F.pad(wav, (0, pad_len))

        # W2V-BERT feature extraction
        if input_features is not None:
            feat = input_features.to(self.device)
        else:
            feat = (
                self.feature_extractor(
                    F.pad(wav[0, :].cpu(), (160, 160)),
                    sampling_rate=self.sr,
                    return_tensors="pt",
                )
                .data["input_features"]
                .to(self.device)
            )

        # Acoustic encoding
        vq_emb = self.encoder(wav.unsqueeze(1))  # (B, T_enc, hidden_dim)
        vq_emb = vq_emb.transpose(1, 2)  # (B, hidden_dim, T_enc)

        # Semantic encoding
        semantic_out = self.semantic_model(feat[:, :, :])
        semantic_target = semantic_out.hidden_states[self.semantic_layer_index]
        semantic_target = semantic_target.transpose(1, 2)  # (B, dim, T_enc)
        semantic_target = pool_semantic_features(semantic_target, self.encoder.hop_length)
        semantic_target = self.semantic_encoder(semantic_target)

        # Fuse (truncate both streams to the shorter length, as in training)
        min_seq = min(semantic_target.shape[2], vq_emb.shape[2])
        vq_emb = torch.cat([semantic_target[:, :, :min_seq], vq_emb[:, :, :min_seq]], dim=1)
        vq_emb = self.fc_prior(vq_emb.transpose(1, 2)).transpose(1, 2)

        if self.firing is None:
            _, vq_code, _ = self.decoder(vq_emb, vq=True)
            return vq_code

        # --- Adaptive firing: fire -> segment aggregation -> slot quantization -> (token, dur) ---
        aux_flux = None
        if "flux" in self.firing.signals:
            aux_flux = mel_flux(wav, sr=self.sr, hop=hop)
        fire = self.firing(vq_emb, flux=aux_flux)

        if self.firing_agg is not None:
            seg = self.firing.segments(vq_emb, fire)
            g = self.firing_agg(vq_emb, seg)  # (1, D, N) segment summaries
            _, vq_code, _ = self.decoder(g, vq=True, vq_mask=seg["slot_mask"])
            ((tokens, durs),) = self.firing.extract_codes_slots(vq_code, seg)
        else:
            # point selection (no aggregation): quantize on the grid and extract firing positions
            _, vq_code, _ = self.decoder(vq_emb, vq=True)
            ((tokens, durs),) = self.firing.extract_codes(vq_code, fire)
        if tokens.dim() == 1:  # normalize the single-stage (Q=1) shape to (Q, N)
            tokens = tokens.unsqueeze(0)
        return {"codes": tokens, "durs": durs}

    @torch.inference_mode()
    def decode(
        self,
        vq_codes: torch.Tensor | EncodedAudio,
        durs: torch.Tensor | None = None,
        depth: int | None = None,
        *,
        window_frames: int | None | _Default = _DEFAULT,
        stride_frames: int | None | _Default = _DEFAULT,
        batch_size: int = 16,
        crossfade_ms: float | _Default = _DEFAULT,
    ) -> torch.Tensor:
        """Decode VQ codes to audio.

        Args:
            vq_codes: for fixed-rate checkpoints, grid codes of shape ``(B, Q, T_enc)``.
                For adaptive-firing checkpoints, codes in firing order of shape
                ``(Q, N)``, or the dict returned by ``encode`` (durs is then
                not needed)
            durs: (N,) long — length of each segment (adaptive firing only).
                Firing positions are pos_0 = 0, pos_i = sum_{j<i} dur_j, and
                sum(durs) = total grid frames
            depth: number of residual-quantizer layers to use (decode with the
                first ``depth`` layers). Must be between 1 and the number of
                supplied layers. None uses all supplied layers.
            window_frames: recompute decoding in windows of this many grid
                frames (300 = 6 s at 50 Hz). Omitted: model config.
                None explicitly decodes the full grid at once.
                Requires a fully causal decoder.
            stride_frames: new frames per window, after the first window.
                Omitted: model config. None uses one third of window_frames.
            batch_size: maximum number of windows decoded together offline.
            crossfade_ms: waveform overlap blended before each window boundary
                (model config, or 10 ms if absent; 0 disables blending).

        Returns:
            Audio tensor of shape ``(B, 1, T_audio)`` on the codec device, at
            ``self.output_sr`` Hz. For the released model, T_audio equals
            ``882 * sum(durs)`` and includes encoder hop padding.

        Raises:
            TypeError: Codes or durations do not have integer tensor dtypes.
            ValueError: Shapes, values, depth, or window settings are invalid.
        """
        if window_frames is _DEFAULT:
            window_frames = self.inference.get("decode_window_frames")
        if stride_frames is _DEFAULT:
            stride_frames = (
                self.inference.get("decode_stride_frames") if window_frames is not None else None
            )
        if crossfade_ms is _DEFAULT:
            crossfade_ms = self.inference.get("decode_crossfade_ms", 10.0)
        if window_frames is None and stride_frames is not None:
            raise ValueError("stride_frames requires window_frames")
        if window_frames is not None:
            validate_positive_integer(window_frames, "window_frames")
            if stride_frames is None:
                stride_frames = window_frames // 3
            validate_positive_integer(stride_frames, "stride_frames")
            if not 0 < stride_frames < window_frames:
                raise ValueError("Require 0 < stride_frames < window_frames")
            validate_positive_integer(batch_size, "batch_size")
            if not math.isfinite(crossfade_ms) or crossfade_ms < 0:
                raise ValueError("crossfade_ms must be finite and nonnegative")
            if self.decoder.head.istft.padding != "causal":
                raise ValueError("Windowed decoding requires a fully causal decoder")
        if isinstance(vq_codes, dict):
            if "codes" not in vq_codes or "durs" not in vq_codes:
                raise ValueError("Encoded audio must contain codes and durs")
            if durs is not None:
                raise ValueError("Pass durations either in encoded audio or as durs, not both")
            durs = vq_codes["durs"]
            vq_codes = vq_codes["codes"]
        validate_codes(
            vq_codes, durs, depth, self.decoder.quantizer, adaptive=self.firing is not None
        )
        vq_codes = vq_codes.to(device=self.device, dtype=torch.long)

        if durs is None:
            # --- fixed rate: decode grid codes directly ---
            idx = vq_codes.transpose(1, 2)  # (B, T, Q)
            if depth is not None:
                idx = idx[..., :depth]
            vq_post_emb = self.decoder.quantizer.get_output_from_indices(idx)
            # Match the decoder projection dtype.
            vq_post_emb = vq_post_emb.to(self.fc_post_a.weight.dtype)
            x = self.fc_post_a(vq_post_emb)  # (B, T, hidden)
            if window_frames is None:
                return self.decoder(x, vq=False)[0]
            return torch.cat(
                [
                    self._decode_windowed(
                        row.unsqueeze(0),
                        list(range(x.shape[1])),
                        window_frames,
                        stride_frames,
                        batch_size,
                        crossfade_ms,
                    )
                    for row in x
                ],
                dim=0,
            )

        # --- Adaptive firing: (token, dur) -> grid expansion -> compose -> decode ---
        durs = durs.to(device=self.device, dtype=torch.long)
        pos = torch.cumsum(durs, dim=0) - durs  # (N,) firing positions (pos[0]=0)
        T = int(durs.sum())

        idx = vq_codes.t().unsqueeze(0)  # (1, N, Q)
        if depth is not None:
            idx = idx[..., :depth]
        slots = self.decoder.quantizer.get_output_from_indices(idx)  # (1, N, D)
        slots = slots.to(self.fc_post_a.weight.dtype)  # Match the decoder projection dtype.

        grid_q = torch.zeros(1, slots.shape[2], T, device=self.device, dtype=slots.dtype)
        grid_q[0, :, pos] = slots[0].t()
        mask_hard = torch.zeros(1, T, dtype=torch.bool, device=self.device)
        mask_hard[0, pos] = True
        fire = {"mask": mask_hard.to(slots.dtype), "mask_hard": mask_hard}

        dec_base = self.firing.compose(grid_q, fire)  # (1, D, T)
        x = self.fc_post_a(dec_base.transpose(1, 2))  # (1, T, hidden)
        if window_frames is None:
            return self.decoder(x, vq=False)[0]
        return self._decode_windowed(
            x, pos.cpu().tolist(), window_frames, stride_frames, batch_size, crossfade_ms
        )

    def _decode_windowed(
        self,
        features: torch.Tensor,
        onsets: list[int],
        window_frames: int,
        stride_frames: int,
        batch_size: int,
        crossfade_ms: float,
    ) -> torch.Tensor:
        """Decode independent, onset-aligned windows and blend their overlap."""
        total = features.shape[1]
        if not total:
            raise ValueError("Cannot decode an empty grid")
        upsample = self.decoder.upsampler
        samples_per_frame = self.decoder.head.istft.hop_length * (
            upsample.total_upsample_factor if upsample is not None else 1
        )
        fade_samples = round(crossfade_ms * self.output_sr / 1000)
        windows = []
        cursor = 0
        while cursor < total:
            end = min(total, window_frames if cursor == 0 else cursor + stride_frames)
            index = bisect_left(onsets, max(0, end - window_frames))
            if index < len(onsets) and onsets[index] <= cursor:
                start = onsets[index]
            else:
                # Keep the owning event when a long interval exceeds the window.
                start = onsets[bisect_right(onsets, cursor) - 1]
            windows.append((start, end, cursor))
            cursor = end

        parts = []
        for offset in range(0, len(windows), batch_size):
            group = windows[offset : offset + batch_size]
            width = max(end - start for start, end, _ in group)
            inputs = torch.cat(
                [
                    F.pad(features[:, start:end], (0, 0, 0, width - (end - start)))
                    for start, end, _ in group
                ],
                dim=0,
            )
            # Right padding cannot affect valid samples in a causal decoder.
            audio = self.decoder(inputs, vq=False)[0]
            if audio.shape[0] != len(group) or audio.shape[-1] != width * samples_per_frame:
                raise RuntimeError("Decoder output does not match the window grid")
            for i, (start, end, cursor) in enumerate(group):
                begin = (cursor - start) * samples_per_frame
                current = audio[i : i + 1]
                overlap = min(fade_samples, begin, parts[-1].shape[-1]) if parts else 0
                if overlap:
                    phase = torch.linspace(
                        0, math.pi, overlap, device=audio.device, dtype=audio.dtype
                    )
                    weight = (1 - phase.cos()) * 0.5 if overlap > 1 else torch.ones_like(phase)
                    blended = (
                        parts[-1][..., -overlap:] * (1 - weight)
                        + current[..., begin - overlap : begin] * weight
                    )
                    parts[-1] = torch.cat((parts[-1][..., :-overlap], blended), dim=-1)
                parts.append(current[..., begin : (end - start) * samples_per_frame])
        return torch.cat(parts, dim=-1)

    @torch.inference_mode()
    def decode_streaming(
        self,
        vq_codes: torch.Tensor | None,
        key: str,
        finish: bool = False,
        durs: torch.Tensor | None = None,
        depth: int | None = None,
        *,
        window_frames: int | None | _Default = _DEFAULT,
        stride_frames: int | None | _Default = _DEFAULT,
        crossfade_ms: float | _Default = _DEFAULT,
    ) -> torch.Tensor | None:
        """Feed VQ codes and get audio output in streaming fashion.

        A streaming session is created automatically on the first call for a
        given *key* and cleaned up when ``finish=True``.

        Args:
            vq_codes: for fixed-rate checkpoints, grid codes of shape ``(B, Q, T)``.
                For adaptive-firing checkpoints, a chunk of codes in firing
                order of shape ``(Q, N)``. Integer indices must be within each
                layer's vocabulary; -1 skips a residual layer. An empty chunk
                or None with finish=True flushes the session.
            key: session identifier (e.g. user ID, session ID).
            finish: if ``True``, flush internal buffers and clean up the
                session after decoding this chunk.  Set this on the last call.
            durs: (n,) long — segment length per token (adaptive firing only).
                In the dur form, a token's segment length is known the moment
                the token arrives, so decoding needs no token lookahead and
                has no trailing-segment special case
            depth: Number of leading quantizer layers to decode, from 1 to
                the number of supplied layers. None uses all supplied layers.
                Must stay fixed within a session.
            window_frames: history window in grid frames. Omitted: the active
                session's setting, or model config for a new session. None
                disables periodic decoder resets.
            stride_frames: interval between history resets, not an output chunk
                size. Omitted: model config; None: one third of window_frames.
            crossfade_ms: audio held for blending at history resets. Omitted:
                model config. Settings and depth stay fixed within a session.

        Returns:
            Audio chunk ``(B, 1, T_audio)`` if audio is available,
            or ``None`` while internal buffers are still filling.
        """
        validate_codes(
            vq_codes,
            durs,
            depth,
            self.decoder.quantizer,
            adaptive=self.firing is not None,
            allow_empty=True,
        )
        active = self._streaming_states.get(key)
        if active is not None and active.get("depth") != depth:
            raise ValueError("depth must stay fixed within a session")
        if vq_codes is not None:
            vq_codes = vq_codes.to(device=self.device, dtype=torch.long)
        saved = active.get("window_settings") if active is not None else None
        if window_frames is _DEFAULT:
            # Omitted settings reuse those of an active session (None for a
            # session started without windowing) and the model config otherwise.
            if saved:
                window_frames = saved[0]
            elif active is not None:
                window_frames = None
            else:
                window_frames = self.inference.get("decode_window_frames")
        if stride_frames is _DEFAULT:
            stride_frames = (
                (saved[1] if saved else self.inference.get("decode_stride_frames"))
                if window_frames is not None
                else None
            )
        if crossfade_ms is _DEFAULT:
            crossfade_ms = saved[2] if saved else self.inference.get("decode_crossfade_ms", 10.0)
        if window_frames is not None:
            validate_positive_integer(window_frames, "window_frames")
            if stride_frames is None:
                stride_frames = window_frames // 3
            validate_positive_integer(stride_frames, "stride_frames")
            if not 0 < stride_frames < window_frames:
                raise ValueError("Require 0 < stride_frames < window_frames")
            if not math.isfinite(crossfade_ms) or crossfade_ms < 0:
                raise ValueError("crossfade_ms must be finite and nonnegative")
            if self.decoder.head.istft.padding != "causal":
                raise ValueError("Windowed streaming requires a fully causal decoder")
            settings = (window_frames, stride_frames, crossfade_ms)
            if active is not None and (saved != settings or active["depth"] != depth):
                raise ValueError("Decoding settings must stay fixed within a session")
            return self._decode_streaming_windowed(vq_codes, durs, key, finish, depth, settings)
        if saved is not None:
            raise ValueError("Cannot disable windowing in an active session")
        if stride_frames is not None:
            raise ValueError("stride_frames requires window_frames")
        if (vq_codes is None or not vq_codes.numel()) and active is None:
            return None
        if self.firing is not None:
            return self._decode_streaming_firing(
                vq_codes,
                key,
                finish=finish,
                durs=durs,
                depth=depth,
            )

        if vq_codes is None or not vq_codes.numel():
            if finish:
                audio = self.decoder.flush_streaming(self._streaming_states.pop(key))
                return audio if audio.shape[-1] else None
            return None
        if depth is not None:
            vq_codes = vq_codes[:, :depth]
        if key not in self._streaming_states:
            self._streaming_states[key] = {
                "backbone_state": None,
                "upsampler_state": None,
                "istft_state": None,
            }
        state = self._streaming_states[key]
        audio_chunk, self._streaming_states[key] = self.decoder.decode_streaming(
            vq_codes, self.fc_post_a, state=state
        )

        self._streaming_states[key]["depth"] = depth
        parts = []
        if audio_chunk.shape[2] > 0:
            parts.append(audio_chunk)

        if finish:
            audio_flush = self.decoder.flush_streaming(self._streaming_states.pop(key))
            if audio_flush.shape[2] > 0:
                parts.append(audio_flush)

        if parts:
            return torch.cat(parts, dim=2)
        return None

    def _stream_projected_grid(self, features: torch.Tensor, state: dict) -> torch.Tensor:
        """Advance a causal decoder once, retaining convolution and attention state."""
        h, state["backbone"] = self.decoder.backbone.forward_streaming(
            features, state.get("backbone")
        )
        if self.decoder.upsampler is not None:
            h, state["upsampler"] = self.decoder.upsampler.forward_streaming(
                h.transpose(1, 2), state.get("upsampler")
            )
        audio, state["head"] = self.decoder.head.streaming_forward(h, state.get("head"))
        return audio

    def _decode_streaming_windowed(self, codes, durs, key, finish, depth, settings):
        """Incremental decoding with periodic, onset-aligned history recomputation."""
        window, stride, fade_ms = settings
        state = self._streaming_states.get(key)
        if codes is None or codes.numel() == 0:
            if finish and state is not None:
                self._streaming_states.pop(key)
                return state["pending"]
            return None

        if self.firing is not None:
            durations = durs.to(device=self.device, dtype=torch.long)
            idx = codes.to(self.device).t().unsqueeze(0)
            if depth is not None:
                idx = idx[..., :depth]
            slots = self.decoder.quantizer.get_output_from_indices(idx).to(
                self.fc_post_a.weight.dtype
            )
            positions = durations.cumsum(0) - durations
            total = int(durations.sum())
            grid = slots.new_zeros(1, slots.shape[-1], total)
            grid[0, :, positions] = slots[0].t()
            mask = torch.zeros(1, total, device=grid.device, dtype=torch.bool)
            mask[0, positions] = True
            frames = self.firing.compose(grid, {"mask": mask.to(grid.dtype), "mask_hard": mask})
            features = self.fc_post_a(frames.transpose(1, 2))
            onsets = positions.cpu().tolist()
        else:
            idx = codes.to(self.device).transpose(1, 2)
            if depth is not None:
                idx = idx[..., :depth]
            slots = self.decoder.quantizer.get_output_from_indices(idx).to(
                self.fc_post_a.weight.dtype
            )
            features = self.fc_post_a(slots)
            onsets = list(range(features.shape[1]))

        upsample = self.decoder.upsampler
        samples = self.decoder.head.istft.hop_length * (
            upsample.total_upsample_factor if upsample is not None else 1
        )
        fade = round(fade_ms * self.output_sr / 1000)
        if state is None:
            state = {
                "window_settings": settings,
                "depth": depth,
                "received": 0,
                "cursor": 0,
                "origin": 0,
                "end": window,
                "last_reset": 0,
                "onsets": [],
                "grid": None,
                "decoder": {},
                "pending": None,
            }
            self._streaming_states[key] = state
        elif features.shape[0] != state["grid"].shape[0]:
            raise ValueError("Batch size must stay fixed within a session")
        state["onsets"].extend(state["received"] + p for p in onsets)
        state["received"] += features.shape[1]
        state["grid"] = (
            features if state["grid"] is None else torch.cat((state["grid"], features), dim=1)
        )
        parts = []
        while state["cursor"] < state["received"]:
            cursor = state["cursor"]
            if cursor == state["end"]:
                state["end"] += stride
                earliest = max(0, state["end"] - window)
                index = bisect_left(state["onsets"], earliest)
                if index < len(state["onsets"]) and state["onsets"][index] <= cursor:
                    start = state["onsets"][index]
                else:
                    start = state["onsets"][bisect_right(state["onsets"], cursor) - 1]
                state["grid"] = state["grid"][:, start - state["origin"] :].clone()
                state["origin"] = start
                state["onsets"] = state["onsets"][bisect_left(state["onsets"], start) :]
                state["decoder"] = {}
                prefix = state["grid"][:, : cursor - start]
                pending = state["pending"]
                if prefix.shape[1]:
                    warm = self._stream_projected_grid(prefix, state["decoder"])
                    if warm.shape[-1] != prefix.shape[1] * samples:
                        raise RuntimeError("Causal decoder did not emit the complete history")
                    overlap = min(
                        fade,
                        warm.shape[-1],
                        pending.shape[-1] if pending is not None else 0,
                        (cursor - state["last_reset"]) * samples,
                    )
                    if overlap:
                        phase = torch.linspace(
                            0, math.pi, overlap, device=warm.device, dtype=warm.dtype
                        )
                        weight = (1 - phase.cos()) * 0.5 if overlap > 1 else torch.ones_like(phase)
                        blended = (
                            pending[..., -overlap:] * (1 - weight) + warm[..., -overlap:] * weight
                        )
                        pending = torch.cat((pending[..., :-overlap], blended), dim=-1)
                if pending is not None and pending.shape[-1]:
                    parts.append(pending)
                state["pending"] = None
                state["last_reset"] = cursor
            end = min(state["received"], state["end"])
            x = state["grid"][:, cursor - state["origin"] : end - state["origin"]]
            audio = self._stream_projected_grid(x, state["decoder"])
            if audio.shape[-1] != (end - cursor) * samples:
                raise RuntimeError("Causal decoder did not emit the complete increment")
            if state["pending"] is not None:
                audio = torch.cat((state["pending"], audio), dim=-1)
            until_reset = (state["end"] - end) * samples
            hold = min(max(0, fade - until_reset), audio.shape[-1])
            ready = audio.shape[-1] - hold
            if ready:
                parts.append(audio[..., :ready])
            state["pending"] = audio[..., ready:].clone() if hold else None
            state["cursor"] = end
        if finish:
            pending = self._streaming_states.pop(key)["pending"]
            if pending is not None and pending.shape[-1]:
                parts.append(pending)
        return torch.cat(parts, dim=-1) if parts else None

    def _decode_streaming_firing(
        self,
        codes: torch.Tensor,
        key: str,
        finish: bool,
        durs: torch.Tensor | None,
        depth: int | None,
    ) -> torch.Tensor | None:
        """Streaming decode of adaptive-firing codes (numerically matches batch
        ``decode(window_frames=None)``, processed chunk by chunk).

        (token, dur) -> compose_stream incrementally synthesizes grid frames ->
        run through the streaming forward of backbone/upsampler/head.
        """
        if key not in self._streaming_states:
            st = {
                "compose_state": None,
                "backbone_state": None,
                "upsampler_state": None,
                "head_state": None,
                "depth": depth,
            }
            self._streaming_states[key] = st
        state = self._streaming_states[key]

        # (token, dur) chunk -> quantizer indices
        if codes is not None and codes.numel() > 0:
            idx = codes.to(self.device).t().unsqueeze(0)  # (1, n, Q)
            if depth is not None:
                idx = idx[..., :depth]
            durs = durs.to(self.device).long()
        else:
            # flush-only call (finish=True drains the buffers only)
            idx = torch.zeros(1, 0, 1, dtype=torch.long, device=self.device)
            durs = torch.zeros(0, dtype=torch.long, device=self.device)

        # indices -> slot vectors -> grid frame composition
        if idx.shape[1] > 0:
            slots = self.decoder.quantizer.get_output_from_indices(idx)[0]  # (n, D)
            slots = slots.to(self.fc_post_a.weight.dtype)  # Match the decoder projection dtype.
        else:
            slots = torch.zeros(0, 1, device=self.device)
        frames, state["compose_state"] = self.firing.compose_stream(
            slots, durs, state["compose_state"]
        )

        parts = []
        x = self.fc_post_a(frames)  # (1, T_chunk, hidden)
        h, state["backbone_state"] = self.decoder.backbone.forward_streaming(
            x, state["backbone_state"], flush=finish
        )
        if h.shape[1] > 0:
            if self.decoder.upsampler is not None:
                u, state["upsampler_state"] = self.decoder.upsampler.forward_streaming(
                    h.transpose(1, 2), state["upsampler_state"], flush=finish
                )
            else:
                u = h
            if u.shape[1] > 0:
                a, state["head_state"] = self.decoder.head.streaming_forward(u, state["head_state"])
                if a.shape[2] > 0:
                    parts.append(a)

        if finish:
            st = self._streaming_states.pop(key)
            a = self.decoder.head.flush_streaming(st["head_state"])
            if a.shape[2] > 0:
                parts.append(a)

        if parts:
            return torch.cat(parts, dim=2)
        return None
