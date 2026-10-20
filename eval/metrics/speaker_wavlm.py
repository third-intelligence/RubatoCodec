"""Speaker similarity metric using the WavLM-Large SV model (UniSpeech finetune).

The model definition is loaded at runtime from a local checkout of
https://github.com/microsoft/UniSpeech (``--unispeech-dir``). Also requires
the ``wavlm_large_finetune.pth`` checkpoint distributed there and the
``s3prl`` package (the WavLM-Large upstream weights are downloaded/cached
under ``~/.cache/s3prl`` on first use).
"""

import importlib.util
import os
import sys
import types
from functools import partial

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio

from .spk_emb_cache import EmbeddingCache, file_sha1

EVAL_SR = 16000

_ECAPA_REL_PATH = os.path.join("downstreams", "speaker_verification", "models", "ecapa_tdnn.py")


def _load_s3prl_upstream(feat_type):
    """Build an s3prl upstream (e.g. "wavlm_large") without network access."""
    # s3prl (<= 0.4.18) calls the removed torchaudio.set_audio_backend at
    # import time in some upstream modules.
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None

    if feat_type.startswith("wavlm"):
        # Import only the WavLM hubconf: `s3prl.hub` would import every
        # upstream, some of which fail on current torchaudio.
        from s3prl.upstream.wavlm import hubconf as wavlm_hubconf

        return getattr(wavlm_hubconf, feat_type)()
    raise ValueError(f"unsupported s3prl upstream: {feat_type}")


def _import_ecapa_module(unispeech_dir):
    """Import UniSpeech's ecapa_tdnn.py from a local checkout."""
    path = os.path.join(unispeech_dir, _ECAPA_REL_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Pass --unispeech-dir pointing to a checkout of "
            "https://github.com/microsoft/UniSpeech"
        )
    # ecapa_tdnn.py imports UpstreamExpert from the sibling utils.py, which
    # requires fairseq. UpstreamExpert is only used for non-s3prl features, so
    # load the file as a package module with a stub utils instead.
    package = "unispeech_sv_models"
    if package not in sys.modules:
        pkg = types.ModuleType(package)
        pkg.__path__ = [os.path.dirname(path)]
        utils = types.ModuleType(f"{package}.utils")

        class UpstreamExpert:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("only s3prl upstreams such as wavlm_large are supported")

        utils.UpstreamExpert = UpstreamExpert
        sys.modules[package] = pkg
        sys.modules[f"{package}.utils"] = utils
    spec = importlib.util.spec_from_file_location(f"{package}.ecapa_tdnn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_model(unispeech_dir, checkpoint_path):
    module = _import_ecapa_module(unispeech_dir)

    # UniSpeech's ECAPA_TDNN builds its upstream via torch.hub.load, which
    # probes GitHub on every call. Redirect s3prl loads to the installed
    # package while the model is constructed.
    orig_hub_load = torch.hub.load

    def _patched_hub_load(repo, model, *args, **kwargs):
        if isinstance(repo, str) and repo.startswith("s3prl/s3prl"):
            return _load_s3prl_upstream(model)
        return orig_hub_load(repo, model, *args, **kwargs)

    torch.hub.load = _patched_hub_load
    try:
        model = module.ECAPA_TDNN_SMALL(
            feat_dim=1024, emb_dim=256, feat_type="wavlm_large", config_path=None
        )
    finally:
        torch.hub.load = orig_hub_load

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # strict=False mirrors the original UniSpeech loader (the checkpoint lacks
    # a few frozen upstream buffers).
    model.load_state_dict(state_dict["model"], strict=False)
    return model


def compute_speaker_similarity_wavlm(
    original_paths,
    reconstructed_paths,
    checkpoint_path,
    unispeech_dir,
    device="cpu",
    cache_dir=None,
):
    """Compute cosine similarity of WavLM-SV speaker embeddings between pairs.

    Args:
        original_paths: List of paths to original wav files.
        reconstructed_paths: List of paths to reconstructed wav files.
        checkpoint_path: Path to wavlm_large_finetune.pth.
        unispeech_dir: Path to a local checkout of microsoft/UniSpeech.
        device: Torch device.
        cache_dir: Optional directory for embedding reuse across runs (embeddings
            are keyed by audio file content and the speaker model; original-side
            embeddings are shared by every arm of the same set).

    Returns:
        List of dicts with 'speaker_similarity_wavlm' key (cosine similarity, -1 to 1).
    """
    model = _build_model(unispeech_dir, checkpoint_path)
    model = model.to(device)
    model.eval()
    # Key the cache on the checkpoint content so a different checkpoint never
    # reuses embeddings computed with another one.
    cache = EmbeddingCache(cache_dir, "wavlm_sv", file_sha1(checkpoint_path) if cache_dir else None)

    results = []
    n = len(original_paths)
    with torch.no_grad():
        for i, (orig_path, recon_path) in enumerate(
            zip(original_paths, reconstructed_paths, strict=True)
        ):
            emb_orig = cache.get_or_compute(
                orig_path, partial(_get_embedding, model, orig_path, device)
            ).to(device)
            emb_recon = cache.get_or_compute(
                recon_path, partial(_get_embedding, model, recon_path, device)
            ).to(device)

            similarity = F.cosine_similarity(emb_orig, emb_recon, dim=-1)
            results.append({"speaker_similarity_wavlm": float(similarity.item())})
            if (i + 1) % 200 == 0 or (i + 1) == n:
                print(f"  Speaker sim (WavLM-SV): {i + 1}/{n}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _get_embedding(model, audio_path, device):
    """Extract a speaker embedding from an audio file (first channel, 16 kHz)."""
    wav, sr = torchaudio.load(audio_path)
    wav = wav[:1]  # first channel, as in the UniSpeech verification script
    if sr != EVAL_SR:
        wav = F_audio.resample(wav, sr, EVAL_SR)
    wav = wav.to(device)
    embedding = model(wav)
    return embedding.squeeze(0)
