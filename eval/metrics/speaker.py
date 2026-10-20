"""Speaker similarity metric using SpeechBrain speaker verification models."""

from functools import partial

import torchaudio

# Monkeypatch for SpeechBrain compatibility with torchaudio >= 2.9
# See: https://github.com/speechbrain/speechbrain/issues/3012
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: [""]

import torch
import torch.nn.functional as F
import torchaudio.functional as F_audio

from .spk_emb_cache import EmbeddingCache

EVAL_SR = 16000


def compute_speaker_similarity(
    original_paths,
    reconstructed_paths,
    model_source="speechbrain/spkrec-ecapa-voxceleb",
    device="cpu",
    cache_dir=None,
):
    """Compute cosine similarity of speaker embeddings between original and reconstructed audio.

    Args:
        original_paths: List of paths to original wav files.
        reconstructed_paths: List of paths to reconstructed wav files.
        model_source: SpeechBrain model source identifier.
        device: Torch device.
        cache_dir: Optional directory for embedding reuse across runs (embeddings
            are keyed by audio file content and the speaker model; original-side
            embeddings are shared by every arm of the same set).

    Returns:
        List of dicts with 'speaker_similarity' key (cosine similarity, -1 to 1).
    """
    from speechbrain.inference.speaker import EncoderClassifier

    classifier = EncoderClassifier.from_hparams(
        source=model_source,
        run_opts={"device": device},
    )
    cache = EmbeddingCache(cache_dir, "ecapa", model_source)

    results = []
    n = len(original_paths)
    for i, (orig_path, recon_path) in enumerate(
        zip(original_paths, reconstructed_paths, strict=True)
    ):
        emb_orig = cache.get_or_compute(
            orig_path, partial(_get_embedding, classifier, orig_path, device)
        ).to(device)
        emb_recon = cache.get_or_compute(
            recon_path, partial(_get_embedding, classifier, recon_path, device)
        ).to(device)

        similarity = F.cosine_similarity(emb_orig, emb_recon, dim=-1)
        results.append({"speaker_similarity": float(similarity.item())})
        if (i + 1) % 200 == 0 or (i + 1) == n:
            print(f"  Speaker sim (ECAPA): {i + 1}/{n}")

    del classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _get_embedding(classifier, audio_path, device):
    """Extract speaker embedding from audio file."""
    wav, sr = torchaudio.load(audio_path)
    if sr != EVAL_SR:
        wav = F_audio.resample(wav, sr, EVAL_SR)
    wav = wav.to(device)
    embedding = classifier.encode_batch(wav)
    return embedding.squeeze()
