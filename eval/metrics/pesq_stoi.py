"""PESQ (wideband and narrowband) and STOI metrics for audio quality evaluation.

Both signals are resampled to 16 kHz and trimmed to their common length.
Pairs are computed in parallel worker processes; each pair is independent, so
results are identical to sequential computation.
"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import soundfile as sf

from .cpu import usable_cpu_count

EVAL_SR = 16000


def _load_and_resample(path, target_sr=EVAL_SR):
    """Load audio file and resample to target_sr if needed."""
    import torch
    import torchaudio.functional as F_audio

    wav, sr = sf.read(path, dtype="float32")
    if sr != target_sr:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        wav_tensor = F_audio.resample(wav_tensor, sr, target_sr)
        wav = wav_tensor.squeeze(0).numpy()
    return wav


def _align_lengths(ref, deg):
    """Trim both signals to the shorter length."""
    min_len = min(len(ref), len(deg))
    return ref[:min_len], deg[:min_len]


def _init_worker():
    import torch

    torch.set_num_threads(1)


def _pair_worker(paths):
    from pesq import pesq as pesq_fn
    from pystoi import stoi as stoi_fn

    orig_path, recon_path = paths
    ref = _load_and_resample(orig_path, EVAL_SR)
    deg = _load_and_resample(recon_path, EVAL_SR)
    ref, deg = _align_lengths(ref, deg)

    result = {"pesq": float("nan"), "pesq_nb": float("nan"), "stoi": float("nan")}

    # PESQ requires at least 0.5s of audio
    try:
        result["pesq"] = float(pesq_fn(EVAL_SR, ref, deg, "wb"))
    except Exception:
        pass

    # narrowband PESQ (some external papers report NB)
    try:
        result["pesq_nb"] = float(pesq_fn(EVAL_SR, ref, deg, "nb"))
    except Exception:
        pass

    try:
        result["stoi"] = float(stoi_fn(ref, deg, EVAL_SR, extended=False))
    except Exception:
        pass

    return result


def compute_pesq_stoi(original_paths, reconstructed_paths, num_workers=None):
    """Compute PESQ (wideband/narrowband) and STOI for each original/reconstructed pair.

    Args:
        original_paths: List of paths to original wav files.
        reconstructed_paths: List of paths to reconstructed wav files (any SR,
            resampled to 16kHz).
        num_workers: Worker processes (default: min(16, available cores)).

    Returns:
        List of dicts with 'pesq', 'pesq_nb' and 'stoi' keys. Values are float
        or NaN on failure.
    """
    pairs = list(zip(original_paths, reconstructed_paths, strict=True))
    if num_workers is None:
        num_workers = min(16, usable_cpu_count())
    num_workers = max(1, min(num_workers, len(pairs)))

    results = []
    # spawn: workers never touch CUDA and stay safe even if the parent
    # initialized it in an earlier metric.
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=get_context("spawn"),
        initializer=_init_worker,
    ) as ex:
        for i, r in enumerate(ex.map(_pair_worker, pairs, chunksize=8)):
            results.append(r)
            if (i + 1) % 200 == 0 or (i + 1) == len(pairs):
                print(f"  PESQ/STOI: {i + 1}/{len(pairs)}")
    return results
