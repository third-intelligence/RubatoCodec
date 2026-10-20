"""Multi-resolution mel spectrogram distance metric."""

import soundfile as sf
import torch
import torchaudio.functional as F_audio

from training.criterions.mel_loss import MultiResolutionMelSpectrogramLoss

EVAL_SR = 16000


def _load_and_resample(path, target_sr):
    """Load audio file and resample to target_sr if needed."""
    wav, sr = sf.read(path, dtype="float32")
    if sr != target_sr:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        wav_tensor = F_audio.resample(wav_tensor, sr, target_sr)
        wav = wav_tensor.squeeze(0).numpy()
    return wav


def compute_mel_distance(original_paths, reconstructed_paths, sample_rate=EVAL_SR, device="cpu"):
    """Compute multi-resolution mel spectrogram L1 distance.

    Args:
        original_paths: List of paths to original wav files.
        reconstructed_paths: List of paths to reconstructed wav files.
        sample_rate: Sample rate for mel computation (both signals resampled to this).
        device: Torch device.

    Returns:
        List of dicts with 'mel_distance' key.
    """
    mel_loss_fn = MultiResolutionMelSpectrogramLoss(sample_rate=sample_rate).to(device)
    mel_loss_fn.eval()

    results = []
    for orig_path, recon_path in zip(original_paths, reconstructed_paths, strict=True):
        ref = _load_and_resample(orig_path, sample_rate)
        deg = _load_and_resample(recon_path, sample_rate)

        # Align lengths
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]

        ref_tensor = torch.from_numpy(ref).unsqueeze(0).to(device)
        deg_tensor = torch.from_numpy(deg).unsqueeze(0).to(device)

        with torch.no_grad():
            loss = mel_loss_fn(deg_tensor, ref_tensor)

        results.append({"mel_distance": float(loss.item())})

    return results
