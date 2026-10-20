"""UTMOS and UTMOSv2 speech quality estimation metrics."""

from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as F_audio

from .cpu import usable_cpu_count

EVAL_SR = 16000


def compute_utmosv2(audio_paths, device="cpu", batch_size=32):
    """Compute UTMOSv2 predicted MOS for each audio file.

    When all paths live in a single directory, use utmosv2's input_dir batch inference
    (DataLoader with batch_size/num_workers); the per-file predict(data=...) fallback
    processes files sequentially and is slower. Preprocessing
    (remove_silent_section etc.) goes through the same cfg on both paths, so they are
    equivalent.

    Args:
        audio_paths: List of paths to audio files.
        device: Torch device.
        batch_size: DataLoader batch size for input_dir mode.

    Returns:
        List of dicts with 'utmosv2' key.
    """
    import utmosv2

    model = utmosv2.create_model(pretrained=True)
    num_workers = usable_cpu_count()

    dirs = {Path(p).parent for p in audio_paths}
    results = None
    if len(dirs) == 1 and all(Path(p).suffix == ".wav" for p in audio_paths):
        preds = model.predict(
            input_dir=str(dirs.pop()),
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            verbose=True,
        )
        by_name = {Path(d["file_path"]).name: d["predicted_mos"] for d in preds}
        if all(Path(p).name in by_name for p in audio_paths):
            results = [{"utmosv2": float(by_name[Path(p).name])} for p in audio_paths]
        else:
            print("utmosv2: input_dir results are missing some paths, falling back to per-file")

    if results is None:
        results = []
        for path in audio_paths:
            wav, sr = sf.read(path, dtype="float32")
            wav_tensor = torch.from_numpy(wav)
            if sr != EVAL_SR:
                wav_tensor = F_audio.resample(wav_tensor.unsqueeze(0), sr, EVAL_SR).squeeze(0)
            mos = model.predict(data=wav_tensor, sr=EVAL_SR, device=device, num_workers=0)
            if isinstance(mos, torch.Tensor):
                mos = mos.item()
            results.append({"utmosv2": float(mos)})

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def compute_utmos(audio_paths, device="cpu"):
    """Compute UTMOS predicted MOS for each audio file.

    Returns:
        List of dicts with 'utmos' key.
    """
    model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    model = model.to(device).eval()

    results = []
    with torch.inference_mode():
        for path in audio_paths:
            wav, sr = sf.read(path, dtype="float32")
            wav_tensor = torch.from_numpy(wav)
            if wav_tensor.ndim > 1:
                wav_tensor = wav_tensor.mean(dim=-1)
            if sr != EVAL_SR:
                wav_tensor = F_audio.resample(wav_tensor.unsqueeze(0), sr, EVAL_SR).squeeze(0)
            score = model(wav_tensor.unsqueeze(0).to(device), EVAL_SR)
            results.append({"utmos": float(score.item())})

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results
