"""DNSMOS (Deep Noise Suppression Mean Opinion Score) metric via torchmetrics."""

import soundfile as sf
import torch
import torchaudio.functional as F_audio
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

from .cpu import usable_cpu_count

EVAL_SR = 16000


def compute_dnsmos(audio_paths, device="cpu"):
    """Compute DNSMOS for each audio file.

    Returns per-sample [p808_mos, mos_sig, mos_bak, mos_ovr].

    Args:
        audio_paths: List of paths to audio files.
        device: Torch device for ONNX inference.

    Returns:
        List of dicts with 'p808_mos', 'sig', 'bak', 'ovr' keys.
    """
    # onnxruntime defaults to one thread per physical core of the machine. When the
    # process may use fewer CPUs (e.g. under a container or job-scheduler CPU limit),
    # the extra threads slow inference down, so match the usable CPUs (scores unchanged).
    num_threads = usable_cpu_count()

    # torchmetrics passes device.index as the CUDAExecutionProvider device_id, so
    # normalize an index-less "cuda" to "cuda:0" (requires onnxruntime-gpu; with a
    # CPU-only build torchmetrics falls back to CPU)
    if isinstance(device, str) and device == "cuda":
        device = "cuda:0"

    import onnxruntime as ort

    providers = ort.get_available_providers()
    print(f"  DNSMOS providers available: {providers}")

    # ORT silently falls back to the CPU EP when the CUDA EP is unavailable
    # (e.g. a CPU-only onnxruntime build), so a successful inference probe alone
    # is not evidence that the GPU is used: require the provider to be present.
    if isinstance(device, str) and device.startswith("cuda"):
        if "CUDAExecutionProvider" not in providers:
            print("  DNSMOS: CUDAExecutionProvider not available, using CPU EP")
            device = "cpu"
        else:
            # Runtime probe with 1 second of silence: some GPU / onnxruntime
            # build combinations pass the provider check but still fail at
            # inference time (e.g. no kernel image for the device), so fall
            # back to the CPU EP on failure.
            try:
                deep_noise_suppression_mean_opinion_score(
                    torch.zeros(EVAL_SR),
                    fs=EVAL_SR,
                    personalized=False,
                    device=device,
                    num_threads=num_threads,
                )
                print(f"  DNSMOS: CUDA EP ok on {device}")
            except Exception as e:
                print(f"  DNSMOS: CUDA EP failed ({type(e).__name__}), fallback to CPU")
                device = "cpu"

    results = []
    for path in audio_paths:
        wav, sr = sf.read(path, dtype="float32")
        wav_tensor = torch.from_numpy(wav)
        if sr != EVAL_SR:
            wav_tensor = F_audio.resample(wav_tensor.unsqueeze(0), sr, EVAL_SR).squeeze(0)

        scores = deep_noise_suppression_mean_opinion_score(
            wav_tensor,
            fs=EVAL_SR,
            personalized=False,
            device=device,
            num_threads=num_threads,
        )
        # scores shape: (4,) -> [p808_mos, mos_sig, mos_bak, mos_ovr]
        results.append(
            {
                "p808_mos": float(scores[0]),
                "sig": float(scores[1]),
                "bak": float(scores[2]),
                "ovr": float(scores[3]),
            }
        )

    return results
