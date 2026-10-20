"""Reconstruct a speech file with RubatoCodec.

Run from the repository root with ``python -m examples.reconstruct input.wav output.wav``.
"""

from argparse import ArgumentParser
from pathlib import Path

import soundfile as sf
import torch

from rubatocodec import RubatoCodec


def reconstruct_audio(
    codec: RubatoCodec,
    wav: torch.Tensor,
    sr: int,
    *,
    depth: int | None = None,
    streaming: bool = False,
) -> torch.Tensor:
    """Encode mono audio and return its offline or streaming reconstruction.

    Args:
        codec: Fully loaded codec (encoding components are required).
        wav: Floating-point mono waveform shaped (samples,).
        sr: Input sample rate in Hz.
        depth: Number of leading quantizer layers to decode, or None for all.
        streaming: Decode one event at a time with the model's default settings.

    Returns:
        Waveform shaped (1, 1, samples) at codec.output_sr on the codec device.
    """
    encoded = codec.encode(wav, sr=sr)
    if streaming:
        if not isinstance(encoded, dict):
            raise ValueError("Streaming reconstruction requires an adaptive-firing checkpoint")
        codes, durs = encoded["codes"], encoded["durs"]
        chunks = []
        for i in range(durs.numel()):
            chunk = codec.decode_streaming(
                codes[:, i : i + 1],
                "reconstruction",
                durs=durs[i : i + 1],
                depth=depth,
                finish=i == durs.numel() - 1,
            )
            if chunk is not None:
                chunks.append(chunk)
        audio = torch.cat(chunks, dim=-1)
    else:
        audio = codec.decode(encoded, depth=depth)

    return audio


def main() -> None:
    """Load mono audio, encode it, and save its reconstruction at the output rate."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input audio file (stereo is mixed to mono)")
    parser.add_argument("output", type=Path, help="Reconstructed WAV file")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, help="Local .safetensors or .ckpt file")
    source.add_argument("--model", default="third-intelligence/RubatoCodec", help="Hub model ID")
    parser.add_argument("--device", default="cuda", help="Inference device, e.g. cuda or cpu")
    parser.add_argument("--depth", type=int, default=None, help="Number of quantizer layers")
    parser.add_argument(
        "--streaming", action="store_true", help="Decode one event at a time (adaptive models)"
    )
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output paths must differ")
    data, sr = sf.read(args.input, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.mean(axis=1))
    codec = (
        RubatoCodec.from_checkpoint(args.checkpoint, device=args.device)
        if args.checkpoint is not None
        else RubatoCodec.from_pretrained(args.model, device=args.device)
    )
    audio = reconstruct_audio(codec, wav, sr, depth=args.depth, streaming=args.streaming)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio[0, 0].cpu().numpy(), codec.output_sr)
    print(f"Saved {args.output} ({codec.output_sr} Hz, {audio.shape[-1] / codec.output_sr:.2f} s)")


if __name__ == "__main__":
    main()
