"""Generate original/reconstructed audio pairs and metadata.json for evaluation.

Run ``python -m examples.reconstruct_dataset manifest.json eval_output/run1``.
The manifest contains a nonempty samples list with audio paths and optional text/idx.
Relative input paths are resolved against the manifest directory.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import soundfile as sf
import torch

from examples.reconstruct import reconstruct_audio
from rubatocodec import RubatoCodec


def load_manifest(path: Path) -> list[tuple[Path, int | str, str | None]]:
    """Validate sample IDs and paths before loading the codec or writing audio."""
    with path.open(encoding="utf-8") as f:
        document = json.load(f)
    samples = document.get("samples") if isinstance(document, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest must contain a nonempty samples list")
    entries = []
    seen = set()
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict) or not isinstance(sample.get("audio"), str):
            raise ValueError(f"Sample {i} must contain an audio path string")
        audio = Path(sample["audio"])
        if not audio.is_absolute():
            audio = path.parent / audio
        if not audio.is_file():
            raise ValueError(f"Sample {i} audio file does not exist: {audio}")
        idx = sample.get("idx", i)
        if isinstance(idx, bool) or not isinstance(idx, (int, str)) or idx in seen:
            raise ValueError(f"Sample {i} idx must be a unique integer or string")
        text = sample.get("text")
        if text is not None and not isinstance(text, str):
            raise ValueError(f"Sample {i} text must be a string or null")
        seen.add(idx)
        entries.append((audio, idx, text))
    return entries


def main() -> None:
    """Reconstruct a manifest into a portable directory accepted by eval.evaluate."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON manifest with samples[].audio")
    parser.add_argument("output", type=Path, help="New or empty evaluation output directory")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, help="Local .safetensors or .ckpt file")
    source.add_argument("--model", default="third-intelligence/RubatoCodec", help="Hub model ID")
    parser.add_argument("--device", default="cuda", help="Inference device, e.g. cuda or cpu")
    parser.add_argument("--depth", type=int, default=None, help="Number of quantizer layers")
    parser.add_argument("--streaming", action="store_true", help="Decode one event at a time")
    args = parser.parse_args()

    try:
        entries = load_manifest(args.manifest)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Output must be a new or empty directory")
    codec = (
        RubatoCodec.from_checkpoint(args.checkpoint, device=args.device)
        if args.checkpoint is not None
        else RubatoCodec.from_pretrained(args.model, device=args.device)
    )
    depth = codec.decoder.quantizer.num_quantizers if args.depth is None else args.depth
    if not 1 <= depth <= codec.decoder.quantizer.num_quantizers:
        parser.error(f"--depth must be between 1 and {codec.decoder.quantizer.num_quantizers}")
    if args.streaming and codec.firing is None:
        parser.error("--streaming requires an adaptive-firing checkpoint")

    (args.output / "original").mkdir(parents=True, exist_ok=True)
    (args.output / "reconstructed").mkdir(exist_ok=True)
    metadata = {
        "config": {
            "model": str(args.checkpoint) if args.checkpoint is not None else args.model,
            "depth": depth,
            "streaming": args.streaming,
            "encoder_sample_rate": codec.sr,
            "output_sample_rate": codec.output_sr,
            "inference": codec.inference,
        },
        "samples": [],
    }
    for i, (path, idx, text) in enumerate(entries):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.mean(axis=1))
        audio = reconstruct_audio(codec, wav, sr, depth=depth, streaming=args.streaming)
        original = f"original/{i:06d}.wav"
        reconstructed = f"reconstructed/{i:06d}.wav"
        # Float WAV avoids clipping/PCM quantization in subsequent quality metrics.
        sf.write(args.output / original, wav.numpy(), sr, subtype="FLOAT")
        sf.write(
            args.output / reconstructed,
            audio[0, 0].cpu().numpy(),
            codec.output_sr,
            subtype="FLOAT",
        )
        sample = {"idx": idx, "original": original, "reconstructed": reconstructed}
        if text is not None:
            sample["text"] = text
        metadata["samples"].append(sample)
        print(f"[{i + 1}/{len(entries)}] {path.name}")

    # Only publish metadata after all pairs were written successfully.
    temporary = args.output / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output / "metadata.json")
    print(f"Saved {args.output / 'metadata.json'}")


if __name__ == "__main__":
    main()
