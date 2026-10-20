"""Convert a PyTorch Lightning checkpoint to safetensors + config.json.

Strips discriminators, optimizers, loss models, and dataset/training config,
keeping only the weights and config needed for inference.

Usage::

    python -m scripts.convert_to_safetensors --ckpt path/to/last.ckpt --output-dir output/
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from safetensors.torch import save_file

from rubatocodec.checkpoint import clean_inference_config, load_lightning_checkpoint


def convert(ckpt_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    config, inference_sd = load_lightning_checkpoint(ckpt_path)
    total_params = sum(v.numel() for v in inference_sd.values())
    print(f"Inference weights: {len(inference_sd)} tensors, {total_params / 1e6:.1f}M params")

    config = clean_inference_config(config)

    # --- save ---
    safetensors_path = output_dir / "model.safetensors"
    config_path = output_dir / "config.json"

    save_file(inference_sd, safetensors_path)
    print(f"Saved weights: {safetensors_path} ({safetensors_path.stat().st_size / 1e6:.1f} MB)")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config:  {config_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert Lightning .ckpt to safetensors for inference")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    convert(args.ckpt, args.output_dir)
