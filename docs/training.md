# Training Guide

The training stack lives in `training/` (PyTorch Lightning + Hydra) and is
installed with the `train` extra:

```bash
uv sync --extra train
```

Configuration is composed by Hydra from `training/config/`:

```
training/config/default.yaml        # top level: log_dir, wandb, init_ckpt
training/config/dataset/default.yaml
training/config/model/default.yaml  # released model (see docs/architecture.md)
training/config/train/default.yaml  # optimization, losses, phase schedule
```

Configuration keys are documented in [config_reference.md](config_reference.md).

## Data

The data module (`training/data_module.py`) reads Hugging Face `datasets`
entries listed under `dataset.train.hf_datasets` and
`dataset.val.hf_datasets`. Both are required: validation computes
`val/mel_loss`, which checkpoint selection monitors. The shipped dataset config
contains only a commented-out example; point it at your own data:

```yaml
# training/config/dataset/default.yaml
train:
  batch_size: 64
  shuffle: true
  hf_datasets:
    - path: "your-org/your-speech-dataset"
      split: "train"
      audio_column: "audio"
val:
  batch_size: 8
  shuffle: false
  hf_datasets:
    - path: "your-org/your-speech-dataset"
      split: "validation"
      audio_column: "audio"
```

For gated or private datasets, authenticate with `hf auth login` or the
`HF_TOKEN` environment variable rather than writing a `token` into the config.
A per-dataset `token` is accepted but is removed from the config saved in
checkpoints and logged to Weights & Biases; Hydra's own copy of the config
(`<log_dir>/hydra/config.yaml`) still contains it.

Each sample is resampled to both 16 kHz (encoder input) and 44.1 kHz
(reconstruction target) and randomly cropped to 6 s (`min_audio_length: 96000`
samples at 16 kHz), with the crop position aligned so it maps to integer
sample indices at both rates. Shorter samples are zero-padded to the crop
length.

## Launching a run

`training/train.py` is a Hydra entry point; every key can be set in the config
files or overridden on the CLI, e.g.:

```bash
uv run python training/train.py \
    log_dir=output/rubatocodec_run1 \
    train.trainer.devices=8 \
    train.ckpt_every_n_steps=20000 \
    wandb.project=rubatocodec wandb.entity=YOUR_ENTITY
```

- Multi-GPU uses PyTorch Lightning DDP (`train.trainer.devices`,
  `train.trainer.num_nodes`).
- Precision is `bf16-mixed` by default; the residual-quantizer chain and the
  spectral-flux component of the firing signal always run in fp32.
- Set `train.epochs=N` to derive `max_steps` (and the scheduler `down_step`)
  from the dataset length instead of specifying steps explicitly.
- A relative `log_dir` is resolved against the launch directory, and Hydra
  writes its run files to `<log_dir>/hydra`.
- If `log_dir` already contains `last.ckpt`, training resumes from it
  automatically (optimizer state and step included). Otherwise
  `init_ckpt=path/to/model.safetensors` or `init_ckpt=path/to/last.ckpt`
  initializes weights without restoring the optimizer or training step.
  Safetensors exports require an adjacent `config.json` matching the training
  model settings. Codec weights are loaded strictly; discriminators and the
  training-only semantic head start from new weights, while frozen teachers
  are loaded from their pretrained sources.

Checkpoints are written to `log_dir` every `ckpt_every_n_steps` steps;
validation runs every `val_check_interval` steps.

## Optimization

Generator and discriminators are trained with separate AdamW optimizers,
stepped alternately. The LR schedule (warmup followed by linear decay,
`gen_schedule_params` / `disc_schedule_params`) is a `LambdaLR` whose lambda
returns the absolute learning rate, hence `lr: 1.0` in the optimizer params.
The factory is `build_warmup_decay_scheduler` in `training/schedulers.py`;
`warmup_step=0` starts directly at `max_lr`.
Gradients are clipped to 1.0.

### Phase schedule

Losses are enabled in three phases over the run
(`train.phase_schedule`, ratios of `max_steps`):

| Phase | Default span | Active losses |
|---|---|---|
| 1 | first 10% | mel + semantic distillation + VQ commitment/codebook |
| 2 | next 75% | + adversarial, discriminator, feature matching |
| 3 | last 15% | + perceptual loss |

### Losses

| Loss | Weight (`train.lambdas`) | Notes |
|---|---|---|
| Multi-resolution mel | 15.0 | 8 resolutions, 5–640 mel bins / window 32–4096 |
| Adversarial (generator) | 1.0 | least-squares GAN, vs. MPD + multi-resolution STFT discriminators |
| Discriminator | 1.0 | HiFi-GAN MPD (periods 2,3,5,7,11,17,23,37) + 8-resolution STFT |
| Feature matching | 1.0 | MPD and STFT-discriminator feature maps; both terms use this weight |
| Semantic distillation | 5.0 | Semantic-head predictions vs. frozen w2v-BERT layer-16 features; event starts only (`sem_loss_fired_only: true`) |
| Perceptual | 1.0 | wav2vec2-xls-r-300m, layers 10–21 (phase 3) |
| VQ commitment/codebook | 1.0 | DAC weights (0.25 / 1.0) applied inside the quantizer |

The semantic head and its sparse-grid input are described in
[architecture.md](architecture.md#semantic-distillation). For fixed-rate
encoders below 50 Hz, teacher features are mean-pooled over the corresponding
frame groups before feature fusion and semantic supervision.

## After training

Convert the Lightning checkpoint to the inference format:

```bash
uv run python -m scripts.convert_to_safetensors \
    --ckpt output/rubatocodec_run1/last.ckpt --output-dir output/rubatocodec_release/
```
