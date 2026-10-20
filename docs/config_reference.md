# Configuration Reference

Hydra composes the training configuration from `training/config/`. The values
below are the shipped defaults; model defaults specify the released architecture.
Inference reads the `config.json` exported by `scripts/convert_to_safetensors.py`,
which specifies the model architecture and inference settings.

## Top level (`training/config/default.yaml`)

| Key | Default | Description |
|---|---|---|
| `log_dir` | `./logs` | Run directory: checkpoints, Hydra output (`<log_dir>/hydra`). Relative paths resolve against the launch directory. |
| `init_ckpt` | `null` | Weights-only initialization from `.ckpt` or `.safetensors`; the latter requires an adjacent `config.json` matching the model settings. |
| `wandb.project` / `entity` / `name` | `rubatocodec` / `null` / `null` | Weights & Biases logging; `name` defaults to the `log_dir` basename. |

## Dataset (`dataset/default.yaml`)

| Key | Default | Description |
|---|---|---|
| `num_workers` | 28 | DataLoader workers. |
| `persistent_workers` | `true` | Keep workers alive between epochs. |
| `audio.sr` | 16000 | Encoder-side sample rate. |
| `audio.output_sr` | 44100 | Decoder-side (reconstruction) sample rate. |
| `min_audio_length` | 96000 | Training crop length in samples at `sr`; shorter samples are zero-padded to it. Crops are random and aligned to integer positions at both sample rates. |

The `train` / `val` sections take the same keys:

| Key | Default | Description |
|---|---|---|
| `batch_size` | 64 (train), 8 (val) | Per-device batch size. |
| `shuffle` | `true` (train), `false` (val) | Shuffle samples each epoch. |
| `hf_datasets` | — | Required for both `train` and `val`. List of Hugging Face datasets: `path`, `split`, optional `name`, `audio_column` (default `"audio"`), `data_files`, `revision`, `token` (prefer `HF_TOKEN`; see [training.md](training.md#data)). |
| `hf_num_proc` | `null` | `num_proc` for `load_dataset`. |
| `hf_cache_dir` | `null` | HF datasets cache directory. |
| `hf_min_sample_rate` | `null` | Minimum source sample rate; `null` disables this check. |
| `hf_skip_failed` / `hf_max_retries` | `true` / 10 | Retry behavior for failed samples. |

## Model (`model/default.yaml`)

### `codec_encoder`

| Key | Default | Description |
|---|---|---|
| `ngf` | 48 | Base channel width. |
| `hidden_dim` | 1024 | Encoder output dimension. |
| `up_ratios` | `[2, 2, 4, 4, 5]` | Downsampling ratios; product = hop size 320 → 50 Hz latent grid at 16 kHz. |
| `dilations` | `[1, 3, 9]` | Residual-unit dilations. |
| `activation` | `snake_beta` | `snake_beta` or `silu`. |

### `codec_decoder`

| Key | Default | Description |
|---|---|---|
| `hidden_dim` | 1024 | Backbone width. |
| `activation` | `snake_beta` | Activation of the conv blocks. |
| `conv_padding_mode` | `causal_full` | `causal_full`: fully causal (left-padded convs, right-trimmed transposed convs, forward-window ISTFT). `causal`: causal convs with a symmetric output head. `symmetric` / `asymmetric`: non-causal. |
| `resnet_kernel_size` | 1 | Kernel size of the backbone ResNet blocks. |
| `embed_kernel_size` | 3 | Kernel size of the backbone embedding conv. |
| `depth` | 12 | Number of Transformer layers. |
| `heads` | 16 | Attention heads. |
| `pos_emb_dim` | 64 | Rotary embedding dimension. |
| `vq_dim` | 2048 | Latent / quantizer dimension. |
| `attn_causal` | `true` | Causal attention. |
| `dropout` / `attn_dropout` | 0.1 / 0.0 | Backbone dropout rates (training only). |
| `mlp_expansion` | 4 | Transformer MLP expansion factor. |
| `upsampler_factors` | `[3]` | Transposed-conv upsampling factors after the backbone. |
| `upsampler_kernel_sizes` | `[9]` | Transposed-conv kernel sizes per stage. |
| `output_n_fft` | 1176 | ISTFT head FFT size. |
| `output_hop_length` | 294 | ISTFT head hop length. |

Consistency constraint: `audio.output_sr` = (`audio.sr` / prod(`up_ratios`)) ×
prod(`upsampler_factors`) × `output_hop_length`. The defaults give
50 Hz × 3 × 294 = 44100.

### `codec_decoder.quantizer`

| Key | Default | Description |
|---|---|---|
| `mode` | `rvq` | `rvq` = `RubatoResidualQuantizer` (cb1, the semantically distilled first layer, plus residual layers). `fsq` = the single FSQ stage of the original X-Codec-2.0 (configured by `fsq_levels` / `fsq_num_quantizers`). |
| `num_quantizers` | 16 | Total layers (cb1 + residual). |
| `codebook_size` | 4096 | Entries per layer. |
| `fsq_levels` | `[8, 8, 8, 8]` | FSQ levels for FSQ-type layers; product must equal `codebook_size`. |
| `cb1_type` | `fsq` | Type of the semantically distilled first layer (`fsq` or `vq`). |
| `residual_type` | `vq` | Type of the residual layers. |
| `quantize_dropout` | 1.0 | Fraction of each batch that draws a random depth ~ U{1..Q}. |
| `vq_codebook_dim` | 8 | Factorized codebook dimension of the VQ layers. |
| `vq_commitment_weight` / `vq_codebook_weight` | 0.25 / 1.0 | VQ loss weights. |
| `cb1_codebook_size` / `cb1_fsq_levels` | `null` (shared) | Optional distinct size for cb1. |

### `firing`

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Adaptive Firing on/off (`false` = fixed-rate codec). |
| `mode` | `adaptive` | `adaptive` = signal-driven Adaptive Firing; `stride` = fixed-interval baseline. |
| `target_ratio` | 0.25 | Average firing ratio relative to the latent grid rate (e.g. 0.25 on the 50 Hz grid gives ≈12.5 events/s). |
| `decision` | `clock` | `clock` = bounded cumulative allocation; `level` = thresholded signal level. |
| `head_in_budget` | `true` | Count the forced first firing inside the clock budget: `round(target_ratio · T) − 1` firings are allocated over `t ≥ 1`, so the event count is `max(1, min(T, round(target_ratio · T)))`. `false` allocates the full budget over all frames, giving ≈ 1 extra event per utterance from the forced first firing. |
| `signals` | `[hnet_ms, flux]` | Firing-signal components, per-sequence mean-normalized and summed. `hnet`: cosine dissimilarity of adjacent latent frames. `hnet_ms`: multi-scale variant of `hnet` over wider frame offsets. `flux`: log-mel spectral flux of the input waveform. `distill`: per-frame reconstruction difficulty distilled from the training loss. |
| `max_gap` | 64 | Saturation index of the offset / duration embeddings (durations themselves are not capped). |
| `aggregate` | `mean` | Pre-quantization interval aggregation: `mean` or `none` (point selection). |

Keys used only by specific signal / decision / aggregation choices:

| Key | Default | Used by | Description |
|---|---|---|---|
| `hnet_ms_offsets` | `[[1,0],[1,1],[2,1]]` | `hnet_ms` | (past, future) frame offsets of the multi-scale dissimilarity. |
| `temp` | 1.0 | `decision: level` | Sigmoid temperature of the straight-through firing mask (training only). |

### `mpd` / `mstft`

Discriminator settings (training only): HiFi-GAN multi-period discriminator
(periods `[2, 3, 5, 7, 11, 17, 23, 37]`) and a multi-resolution STFT
discriminator (8 resolutions, FFT sizes 216–6328).

## Training (`train/default.yaml`)

| Key | Default | Description |
|---|---|---|
| `compile` | `false` | `torch.compile` toggle. |
| `epochs` | `null` | If set, `max_steps` (and scheduler `down_step`) are derived from the dataset length. |
| `ckpt_every_n_steps` | 20000 | Checkpoint frequency. |
| `freeze_encoder` | `false` | Freeze encoder / semantic encoder / `fc_prior`. |
| `sem_loss_fired_only` | `true` | Restrict the semantic-distillation loss to event starts. |
| `trainer.accelerator` | `gpu` | Standard PyTorch Lightning `Trainer` argument (as are the rest of `trainer.*`). |
| `trainer.devices` / `trainer.num_nodes` | 1 / 1 | DDP device / node count. |
| `trainer.max_steps` / `trainer.min_steps` | 100000 | Training length in optimizer steps. Set per run; the default is not the released model's training length. |
| `trainer.precision` | `bf16-mixed` | Mixed-precision mode. |
| `trainer.val_check_interval` | 2000 | Validation interval in steps. |
| `trainer.limit_val_batches` | 2560 | Validation batches per validation run. |
| `trainer.num_sanity_val_steps` | 0 | Sanity-check validation steps before training. |
| `trainer.limit_train_batches` | 1.0 | Training batches per epoch: a fraction as a float, or a batch count as an integer. |
| `trainer.log_every_n_steps` | 10 | Logging interval in steps. |
| `lambdas.*` | see [training.md](training.md) | Loss weights. |
| `use_mel_loss` / `use_feat_match_loss` | `true` | Loss toggles. |
| `phase_schedule.*` | 0.1 / 0.75 / 0.15 | Phase ratios (see [training.md](training.md)); auto-normalized if they do not sum to 1. |
| `semantic_model` / `semantic_layer_index` | `facebook/w2v-bert-2.0` / 16 | Distillation teacher (also the encode-side semantic model). |
| `perceptual_model` / `perceptual_layers` | `facebook/wav2vec2-xls-r-300m` / 10–21 | Perceptual-loss feature model. |
| `mel_loss_params` | 8 resolutions | `n_mels` 5–640, windows 32–4096. |
| `gen_optim_params` / `disc_optim_params` | `lr: 1.0`, betas 0.8/0.9 | AdamW; the LR schedule lambda returns the absolute LR (see [training.md](training.md)). |
| `gen_grad_clip` / `disc_grad_clip` | 1.0 | Gradient clipping. |
| `gen_schedule_params` / `disc_schedule_params` | warmup 3000, `max_lr` 1e-4, `min_lr` 2e-5 | Set `down_step` to match `max_steps` per run. |

Training uses manual optimization; gradient accumulation is not supported.
