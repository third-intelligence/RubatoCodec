# Inference Guide

The inference API is the `RubatoCodec` class (`rubatocodec/model.py`). It is
installed by the base package:

```bash
uv sync            # inference dependencies only
```

## Loading a model

```python
from rubatocodec import RubatoCodec

# from a local safetensors export (preferred)
codec = RubatoCodec.from_checkpoint("path/to/model.safetensors", device="cuda")

# from the Hugging Face Hub (expects model.safetensors + config.json)
codec = RubatoCodec.from_pretrained("third-intelligence/RubatoCodec", device="cuda")

# directly from a PyTorch Lightning .ckpt (slower; loads the full checkpoint)
codec = RubatoCodec.from_checkpoint("path/to/last.ckpt", device="cuda")
```

By default, loading also downloads the frozen `facebook/w2v-bert-2.0` semantic
model from the Hugging Face Hub (used on the encode side only).

### Loading for decoding only

If codes and durations are already available, load only the decoding components:

```python
codec = RubatoCodec.from_pretrained(
    "third-intelligence/RubatoCodec", device="cuda", decoder_only=True
)
# The same option works with local .safetensors and .ckpt files:
codec = RubatoCodec.from_checkpoint(
    "path/to/model.safetensors", device="cuda", decoder_only=True
)
audio = codec.decode(enc, depth=8)  # enc was produced by a full codec instance
```

This skips the encoders and semantic model, supporting `decode` and
`decode_streaming` but not `encode`.

## Encoding

```python
import soundfile as sf
import torch

data, sr = sf.read("speech.wav", dtype="float32")  # mono waveform
wav = torch.from_numpy(data)

enc = codec.encode(wav, sr=sr)  # resampled internally to codec.sr (16 kHz) if needed
enc["codes"]  # (Q=16, K) int64: quantizer indices per event; layer 0 has semantic supervision
enc["durs"]  # (K,)   int64: event duration in 50 Hz grid frames
```

Notes:

- Omitting `sr` assumes the input is already at `codec.sr` (16 kHz). Internal
  resampling can differ marginally from an external resampler.
- Encoding supports batch size 1 per call.
- Pass finite floating-point mono audio shaped `(T,)` or `(1, T)`.
  Stereo audio must be mixed to mono before encoding.
- Encoding has no streaming mode.

## Decoding

```python
audio = codec.decode(enc)  # (1, 1, T_audio) at 44.1 kHz, full depth
audio = codec.decode(enc, depth=8)  # first 8 quantizer layers (~1.2 kbit/s)
audio = codec.decode(enc["codes"], durs=enc["durs"])  # equivalent explicit form

sf.write("recon.wav", audio[0, 0].cpu().numpy(), codec.output_sr)
```

`depth` selects how many residual-quantizer layers are used (1–16). Because the
quantizer is trained with quantizer dropout, every prefix depth yields a valid
reconstruction. Increasing depth retains more residual information.

The output length is exactly `882 · sum(durs)` samples at 44.1 kHz
(20 ms per grid frame).

### Inference settings

`decode(enc)` uses the model's recommended settings from `config.json`.
Both `.ckpt` and safetensors loading apply the same defaults when settings are
absent. For the released model these are `decode_window_frames=300`,
`decode_stride_frames=100`, `decode_crossfade_ms=10`, and a six-second attention
window. Saved settings and explicit API arguments override these defaults.
For example:

```python
audio = codec.decode(enc, batch_size=16)
# Decode the entire grid without periodic window resets:
audio = codec.decode(enc, window_frames=None)
```

See the `decode` API docstring for optional decoding controls.

Use integer tensors for codes and durations, with positive durations and
`depth` between 1 and the number of supplied code layers.

## Streaming decode

`decode_streaming` accepts one event or a chunk of events and returns the
corresponding audio incrementally. Use the same `key` for a stream and set
`finish=True` on the final call to emit any remaining audio and close the session.

```python
codes, durs = enc["codes"], enc["durs"]
K = codes.shape[1]
key = "session-0"

out = []
for i in range(K):  # e.g. events arriving one by one from a TTS language model
    chunk = codec.decode_streaming(
        codes[:, i : i + 1],
        key,
        durs=durs[i : i + 1],
        finish=(i == K - 1),
    )
    if chunk is not None:
        out.append(chunk)
audio = torch.cat(out, dim=2)
```

- Chunks may contain any number of events (including a flush-only call with an
  empty chunk and `finish=True`).
- Total output length is `882 · sum(durs)` samples.
- `depth` and decoding settings must stay fixed within a session.
- One session decodes one utterance stream; concurrent sessions are separate
  `key`s.

## Reconstructing a file

From a repository checkout:

```bash
uv run python -m examples.reconstruct speech.wav recon.wav --depth 8
uv run python -m examples.reconstruct speech.wav recon.wav --streaming
uv run python -m examples.reconstruct speech.wav recon.wav \
    --checkpoint path/to/model.safetensors --device cuda
```

The example mixes stereo input to mono, resamples through `encode`, and saves
the result at `codec.output_sr`. Output includes any padding to the encoder hop.

For multiple utterances and a `metadata.json` accepted by the evaluation CLI,
see [Running evaluation](evaluation.md#running-evaluation).


## Long recordings

Set `low_memory_encode=True` on `RubatoCodec.from_pretrained` or
`RubatoCodec.from_checkpoint` to reduce Wav2Vec2-BERT's temporary attention
memory during encoding. The default is `False`; existing calls retain their
original behavior. With `decoder_only=True`, this option has no effect and does
not load the semantic model.

For semantic sequences longer than 4,096 frames (about 82 seconds with the
default 16 kHz W2V-BERT feature extractor), attention is computed in tiles of
128 queries. Each query still attends to **all keys**. The implementation
projects the small clipped-relative-position table before gathering scores,
avoiding a `[time, time, head_dim]` embedding tensor. It does not split the
waveform, shorten context, or change model weights or precision. Short inputs
use the original attention implementation.

This reduces attention's peak temporary memory, not its quadratic computational
cost. Acoustic encoding and other tensors still grow with recording length;
this option does not guarantee that arbitrarily long audio fits on every GPU.
It is separate from `attn_window_sec`, which controls codec attention windows.

The optimized path supports inference with Wav2Vec2-BERT `relative_key`
attention and `output_attentions=False`. It rejects other position-embedding
modes when enabled. The internal implementation was validated with Transformers
4.57.6; other versions should be checked before deployment. Floating-point
kernel differences can affect rounding, so bitwise-identical codes across all
hardware, precision settings and inputs are not guaranteed.
