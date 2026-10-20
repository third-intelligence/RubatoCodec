import math
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio.functional as F_audio
from torch.utils.data import ConcatDataset, Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from transformers import AutoFeatureExtractor

try:
    import datasets as hf_datasets_lib
    from datasets import Audio as HFAudioFeature
except ImportError:
    hf_datasets_lib = None
    HFAudioFeature = None


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def get_loader(self, phase):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size

        # Cache dataset so repeated train_dataloader() calls don't re-mmap
        cache_attr = f"_ds_cache_{phase}"
        ds = getattr(self, cache_attr, None)
        if ds is None:
            ds = self._create_hf_dataset(phase_cfg)
            setattr(self, cache_attr, ds)

        # Use StatefulDistributedSampler in DDP so shuffle state is checkpointed.
        sampler = None
        shuffle = phase_cfg.shuffle
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = StatefulDistributedSampler(ds, shuffle=shuffle)
            shuffle = False  # sampler handles shuffling

        dl = StatefulDataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.cfg.dataset.get("num_workers", 28),
            collate_fn=codec_collate_fn,
            pin_memory=True,
            persistent_workers=self.cfg.dataset.get("persistent_workers", True),
        )

        return dl

    def _create_hf_dataset(self, phase_cfg):
        num_proc = phase_cfg.get("hf_num_proc", None)
        cache_dir = phase_cfg.get("hf_cache_dir", None)
        min_sample_rate = phase_cfg.get("hf_min_sample_rate", None)
        skip_failed = phase_cfg.get("hf_skip_failed", True)
        max_retries = phase_cfg.get("hf_max_retries", 10)

        datasets_list = []
        for spec in phase_cfg.hf_datasets:
            ds = HFAudioDataset(
                spec=spec,
                cfg=self.cfg,
                num_proc=num_proc,
                cache_dir=cache_dir,
                min_sample_rate=min_sample_rate,
                skip_failed=skip_failed,
                max_retries=max_retries,
            )
            datasets_list.append(ds)

        if len(datasets_list) == 1:
            return datasets_list[0]
        return ConcatDataset(datasets_list)

    def train_dataloader(self):
        return self.get_loader("train")

    def val_dataloader(self):
        return self.get_loader("val")


def codec_collate_fn(bs):
    wavs = torch.stack([b["wav"] for b in bs])
    wavs_high = torch.stack([b["wav_high"] for b in bs])
    feats = torch.stack([b["feat"] for b in bs])
    return {
        "wav": wavs,
        "wav_high": wavs_high,
        "feats": feats,
    }


class HFAudioDataset(Dataset):
    """Dataset that loads audio from HuggingFace datasets."""

    def __init__(
        self,
        spec,
        cfg,
        num_proc=None,
        cache_dir=None,
        min_sample_rate=None,
        skip_failed=True,
        max_retries=10,
    ):
        if hf_datasets_lib is None:
            raise RuntimeError(
                "datasets is not installed. Install with: pip install 'datasets[audio]'"
            )

        self.sr = cfg.dataset.audio.sr
        self.output_sr = getattr(cfg.dataset.audio, "output_sr", self.sr)
        self.min_audio_length = cfg.dataset.min_audio_length
        self.min_sample_rate = min_sample_rate
        self.skip_failed = skip_failed
        self.max_retries = max(1, max_retries)

        self._semantic_pad = int(self.sr * 0.01)
        self.audio_column = spec.get("audio_column", "audio")

        semantic_model_name = getattr(cfg.train, "semantic_model", "facebook/w2v-bert-2.0")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(semantic_model_name)

        print(
            f"Loading HF dataset: {spec['path']} (split={spec['split']}, name={spec.get('name')})"
        )
        self.ds = hf_datasets_lib.load_dataset(
            spec["path"],
            spec.get("name", None),
            split=spec["split"],
            data_files=spec.get("data_files", None),
            revision=spec.get("revision", None),
            token=spec.get("token", None),
            num_proc=num_proc,
            cache_dir=cache_dir,
        )

        if HFAudioFeature is not None:
            if self.audio_column in self.ds.column_names:
                if not isinstance(self.ds.features[self.audio_column], HFAudioFeature):
                    self.ds = self.ds.cast_column(self.audio_column, HFAudioFeature())
            else:
                raise ValueError(
                    f"Column '{self.audio_column}' not found in dataset '{spec['path']}'. "
                    f"Available columns: {self.ds.column_names}"
                )

        print(f"Loaded HF dataset: {spec['path']} ({len(self.ds)} samples)")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if not self.skip_failed:
            return self._get_item(idx)

        total = len(self)
        for attempt in range(self.max_retries):
            tried_idx = idx if attempt == 0 else random.randint(0, total - 1)
            try:
                return self._get_item(tried_idx)
            except Exception as exc:
                if attempt > 0:
                    print(
                        f"Warning: Failed to load HF sample idx={idx} "
                        f"(tried={tried_idx}, attempt={attempt + 1}/{self.max_retries}): {exc}"
                    )
        # All retries exhausted — return a silent sample to avoid crashing DDP
        print(
            f"ERROR: All {self.max_retries} retries failed for idx={idx}, "
            f"returning silent audio to keep training alive"
        )
        return self._silent_sample()

    def _silent_sample(self):
        """Return a valid silent sample to avoid crashing DDP on persistent errors."""
        min_length_low = self.min_audio_length
        min_length_high = int(min_length_low * self.output_sr / self.sr)
        wav = torch.zeros(min_length_low)
        wav_high = torch.zeros(min_length_high)

        wav_pad = F.pad(wav, (self._semantic_pad, self._semantic_pad))
        feat = self.feature_extractor(wav_pad, sampling_rate=self.sr, return_tensors="pt").data[
            "input_features"
        ]

        return {"wav": wav, "wav_high": wav_high, "feat": feat}

    def _get_item(self, idx):
        sample = self.ds[idx]
        audio = sample[self.audio_column]

        # Extract audio array and sample rate
        wav_np = audio["array"]
        sr = int(audio["sampling_rate"])

        # min_sample_rate check
        if self.min_sample_rate is not None and sr < self.min_sample_rate:
            raise ValueError(f"Sample rate too low: {sr} < {self.min_sample_rate}")

        # Convert to tensor [1, T]
        wav_raw = torch.from_numpy(np.asarray(wav_np)).float().unsqueeze(0)

        # Resample to both output_sr (44.1kHz) and sr (16kHz) directly from source
        wav_high_full = (
            F_audio.resample(wav_raw, sr, self.output_sr)[0] if sr != self.output_sr else wav_raw[0]
        )
        wav_low_full = F_audio.resample(wav_raw, sr, self.sr)[0] if sr != self.sr else wav_raw[0]

        # Crop alignment: ensure crop position maps to integer samples at both rates
        # gcd(output_sr, sr) gives the base rate; output_sr / gcd is the crop stride
        crop_stride = self.output_sr // math.gcd(self.output_sr, self.sr)

        # Compute min_audio_length at output_sr scale, rounded up to crop_stride
        min_length_high = int(self.min_audio_length * self.output_sr / self.sr)
        min_length_high = ((min_length_high + crop_stride - 1) // crop_stride) * crop_stride

        length = wav_high_full.shape[0]
        if length < min_length_high:
            wav_high_full = F.pad(wav_high_full, (0, min_length_high - length))
            min_length_low = min_length_high * self.sr // self.output_sr
            if wav_low_full.shape[0] < min_length_low:
                wav_low_full = F.pad(wav_low_full, (0, min_length_low - wav_low_full.shape[0]))
            length = wav_high_full.shape[0]

        # Random crop at crop_stride-aligned position
        max_start = (length - min_length_high) // crop_stride
        i_high = random.randint(0, max_start) * crop_stride
        wav_high = wav_high_full[i_high : i_high + min_length_high]

        # Corresponding crop in low-rate version
        i_low = i_high * self.sr // self.output_sr
        min_length_low = min_length_high * self.sr // self.output_sr
        wav = wav_low_full[i_low : i_low + min_length_low]

        wav_pad = F.pad(wav, (self._semantic_pad, self._semantic_pad))
        feat = self.feature_extractor(wav_pad, sampling_rate=self.sr, return_tensors="pt").data[
            "input_features"
        ]

        return {
            "wav": wav,  # 16kHz for encoder/semantic
            "wav_high": wav_high,  # 44.1kHz for GT loss
            "feat": feat,
        }
