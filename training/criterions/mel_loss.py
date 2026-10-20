import torch.nn as nn
from torchaudio.transforms import MelSpectrogram


class MultiResolutionMelSpectrogramLoss(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_mels: list[int] | None = None,
        window_lengths: list[int] | None = None,
        clamp_eps: float = 1e-5,
        pow: float = 1.0,
    ):
        super().__init__()
        if n_mels is None:
            n_mels = [5, 10, 20, 40, 80, 160, 320]
        if window_lengths is None:
            window_lengths = [32, 64, 128, 256, 512, 1024, 2048]
        self.mel_transforms = nn.ModuleList(
            [
                MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=window_length,
                    hop_length=window_length // 4,
                    n_mels=n_mel,
                    power=1.0,
                    center=True,
                    norm="slaney",
                    mel_scale="slaney",
                )
                for n_mel, window_length in zip(n_mels, window_lengths, strict=False)
            ]
        )
        self.loss_fn = nn.L1Loss()
        self.clamp_eps = clamp_eps
        self.pow = pow

    def forward(self, x, y):
        loss = 0.0
        for mel_transform in self.mel_transforms:
            x_mel = mel_transform(x)
            y_mel = mel_transform(y)
            log_x_mel = x_mel.clamp(self.clamp_eps).pow(self.pow).log10()
            log_y_mel = y_mel.clamp(self.clamp_eps).pow(self.pow).log10()
            loss += self.loss_fn(log_x_mel, log_y_mel)
        return loss
