import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class PerceptualLoss(nn.Module):
    """Perceptual loss using wav2vec2 hidden states.

    Computes MSE between hidden states of a frozen wav2vec2 model
    for generated and ground-truth waveforms.

    Args:
        model_name: HuggingFace model name (default: facebook/wav2vec2-xls-r-300m)
        layers: List of hidden layer indices to use for loss computation.
            Defaults to layers 10-21 (12 layers).
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-xls-r-300m",
        layers: list[int] | None = None,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.requires_grad_(False)
        self.layers = layers if layers is not None else list(range(10, 22))

    def forward(self, gen_wav: torch.Tensor, gt_wav: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss.

        Args:
            gen_wav: Generated waveform [B, T] at 16kHz. Gradients flow through this.
            gt_wav: Ground truth waveform [B, T] at 16kHz.

        Returns:
            Mean MSE loss over selected hidden layers.
        """
        with torch.no_grad():
            gt_hidden = self.model(gt_wav, output_hidden_states=True).hidden_states

        gen_hidden = self.model(gen_wav, output_hidden_states=True).hidden_states

        losses = []
        for layer_idx in self.layers:
            losses.append(F.mse_loss(gen_hidden[layer_idx], gt_hidden[layer_idx].detach()))

        return torch.stack(losses).mean()
