# Adapted from BigCodec's module/mpd.py (MIT), via X-Codec-2.0.
# https://github.com/Aria-K-Alethia/BigCodec
# See incl_licenses/LICENSE_bigcodec and incl_licenses/LICENSE_xcodec2.

import copy

import torch
import torch.nn.functional as F


class HiFiGANPeriodDiscriminator(torch.nn.Module):
    """HiFiGAN period discriminator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        period=3,
        kernel_sizes=None,
        channels=32,
        downsample_scales=None,
        channel_increasing_factor=4,
        max_downsample_channels=1024,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params=None,
        use_weight_norm=True,
    ):
        """Initialize HiFiGANPeriodDiscriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            period (int): Period.
            kernel_sizes (list): Kernel sizes of initial conv layers and the final conv layer.
            channels (int): Number of initial channels.
            downsample_scales (list): List of downsampling scales.
            max_downsample_channels (int): Number of maximum downsampling channels.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [5, 3]
        if downsample_scales is None:
            downsample_scales = [3, 3, 3, 3, 1]
        if nonlinear_activation_params is None:
            nonlinear_activation_params = {"negative_slope": 0.1}
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1, "Kernel size must be odd number."
        assert kernel_sizes[1] % 2 == 1, "Kernel size must be odd number."

        self.period = period
        self.convs = torch.nn.ModuleList()
        in_chs = in_channels
        out_chs = channels
        for downsample_scale in downsample_scales:
            self.convs += [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_chs,
                        out_chs,
                        (kernel_sizes[0], 1),
                        (downsample_scale, 1),
                        padding=((kernel_sizes[0] - 1) // 2, 0),
                    ),
                    getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                )
            ]
            in_chs = out_chs
            out_chs = min(out_chs * channel_increasing_factor, max_downsample_channels)
        self.output_conv = torch.nn.Conv2d(
            in_chs,
            out_channels,
            (kernel_sizes[1] - 1, 1),
            1,
            padding=((kernel_sizes[1] - 1) // 2, 0),
        )

        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).

        Returns:
            list: List of each layer's tensors.
        """
        b, c, t = x.shape
        n_pad = (self.period - t % self.period) % self.period
        x = F.pad(x, (0, n_pad), "reflect")
        t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        outs = []
        for layer in self.convs:
            x = layer(x)
            outs += [x]
        x = self.output_conv(x)
        x = torch.flatten(x, 1, -1)
        outs += [x]

        return outs

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)


class HiFiGANMultiPeriodDiscriminator(torch.nn.Module):
    def __init__(
        self,
        periods=None,
        **kwargs,
    ):
        """Initialize HiFiGANMultiPeriodDiscriminator module.

        Args:
            periods (list): List of periods.
            **kwargs: Parameters passed to each HiFiGANPeriodDiscriminator
                (the period parameter is overwritten per discriminator).
        """
        super().__init__()
        if periods is None:
            periods = [2, 3, 5, 7, 11]
        self.discriminators = torch.nn.ModuleList()
        for period in periods:
            params = copy.deepcopy(kwargs)
            params["period"] = period
            self.discriminators += [HiFiGANPeriodDiscriminator(**params)]

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input waveform (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]

        return outs
