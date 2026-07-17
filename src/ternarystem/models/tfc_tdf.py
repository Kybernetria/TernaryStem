"""Configurable joint four-output mixed-precision TFC-TDF U-Net."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ternarystem.audio import STFT, mixture_consistency
from ternarystem.quant import (
    TernaryConv2d,
    TernaryLinear,
    W4A8Conv2d,
    W4A8Linear,
    W8A8Conv2d,
    W8A8Linear,
)

PRECISIONS = {"fp32", "ternary", "w4a8", "w8a8"}
LAYER_FAMILIES = {
    "tdf_linear",
    "bottleneck_conv",
    "encoder_conv",
    "decoder_conv",
    "projections",
}


@dataclass(frozen=True)
class SeparatorConfig:
    channels: tuple[int, ...] = (24, 48, 96)
    tdf_bottleneck: int = 16
    n_fft: int = 4096
    hop_length: int = 1024
    frequency_bins: int = 1024
    sources: int = 4
    # Keys may be a family name or an exact module path. Unspecified layers are FP32.
    layer_precisions: dict[str, str] = field(default_factory=dict)
    zero_ratio: float = 0.4
    ternary_method: str = "adaptive"
    w4_group_size: int | None = 32
    activation_method: str = "ema"
    # ``direct_estimate`` preserves the original checkpoint behavior. ``complex_mask``
    # bounds real and imaginary mask components with tanh before complex multiplication.
    output_parameterization: str = "direct_estimate"

    def precision_for(self, family: str, path: str) -> str:
        precision = self.layer_precisions.get(path, self.layer_precisions.get(family, "fp32"))
        if precision not in PRECISIONS:
            raise ValueError(f"invalid precision {precision!r} for {path}")
        return precision


class LayerFactory:
    def __init__(self, config: SeparatorConfig) -> None:
        self.config = config

    def _kwargs(self, family: str, path: str) -> tuple[str, dict]:
        precision = self.config.precision_for(family, path)
        common = {"activation_method": self.config.activation_method}
        if precision == "ternary":
            common.update(method=self.config.ternary_method, zero_ratio=self.config.zero_ratio)
        elif precision == "w4a8":
            common["group_size"] = self.config.w4_group_size
        return precision, common

    def conv(self, family: str, path: str, *args, **kwargs) -> nn.Conv2d:
        precision, quant_kwargs = self._kwargs(family, path)
        classes = {
            "fp32": nn.Conv2d,
            "ternary": TernaryConv2d,
            "w4a8": W4A8Conv2d,
            "w8a8": W8A8Conv2d,
        }
        if precision != "fp32":
            kwargs.update(quant_kwargs)
        return classes[precision](*args, **kwargs)

    def linear(self, family: str, path: str, *args, **kwargs) -> nn.Linear:
        precision, quant_kwargs = self._kwargs(family, path)
        classes = {
            "fp32": nn.Linear,
            "ternary": TernaryLinear,
            "w4a8": W4A8Linear,
            "w8a8": W8A8Linear,
        }
        if precision != "fp32":
            kwargs.update(quant_kwargs)
        return classes[precision](*args, **kwargs)


class TFCBlock(nn.Module):
    def __init__(self, channels: int, factory: LayerFactory, family: str, prefix: str) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            factory.conv(family, f"{prefix}.layers.0", channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            factory.conv(family, f"{prefix}.layers.3", channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return F.gelu(inputs + self.layers(inputs))


class TDFBlock(nn.Module):
    def __init__(
        self, channels: int, frequencies: int, bottleneck: int, factory: LayerFactory, prefix: str
    ) -> None:
        super().__init__()
        hidden = max(bottleneck, frequencies // bottleneck)
        self.norm = nn.LayerNorm(frequencies)
        self.layers = nn.Sequential(
            factory.linear("tdf_linear", f"{prefix}.layers.0", frequencies, hidden),
            nn.GELU(),
            factory.linear("tdf_linear", f"{prefix}.layers.2", hidden, frequencies),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        transposed = inputs.permute(0, 1, 3, 2)
        return inputs + self.layers(self.norm(transposed)).permute(0, 1, 3, 2)


class TFCTDFBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        frequencies: int,
        bottleneck: int,
        factory: LayerFactory,
        family: str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.tfc = TFCBlock(channels, factory, family, f"{prefix}.tfc")
        self.tdf = TDFBlock(channels, frequencies, bottleneck, factory, f"{prefix}.tdf")

    def forward(self, inputs: Tensor) -> Tensor:
        return self.tdf(self.tfc(inputs))


class TFCTDFUNet(nn.Module):
    """Maps stereo complex features ``[B,4,F,T]`` to source complex estimates."""

    def __init__(self, config: SeparatorConfig) -> None:
        super().__init__()
        if not config.channels or config.frequency_bins > config.n_fft // 2 + 1:
            raise ValueError("invalid channels or frequency truncation")
        if config.output_parameterization not in {"direct_estimate", "complex_mask"}:
            raise ValueError(f"invalid output parameterization: {config.output_parameterization!r}")
        unknown = set(config.layer_precisions) - LAYER_FAMILIES
        exact_boundary_paths = {"input_projection", "output_projection"}
        # Other exact module paths contain a dot; reject likely misspelled family keys.
        invalid = {key for key in unknown if "." not in key and key not in exact_boundary_paths}
        if invalid:
            raise ValueError(f"unknown layer precision families: {sorted(invalid)}")
        self.config = config
        factory = LayerFactory(config)
        self.input_projection = factory.conv("projections", "input_projection", 4, config.channels[0], 1)
        frequencies = [config.frequency_bins]
        for _ in config.channels[1:]:
            frequencies.append(frequencies[-1] // 2)
        self.encoder = nn.ModuleList()
        previous = config.channels[0]
        last_encoder = len(config.channels) - 1
        for index, (channels, bins) in enumerate(zip(config.channels, frequencies)):
            family = "bottleneck_conv" if index == last_encoder else "encoder_conv"
            prefix = f"encoder.{index}"
            transition = (
                nn.Identity()
                if previous == channels
                else factory.conv(family, f"{prefix}.0", previous, channels, 1)
            )
            block = TFCTDFBlock(
                channels, bins, config.tdf_bottleneck, factory, family, f"{prefix}.1"
            )
            self.encoder.append(nn.Sequential(transition, block))
            previous = channels
        self.decoder = nn.ModuleList()
        for decoder_index, index in enumerate(range(len(config.channels) - 2, -1, -1)):
            channels = config.channels[index]
            prefix = f"decoder.{decoder_index}"
            self.decoder.append(
                nn.Sequential(
                    factory.conv(
                        "decoder_conv",
                        f"{prefix}.0",
                        config.channels[index + 1] + channels,
                        channels,
                        1,
                    ),
                    TFCTDFBlock(
                        channels,
                        frequencies[index],
                        config.tdf_bottleneck,
                        factory,
                        "decoder_conv",
                        f"{prefix}.1",
                    ),
                )
            )
        self.output_projection = factory.conv(
            "projections", "output_projection", config.channels[0], config.sources * 2 * 2, 1
        )

    def forward(self, features: Tensor) -> Tensor:
        current = self.input_projection(features)
        skips = []
        for index, block in enumerate(self.encoder):
            current = block(current)
            skips.append(current)
            if index + 1 < len(self.encoder):
                current = F.avg_pool2d(current, 2)
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            current = F.interpolate(current, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            current = block(torch.cat((current, skip), dim=1))
        output = self.output_projection(current)
        batch, _, frequencies, frames = output.shape
        output = output.reshape(batch, self.config.sources, 2, 2, frequencies, frames)
        return torch.complex(output[:, :, :, 0], output[:, :, :, 1])


class Separator(nn.Module):
    def __init__(self, config: SeparatorConfig = SeparatorConfig()) -> None:
        super().__init__()
        self.config = config
        self.stft = STFT(config.n_fft, config.hop_length)
        self.network = TFCTDFUNet(config)

    def spectrograms(self, waveform: Tensor) -> Tensor:
        # STFT, output parameterization, padding, and consistency are explicit FP32
        # boundaries even when the core is run under mixed-precision autocast.
        mixture = self.stft.analysis(waveform.float()).to(torch.complex64)
        kept = mixture[..., : self.config.frequency_bins, :]
        parts = torch.view_as_real(kept).permute(0, 1, 4, 2, 3).flatten(1, 2)
        raw = self.network(parts)
        raw = torch.complex(raw.real.float(), raw.imag.float())
        if self.config.output_parameterization == "complex_mask":
            # Separately bounded Cartesian components avoid unbounded ratio masks while
            # retaining phase rotation and a smooth gradient everywhere.
            masks = torch.complex(torch.tanh(raw.real), torch.tanh(raw.imag))
            estimates = masks * kept.unsqueeze(1)
        else:
            estimates = raw
        if self.config.frequency_bins < mixture.shape[-2]:
            estimates = F.pad(estimates, (0, 0, 0, mixture.shape[-2] - self.config.frequency_bins))
        return mixture_consistency(estimates, mixture)

    def forward(self, waveform: Tensor) -> Tensor:
        estimates = self.spectrograms(waveform.float())
        return self.stft.synthesis(estimates, waveform.shape[-1])
