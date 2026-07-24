"""Full-spectrum odd-safe TernaryStem-v2 FP architecture candidate."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternarystem.audio import STFT, mixture_consistency

from .interface import SOURCE_ORDER
from .tfc_tdf import PRECISIONS, LayerFactory

TERNARYSTEM_V2_ARCHITECTURE_ID = "ternarystem_v2_fullband_temporal_v1"
TERNARYSTEM_V2_EXPECTED_TRAINABLE_PARAMETERS = 2_398_783
V2_LAYER_FAMILIES = {
    "tdf_linear",
    "bottleneck_conv",
    "encoder_conv",
    "decoder_conv",
    "temporal_conv",
    "projections",
}


@dataclass(frozen=True)
class TernaryStemV2Config:
    channels: tuple[int, ...] = (32, 64, 128)
    tdf_reduction: int = 16
    tdf_min_hidden: int = 16
    temporal_dilations: tuple[int, ...] = (1, 2, 4, 8)
    temporal_kernel: int = 7
    n_fft: int = 4096
    hop_length: int = 1024
    frequency_bins: int = 2049
    sources: int = 4
    layer_precisions: dict[str, str] = field(default_factory=dict)
    zero_ratio: float = 0.4
    ternary_method: str = "adaptive"
    w4_group_size: int | None = 32
    activation_method: str = "ema"
    output_parameterization: str = "complex_mask"
    expected_trainable_parameters: int | None = TERNARYSTEM_V2_EXPECTED_TRAINABLE_PARAMETERS

    def precision_for(self, family: str, path: str) -> str:
        precision = self.layer_precisions.get(path, self.layer_precisions.get(family, "fp32"))
        if precision not in PRECISIONS:
            raise ValueError(f"invalid precision {precision!r} for {path}")
        return precision


class ResidualTFCTDFV3(nn.Module):
    """Pre-normalized residual TFC and frequency-transform branches."""

    def __init__(
        self,
        channels: int,
        frequencies: int,
        config: TernaryStemV2Config,
        factory: LayerFactory,
        family: str,
        prefix: str,
    ) -> None:
        super().__init__()
        hidden = max(config.tdf_min_hidden, frequencies // config.tdf_reduction)
        self.tfc_norm = nn.GroupNorm(1, channels)
        self.tfc = nn.Sequential(
            factory.conv(family, f"{prefix}.tfc.0", channels, channels, 3, padding=1),
            nn.GELU(),
            factory.conv(family, f"{prefix}.tfc.2", channels, channels, 3, padding=1),
        )
        self.tdf_norm = nn.LayerNorm(frequencies)
        self.tdf = nn.Sequential(
            factory.linear("tdf_linear", f"{prefix}.tdf.0", frequencies, hidden),
            nn.GELU(),
            factory.linear("tdf_linear", f"{prefix}.tdf.2", hidden, frequencies),
        )
        self.output_norm = nn.GroupNorm(1, channels)

    def forward(self, inputs: Tensor) -> Tensor:
        current = inputs + self.tfc(self.tfc_norm(inputs))
        transposed = current.permute(0, 1, 3, 2)
        current = current + self.tdf(self.tdf_norm(transposed)).permute(0, 1, 3, 2)
        return F.gelu(self.output_norm(current))


class LongRangeTemporalStage(nn.Module):
    def __init__(
        self, channels: int, config: TernaryStemV2Config, factory: LayerFactory
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for index, dilation in enumerate(config.temporal_dilations):
            padding = dilation * (config.temporal_kernel - 1) // 2
            self.layers.append(
                nn.Sequential(
                    nn.GroupNorm(1, channels),
                    factory.conv(
                        "temporal_conv",
                        f"temporal.layers.{index}.1",
                        channels,
                        channels,
                        (1, config.temporal_kernel),
                        padding=(0, padding),
                        dilation=(1, dilation),
                    ),
                    nn.GELU(),
                    factory.conv(
                        "temporal_conv",
                        f"temporal.layers.{index}.3",
                        channels,
                        channels,
                        1,
                    ),
                )
            )

    def forward(self, inputs: Tensor) -> Tensor:
        current = inputs
        for layer in self.layers:
            current = current + layer(current)
        return current

    @property
    def receptive_field_frames(self) -> int:
        return 1 + sum(
            (layer[1].kernel_size[1] - 1) * layer[1].dilation[1] for layer in self.layers
        )


class TernaryStemV2Core(nn.Module):
    def __init__(self, config: TernaryStemV2Config) -> None:
        super().__init__()
        if not config.channels or config.frequency_bins != config.n_fft // 2 + 1:
            raise ValueError("TernaryStem-v2 requires non-empty channels and the full real STFT")
        if config.output_parameterization != "complex_mask":
            raise ValueError("TernaryStem-v2 requires bounded complex masks")
        unknown = set(config.layer_precisions) - V2_LAYER_FAMILIES
        invalid = {key for key in unknown if "." not in key and key not in {"input_projection", "output_projection"}}
        if invalid:
            raise ValueError(f"unknown layer precision families: {sorted(invalid)}")
        self.config = config
        factory = LayerFactory(config)
        self.input_projection = factory.conv(
            "projections", "input_projection", 4, config.channels[0], 1
        )
        frequencies = [config.frequency_bins]
        for _ in config.channels[1:]:
            frequencies.append((frequencies[-1] + 1) // 2)
        self.encoder = nn.ModuleList()
        previous = config.channels[0]
        for index, (channels, bins) in enumerate(zip(config.channels, frequencies)):
            family = "bottleneck_conv" if index == len(config.channels) - 1 else "encoder_conv"
            prefix = f"encoder.{index}"
            transition = (
                nn.Identity()
                if previous == channels
                else factory.conv(family, f"{prefix}.transition", previous, channels, 1)
            )
            self.encoder.append(
                nn.ModuleDict(
                    {
                        "transition": transition,
                        "block": ResidualTFCTDFV3(
                            channels, bins, config, factory, family, f"{prefix}.block"
                        ),
                    }
                )
            )
            previous = channels
        self.temporal = LongRangeTemporalStage(config.channels[-1], config, factory)
        self.decoder = nn.ModuleList()
        for decoder_index, index in enumerate(range(len(config.channels) - 2, -1, -1)):
            channels = config.channels[index]
            prefix = f"decoder.{decoder_index}"
            self.decoder.append(
                nn.ModuleDict(
                    {
                        "fusion": factory.conv(
                            "decoder_conv",
                            f"{prefix}.fusion",
                            config.channels[index + 1] + channels,
                            channels,
                            1,
                        ),
                        "block": ResidualTFCTDFV3(
                            channels,
                            frequencies[index],
                            config,
                            factory,
                            "decoder_conv",
                            f"{prefix}.block",
                        ),
                    }
                )
            )
        self.output_projection = factory.conv(
            "projections", "output_projection", config.channels[0], config.sources * 4, 1
        )
        factory.validate_selectors()

    def forward(self, features: Tensor) -> Tensor:
        current = self.input_projection(features)
        skips = []
        for index, stage in enumerate(self.encoder):
            current = stage["block"](stage["transition"](current))
            skips.append(current)
            if index + 1 < len(self.encoder):
                current = F.avg_pool2d(current, kernel_size=(2, 1), ceil_mode=True)
        current = self.temporal(current)
        for stage, skip in zip(self.decoder, reversed(skips[:-1])):
            current = F.interpolate(current, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            current = stage["block"](stage["fusion"](torch.cat((current, skip), dim=1)))
        output = self.output_projection(current).float()
        batch, _, frequencies, frames = output.shape
        output = output.reshape(batch, self.config.sources, 2, 2, frequencies, frames)
        return torch.complex(output[:, :, :, 0], output[:, :, :, 1])


class TernaryStemV2System(nn.Module):
    architecture_id = TERNARYSTEM_V2_ARCHITECTURE_ID
    source_order = SOURCE_ORDER
    loss_profile = "legacy"

    def __init__(self, config: TernaryStemV2Config | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else TernaryStemV2Config()
        self.stft = STFT(self.config.n_fft, self.config.hop_length)
        self.network = TernaryStemV2Core(self.config)
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        if (
            self.config.expected_trainable_parameters is not None
            and trainable != self.config.expected_trainable_parameters
        ):
            raise ValueError(
                f"TernaryStem-v2 trainable count {trainable} does not match frozen "
                f"{self.config.expected_trainable_parameters}"
            )

    def spectrograms(self, waveform: Tensor) -> Tensor:
        mixture = self.stft.analysis(waveform.float()).to(torch.complex64)
        parts = torch.view_as_real(mixture).permute(0, 1, 4, 2, 3).flatten(1, 2)
        raw = self.network(parts)
        masks = torch.complex(torch.tanh(raw.real.float()), torch.tanh(raw.imag.float()))
        return mixture_consistency(masks * mixture.unsqueeze(1), mixture)

    def forward(self, waveform: Tensor) -> Tensor:
        return self.stft.synthesis(self.spectrograms(waveform), waveform.shape[-1])
