"""Canonical project adapter for the pinned upstream SCNet core."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternarystem.audio import STFT, mixture_consistency

from ._vendor.scnet.SCNet import SCNet as UpstreamSCNet
from .interface import SOURCE_ORDER

SCNET_ARCHITECTURE_ID = "scnet_5d95bf96_adapted_v1"
SCNET_UPSTREAM_SOURCE_ORDER = ("drums", "bass", "other", "vocals")
SCNET_EXPECTED_TRAINABLE_PARAMETERS = 10_578_768


@dataclass(frozen=True)
class SCNetConfig:
    sources: tuple[str, ...] = SCNET_UPSTREAM_SOURCE_ORDER
    audio_channels: int = 2
    dims: tuple[int, ...] = (4, 32, 64, 128)
    nfft: int = 4096
    hop_size: int = 1024
    win_size: int = 4096
    normalized: bool = True
    band_SR: tuple[float, ...] = (0.175, 0.392, 0.433)
    band_stride: tuple[int, ...] = (1, 4, 16)
    band_kernel: tuple[int, ...] = (3, 4, 16)
    conv_depths: tuple[int, ...] = (3, 2, 1)
    compress: int = 4
    conv_kernel: int = 3
    num_dplayer: int = 6
    expand: int = 1

    def upstream_kwargs(self) -> dict:
        return {
            "sources": list(self.sources),
            "audio_channels": self.audio_channels,
            "dims": list(self.dims),
            "nfft": self.nfft,
            "hop_size": self.hop_size,
            "win_size": self.win_size,
            "normalized": self.normalized,
            "band_SR": list(self.band_SR),
            "band_stride": list(self.band_stride),
            "band_kernel": list(self.band_kernel),
            "conv_depths": list(self.conv_depths),
            "compress": self.compress,
            "conv_kernel": self.conv_kernel,
            "num_dplayer": self.num_dplayer,
            "expand": self.expand,
        }


class SCNetSystem(nn.Module):
    """Upstream-equivalent core plus canonical source order and mixture consistency.

    The core topology and output head are unchanged. The adapter reorders upstream
    ``drums,bass,other,vocals`` waveforms to the project order and distributes the
    additive waveform residual equally. Consequently, system output is not claimed
    to be byte-for-byte equivalent to raw upstream output.
    """

    architecture_id = SCNET_ARCHITECTURE_ID
    source_order = SOURCE_ORDER
    upstream_source_order = SCNET_UPSTREAM_SOURCE_ORDER
    loss_profile = "scnet_spec_rmse"

    def __init__(self, config: SCNetConfig | None = None) -> None:
        super().__init__()
        config = config if config is not None else SCNetConfig()
        if config.sources != SCNET_UPSTREAM_SOURCE_ORDER:
            raise ValueError(
                f"pinned SCNet core requires source order {SCNET_UPSTREAM_SOURCE_ORDER!r}"
            )
        self.config = config
        self.core = UpstreamSCNet(**config.upstream_kwargs())
        self.stft = STFT(config.nfft, config.hop_size)
        self._canonical_indices = tuple(config.sources.index(name) for name in SOURCE_ORDER)

    def upstream_forward(self, waveform: Tensor) -> Tensor:
        return self.core(waveform.float())

    def training_estimates(self, waveform: Tensor) -> Tensor:
        """Return canonical direct estimates, preserving the upstream training head."""
        return self.upstream_forward(waveform)[:, self._canonical_indices]

    def training_loss(self, estimates: Tensor, targets: Tensor) -> Tensor:
        """Pinned upstream spectral RMSE objective with its exact STFT semantics."""
        length = estimates.shape[-1]
        estimate_spectrum = torch.stft(
            estimates.reshape(-1, length),
            n_fft=self.config.nfft,
            hop_length=self.config.hop_size,
            win_length=self.config.win_size,
            center=True,
            normalized=self.config.normalized,
            return_complex=True,
        )
        target_spectrum = torch.stft(
            targets.reshape(-1, length),
            n_fft=self.config.nfft,
            hop_length=self.config.hop_size,
            win_length=self.config.win_size,
            center=True,
            normalized=self.config.normalized,
            return_complex=True,
        )
        squared = F.mse_loss(
            torch.view_as_real(estimate_spectrum),
            torch.view_as_real(target_spectrum),
            reduction="none",
        )
        return squared.mean(dim=tuple(range(1, squared.ndim))).sqrt().mean()

    def forward(self, waveform: Tensor) -> Tensor:
        canonical = self.training_estimates(waveform)
        return mixture_consistency(canonical, waveform.float())

    def spectrograms(self, waveform: Tensor) -> Tensor:
        return self.stft.analysis(self.forward(waveform))
