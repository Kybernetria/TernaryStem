"""FP32 analysis/synthesis and source consistency operations."""

from __future__ import annotations

import torch
from torch import Tensor


class STFT:
    def __init__(self, n_fft: int = 4096, hop_length: int = 1024, center: bool = True) -> None:
        if n_fft <= 0 or hop_length <= 0:
            raise ValueError("n_fft and hop_length must be positive")
        self.n_fft, self.hop_length, self.center = n_fft, hop_length, center

    def _window(self, reference: Tensor) -> Tensor:
        return torch.hann_window(
            self.n_fft, periodic=True, dtype=torch.float32, device=reference.device
        )

    def analysis(self, waveform: Tensor) -> Tensor:
        """Transform ``[..., samples]`` FP32 audio to a complex spectrogram."""
        waveform = waveform.float()
        shape = waveform.shape[:-1]
        flat = waveform.reshape(-1, waveform.shape[-1])
        spectrum = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self._window(flat),
            center=self.center,
            return_complex=True,
        )
        return spectrum.reshape(*shape, *spectrum.shape[-2:])

    def synthesis(self, spectrum: Tensor, length: int) -> Tensor:
        """Invert a complex spectrogram to ``[..., samples]`` FP32 audio."""
        shape = spectrum.shape[:-2]
        flat = spectrum.reshape(-1, *spectrum.shape[-2:])
        waveform = torch.istft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self._window(flat.real),
            center=self.center,
            length=length,
        )
        return waveform.reshape(*shape, length)


def mixture_consistency(estimates: Tensor, mixture: Tensor) -> Tensor:
    """Orthogonally distribute additive residual over the source dimension (dim=1)."""
    if estimates.ndim != mixture.ndim + 1:
        raise ValueError("estimates must have one source dimension more than mixture")
    residual = mixture - estimates.sum(dim=1)
    return estimates + residual.unsqueeze(1) / estimates.shape[1]
