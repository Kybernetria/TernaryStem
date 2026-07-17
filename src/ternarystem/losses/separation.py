"""Source-separation objectives and diagnostic SDR."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def global_sdr(estimate: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    dims = tuple(range(2, target.ndim))
    signal = target.square().sum(dim=dims)
    error = (target - estimate).square().sum(dim=dims)
    return (10 * torch.log10((signal + eps) / (error + eps))).mean()


def complex_l1(estimate: Tensor, target: Tensor) -> Tensor:
    return (torch.view_as_real(estimate) - torch.view_as_real(target)).abs().mean()


def multiresolution_stft_loss(
    estimate: Tensor,
    target: Tensor,
    resolutions: tuple[tuple[int, int], ...] = ((512, 128), (1024, 256), (2048, 512)),
) -> Tensor:
    estimate = estimate.flatten(0, -2)
    target = target.flatten(0, -2)
    total = estimate.new_tensor(0.0)
    for n_fft, hop in resolutions:
        window = torch.hann_window(n_fft, device=estimate.device)
        est = torch.stft(estimate, n_fft, hop, window=window, return_complex=True)
        ref = torch.stft(target, n_fft, hop, window=window, return_complex=True)
        total = total + F.l1_loss(est.abs(), ref.abs())
    return total / len(resolutions)
