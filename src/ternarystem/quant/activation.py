"""Symmetric activation fake quantization with static, EMA, or learned clipping."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ActivationFakeQuant(nn.Module):
    def __init__(
        self,
        bits: int = 8,
        method: str = "ema",
        clip: float = 6.0,
        momentum: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if bits < 2 or bits > 16:
            raise ValueError("bits must be between 2 and 16")
        if method not in {"static", "ema", "learned"}:
            raise ValueError("method must be static, ema, or learned")
        self.bits, self.method, self.momentum, self.eps = bits, method, momentum, eps
        self.register_buffer("range", torch.tensor(float(clip)))
        initial_log_clip = torch.log(torch.expm1(torch.tensor(float(clip))))
        self.log_clip = nn.Parameter(initial_log_clip) if method == "learned" else None
        self.register_buffer("saturation_rate", torch.tensor(0.0))

    def _clip(self, inputs: Tensor) -> Tensor:
        observed = inputs.detach().abs().amax().clamp_min(self.eps)
        if self.method == "learned":
            return F.softplus(self.log_clip).clamp_min(self.eps)
        if self.training:
            if self.method == "ema":
                self.range.mul_(self.momentum).add_(observed * (1.0 - self.momentum))
            elif self.method == "static":
                self.range.copy_(torch.maximum(self.range, observed))
        return self.range.clamp_min(self.eps)

    def forward(self, inputs: Tensor) -> Tensor:
        clip = self._clip(inputs)
        qmax = (1 << (self.bits - 1)) - 1
        scale = clip / qmax
        clipped = inputs.clamp(-clip, clip)
        quantized = (clipped / scale).round().clamp(-qmax, qmax) * scale
        with torch.no_grad():
            self.saturation_rate.copy_((inputs.detach().abs() > clip.detach()).float().mean())
        return inputs + (quantized - inputs).detach()
