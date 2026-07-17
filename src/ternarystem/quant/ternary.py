"""Reference fake-ternary weights and symmetric INT8 activation QAT."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .activation import ActivationFakeQuant


def _reduce_dims(weight: Tensor) -> tuple[int, ...]:
    if weight.ndim < 2:
        raise ValueError("weights must have an output-channel dimension")
    return tuple(range(1, weight.ndim))


def threshold_for_zero_ratio(weight: Tensor, zero_ratio: float) -> Tensor:
    """Per-output-channel magnitude quantile, shaped for broadcasting."""
    if not 0.0 <= zero_ratio < 1.0:
        raise ValueError("zero_ratio must be in [0, 1)")
    flat = weight.detach().abs().flatten(1)
    delta = torch.quantile(flat, zero_ratio, dim=1, keepdim=True)
    return delta.reshape((weight.shape[0],) + (1,) * (weight.ndim - 1))


def ternary_values(
    weight: Tensor,
    *,
    method: str = "adaptive",
    zero_ratio: float = 0.4,
    threshold: Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return dequantized weights, per-output scale, and {-1,0,1} values."""
    dims = _reduce_dims(weight)
    detached = weight.detach()
    if method == "adaptive":
        delta = threshold_for_zero_ratio(detached, zero_ratio) if threshold is None else threshold
        values = detached.sign() * (detached.abs() > delta).to(detached.dtype)
        numerator = (detached * values).sum(dim=dims, keepdim=True)
        denominator = values.square().sum(dim=dims, keepdim=True).clamp_min(1.0)
        scale = numerator / denominator
    elif method == "absmean":
        scale = detached.abs().mean(dim=dims, keepdim=True).clamp_min(eps)
        values = (detached / scale).round().clamp(-1, 1)
        delta = 0.5 * scale
    else:
        raise ValueError(f"unknown ternary method: {method}")
    return scale * values, scale, values


def fake_ternary(weight: Tensor, **kwargs) -> Tensor:
    """Ternary forward with identity straight-through gradient for latent weights."""
    quantized, _, _ = ternary_values(weight, **kwargs)
    return weight + (quantized - weight).detach()


@dataclass(frozen=True)
class TernaryStats:
    zero: float
    positive: float
    negative: float
    scale_mean: float


def ternary_stats(weight: Tensor, **kwargs) -> TernaryStats:
    _, scale, values = ternary_values(weight, **kwargs)
    count = values.numel()
    return TernaryStats(
        zero=float((values == 0).sum() / count),
        positive=float((values > 0).sum() / count),
        negative=float((values < 0).sum() / count),
        scale_mean=float(scale.mean()),
    )


class TernaryLinear(nn.Linear):
    def __init__(
        self,
        *args,
        method: str = "adaptive",
        zero_ratio: float = 0.4,
        activation_method: str = "ema",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ternary_method = method
        self.zero_ratio = zero_ratio
        self.activation_quant = ActivationFakeQuant(bits=8, method=activation_method)

    def forward(self, inputs: Tensor) -> Tensor:
        weight = fake_ternary(
            self.weight, method=self.ternary_method, zero_ratio=self.zero_ratio
        )
        return F.linear(self.activation_quant(inputs), weight, self.bias)


class TernaryConv2d(nn.Conv2d):
    def __init__(
        self,
        *args,
        method: str = "adaptive",
        zero_ratio: float = 0.4,
        activation_method: str = "ema",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ternary_method = method
        self.zero_ratio = zero_ratio
        self.activation_quant = ActivationFakeQuant(bits=8, method=activation_method)

    def forward(self, inputs: Tensor) -> Tensor:
        weight = fake_ternary(
            self.weight, method=self.ternary_method, zero_ratio=self.zero_ratio
        )
        return self._conv_forward(self.activation_quant(inputs), weight, self.bias)
