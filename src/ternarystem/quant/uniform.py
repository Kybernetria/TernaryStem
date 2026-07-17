"""Uniform symmetric weight fake quantization for W4A8 and W8A8 QAT."""

from __future__ import annotations

import math

from torch import Tensor, nn
import torch.nn.functional as F

from .activation import ActivationFakeQuant


def symmetric_weight_values(
    weight: Tensor, *, bits: int, group_size: int | None = None, eps: float = 1e-8
) -> tuple[Tensor, Tensor, Tensor]:
    """Return dequantized weights, scales, and signed integers.

    Scales are per output channel. When ``group_size`` is set, each flattened
    input-channel/kernel group within an output channel receives its own scale.
    The latent tensor is never modified.
    """
    if weight.ndim < 2:
        raise ValueError("weights must have an output-channel dimension")
    if bits not in {4, 8}:
        raise ValueError("only symmetric 4-bit and 8-bit weights are supported")
    inner = weight[0].numel()
    if group_size is None or group_size == 0:
        group_size = inner
    if group_size < 1:
        raise ValueError("group_size must be positive or None")
    detached = weight.detach().flatten(1)
    groups = math.ceil(inner / group_size)
    padded_inner = groups * group_size
    if padded_inner != inner:
        detached = F.pad(detached, (0, padded_inner - inner))
    grouped = detached.reshape(weight.shape[0], groups, group_size)
    qmax = (1 << (bits - 1)) - 1
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / qmax
    values = (grouped / scale).round().clamp(-qmax, qmax)
    dequantized = (values * scale).reshape(weight.shape[0], padded_inner)[:, :inner]
    integers = values.reshape(weight.shape[0], padded_inner)[:, :inner]
    return dequantized.reshape_as(weight), scale.squeeze(-1), integers.reshape_as(weight)


def fake_symmetric_weight(weight: Tensor, **kwargs) -> Tensor:
    """Uniform quantized forward with identity STE for the latent FP32 weight."""
    quantized, _, _ = symmetric_weight_values(weight, **kwargs)
    return weight + (quantized - weight).detach()


class QuantLinear(nn.Linear):
    def __init__(
        self,
        *args,
        weight_bits: int,
        group_size: int | None = None,
        activation_method: str = "ema",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.activation_quant = ActivationFakeQuant(bits=8, method=activation_method)

    def forward(self, inputs: Tensor) -> Tensor:
        weight = fake_symmetric_weight(
            self.weight, bits=self.weight_bits, group_size=self.group_size
        )
        return F.linear(self.activation_quant(inputs), weight, self.bias)


class QuantConv2d(nn.Conv2d):
    def __init__(
        self,
        *args,
        weight_bits: int,
        group_size: int | None = None,
        activation_method: str = "ema",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.activation_quant = ActivationFakeQuant(bits=8, method=activation_method)

    def forward(self, inputs: Tensor) -> Tensor:
        weight = fake_symmetric_weight(
            self.weight, bits=self.weight_bits, group_size=self.group_size
        )
        return self._conv_forward(self.activation_quant(inputs), weight, self.bias)


class W4A8Linear(QuantLinear):
    def __init__(self, *args, group_size: int | None = 32, **kwargs) -> None:
        super().__init__(*args, weight_bits=4, group_size=group_size, **kwargs)


class W4A8Conv2d(QuantConv2d):
    def __init__(self, *args, group_size: int | None = 32, **kwargs) -> None:
        super().__init__(*args, weight_bits=4, group_size=group_size, **kwargs)


class W8A8Linear(QuantLinear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, weight_bits=8, group_size=None, **kwargs)


class W8A8Conv2d(QuantConv2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, weight_bits=8, group_size=None, **kwargs)
