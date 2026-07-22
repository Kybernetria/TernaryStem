"""Compact QAT telemetry collected at epoch boundaries."""

from __future__ import annotations

from torch import nn

from ternarystem.quant import TernaryConv2d, TernaryLinear, ternary_stats


def ternary_training_summary(model: nn.Module) -> dict:
    layers = []
    covered = 0
    total = sum(parameter.numel() for parameter in model.parameters())
    for name, module in model.named_modules():
        if not isinstance(module, (TernaryConv2d, TernaryLinear)):
            continue
        statistics = ternary_stats(
            module.weight,
            method=module.ternary_method,
            zero_ratio=module.zero_ratio,
        )
        parameters = module.weight.numel()
        covered += parameters
        layers.append(
            {
                "name": name,
                "parameters": parameters,
                "zero_fraction": statistics.zero,
                "positive_fraction": statistics.positive,
                "negative_fraction": statistics.negative,
                "scale_mean": statistics.scale_mean,
                "activation_saturation": float(module.activation_quant.saturation_rate),
            }
        )
    return {
        "ternary_weight_parameters": covered,
        "coverage_of_all_parameters": covered / max(1, total),
        "layers": layers,
    }
