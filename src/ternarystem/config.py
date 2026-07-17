from __future__ import annotations

from pathlib import Path

import yaml

from ternarystem.models import SeparatorConfig


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def model_config(config: dict) -> SeparatorConfig:
    values = dict(config["model"])
    values["channels"] = tuple(values["channels"])
    if "quantized" in values:
        raise ValueError("model.quantized was replaced by quant.layer_precisions")
    quant = config.get("quant", {})
    values["layer_precisions"] = dict(quant.get("layer_precisions", {}))
    values["zero_ratio"] = quant.get("target_zero_ratio", 0.4)
    values["ternary_method"] = quant.get("threshold", "adaptive")
    values["w4_group_size"] = quant.get("w4_group_size", 32)
    values["activation_method"] = quant.get("activation_method", "ema")
    return SeparatorConfig(**values)
