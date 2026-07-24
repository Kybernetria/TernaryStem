from __future__ import annotations

from pathlib import Path

import yaml

from ternarystem.models.registry import resolve_model_config


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError("configuration root must be a mapping")
    return config


def model_config(config: dict):
    """Resolve an architecture-specific model config, migrating legacy configs narrowly."""
    return resolve_model_config(config)
