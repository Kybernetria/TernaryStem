"""Checkpoint loading policies for exact resume and FP-to-QAT warm starts."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict:
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("training checkpoint must contain a state_dict")
    return payload


def warm_start_model(model: nn.Module, payload: dict) -> list[str]:
    """Load shared FP weights while allowing only new fake-quantizer state.

    A warm start intentionally does not restore optimizer or epoch state. Missing
    keys are accepted only below ``activation_quant`` modules, which do not exist
    in the source FP model.
    """
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected warm-start keys: {incompatible.unexpected_keys}")
    invalid_missing = [
        key for key in incompatible.missing_keys if ".activation_quant." not in key
    ]
    if invalid_missing:
        raise ValueError(f"missing non-quantizer warm-start keys: {invalid_missing}")
    return list(incompatible.missing_keys)


def resume_training(model: nn.Module, optimizer: torch.optim.Optimizer, payload: dict) -> int:
    """Restore model and optimizer exactly and return the next epoch index."""
    if "optimizer" not in payload or "epoch" not in payload:
        raise ValueError("resume checkpoint must contain optimizer and epoch")
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["epoch"]) + 1
