"""Checkpoint loading policies for exact resume and FP-to-QAT warm starts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ternarystem.models.registry import checkpoint_architecture_id


def _validate_architecture(model: nn.Module, payload: dict) -> None:
    expected = getattr(model, "architecture_id", None)
    actual = checkpoint_architecture_id(payload)
    if expected is None:
        raise ValueError("model does not declare an architecture_id")
    if actual != expected:
        raise ValueError(f"checkpoint architecture {actual!r} does not match model {expected!r}")


def capture_rng_state() -> dict[str, Any]:
    """Capture all training RNGs using weights-only-loadable primitive payloads."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("checkpoint RNG state is missing or malformed")

    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, (list, tuple)) else value

    random.setstate(tuples(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG state does not match available CUDA devices")
        torch.cuda.set_rng_state_all(cuda_states)


def deterministic_backend_state() -> dict[str, Any]:
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


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
    _validate_architecture(model, payload)
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected warm-start keys: {incompatible.unexpected_keys}")
    invalid_missing = [
        key for key in incompatible.missing_keys if ".activation_quant." not in key
    ]
    if invalid_missing:
        raise ValueError(f"missing non-quantizer warm-start keys: {invalid_missing}")
    return list(incompatible.missing_keys)


def resume_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    payload: dict,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> int:
    """Restore model, optimizer, and configured scheduler exactly."""
    if "optimizer" not in payload or "epoch" not in payload:
        raise ValueError("resume checkpoint must contain optimizer and epoch")
    _validate_architecture(model, payload)
    if scheduler is not None and not isinstance(payload.get("scheduler"), dict):
        raise ValueError("resume checkpoint must contain configured scheduler state")
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    schema_version = int(payload.get("checkpoint_schema_version", 0))
    if schema_version >= 2:
        restore_rng_state(payload.get("rng_state"))
    return int(payload["epoch"]) + 1
