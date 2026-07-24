"""Learning-rate scheduler construction from resolved training configuration."""

from __future__ import annotations

import torch


def resolve_stop_epoch(data_config: dict, train_config: dict, stop_chunks: int | None) -> int:
    maximum = int(train_config["epochs"])
    if stop_chunks is None:
        return maximum
    chunks_per_epoch = int(data_config["epoch_chunks"])
    if stop_chunks <= 0 or stop_chunks % chunks_per_epoch:
        raise ValueError("rung stop must be a positive multiple of data.epoch_chunks")
    stop_epoch = stop_chunks // chunks_per_epoch
    if stop_epoch > maximum:
        raise ValueError("rung stop exceeds the immutable maximum training horizon")
    return stop_epoch


def build_scheduler(
    optimizer: torch.optim.Optimizer, train_config: dict
) -> torch.optim.lr_scheduler.LRScheduler | None:
    config = train_config.get("scheduler")
    if config is None or config == "none" or config.get("name", "none") == "none":
        return None
    name = config.get("name")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("t_max", train_config["epochs"])),
            eta_min=float(config.get("eta_min", 0.0)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config["step_size"]),
            gamma=float(config.get("gamma", 0.1)),
        )
    raise ValueError(f"unknown learning-rate scheduler: {name!r}")
