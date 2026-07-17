"""Learning-rate scheduler construction from resolved training configuration."""

from __future__ import annotations

import torch


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
