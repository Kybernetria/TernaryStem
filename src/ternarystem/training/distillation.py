"""Optional frozen Demucs teacher support for waveform-output distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
import torch.nn.functional as F

from ternarystem.audio import mixture_consistency
from ternarystem.data import STEMS


@dataclass(frozen=True)
class DistillationConfig:
    enabled: bool = False
    teacher: str = "htdemucs_ft"
    weight: float = 0.0
    every_n_steps: int = 1
    enforce_mixture_consistency: bool = True


class SeparationTeacher(Protocol):
    def __call__(self, mixture: Tensor) -> Tensor: ...


def distillation_config(config: dict) -> DistillationConfig:
    raw = config.get("distillation") or {}
    resolved = DistillationConfig(
        enabled=bool(raw.get("enabled", False)),
        teacher=str(raw.get("teacher", "htdemucs_ft")),
        weight=float(raw.get("weight", 0.0)),
        every_n_steps=int(raw.get("every_n_steps", 1)),
        enforce_mixture_consistency=bool(raw.get("enforce_mixture_consistency", True)),
    )
    if resolved.enabled and resolved.weight <= 0:
        raise ValueError("enabled distillation requires a positive weight")
    if resolved.every_n_steps < 1:
        raise ValueError("distillation.every_n_steps must be positive")
    return resolved


def reorder_teacher_sources(
    estimates: Tensor,
    teacher_sources: list[str] | tuple[str, ...],
    target_sources: tuple[str, ...] = STEMS,
) -> Tensor:
    if estimates.ndim != 4 or estimates.shape[1] != len(teacher_sources):
        raise ValueError("teacher estimates must have shape [batch, sources, channels, time]")
    if set(teacher_sources) != set(target_sources):
        raise ValueError(
            f"teacher sources {list(teacher_sources)} do not match required {list(target_sources)}"
        )
    indices = [teacher_sources.index(source) for source in target_sources]
    return estimates[:, indices]


def prepare_teacher_targets(
    estimates: Tensor, mixture: Tensor, enforce_consistency: bool = True
) -> Tensor:
    # Demucs runs under inference_mode; clone outside it so autograd may safely save
    # this tensor when differentiating the student loss.
    estimates = estimates.detach().clone().float()
    if enforce_consistency:
        estimates = mixture_consistency(estimates, mixture.float())
    return estimates.detach()


def waveform_distillation_l1(student: Tensor, teacher: Tensor) -> Tensor:
    if student.shape != teacher.shape:
        raise ValueError("student and teacher waveforms must have identical shapes")
    return F.l1_loss(student, teacher)


class DemucsTeacher:
    """Frozen deterministic Demucs model with project stem ordering and normalization."""

    def __init__(self, name: str, device: torch.device, sample_rate: int = 44100) -> None:
        try:
            from demucs.apply import apply_model
            from demucs.pretrained import get_model
        except ImportError as error:
            raise RuntimeError(
                "Demucs distillation requires: pip install -e '.[teacher]'"
            ) from error
        self._apply_model = apply_model
        self.model = get_model(name)
        if self.model.samplerate != sample_rate or self.model.audio_channels != 2:
            raise ValueError("teacher must use stereo 44.1 kHz audio")
        self.sources = list(self.model.sources)
        if set(self.sources) != set(STEMS):
            raise ValueError(f"unsupported Demucs sources: {self.sources}")
        self.device = device
        # Keep the four-model htdemucs_ft bag on CPU between calls. Demucs apply_model
        # moves one bag member to the execution device at a time and then offloads it.
        self.model.eval().to("cpu")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, mixture: Tensor) -> Tensor:
        # Match demucs.api.Separator normalization, independently for each batch item.
        mixture = mixture.float()
        reference = mixture.mean(dim=1)
        mean = reference.mean(dim=-1, keepdim=True)
        std = reference.std(dim=-1, keepdim=True).clamp_min(1e-8)
        normalized = (mixture - mean[:, None, :]) / std[:, None, :]
        try:
            estimates = self._apply_model(
                self.model,
                normalized,
                shifts=0,
                split=False,
                overlap=0.0,
                progress=False,
                device=self.device,
            )
        finally:
            # apply_model already does this for bags; this also handles a single model.
            self.model.to("cpu")
        estimates = estimates * std[:, None, None, :] + mean[:, None, None, :]
        return reorder_teacher_sources(estimates, self.sources)


def build_teacher(config: DistillationConfig, device: torch.device) -> SeparationTeacher | None:
    if not config.enabled:
        return None
    return DemucsTeacher(config.teacher, device)
