"""Architecture-neutral contracts shared by training and deployment tooling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from torch import Tensor, nn

SOURCE_ORDER = ("vocals", "drums", "bass", "other")


@dataclass(frozen=True)
class ArchitectureIdentity:
    """Immutable identity recorded in configs, checkpoints, and experiment records."""

    architecture_id: str
    schema_version: int
    provenance_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "schema_version": self.schema_version,
            "provenance_sha256": self.provenance_sha256,
        }


@runtime_checkable
class SeparatorSystem(Protocol):
    """Waveform separator behavior required by common project tooling."""

    config: Any

    def spectrograms(self, waveform: Tensor) -> Tensor: ...

    def forward(self, waveform: Tensor) -> Tensor: ...

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]: ...

    def load_state_dict(self, state_dict: dict[str, Tensor], strict: bool = True) -> Any: ...


@dataclass(frozen=True)
class ArchitectureAdapter:
    """Construction and metadata hooks for one complete separator system."""

    identity: ArchitectureIdentity
    source_order: tuple[str, ...]
    config_from_mapping: Callable[[dict[str, Any], dict[str, Any]], Any]
    build: Callable[[Any], nn.Module]
    inventory_operator_types: tuple[type[nn.Module], ...]

    def parameter_metadata(self, model: nn.Module) -> dict[str, Any]:
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        return {
            "architecture": self.identity.as_dict(),
            "source_order": list(self.source_order),
            "trainable_parameters": trainable,
            "total_parameters": total,
        }
