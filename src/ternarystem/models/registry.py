"""Fail-closed separator architecture registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from .interface import SOURCE_ORDER, ArchitectureAdapter, ArchitectureIdentity
from .scnet import SCNET_ARCHITECTURE_ID, SCNetConfig, SCNetSystem
from .ternarystem_v2 import (
    TERNARYSTEM_V2_ARCHITECTURE_ID,
    TernaryStemV2Config,
    TernaryStemV2System,
)
from .tfc_tdf import Separator, SeparatorConfig

LEGACY_ARCHITECTURE_ID = "legacy_tfc_tdf_v1"
SCNET_PROVENANCE_SHA256 = "b0f5a840238d03f8fd9238d7cbda9967d9270339170dde293afa157610d98eb3"


def _legacy_config(model_values: dict[str, Any], quant: dict[str, Any]) -> SeparatorConfig:
    values = dict(model_values)
    values.pop("architecture", None)
    values["channels"] = tuple(values["channels"])
    if "quantized" in values:
        raise ValueError("model.quantized was replaced by quant.layer_precisions")
    values.setdefault("layer_precisions", dict(quant.get("layer_precisions", {})))
    values.setdefault("zero_ratio", quant.get("target_zero_ratio", 0.4))
    values.setdefault("ternary_method", quant.get("threshold", "adaptive"))
    values.setdefault("w4_group_size", quant.get("w4_group_size", 32))
    values.setdefault("activation_method", quant.get("activation_method", "ema"))
    return SeparatorConfig(**values)


def _v2_config(model_values: dict[str, Any], quant: dict[str, Any]) -> TernaryStemV2Config:
    values = dict(model_values)
    values.pop("architecture", None)
    for key in ("channels", "temporal_dilations"):
        if key in values:
            values[key] = tuple(values[key])
    values["layer_precisions"] = dict(quant.get("layer_precisions", {}))
    values["zero_ratio"] = quant.get("target_zero_ratio", 0.4)
    values["ternary_method"] = quant.get("threshold", "adaptive")
    values["w4_group_size"] = quant.get("w4_group_size", 32)
    values["activation_method"] = quant.get("activation_method", "ema")
    return TernaryStemV2Config(**values)


def _scnet_config(model_values: dict[str, Any], quant: dict[str, Any]) -> SCNetConfig:
    values = dict(model_values)
    values.pop("architecture", None)
    selected_precisions = quant.get("layer_precisions", {})
    if selected_precisions:
        raise ValueError("pinned SCNet is FP-only and does not accept layer precision selectors")
    for key in (
        "sources",
        "dims",
        "band_SR",
        "band_stride",
        "band_kernel",
        "conv_depths",
    ):
        if key in values:
            values[key] = tuple(values[key])
    return SCNetConfig(**values)


_REGISTRY = {
    LEGACY_ARCHITECTURE_ID: ArchitectureAdapter(
        identity=ArchitectureIdentity(LEGACY_ARCHITECTURE_ID, schema_version=1),
        source_order=SOURCE_ORDER,
        config_from_mapping=_legacy_config,
        build=Separator,
        inventory_operator_types=(nn.Conv2d, nn.Linear),
    ),
    TERNARYSTEM_V2_ARCHITECTURE_ID: ArchitectureAdapter(
        identity=ArchitectureIdentity(TERNARYSTEM_V2_ARCHITECTURE_ID, schema_version=1),
        source_order=SOURCE_ORDER,
        config_from_mapping=_v2_config,
        build=TernaryStemV2System,
        inventory_operator_types=(nn.Conv2d, nn.Linear),
    ),
    SCNET_ARCHITECTURE_ID: ArchitectureAdapter(
        identity=ArchitectureIdentity(
            SCNET_ARCHITECTURE_ID,
            schema_version=1,
            provenance_sha256=SCNET_PROVENANCE_SHA256,
        ),
        source_order=SOURCE_ORDER,
        config_from_mapping=_scnet_config,
        build=SCNetSystem,
        inventory_operator_types=(nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.LSTM),
    ),
}


def architecture_id(config: Mapping[str, Any]) -> str:
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("configuration must contain a model mapping")
    selected = model.get("architecture", LEGACY_ARCHITECTURE_ID)
    if not isinstance(selected, str) or not selected:
        raise ValueError("model.architecture must be a non-empty string")
    return selected


def get_architecture(selected: str) -> ArchitectureAdapter:
    try:
        return _REGISTRY[selected]
    except KeyError as error:
        raise ValueError(f"unknown model architecture: {selected!r}") from error


def registered_architectures() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def resolve_model_config(config: Mapping[str, Any]) -> Any:
    selected = architecture_id(config)
    adapter = get_architecture(selected)
    model = config["model"]
    quant = config.get("quant", {})
    if not isinstance(quant, Mapping):
        raise TypeError("quant configuration must be a mapping")
    return adapter.config_from_mapping(dict(model), dict(quant))


def build_separator(config: Mapping[str, Any]) -> nn.Module:
    adapter = get_architecture(architecture_id(config))
    return adapter.build(resolve_model_config(config))


def architecture_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    return get_architecture(architecture_id(config)).identity.as_dict()


def checkpoint_architecture_id(payload: Mapping[str, Any]) -> str:
    identity = payload.get("architecture")
    if isinstance(identity, Mapping):
        selected = identity.get("architecture_id")
        if isinstance(selected, str):
            return selected
    resolved = payload.get("resolved_config")
    if isinstance(resolved, Mapping) and isinstance(resolved.get("model"), Mapping):
        return architecture_id(resolved)
    # Schema-less historical checkpoints predate architecture identity. They are
    # narrowly assigned to the only architecture that existed at that time.
    if isinstance(payload.get("config"), Mapping) or isinstance(payload.get("state_dict"), Mapping):
        return LEGACY_ARCHITECTURE_ID
    raise ValueError("checkpoint has no architecture identity or compatible legacy config")


def config_from_checkpoint(payload: Mapping[str, Any]) -> Any:
    selected = checkpoint_architecture_id(payload)
    get_architecture(selected)
    resolved = payload.get("resolved_config")
    if isinstance(resolved, Mapping) and isinstance(resolved.get("model"), Mapping):
        if architecture_id(resolved) != selected:
            raise ValueError("checkpoint architecture conflicts with resolved configuration")
        return resolve_model_config(resolved)
    raw = payload.get("config")
    if selected != LEGACY_ARCHITECTURE_ID or not isinstance(raw, Mapping):
        raise ValueError("checkpoint does not contain a resolvable model configuration")
    return _legacy_config(dict(raw), {})


def build_from_checkpoint(payload: Mapping[str, Any]) -> nn.Module:
    selected = checkpoint_architecture_id(payload)
    return get_architecture(selected).build(config_from_checkpoint(payload))
