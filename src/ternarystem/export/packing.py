"""Deterministic offline packing for ternary, W4, W8, and FP32 tensors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ternarystem.quant import symmetric_weight_values, ternary_values

_ENCODING = {-1: 2, 0: 0, 1: 1}


def pack_ternary(values: Tensor) -> bytes:
    flat = values.detach().to(device="cpu", dtype=torch.int8).numpy().reshape(-1)
    if not np.isin(flat, (-1, 0, 1)).all():
        raise ValueError("input contains non-ternary values")
    encoded = np.zeros((flat.size + 3) // 4, dtype=np.uint8)
    for index, value in enumerate(flat):
        encoded[index // 4] |= _ENCODING[int(value)] << (2 * (index % 4))
    return encoded.tobytes()


def unpack_ternary(packed: bytes, count: int) -> Tensor:
    raw = np.frombuffer(packed, dtype=np.uint8)
    decoded = np.empty(count, dtype=np.int8)
    table = np.array([0, 1, -1, 0], dtype=np.int8)
    for index in range(count):
        decoded[index] = table[(raw[index // 4] >> (2 * (index % 4))) & 3]
    return torch.from_numpy(decoded)


def pack_int4(values: Tensor) -> bytes:
    """Pack signed [-7, 7] integers, low nibble first, using two's complement."""
    flat = values.detach().to(device="cpu", dtype=torch.int8).numpy().reshape(-1)
    if np.any(flat < -7) or np.any(flat > 7):
        raise ValueError("symmetric INT4 values must be in [-7, 7]")
    encoded = np.zeros((flat.size + 1) // 2, dtype=np.uint8)
    for index, value in enumerate(flat):
        encoded[index // 2] |= (int(value) & 0xF) << (4 * (index % 2))
    return encoded.tobytes()


def unpack_int4(packed: bytes, count: int) -> Tensor:
    raw = np.frombuffer(packed, dtype=np.uint8)
    decoded = np.empty(count, dtype=np.int8)
    for index in range(count):
        value = int(raw[index // 2] >> (4 * (index % 2))) & 0xF
        decoded[index] = value - 16 if value >= 8 else value
    return torch.from_numpy(decoded)


def pack_bitnet_i2s(values: Tensor) -> bytes:
    """Pack output rows in BitNet's interleaved I2_S AVX2/NEON layout."""
    rows = values.detach().to(device="cpu", dtype=torch.int8).flatten(1).numpy()
    if rows.shape[1] % 128:
        raise ValueError("BitNet I2_S requires inner dimension divisible by 128")
    if not np.isin(rows, (-1, 0, 1)).all():
        raise ValueError("input contains non-ternary values")
    packed = np.zeros((rows.shape[0], rows.shape[1] // 4), dtype=np.uint8)
    for row in range(rows.shape[0]):
        for block in range(0, rows.shape[1], 128):
            for lane in range(32):
                byte = 0
                for quarter in range(4):
                    code = int(rows[row, block + quarter * 32 + lane]) + 1
                    byte |= code << (6 - 2 * quarter)
                packed[row, block // 4 + lane] = byte
    return packed.tobytes()


def unpack_bitnet_i2s(packed: bytes, rows: int, inner: int) -> Tensor:
    if inner % 128:
        raise ValueError("BitNet I2_S requires inner dimension divisible by 128")
    raw = np.frombuffer(packed, dtype=np.uint8).reshape(rows, inner // 4)
    values = np.empty((rows, inner), dtype=np.int8)
    for row in range(rows):
        for block in range(0, inner, 128):
            for lane in range(32):
                byte = int(raw[row, block // 4 + lane])
                for quarter in range(4):
                    values[row, block + quarter * 32 + lane] = (
                        (byte >> (6 - 2 * quarter)) & 3
                    ) - 1
    return torch.from_numpy(values)


def _family(name: str, last_encoder: int) -> tuple[str | None, str]:
    path = name.removeprefix("network.").removesuffix(".weight")
    if path in {"input_projection", "output_projection"}:
        return "projections", path
    if ".tdf.layers." in path:
        return "tdf_linear", path
    if path.startswith("decoder."):
        return "decoder_conv", path
    if path.startswith(f"encoder.{last_encoder}."):
        return "bottleneck_conv", path
    if path.startswith("encoder."):
        return "encoder_conv", path
    return None, path


def _last_encoder_index(names: list[str]) -> int:
    indices = []
    for name in names:
        path = name.removeprefix("network.")
        if path.startswith("encoder."):
            indices.append(int(path.split(".")[1]))
    return max(indices, default=-1)


def export_state_dict(
    state_dict: dict[str, Tensor],
    destination: str | Path,
    zero_ratio: float = 0.4,
    method: str = "adaptive",
    packing: str = "native",
    layer_precisions: dict[str, str] | None = None,
    w4_group_size: int | None = 32,
) -> dict:
    """Write deterministic packed tensors and an explicit versioned manifest."""
    if packing not in {"native", "bitnet_i2s"}:
        raise ValueError("packing must be native or bitnet_i2s")
    layer_precisions = dict(layer_precisions or {})
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    manifest = {
        "format": "ternarystem-packed",
        "version": 2,
        "byte_order": "little",
        "activation_precision": "int8-symmetric-for-quantized-operators",
        "accumulator_precision": "int32",
        "ternary_method": method,
        "zero_ratio": zero_ratio,
        "w4_group_size": w4_group_size,
        "requested_ternary_packing": packing,
        "layer_precisions": layer_precisions,
        "tensors": {},
    }
    names = sorted(state_dict)
    last_encoder = _last_encoder_index(names)
    for index, name in enumerate(names):
        tensor = state_dict[name]
        key = f"tensor_{index}"
        family, path = _family(name, last_encoder)
        precision = layer_precisions.get(path, layer_precisions.get(family, "fp32"))
        is_weight = tensor.ndim >= 2 and name.endswith("weight") and family is not None
        metadata = {"key": key, "shape": list(tensor.shape), "family": family}
        if not is_weight or precision == "fp32":
            arrays[f"{key}_fp32"] = tensor.detach().cpu().numpy().astype("<f4")
            metadata.update(kind="fp32", precision="fp32")
        elif precision == "ternary":
            _, scale, values = ternary_values(tensor, zero_ratio=zero_ratio, method=method)
            inner = values[0].numel()
            use_bitnet = packing == "bitnet_i2s" and inner % 128 == 0
            packed = pack_bitnet_i2s(values) if use_bitnet else pack_ternary(values)
            arrays[f"{key}_packed"] = np.frombuffer(packed, dtype=np.uint8)
            arrays[f"{key}_scale"] = scale.reshape(tensor.shape[0], -1).cpu().numpy().astype("<f4")
            kind = "ternary2-bitnet-i2s" if use_bitnet else "ternary2-native"
            metadata.update(kind=kind, precision="ternary", scale_axis=0, bits=2)
        elif precision in {"w4a8", "w8a8"}:
            bits = 4 if precision == "w4a8" else 8
            group_size = w4_group_size if bits == 4 else None
            _, scale, values = symmetric_weight_values(
                tensor, bits=bits, group_size=group_size
            )
            if bits == 4:
                packed = pack_int4(values)
                arrays[f"{key}_packed"] = np.frombuffer(packed, dtype=np.uint8)
                kind = "int4-symmetric-nibble"
            else:
                arrays[f"{key}_packed"] = values.cpu().numpy().astype(np.int8).reshape(-1)
                kind = "int8-symmetric"
            arrays[f"{key}_scale"] = scale.cpu().numpy().astype("<f4")
            metadata.update(
                kind=kind,
                precision=precision,
                bits=bits,
                scale_axis=0,
                group_size=tensor[0].numel() if group_size is None else group_size,
                groups_per_output=int(scale.shape[1]),
            )
        else:
            raise ValueError(f"unknown precision {precision!r} for {name}")
        manifest["tensors"][name] = metadata
    np.savez(destination, **arrays)
    manifest_path = destination.with_suffix(destination.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
