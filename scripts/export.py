#!/usr/bin/env python3
"""Export a checkpoint into deterministic offline-packed mixed-precision tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ternarystem.export import export_state_dict
from ternarystem.models import config_from_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--zero-ratio", type=float)
parser.add_argument("--method", choices=("adaptive", "absmean"))
parser.add_argument("--packing", choices=("native", "bitnet_i2s"), default="native")
parser.add_argument(
    "--precision",
    action="append",
    default=[],
    metavar="FAMILY_OR_PATH=PRECISION",
    help="override checkpoint precision selection; may be repeated",
)
args = parser.parse_args()
payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
state = payload.get("state_dict", payload)
config = config_from_checkpoint(payload)
precisions = dict(getattr(config, "layer_precisions", {}))
for item in args.precision:
    try:
        key, value = item.split("=", 1)
    except ValueError as error:
        raise SystemExit(f"invalid --precision {item!r}; expected KEY=VALUE") from error
    precisions[key] = value
export_state_dict(
    state,
    args.output,
    getattr(config, "zero_ratio", 0.4) if args.zero_ratio is None else args.zero_ratio,
    getattr(config, "ternary_method", "adaptive") if args.method is None else args.method,
    args.packing,
    precisions,
    getattr(config, "w4_group_size", 32),
)
