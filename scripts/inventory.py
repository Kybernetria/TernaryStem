#!/usr/bin/env python3
"""Capture architecture-neutral operator calls, shapes, and sequential costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from ternarystem.config import load_config, model_config
from ternarystem.models import architecture_identity, build_separator, get_architecture

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/experiment.yaml")
parser.add_argument("--frames", type=int, default=256)
parser.add_argument("--output", type=Path, default=Path("results/operator_shapes.json"))
args = parser.parse_args()
resolved_config = load_config(args.config)
config = model_config(resolved_config)
system = build_separator(resolved_config).eval()
adapter = get_architecture(architecture_identity(resolved_config)["architecture_id"])
records = []
call_counts: dict[str, int] = {}


def shape(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [shape(item) for item in value]
    return None


def estimated_macs(module: nn.Module, inputs, output) -> int | None:
    result = output[0] if isinstance(output, tuple) else output
    if not isinstance(result, torch.Tensor):
        return None
    if isinstance(module, nn.Linear):
        return result.numel() * module.in_features
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
        kernel = 1
        for size in module.kernel_size:
            kernel *= size
        return result.numel() * (module.in_channels // module.groups) * kernel
    if isinstance(module, nn.LSTM):
        sequence = inputs[0]
        batch = sequence.shape[0] if module.batch_first else sequence.shape[1]
        steps = sequence.shape[1] if module.batch_first else sequence.shape[0]
        directions = 2 if module.bidirectional else 1
        per_step = 4 * module.hidden_size * (module.input_size + module.hidden_size)
        return batch * steps * directions * per_step
    return None


def hook(name):
    def capture(module, inputs, output):
        call_index = call_counts.get(name, 0)
        call_counts[name] = call_index + 1
        records.append(
            {
                "name": name,
                "call_index": call_index,
                "operator": type(module).__name__,
                "input_shape": shape(inputs[0]),
                "parameter_shapes": [list(parameter.shape) for parameter in module.parameters()],
                "output_shape": shape(output),
                "estimated_macs": estimated_macs(module, inputs, output),
            }
        )

    return capture


handles = [
    module.register_forward_hook(hook(name))
    for name, module in system.named_modules()
    if isinstance(module, adapter.inventory_operator_types)
]
hop = int(getattr(config, "hop_length", getattr(config, "hop_size", 1024)))
samples = max(2 * hop, (args.frames - 1) * hop)
waveform = torch.randn(1, 2, samples)
with torch.inference_mode():
    system(waveform)
for handle in handles:
    handle.remove()
payload = {
    "architecture": adapter.identity.as_dict(),
    "source_order": list(adapter.source_order),
    "parameter_metadata": adapter.parameter_metadata(system),
    "config": vars(config),
    "frames_requested": args.frames,
    "waveform_samples": samples,
    "mac_convention": "multiply-accumulate counted once; FFT and elementwise costs excluded",
    "operators": records,
    "estimated_total_macs": sum(item["estimated_macs"] or 0 for item in records),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, indent=2) + "\n")
print(f"wrote {len(records)} operator calls to {args.output}")
