#!/usr/bin/env python3
"""Capture real Conv2D/Linear input and output shapes for kernel planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from ternarystem.config import load_config, model_config
from ternarystem.models import TFCTDFUNet

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/experiment.yaml")
parser.add_argument("--frames", type=int, default=256)
parser.add_argument("--output", type=Path, default=Path("results/operator_shapes.json"))
args = parser.parse_args()
config = model_config(load_config(args.config))
model = TFCTDFUNet(config).eval()
records = []


def hook(name):
    def capture(module, inputs, output):
        records.append(
            {
                "name": name,
                "operator": type(module).__name__,
                "input_shape": list(inputs[0].shape),
                "weight_shape": list(module.weight.shape),
                "output_shape": list(output.shape),
            }
        )

    return capture


handles = [
    module.register_forward_hook(hook(name))
    for name, module in model.named_modules()
    if isinstance(module, (nn.Conv2d, nn.Linear))
]
features = torch.randn(1, 4, config.frequency_bins, args.frames)
with torch.inference_mode():
    model(features)
for handle in handles:
    handle.remove()
payload = {"config": vars(config), "frames": args.frames, "operators": records}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, indent=2) + "\n")
print(f"wrote {len(records)} operator calls to {args.output}")
