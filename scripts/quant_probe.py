#!/usr/bin/env python3
"""Deterministic synthetic learning probe for ternary QAT plumbing.

This is a software sanity check, not Gate 0 music-separation evidence.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from ternarystem.quant import ActivationFakeQuant, TernaryLinear, ternary_stats

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=("adaptive", "absmean"), default="adaptive")
parser.add_argument("--zero-ratio", type=float, default=0.4)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=20250218)
args = parser.parse_args()
torch.manual_seed(args.seed)
inputs = torch.randn(512, 32)
teacher = torch.randn(16, 32) / 32**0.5
targets = inputs @ teacher.T
layer = TernaryLinear(
    32, 16, method=args.method, zero_ratio=args.zero_ratio, bias=True
)
activation = ActivationFakeQuant(method="ema")
optimizer = torch.optim.AdamW(layer.parameters(), lr=3e-2)

with torch.no_grad():
    initial_loss = float(F.mse_loss(layer(activation(inputs)), targets))
for _ in range(args.steps):
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(layer(activation(inputs)), targets)
    loss.backward()
    optimizer.step()
with torch.no_grad():
    final_loss = float(F.mse_loss(layer(activation(inputs)), targets))
stats = ternary_stats(
    layer.weight, method=args.method, zero_ratio=args.zero_ratio
)
print(
    json.dumps(
        {
            "probe": "synthetic_linear_regression",
            "gate_0_evidence": False,
            "method": args.method,
            "steps": args.steps,
            "initial_mse": initial_loss,
            "final_mse": final_loss,
            "improvement": initial_loss / final_loss,
            "weight_stats": vars(stats),
            "activation_saturation_rate": float(activation.saturation_rate),
        },
        indent=2,
    )
)
if not final_loss < initial_loss:
    raise SystemExit("ternary probe did not learn")
