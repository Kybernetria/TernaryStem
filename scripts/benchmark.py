#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from ternarystem.quant import ternary_values

parser = argparse.ArgumentParser(description="Reference PyTorch operator-shape benchmark")
parser.add_argument("--rows", type=int, default=256)
parser.add_argument("--inner", type=int, default=512)
parser.add_argument("--cols", type=int, default=256)
parser.add_argument("--runs", type=int, default=10)
args = parser.parse_args()
x = torch.randint(-127, 128, (args.rows, args.inner), dtype=torch.int32)
w = torch.randn(args.cols, args.inner)
_, _, ternary = ternary_values(w)
times = []
for _ in range(args.runs + 1):
    start = time.perf_counter()
    _ = x @ ternary.to(torch.int32).T
    elapsed = time.perf_counter() - start
    times.append(elapsed)
warmed = sorted(times[1:])
p95 = warmed[min(len(warmed) - 1, int(0.95 * len(warmed)))]
print(json.dumps({"median_seconds": statistics.median(warmed), "p95_seconds": p95, "shape": vars(args)}))
