#!/usr/bin/env python3
"""Benchmark PyTorch's production oneDNN/FBGEMM quantized Linear operator."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=64)
parser.add_argument("--outputs", type=int, default=256)
parser.add_argument("--inner", type=int, default=512)
parser.add_argument("--runs", type=int, default=100)
parser.add_argument("--engine", choices=torch.backends.quantized.supported_engines, default="onednn")
parser.add_argument("--threads", type=int, default=1)
args = parser.parse_args()
torch.set_num_threads(args.threads)
torch.backends.quantized.engine = args.engine
# Fixed unit scales isolate kernel/layout performance; quantization and prepacking are offline here.
unsigned_activation = args.engine in {"fbgemm", "x86", "qnnpack"}
inputs = torch.quantize_per_tensor(
    torch.randint(-127, 128, (args.rows, args.inner), dtype=torch.int8).float(),
    scale=1.0,
    zero_point=128 if unsigned_activation else 0,
    dtype=torch.quint8 if unsigned_activation else torch.qint8,
)
weights = torch.quantize_per_channel(
    torch.randint(-127, 128, (args.outputs, args.inner), dtype=torch.int8).float(),
    scales=torch.ones(args.outputs),
    zero_points=torch.zeros(args.outputs, dtype=torch.int64),
    axis=0,
    dtype=torch.qint8,
)
packed = torch.ops.quantized.linear_prepack(weights, None)


def operator():
    return torch.ops.quantized.linear(inputs, packed, 1.0, 0)


for _ in range(10):
    operator()
times = []
for _ in range(args.runs):
    start = time.perf_counter_ns()
    operator()
    times.append((time.perf_counter_ns() - start) / 1e6)
times.sort()
print(
    json.dumps(
        {
            "operator": "torch.ops.quantized.linear",
            "engine": args.engine,
            "rows": args.rows,
            "outputs": args.outputs,
            "inner": args.inner,
            "threads": args.threads,
            "activation_dtype": str(inputs.dtype),
            "median_ms": statistics.median(times),
            "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
            "includes": "operator dispatch and output requantization",
            "excludes": "input quantization and weight prepacking",
        }
    )
)
