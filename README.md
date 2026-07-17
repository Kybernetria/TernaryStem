# TernaryStem

Research implementation of a joint four-stem TFC-TDF separator with mixed ternary, W4A8, W8A8, and FP32 precision plus packed native ternary kernels.

> This repository is an experimental scaffold. It does not yet include trained weights and has not passed Gate 0 in [the plan](docs/PLAN.md).

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Smoke tests

```bash
python scripts/train.py --config configs/experiment.yaml --dry-run
python scripts/benchmark.py --rows 256 --inner 512 --cols 256
cmake -S runtime -B build/runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/runtime
./build/runtime/ternary_bench
```

An optional, non-vendored BitNet.cpp comparison can be enabled after building an MIT-licensed `microsoft/BitNet` checkout:

```bash
cmake -S runtime -B build/bitnet -DBITNET_ROOT=/path/to/BitNet -DCMAKE_BUILD_TYPE=Release
cmake --build build/bitnet --target bitnet_comparison
./build/bitnet/bitnet_comparison 64 256 512
```

## Training

Full training is intended for a remote GPU with MUSDB18-HQ available locally to that host:

```bash
python scripts/train.py --config configs/experiment.yaml \
  --data-root /datasets/MUSDB18-HQ/train --output-dir runs/baseline
```

Runs save resumable `latest.pt` and structured `experiment.json` artifacts. Precision is selected under `quant.layer_precisions` by layer family or exact module path. Reduced remote smoke configurations are available in `configs/smoke/`; input/output projections and FP32 reconstruction boundaries remain FP32 unless explicitly overridden.

Warm-start a QAT smoke run from the matched FP checkpoint with a fresh optimizer:

```bash
python scripts/train.py --config configs/smoke/mixed.yaml \
  --init-checkpoint runs/fp32/best.pt --data-root /datasets/MUSDB18-HQ/train \
  --output-dir runs/mixed
```

Use `--resume runs/mixed/latest.pt` only to restore the exact same configuration and optimizer after interruption.

Run immediate development-split layer-family sensitivity diagnostics from an FP checkpoint with:

```bash
python scripts/sensitivity.py model.pt --data-root /datasets/MUSDB18-HQ/train \
  --precision w4a8 --output runs/sensitivity-w4a8.json
```

Export weights into the deterministic mixed-precision packed format with:

```bash
python scripts/export.py model.pt model.npz
```

## Inference

```bash
python scripts/separate.py mixture.wav --checkpoint model.pt --output-dir separated
```

Capture candidate operator shapes with:

```bash
python scripts/inventory.py --config configs/experiment.yaml
```

The benchmark contract is frozen in [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md). See [`docs/STATUS.md`](docs/STATUS.md) for completed work and open gates. Dataset audio, generated stems, checkpoints, and binaries must not be committed.
