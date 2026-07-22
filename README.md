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

Full training is intended for a remote GPU with MUSDB18-HQ available locally to that host. For a guarded Vast.ai bootstrap, detached FP-to-selective-ternary pipeline, atomic recovery, and off-instance backup workflow, follow [`docs/VAST_AI.md`](docs/VAST_AI.md).

Manual training:

```bash
python scripts/train.py --config configs/experiment.yaml \
  --data-root /datasets/MUSDB18-HQ/train --output-dir runs/baseline
```

Runs save resumable `latest.pt` and structured `experiment.json` artifacts. Schema-v3 validation records energy-aggregated development `global_sdr`, historical mean-chunk SDR, L1 by stem, and an equal-share baseline; none are BSSEval. Re-evaluate an existing checkpoint with `scripts/validate.py`. Precision is selected under `quant.layer_precisions` by layer family or exact module path. Reduced remote smoke configurations are available in `configs/smoke/`. Matched longer direct-estimate and bounded complex-mask FP32 configurations are in `configs/remote/`; input/output projections and FP32 reconstruction boundaries remain FP32 unless explicitly overridden. Optional `train.amp: fp16` or `bf16` autocasts the neural core while retaining explicit FP32 signal-processing/reconstruction boundaries and resumable gradient-scaler state.

Warm-start a QAT smoke run from the matched FP checkpoint with a fresh optimizer:

```bash
python scripts/train.py --config configs/smoke/mixed.yaml \
  --init-checkpoint runs/fp32/best.pt --data-root /datasets/MUSDB18-HQ/train \
  --output-dir runs/mixed
```

Use `--resume runs/mixed/latest.pt` only to restore the exact same configuration, optimizer, and scheduler after interruption.

Optional output distillation from the frozen `htdemucs_ft` teacher is available without adding the teacher to student checkpoints or inference. Install it separately and run the bounded smoke before enabling it for a larger experiment:

```bash
pip install -e '.[teacher]'
python scripts/train.py --config configs/smoke/htdemucs_distillation.yaml \
  --data-root /datasets/MUSDB18-HQ/train --output-dir runs/distillation-smoke
```

MUSDB ground truth remains the primary objective. The deterministic teacher uses no shifts, is projected to mixture consistency, and is only evaluated every `distillation.every_n_steps`; compare matched runs before retaining it.

Run immediate development-split layer-family sensitivity diagnostics from an FP checkpoint with:

```bash
python scripts/sensitivity.py model.pt --data-root /datasets/MUSDB18-HQ/train \
  --precision w4a8 --output runs/sensitivity-w4a8.json
```

Compare one or more experiment records (best/final epoch, equal-share baseline, output mode, and FP32/QAT classification) with:

```bash
python scripts/compare_runs.py runs/direct/experiment.json runs/mask/experiment.json
```

All comparison values are development `global_sdr`, never BSSEval.

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

The benchmark contract is frozen in [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md). See [`docs/STATUS.md`](docs/STATUS.md) for completed work and open gates. To resume work in a fresh assistant session, copy [`docs/NEW_SESSION_PROMPT.md`](docs/NEW_SESSION_PROMPT.md). Dataset audio, generated stems, checkpoints, and binaries must not be committed.
