# TernaryStem Development Plan

## 0. Execution Status (updated 2026-07-17)

Phase 0 software scaffolding and initial x86-64 kernel probes are complete. The quality portion of Gate 0 remains unevaluated because MUSDB18-HQ training requires a remote GPU environment.

The deployment portion of Gate 0 has **not passed for an all-ternary runtime**. On an AMD Ryzen 7 7730U, an exact BitNet.cpp I2_S adapter ranged from 0.69x to 3.49x versus production FBGEMM INT8 and missed the required 1.3x speedup on four of five tested shapes. See `docs/GATE0_REPORT.md`.

Consequently, development proceeds on two tracks:

1. Continue ternary QAT as a quality research question.
2. Use selective per-layer deployment: BitNet I2_S only for shapes where it wins, W4A8/W8A8 for the remaining quantized core, and FP32 for sensitive boundaries/norms.

No end-to-end latency or separation-quality claim has been established. A first remote MUSDB18-HQ development smoke run and matched continuation are recorded in `results/remote/2026-07-17-selective-ternary/`: selective TDF/bottleneck ternary QAT trailed its FP32 control by 0.0224 dB on diagnostic `global_sdr`, but the FP model itself remained at negative SDR and poor separation quality. This supports recoverable selective-QAT plumbing, not Gate 1 or Gate 2. The local training path now includes per-stem development diagnostics and an equal-share baseline, direct-estimate and bounded complex-mask output modes, resumable learning-rate scheduling, expanded environment/checkpoint metadata, and matched longer FP32 configurations. These additions have only synthetic/local test coverage; the next decisive milestone remains a capable remotely trained FP32 music-separation baseline, followed by matched W4A8/W8A8 and mixed-precision runs.

## 1. Objective

Build and evaluate an open-source, four-stem music source separator for stereo 44.1 kHz audio using ternary core weights (`{-1, 0, 1}`), INT8 activations, and a native laptop-CPU runtime.

The project has two independent research questions:

1. Can ternary quantization retain enough regression accuracy for music source separation?
2. Can packed ternary kernels outperform optimized INT8 kernels for the model's actual convolution and TDF shapes?

The final stretch targets are:

- Quality: match HT-Demucs under a clearly defined, comparable evaluation protocol.
- Latency: process a three-minute song in less than 10 seconds on a specified consumer CPU.
- Portability: x86-64 AVX2 and ARM64 NEON baseline implementations, with optional newer ISA paths.

These are stretch targets, not assumptions. Intermediate gates below determine whether they remain feasible.

## 2. Fixed Technical Direction

### 2.1 Candidate model

Use a joint four-output TFC-TDF U-Net rather than four independent source models:

```text
Stereo waveform
  -> FP32 STFT
  -> complex feature projection
  -> shared TFC-TDF encoder/core/decoder
  -> four complex masks or complex spectrogram estimates
  -> mixture-consistency projection
  -> FP32 iSTFT
  -> vocals, drums, bass, other
```

The first implementation must be configurable so channel counts, depth, TDF bottlenecks, frequency truncation, and quantized layer coverage can be changed without rewriting the model.

### 2.2 Precision policy

The original all-ternary core remains an experimental target, not the default deployment assumption. Following the interim Gate 0 result, export and runtime must support per-layer precision selected from ternary, W4A8, W8A8, and FP32.

| Component | Training | CPU deployment |
|---|---:|---:|
| STFT/iSTFT | FP32 | FP32 |
| Input/output projections | FP32 or FP16 | FP32 initially; test INT8 later |
| Core Conv2D/Linear weights | Latent FP32 + fake ternary/W4/W8 | Selective packed ternary, W4, or INT8 |
| Core activations | FP16/FP32 fake-quantized to INT8 | INT8 |
| Accumulators | FP32 simulation | INT32 |
| Per-output-channel scales | FP32 | FP32 |
| Norms and sensitive nonlinearities | FP32 | FP32 or optimized mixed precision |
| Final masks/reconstruction | FP32 | FP32 |

Do not keep the entire decoder high precision by default. Use layer-sensitivity experiments to decide which decoder layers, if any, must remain high precision.

### 2.3 Ternary quantization

Maintain a latent full-precision weight `W`. For each output channel:

```text
T = sign(W) * 1[abs(W) > delta]
alpha = sum(W * T) / max(sum(T^2), 1)
W_q = alpha * T
```

Use an STE for the backward pass. Compare this learned/adaptive scheme against BitNet-style abs-mean quantization.

The zero ratio is an experimental variable, not a fixed requirement. Test at least 20%, 40%, and 60%. A sparsity setting is accepted only if it improves measured native-kernel latency without unacceptable SDR loss.

Use symmetric INT8 activation quantization. Compare static calibration, EMA range tracking, and learned clipping before selecting one method.

## 3. Benchmark Contract

Freeze this contract before reporting model improvements.

### 3.1 Data

- Dataset: MUSDB18-HQ.
- Official test set: all 50 test tracks, never used for tuning.
- Development split: fixed 86-train/14-validation split from the official 100 training tracks.
- Save the split, random seed, and track names in version control.
- Primary result: MUSDB18-HQ only, with no additional audio data.
- Dynamic augmentation: cross-song stem remixing, gain, stereo channel swap, polarity, and optional pitch/time transforms.
- Define an epoch as a fixed number of sampled chunks; do not describe dynamic mixtures as a finite augmented dataset.

Confirm dataset and derived-weight distribution terms before publishing checkpoints.

### 3.2 Quality metrics

Report all of the following:

- museval/BSSEval v4 SDR per stem and overall.
- A clearly named global or utterance-level SDR for training diagnostics.
- Median and mean where appropriate; do not compare unlike SDR definitions.
- Optional artifact checks: high-frequency error, silent-source leakage, mixture consistency, and listening samples.

Record whether results use overlap-add, test-time shifts, ensembles, source-specific fine-tuning, or extra data.

### 3.3 Runtime metrics

Use a fixed three-minute, stereo, 44.1 kHz WAV input and report:

- End-to-end wall time, including STFT, model, iSTFT, and overlap-add.
- Warm and cold runs.
- Median and p95 across at least ten warm runs.
- Thread count, CPU model, ISA, compiler, power mode, and memory use.
- Per-operator timing and packed model size.

Initial reference machines:

- x86-64: AVX2-capable Intel Core i5/i7 or AMD Ryzen 5/7.
- ARM64: Apple M1 or newer using NEON.

AVX-512, VNNI, and ARM DotProd are optional optimized paths and cannot be required for baseline compatibility.

## 4. Work Plan

## Phase 0 — Feasibility and Contracts (Week 1)

### Tasks

1. Create the repository structure, reproducible environment, CI, configuration system, and test harness.
2. Freeze the dataset split, SDR definitions, inference settings, and reference hardware.
3. Reproduce inference from one published lightweight TFC-TDF/DTTNet checkpoint and profile its operator shapes.
4. Implement Python reference ternary quantization and validate its scale, threshold, zero ratio, and STE gradients.
5. Build standalone C++ microbenchmarks for representative TDF GEMM and Conv2D shapes:
   - FP32 baseline
   - optimized INT8 baseline
   - packed ternary scalar reference
   - AVX2 ternary prototype
   - NEON ternary prototype, when ARM hardware is available
6. Run a small quantization probe on a reduced model and training subset.

### Deliverables

- `docs/BENCHMARK_PROTOCOL.md`
- Fixed dataset split and experiment seeds
- Operator-shape inventory
- Quantization unit tests
- Kernel benchmark report
- Initial FP32, INT8, and ternary quality/latency measurements

### Gate 0

Continue with native ternary deployment only if:

- Fake-ternary training shows recoverable learning on the reduced task.
- A representative packed ternary kernel is at least 1.3x faster than the chosen optimized INT8 baseline, or shows a credible path to that result.
- Packing is offline and runtime unpack/LUT overhead does not erase the bandwidth benefit.

If the gate fails, retain the model work but pivot deployment experiments to W4A8 or W8A8.

**Interim decision:** the all-ternary deployment gate failed on tested x86 shapes. The required pivot is active. Selective BitNet dispatch remains allowed where a matched benchmark demonstrates a win; it does not count as evidence for an all-ternary runtime.

## Phase 1 — Floating-Point Baseline (Weeks 2–3)

### Tasks

1. Implement FP32 STFT/iSTFT with exact reconstruction tests.
2. Implement configurable TFC, TDF, encoder, bottleneck, decoder, complex output heads, and mixture consistency.
3. Implement chunking and overlap-add with boundary artifact tests.
4. Build streaming MUSDB18-HQ loading and dynamic stem remixing without pre-generating a large augmented dataset.
5. Add losses and ablate them on validation data:
   - waveform L1
   - complex spectrogram L1
   - multi-resolution STFT loss
   - optional SDR-oriented loss
   - mixture-consistency loss or projection
6. Train small, medium, and latency-oriented floating-point configurations.
7. Profile CPU inference before quantization and identify the dominant operators.

### Deliverables

- Reproducible FP32 training and inference commands
- Baseline checkpoints and training curves
- Per-stem validation report
- Model card containing parameter count, MAC estimate, peak memory, and real CPU profile

### Gate 1

Proceed to expensive QAT when one joint model:

- Produces all four stems without shape, phase, or overlap-add failures.
- Reaches at least 7.5 dB overall SDR on the agreed development metric, or provides enough evidence that scaling the selected configuration will do so.
- Has a compute profile compatible with the latency target after plausible kernel speedups.

If quality is insufficient, improve the floating-point architecture before introducing ternary error.

## Phase 2 — Quantization-Aware Training (Weeks 4–6)

### Stage A: sensitivity and warm-start experiments

1. Start from the best FP checkpoint.
2. Quantize one layer family at a time:
   - TDF Linear layers
   - bottleneck convolutions
   - encoder convolutions
   - decoder convolutions
   - first and final projections
3. Measure immediate and recovered SDR loss.
4. Compare per-tensor and per-output-channel scaling.
5. Compare activation FP16 simulation and W1.58A8 fake quantization.

### Stage B: QAT recipe search

Compare:

- Warm-start QAT versus ternary training from scratch.
- Immediate ternarization versus progressive layer conversion.
- Fixed, learned, and zero-ratio-controlled thresholds.
- 20%, 40%, and 60% target zero ratios.
- BitNet abs-mean versus reconstruction-optimal per-channel scale.
- No sparsity regularizer versus weak scheduled regularization.

Use short runs to eliminate weak configurations, then conduct one full training run on the selected recipe.

### Training safeguards

Log per layer:

- zero, positive, and negative weight fractions
- latent and quantized weight distributions
- scale and threshold values
- activation clipping and saturation rates
- gradient norms
- SDR by stem

Keep task loss dominant. Quantization reconstruction and sparsity penalties are auxiliary losses and must not overwhelm source-separation learning.

### Deliverables

- QAT modules with deterministic export behavior
- Ablation report
- Best ternary checkpoint
- Packed-weight export manifest

### Gate 2

Target thresholds:

- Preferred: no more than 0.5 dB overall SDR loss relative to the matched FP model.
- Maximum acceptable for continued stretch-target work: 1.0 dB loss.
- Stable zero ratio and activation saturation across validation tracks.

If loss exceeds 1.0 dB, use a mixed W1.58/W4/FP model based on layer sensitivity rather than forcing every layer to ternary.

## Phase 3 — Export and Native Runtime (Weeks 7–9)

### Runtime architecture

Start with ONNX Runtime for graph orchestration and custom packed operators:

- `TernaryGemm`
- `TernaryConv2D`

If custom-op dispatch, tensor conversion, or im2col overhead is too high, replace ONNX Runtime with a small purpose-built static executor. ONNX compatibility is secondary to end-to-end performance.

### Kernel paths

Implement in this order:

1. Portable scalar correctness reference.
2. x86-64 AVX2.
3. ARM64 NEON.
4. Optional AVX-512/VNNI and ARM DotProd paths.

Requirements:

- Pack weights once during model export, never during inference.
- Store scales and packed weights in output-channel/tile order.
- Tune tile sizes using actual model shapes and cache sizes.
- Fuse bias, scaling, requantization, and supported activation operations.
- Avoid generic im2col where a direct packed convolution is measurably faster.
- Add runtime ISA detection and deterministic fallback.
- Compare every optimized kernel against the scalar reference bit-for-bit or within documented numerical tolerance.

### Deliverables

- Versioned packed model format
- Exporter from PyTorch checkpoint
- Native CLI and library API
- Scalar, AVX2, and NEON backends
- Kernel and end-to-end benchmarks

### Gate 3

- Native output agrees with fake-quantized PyTorch within the defined tolerance.
- Ternary operators beat the optimized INT8 baseline on end-to-end model time, not only isolated microbenchmarks.
- No runtime weight repacking or full dequantization occurs.
- First latency milestone: under 30 seconds for a three-minute track on at least one reference laptop.
- Stretch latency milestone: under 10 seconds.

## Phase 4 — Validation and Release (Week 10)

### Tasks

1. Run the untouched MUSDB18-HQ test set once the model and runtime are frozen.
2. Report per-stem and overall metrics using the benchmark contract.
3. Benchmark at least one x86-64 AVX2 and one Apple ARM64 laptop.
4. Test difficult cases:
   - silent or nearly silent sources
   - dense cymbals and high-frequency content
   - hard-panned material
   - vocal effects and vocal chops
   - clipping and unusually quiet mixtures
   - songs longer than the training chunks
5. Conduct blind listening comparisons against the FP model and HT-Demucs.
6. Verify installation, model conversion, inference, licensing notices, and reproducibility from a clean machine.

### Deliverables

- Test-set quality report
- Cross-platform benchmark report
- Released source, packed checkpoint, model card, and CLI
- Documented limitations and fallback INT8/mixed-precision configuration

## 5. Acceptance Criteria

### Minimum viable research result

- Reproducible joint four-stem FP model.
- Working W1.58A8 QAT implementation.
- Native packed ternary kernels on AVX2 and NEON.
- Complete quality/latency comparison against FP32 and INT8.
- Honest negative result is acceptable if ternary audio regression or kernel economics fail a gate.

### Successful first release

- Overall SDR of at least 7.5 dB under the frozen MUSDB18-HQ-only protocol.
- Ternary model loses no more than 0.5 dB against its FP counterpart.
- Under 30 seconds for a three-minute track on a reference laptop.
- Reproducible one-command inference and published benchmark details.

### Stretch release

- Quality comparable to HT-Demucs under an equivalent data and evaluation protocol.
- Under 10 seconds for a three-minute track.
- No test-time ensemble or shift multiplication hidden in the latency comparison.

## 6. Repository Layout

```text
TernaryStem/
  README.md
  pyproject.toml
  configs/
    data/
    model/
    train/
    quant/
  docs/
    PLAN.md
    BENCHMARK_PROTOCOL.md
    MODEL_CARD.md
  src/ternarystem/
    audio/
    data/
    models/
    quant/
    losses/
    export/
    evaluation/
  scripts/
    train.py
    evaluate.py
    separate.py
    export.py
    benchmark.py
  runtime/
    CMakeLists.txt
    include/
    src/
    kernels/
      scalar/
      avx2/
      neon/
      avx512/
    tests/
    benchmarks/
  tests/
    unit/
    integration/
    audio/
  results/
    README.md
```

Large datasets, checkpoints, generated stems, and benchmark binaries must not be committed.

## 6.1 Training Execution Contract

Full training is not expected to run on the local CPU development machine.

- Local machine: unit/integration tests, synthetic and tiny-data probes, export, kernel development, and CPU inference benchmarks.
- Remote GPU machine: MUSDB18-HQ storage, reduced/full FP32 training, QAT, layer sensitivity sweeps, and full validation.
- Transfer only source commits, resolved configs, experiment JSON, checkpoints, packed models, and hashes. Never commit datasets or credentials.
- Every remote run must be resumable from `latest.pt`, preserve `best.pt`, and record the source commit and dirty status.
- Before a full run, execute a reduced configuration that proves decreasing loss, improving validation SDR, correct stem shapes, and successful checkpoint resume.

## 7. Experiment Tracking

Every experiment must record:

- Git commit and dirty status
- full resolved configuration
- random seeds
- dataset split hash
- hardware and software versions
- precision and quantized-layer coverage
- per-layer ternary statistics
- quality metrics
- end-to-end and per-operator latency
- checkpoint and packed-model hashes

Use a local structured format as the source of truth; external tracking services may mirror it but must not be required for reproducibility.

## 8. Principal Risks and Responses

| Risk | Early signal | Response |
|---|---|---|
| Ternary quality collapse | More than 1 dB loss after QAT recovery | Keep sensitive layers W4/FP; distill from FP model; increase width selectively |
| Ternary slower than INT8 | Microbenchmark or end-to-end speedup below 1.3x | Change tensor shapes/layouts, use LUT kernel, or ship W4A8/W8A8 |
| Decoder dominates runtime | Profile shows high-precision decoder is the bottleneck | Ternarize internal decoder layers; retain only final projection in FP32 |
| Unstructured zeros provide no speedup | Latency unchanged across zero ratios | Stop sparsity targeting; optimize dense packed ternary execution |
| Packing/unpacking dominates | High frontend or conversion time | Offline packing, fused kernels, static tensor layouts |
| FP16 CPU path regresses | Conversion overhead or unsupported ISA | Use INT8 core with FP32 boundaries |
| STFT/chunk overhead dominates | Kernels are fast but end-to-end time is not | Tune FFT library, hop size, overlap, batching, and threading |
| Benchmark is incomparable | Different data, SDR, shifts, or extra data | Enforce the benchmark contract and report all inference multipliers |
| MUSDB licensing blocks release | Unclear checkpoint redistribution terms | Resolve before public checkpoint release; publish code/configs regardless |

## 9. Immediate Next Actions

1. Run the matched long-budget direct-estimate and bounded complex-mask FP32 configurations remotely, and compare both against their equal-share development baselines; the first remote smoke remained at negative diagnostic SDR.
2. Use the now-recorded GPU/software metadata and latest/best checkpoint hashes, and verify exact scheduler resume on the remote host before committing a long run.
3. Improve the FP architecture/training recipe based on per-stem development diagnostics until it produces meaningful separation; do not begin expensive QAT from another incapable FP baseline.
4. Run W4A8 and W8A8 family sensitivity from the matched capable FP checkpoint, preserving projections in FP32.
5. Repeat selective ternary QAT at 20%, 40%, and 60% zero targets and compare adaptive versus BitNet-style quantization only after the FP baseline is adequate.
6. Train matched W4A8, W8A8, and mixed continuations selected by sensitivity, each with an equal-compute FP control.
7. Train and profile the selected FP32 baseline before making broad deployment decisions.
8. Add matched scale/bias/requantization to BitNet and FBGEMM comparisons and build per-shape selective dispatch.
9. Implement and benchmark native offline-packed W4/W8 operator paths before making deployment-speed claims.
10. Evaluate the best mixed-precision model against its matched FP32 checkpoint; do not touch the official test set until the recipe is frozen.
11. Validate NEON economics on ARM hardware after x86 precision selection is stable.
