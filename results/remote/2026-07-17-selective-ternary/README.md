# Selective ternary remote development smoke — 2026-07-17

## Scope

This is a MUSDB18-HQ **development-split diagnostic**, not an official test-set evaluation. It used the frozen 86/14 split, seed `20250218`, 56 fixed six-second validation chunks for training runs, and the separately named `global_sdr` metric. No museval/BSSEval result, released checkpoint, end-to-end latency, or separation-quality claim follows from this experiment.

The records identify commit `2b1890fe8dd00b54c7ffff47d819195672be242c` with a clean working tree. The remote record did not capture the GPU model or PyTorch/CUDA versions, so those fields remain unknown rather than inferred.

## FP32 warm-up

The 632,208-parameter FP32 model trained from scratch for 30 short epochs of 256 dynamically sampled chunks. Its best validation diagnostic occurred at epoch 28:

| Metric | Epoch 0 | Best |
|---|---:|---:|
| validation `global_sdr` | -6.4890 dB | -3.5326 dB |
| validation waveform L1 | 0.038759 | 0.030563 |

The absolute diagnostic SDR remained negative. This run demonstrates learning and provides a matched initialization, not useful separator quality.

## Immediate ternary sensitivity

Each family was independently fake-quantized from the FP checkpoint over 28 development chunks. Coverage percentages below use eligible multi-dimensional weights.

| Family | Eligible coverage | Diagnostic SDR delta |
|---|---:|---:|
| TDF Linear | 54.1% | -0.1147 dB |
| Bottleneck convolution | 27.5% | -0.1485 dB |
| Encoder convolution | 8.5% | -1.6619 dB |
| Decoder convolution | 9.7% | -1.9344 dB |
| First/final projections | 0.1% | -2.2781 dB |

This selected TDF Linear and bottleneck convolution for ternary QAT while retaining encoder, decoder, projections, norms, and signal-processing boundaries in FP32. The selected families comprise 81.63% of eligible weights and 80.10% of all model parameters. The sensitivity deltas are one-family immediate perturbations and are not assumed to add linearly.

## Matched ten-epoch continuation

Both branches warm-started from the same FP checkpoint, whose SHA-256 is `0967a01ba029d9b7520b931e60ae62b46707247cc55a6297095469dfbc310395`. Both used a fresh optimizer, learning rate `1e-4`, ten epochs, and otherwise matched data/loss settings.

| Branch | Best epoch | Validation `global_sdr` | Validation waveform L1 |
|---|---:|---:|---:|
| FP32 control | 8 | -3.144536 dB | 0.02919223 |
| Selective ternary QAT | 9 | -3.166944 dB | 0.02920718 |
| Ternary minus FP32 | — | **-0.022408 dB** | **+0.00001495** |

This is evidence that selective ternary QAT recovered the measured perturbation on this reduced development task. It does **not** satisfy Gate 1 or Gate 2: the FP reference itself lacks meaningful separation quality, best-epoch selection used development diagnostics, and BSSEval was not run.

## Artifacts

- `fp32-large-experiment.json`: FP warm-up record
- `fp32-control-experiment.json`: matched FP continuation
- `selective-ternary-experiment.json`: matched selective-QAT continuation
- `ternary-sensitivity.json`: per-family immediate sensitivity and quantization statistics
- `checkpoint-sha256.txt`: hashes of uncommitted checkpoints retained outside Git
- `source-archive-sha256.txt`: hash of the transfer archive

No dataset audio, generated audio, or checkpoint is committed.
