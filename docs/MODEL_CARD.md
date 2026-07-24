# TernaryStem Model Card

## Status

Research scaffold with recorded remote development experiments, including one stopped RTX 5090 FP32 complex-mask baseline. No checkpoint is released and no separation-quality claim is made.

## Intended use

Research on four-source separation (vocals, drums, bass, other) and low-precision CPU inference for stereo 44.1 kHz music. Not intended for safety-critical use or for asserting ownership of separated material.

## Architecture

A configurable joint TFC-TDF U-Net predicts either four direct complex source spectrograms or four stereo complex masks. Direct estimate is the default and preserves existing checkpoint behavior. Complex-mask mode bounds the real and imaginary mask components independently with `tanh`, then multiplies the resulting complex mask by the retained mixture spectrogram. This smooth Cartesian bound avoids unbounded complex ratio masks while retaining phase rotation. Frequency padding, final estimate construction, additive mixture consistency, and iSTFT remain FP32; both modes preserve exact waveform mixture consistency within the frozen numerical tolerance.

Per-layer deployment precision can be FP32, ternary, W4A8, or W8A8. Quantization-aware layers retain latent FP32 weights and use fake-quantized weights and symmetric INT8 activations in forward passes. Projections, norms, STFT/iSTFT, and reconstruction remain FP32 by default.

Deployment #2 adds two unreleased FP candidates. `ternarystem_v2_fullband_temporal_v1` has exactly 2,398,783 trainable parameters, models all 2,049 bins with odd-safe frequency scaling, uses residual TFC-TDF-v3-style blocks and a 91-frame explicit temporal stage, and retains bounded Cartesian complex masks. `scnet_5d95bf96_adapted_v1` preserves a byte-pinned MIT SCNet core with exactly 10,578,768 trainable parameters, then reorders its direct waveform estimates and applies waveform mixture consistency. The adapter is not claimed as an exact published-system reproduction. Neither candidate has paid quality evidence.

## Training and evaluation

The benchmark contract uses MUSDB18-HQ and the frozen 86/14 development split in `BENCHMARK_PROTOCOL.md`. Dataset audio and weights are not distributed here. Redistribution terms must be reviewed before publishing a trained checkpoint.

The first recorded remote smoke experiment is under `results/remote/2026-07-17-selective-ternary/`. Its 632,208-parameter FP warm-up remained at -3.5326 dB validation diagnostic `global_sdr`. Matched ten-epoch continuations reached -3.1445 dB for FP32 and -3.1669 dB when TDF Linear and bottleneck convolution families used selective ternary QAT. The -0.0224 dB difference is evidence for reduced-task QAT recovery only. It is not museval/BSSEval and the negative absolute diagnostic confirms that this is not a useful released separator.

A bounded-complex-mask FP32 baseline is recorded under `results/remote/2026-07-23-vast-fp32-plateau/`. The 632,208-parameter model completed 31 epochs / 310,000 chunks on an RTX 5090. Its best checkpoint was record epoch index 23 (the 24th completed epoch, after 240,000 chunks) at 5.4480 dB development `global_sdr`, with same-checkpoint vocals/drums/bass/other values of 6.1231/5.9805/5.4315/4.2568 dB. This failed the 7.5 dB FP gate, the run was stopped, and no sensitivity or QAT ran.

The official MUSDB test set has not been evaluated. Training records emit explicitly labeled development-only overall/per-stem `global_sdr`, per-stem waveform L1, and a trivial equal-share (`mixture / 4`) baseline. These are not museval/BSSEval.

## Limitations

- No quality-qualified or published checkpoint exists.
- The FP architecture/training recipe requires substantial improvement before quantization quality gates can be evaluated.
- The historical remote smoke record omitted GPU identity and PyTorch/CUDA versions; the updated recorder cannot retroactively supply them and leaves missing values unknown.
- Direct-estimate and complex-mask pilots are not sufficient to establish an architecture winner; the qualifying complex-mask baseline failed Gate 1.
- Native packed ternary kernels exist, but all-ternary deployment failed the interim optimized INT8 comparison.
- Native W4/W8 operators and end-to-end mixed-precision runtime integration are incomplete.
- No end-to-end model latency claim exists.
