# TernaryStem Model Card

## Status

Research scaffold with one recorded remote development smoke experiment. No checkpoint is released and no separation-quality claim is made.

## Intended use

Research on four-source separation (vocals, drums, bass, other) and low-precision CPU inference for stereo 44.1 kHz music. Not intended for safety-critical use or for asserting ownership of separated material.

## Architecture

A configurable joint TFC-TDF U-Net predicts four complex source spectrograms. An additive mixture-consistency projection is applied before FP32 iSTFT. Per-layer deployment precision can be FP32, ternary, W4A8, or W8A8. Quantization-aware layers retain latent FP32 weights and use fake-quantized weights and symmetric INT8 activations in forward passes. Projections, norms, STFT/iSTFT, and reconstruction remain FP32 by default.

## Training and evaluation

The benchmark contract uses MUSDB18-HQ and the frozen 86/14 development split in `BENCHMARK_PROTOCOL.md`. Dataset audio and weights are not distributed here. Redistribution terms must be reviewed before publishing a trained checkpoint.

The first recorded remote smoke experiment is under `results/remote/2026-07-17-selective-ternary/`. Its 632,208-parameter FP warm-up remained at -3.5326 dB validation diagnostic `global_sdr`. Matched ten-epoch continuations reached -3.1445 dB for FP32 and -3.1669 dB when TDF Linear and bottleneck convolution families used selective ternary QAT. The -0.0224 dB difference is evidence for reduced-task QAT recovery only. It is not museval/BSSEval and the negative absolute diagnostic confirms that this is not a useful released separator.

The official MUSDB test set has not been evaluated.

## Limitations

- No quality-qualified or published checkpoint exists.
- The FP architecture/training recipe requires substantial improvement before quantization quality gates can be evaluated.
- The remote smoke record omitted GPU identity and PyTorch/CUDA versions.
- Native packed ternary kernels exist, but all-ternary deployment failed the interim optimized INT8 comparison.
- Native W4/W8 operators and end-to-end mixed-precision runtime integration are incomplete.
- No end-to-end model latency claim exists.
