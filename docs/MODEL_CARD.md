# TernaryStem Model Card

## Status

Architecture scaffold only; no checkpoint is released and no quality claim is made.

## Intended use

Research on four-source separation (vocals, drums, bass, other) and low-precision CPU inference for stereo 44.1 kHz music. Not intended for safety-critical use or for asserting ownership of separated material.

## Architecture

A configurable joint TFC-TDF U-Net predicts four complex source spectrograms. An additive mixture-consistency projection is applied before FP32 iSTFT. Core layers can use latent FP32 weights with fake ternary forward passes and symmetric INT8 fake-quantized activations.

## Training and evaluation

Planned data is MUSDB18-HQ under `BENCHMARK_PROTOCOL.md`. Dataset audio and weights are not distributed here. Redistribution terms must be reviewed before publishing a trained checkpoint.

## Limitations

The current code is untrained, CPU runtime integration is incomplete, and packed kernels have not been benchmarked against an optimized library INT8 implementation. Gate 0 remains open.
