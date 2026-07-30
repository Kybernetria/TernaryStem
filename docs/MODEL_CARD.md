# TernaryStem Model Card

## Status

Concluded, unmaintained research experiment with recorded remote development runs and proportional native-runtime feasibility work. No checkpoint is released and no separation-quality claim is made. The repository is source/history only.

## Intended use

Research on four-source separation (vocals, drums, bass, other) and low-precision CPU inference for stereo 44.1 kHz music. Not intended for safety-critical use or for asserting ownership of separated material.

## Architecture

A configurable joint TFC-TDF U-Net predicts either four direct complex source spectrograms or four stereo complex masks. Direct estimate is the default and preserves existing checkpoint behavior. Complex-mask mode bounds the real and imaginary mask components independently with `tanh`, then multiplies the resulting complex mask by the retained mixture spectrogram. This smooth Cartesian bound avoids unbounded complex ratio masks while retaining phase rotation. Frequency padding, final estimate construction, additive mixture consistency, and iSTFT remain FP32; both modes preserve exact waveform mixture consistency within the frozen numerical tolerance.

Per-layer deployment precision can be FP32, ternary, W4A8, or W8A8. Quantization-aware layers retain latent FP32 weights and use fake-quantized weights and symmetric INT8 activations in forward passes. Projections, norms, STFT/iSTFT, and reconstruction remain FP32 by default.

Deployment #2 adds two unreleased FP candidates. `ternarystem_v2_fullband_temporal_v1` has exactly 2,398,783 trainable parameters, models all 2,049 bins with odd-safe frequency scaling, uses residual TFC-TDF-v3-style blocks and a 91-frame explicit temporal stage, and retains bounded Cartesian complex masks. `scnet_5d95bf96_adapted_v1` preserves a byte-pinned MIT SCNet core with exactly 10,578,768 trainable parameters, then reorders its direct waveform estimates and applies waveform mixture consistency. The adapter is not claimed as an exact published-system reproduction. Neither candidate has paid quality evidence.

## Training and evaluation

The benchmark contract used MUSDB18-HQ and the frozen 86/14 development split in `BENCHMARK_PROTOCOL.md`. Dataset audio and weights are not distributed here. The official [MUSDB18-HQ record](https://doi.org/10.5281/zenodo.3338373) limits the material to educational use and identifies heterogeneous underlying rights: DSD100/Mixing Secrets, CC BY-NC-SA 4.0 MedleyDB tracks, Native Instruments tracks, and CC BY-NC-SA 3.0 Easton Ellises tracks. Neither that record nor the [MUSDB documentation](https://sigsep.github.io/datasets/musdb.html) clearly grants unrestricted redistribution rights for trained weights. The project therefore publishes hashes and provenance, not model files.

The first recorded remote smoke experiment is under `results/remote/2026-07-17-selective-ternary/`. Its 632,208-parameter FP warm-up remained at -3.5326 dB validation diagnostic `global_sdr`. Matched ten-epoch continuations reached -3.1445 dB for FP32 and -3.1669 dB when TDF Linear and bottleneck convolution families used selective ternary QAT. The -0.0224 dB difference is evidence for reduced-task QAT recovery only. It is not museval/BSSEval and the negative absolute diagnostic confirms that this is not a useful released separator.

A bounded-complex-mask FP32 baseline is recorded under `results/remote/2026-07-23-vast-fp32-plateau/`. The 632,208-parameter model completed 31 epochs / 310,000 chunks on an RTX 5090. Its best checkpoint was record epoch index 23 (the 24th completed epoch, after 240,000 chunks) at 5.4480 dB development `global_sdr`, with same-checkpoint vocals/drums/bass/other values of 6.1231/5.9805/5.4315/4.2568 dB. This failed the 7.5 dB FP gate, the run was stopped, and no sensitivity or QAT ran.

The official MUSDB test set has not been evaluated. Training records emit explicitly labeled development-only overall/per-stem `global_sdr`, per-stem waveform L1, and a trivial equal-share (`mixture / 4`) baseline. These are not museval/BSSEval.

## Preserved artifact identities

The following privately retained artifacts were judged potentially useful for future research but are intentionally not downloadable from this public archive:

| Artifact | Size | SHA-256 |
|---|---:|---|
| Candidate-A pre-QAT best checkpoint | 29,167,307 bytes | `c66f1df99948d0c4abcfa09f4c1188b6246ac70cf1e8003c2880589e90bfd23c` |
| Candidate-A final QAT best checkpoint | 28,972,811 bytes | `4f4098b2c708525918ad182bd5b077d47b5b975518c26e4f46b1ce9ee0ae8a4b` |
| Candidate-A canonical packed TSRC | 2,740,800 bytes | `067d88c5448a37ac46372769e21ce50ef4ce1b01af5682372c8e428e915359a4` |

These identities do not imply that the artifacts are safe to deserialize, licensed for redistribution, quality-qualified, or recoverable from the repository. Raw PyTorch checkpoints are pickle-based and should not be loaded from an untrusted source.

## Limitations

- No quality-qualified or published checkpoint exists.
- The FP architecture/training recipe requires substantial improvement before quantization quality gates can be evaluated.
- The historical remote smoke record omitted GPU identity and PyTorch/CUDA versions; the updated recorder cannot retroactively supply them and leaves missing values unknown.
- Direct-estimate and complex-mask pilots are not sufficient to establish an architecture winner; the qualifying complex-mask baseline failed Gate 1.
- Native packed ternary kernels exist, but all-ternary deployment failed the interim optimized INT8 comparison.
- Native W4/W8 operators and end-to-end mixed-precision runtime integration are incomplete.
- No end-to-end model latency claim exists.
