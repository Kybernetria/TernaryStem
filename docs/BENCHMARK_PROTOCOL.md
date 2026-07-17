# Benchmark Protocol (v1.0)

This file is frozen for the first development cycle. Changes require a version bump and invalidate direct comparisons with earlier records.

## Data and split

- Dataset: MUSDB18-HQ, stereo, 44.1 kHz.
- Development: official 100-track training partition split into 86 training and the 14 validation tracks listed in `src/ternarystem/data/musdb18_split.json`.
- Test: all 50 official test tracks, untouched until model/runtime freeze.
- Primary results use no extra audio. Every run records seed `20250218`, split SHA-256, and exact track names.
- An epoch is 10,000 sampled chunks by default, not a pass over a finite augmented dataset.
- Permitted dynamic augmentation: cross-track stem remixing, gain, channel swap, polarity, and declared pitch/time transforms.

## Quality

Primary publication metrics are museval BSSEval v4 SDR for vocals, drums, bass, other, and their unweighted mean. Report track-level median and mean. Training diagnostics use the separately named `global_sdr`, defined as `10 log10(sum(target²) / (sum((target-estimate)²) + 1e-8))` over channels and time; it must not be called BSSEval SDR.

Record chunk size, overlap, window, shifts, ensembles, source-specific tuning, and extra data. Default inference uses 6 s chunks, 50% overlap, Hann synthesis weighting, no shifts, and no ensemble. Report silent-target handling explicitly.

## Runtime

Input is a checked-in-by-hash/generated three-minute stereo 44.1 kHz PCM WAV. Timing spans file decode, STFT, model, iSTFT, overlap-add, and output preparation; also report model-only timing. Run one cold pass and at least ten warm passes, reporting median and p95.

Record input SHA-256, CPU model, OS, ISA, compiler and flags, thread count/affinity, power mode, RAM, peak RSS, packed model size, commit/dirty state, and software versions. Weight packing is export-time and excluded only because packed weights are the deployment artifact; runtime repacking is forbidden.

Baseline ISA targets are x86-64 AVX2 and ARM64 NEON. Optional AVX-512, VNNI, DotProd, shifts, or ensembles must have separate rows. The optimized INT8 baseline must use the same shapes, threads, and fused epilogue as closely as practical.

## Numerical agreement

- FP32 STFT/iSTFT reconstruction: max absolute error <= `2e-5` for normalized test signals.
- Packed scalar kernel versus mathematical integer reference: exact INT32 accumulators.
- SIMD versus scalar: exact INT32 accumulators before scaling; post-scale max absolute error <= `1e-5`.
- Native model versus exported fake-quantized PyTorch: max absolute spectrogram error <= `2e-4` and waveform error <= `5e-4`, unless an operator documents a tighter or shape-dependent bound.

## Required record

Each experiment stores resolved config, UTC time, commit and dirty status, seeds, split hash, host/software details, precision coverage, per-layer ternary statistics, quality metrics, timings, and SHA-256 hashes for checkpoint and packed model. JSON is the local source of truth. Missing values are `null`, never fabricated.
