# Native runtime experiment history

This is a compact historical record of local native-runtime research conducted after commit `8110cb4fc8a0f6f4f7b6fa76877654f84870248c`. The experimental implementation, bulk machine evidence, audio, checkpoints, native model containers, captures, and binaries are intentionally **not published on `main`**.

A phase result below applies only to its recorded narrow scope. It is not a claim of perceptual equivalence, complete-track validation, portability, production readiness, or general acceleration.

## Phase 1 — scalar packed-QAT contract

A C++17, one-thread, ISA-neutral scalar prototype implemented strict TSRC loading, exact ties-to-even INT8 activation quantization, tensor-flat ternary decoding, INT32 accumulation, and ordered FP32 epilogues. All 24 selected Candidate-A operator calls passed exact activation-code, ternary-code, and INT32 checks across the registered calibration and held-out profiles.

## Phase 2 — fixed graph and sampled waveform validation

The prototype was extended with the fixed graph, STFT/iSTFT, masks, mixture consistency, chunking/overlap-add, and caller-owned arenas. Calibration and synthetic records passed the frozen gates. The frozen Phase 2 tolerance file had SHA-256 `0f26212dd35a1a76b38a86295bb97d2ed6ecdeaf5f3e39e4d8a445a617f24821`; remediation retained those limits rather than adopting held-out-informed changes.

A complete-track attempt was externally interrupted before atomic evidence publication and was deliberately not restarted. Revised proportional validation used four deterministic six-second Falcon samples starting at 0, 120, 180, and 243 seconds. Those samples passed the unchanged limits; they do not establish complete-track or assembled multi-chunk parity.

## Phase 3 — proportional profiling

Exactly one normal six-second ANiMAL fixture and one 44,100-valid-sample padded edge fixture were timed. Selected ternary operations accounted for about 80.34% of measured execution, FP32 temporal convolutions about 17.85%, and normalization/elementwise work about 1.18%. This justified only a narrow selected-operation feasibility comparison.

## Phase 4A — isolated packed AVX2 feasibility

An opt-in AVX2 backend was isolated in backend-specific translation units while scalar loading, execution, and dispatch remained baseline-ISA. All-24 exact gates covered 379,076,544 activation codes, 1,849,152 ternary codes, and 328,130,208 INT32 accumulators; complete outputs were byte-identical in the measured fixtures.

The single six-second observation measured 6.439x selected-operation and 2.992x complete-chunk speedups over scalar. The padded-tail observation measured 6.771x and 3.198x. These are one-platform feasibility observations, not benchmark distributions.

## Phase 4B — predecoded INT8 comparator

A second opt-in AVX2 design retained canonical packed weights but expanded them once during context preparation. Execute performed no allocation, decoding, packing, or prepacking. Three cyclic-order trials per backend/profile found the predecoded representation faster than packed AVX2:

- Six seconds: 1.424x selected-operation and 1.210x complete-chunk advantage.
- Padded tail: 1.444x selected-operation and 1.061x complete-chunk advantage.

The tradeoff was approximately 6.5–6.9 ms one-time preparation and 1,849,216 additional runtime-arena bytes. This supports a future opt-in design investigation only.

## Current disposition

- Packed QAT remained the semantic identity; load-time expansion changed representation, not arithmetic.
- Scalar remained the default and unsupported forced AVX2 requests failed closed.
- The complete-track attempt remains interrupted, not passed or numerically failed.
- Formal Gate 0, quality, deployment, portability, energy, threading, and production gates remain open.
- No Phase 5 work was started.

The canonical external TSRC used by the experiments had SHA-256 `067d88c5448a37ac46372769e21ce50ef4ce1b01af5682372c8e428e915359a4`. Bulk evidence and implementation snapshots remain outside the published branch to keep the repository small. Model files were not published because MUSDB18-HQ permits educational use but does not clearly grant unrestricted redistribution rights for trained weights; the [model card](MODEL_CARD.md) records the privately retained artifact identities.
