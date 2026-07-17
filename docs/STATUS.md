# Implementation Status

## Completed locally

- Repository/package/config/CI scaffold.
- Frozen benchmark protocol and canonical musdb 86/14 split policy.
- FP32 STFT/iSTFT, overlap-add, mixture consistency, configurable joint TFC-TDF U-Net, losses, and inference CLI.
- Adaptive and abs-mean ternary references, identity STE, per-output scales, zero-ratio control, and static/EMA/learned symmetric activation fake quantization.
- Symmetric W8 per-output-channel and grouped/per-output-channel W4 fake quantization with identity STE and latent FP32 weights.
- Per-family and exact-layer precision selection across ternary, W4A8, W8A8, and FP32; projections, norms, STFT/iSTFT, and reconstruction remain FP32 by default.
- Deterministic ternary, signed-nibble W4, and W8 export references with versioned precision/scale/group metadata and a local experiment-record schema.
- Exact scalar packed-ternary and scalar INT8 GEMM references with CMake correctness tests.
- Runtime-dispatched AVX2 packed-ternary and INT8 prototypes with exact vector-path tests.
- Unit/integration tests, synthetic QAT probes, and a captured operator-shape inventory.
- Streaming MUSDB chunk/remix augmentation, fixed validation chunks, best/latest checkpoints, exact optimizer/epoch resume, and separate FP-to-QAT warm-start loading with fresh optimizer state.
- Reduced FP32, ternary-QAT, W4A8, W8A8, and mixed-precision remote smoke configurations; all resolve successfully in local dry runs.
- A development-split layer-family sensitivity command that records immediate diagnostic loss/global-SDR deltas, parameter coverage, quantization statistics, activation saturation, and resolved configurations. It has not yet been run on MUSDB audio.
- Reproducible oneDNN/FBGEMM quantized Linear benchmark harness and recorded FBGEMM results.
- Optional exact BitNet.cpp I2_S benchmark adapter and matching exporter layout.

## Verification in this environment

C++ configuration/build and scalar/AVX2 correctness tests pass. The full Python suite passes (28 tests), and Ruff passes. All five remote smoke configurations pass local model-construction dry runs. FP-to-QAT warm starts and exact model/optimizer/epoch resume pass synthetic unit tests; no remote checkpoint resume has been exercised yet. Python dependencies were installed in an isolated CPU-only virtual environment under `/tmp` with pip caching disabled, avoiding the home-directory quota.

## Open gates

Gate 0 is **not passed**. Against production FBGEMM INT8, the exact BitNet.cpp adapter ranges from 0.69x to 3.49x and misses the required 1.3x on four of five tested shapes; see `docs/GATE0_REPORT.md`. Selective BitNet dispatch remains plausible. W4A8/W8A8 training and export plumbing now exists, but no native W4/W8 runtime benchmark or music-quality result has been measured. Published-checkpoint reproduction, reduced music-data training, layer-sensitivity execution, matched fused epilogues, and ARM NEON measurements remain open. No quality or optimized latency values have been invented. Phases 1–4 remain experimental work, not completed deliverables.
