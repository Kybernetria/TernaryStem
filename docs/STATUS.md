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
- Streaming MUSDB chunk/remix augmentation, fixed validation chunks, best/latest checkpoints, exact optimizer/epoch/scheduler resume, and separate FP-to-QAT warm-start loading with fresh optimizer/scheduler state.
- Schema-v3 development diagnostics now report canonical energy-aggregated overall/per-stem `global_sdr`, historical mean-chunk SDR, waveform L1, and the matched `mixture / 4` equal-share baseline. Records and console output label all diagnostics as not BSSEval.
- Checkpoint-compatible output parameterization supports the original direct complex estimate and a bounded Cartesian complex mask applied to the retained mixture spectrogram. Both preserve FP32 reconstruction, frequency padding, source/stereo shapes, and waveform mixture consistency.
- Matched long-run FP32 direct-estimate and complex-mask configurations use the same split, seed, model capacity, validation chunks, budget, and cosine learning-rate schedule.
- Experiment schema v2 records the selected device, PyTorch/CUDA availability and versions, GPU model when available, relevant package versions, and post-write hashes for both latest and best checkpoints. A compact comparison command summarizes model versus equal-share, output mode, FP32/mixed precision, and best/final development diagnostics.
- Reduced FP32, medium FP32, ternary-QAT, W4A8, W8A8, and mixed-precision remote configurations; all resolve successfully in local dry runs. Remote records now verify the larger 632,208-parameter configuration used in the first music-data experiment.
- A development-split layer-family sensitivity command that records immediate diagnostic loss/global-SDR deltas, parameter coverage, quantization statistics, activation saturation, and resolved configurations. Ternary sensitivity has run on MUSDB development audio; W4/W8 sensitivity remains open.
- Reproducible oneDNN/FBGEMM quantized Linear benchmark harness and recorded FBGEMM results.
- Optional exact BitNet.cpp I2_S benchmark adapter and matching exporter layout.

## Verified remote development smoke

A user-operated remote GPU run on the frozen MUSDB18-HQ development split is recorded under `results/remote/2026-07-17-selective-ternary/`. A 632,208-parameter FP model learned over 30 short epochs but remained at a poor absolute validation diagnostic `global_sdr` of -3.5326 dB. Immediate family sensitivity selected TDF Linear and bottleneck convolution for ternary QAT, covering 81.63% of eligible weights while preserving sensitive encoder/decoder/projection families in FP32.

Matched ten-epoch continuations from the same FP checkpoint reached -3.1445 dB for FP32 and -3.1669 dB for selective ternary QAT, a -0.0224 dB diagnostic difference. This demonstrates recovery on a reduced development task only. It is not BSSEval, does not establish useful separation, and does not pass Gate 1 or Gate 2. The remote record omitted GPU identity and PyTorch/CUDA versions; no remote latency statement is made.

## Colab engineering pilot (operator-retained artifacts)

A T4 pilot validated the 632,208-parameter complex-mask shape with six-second chunks, batch four, full losses, checkpoint recovery, FP16 autocast, selective ternary QAT, chunked overlap-add inference, silence handling, and deterministic export. FP32 reached 2.5480 dB energy-aggregated development global SDR versus a 1.2470 dB equal-share baseline after five short epochs. FP16 was 27% faster and 0.0096 dB lower over that budget, which is not proof of equal full convergence. Eight selective-QAT updates covered 80.10% of parameters and measured -0.1245 dB versus the matched FP checkpoint; the FP model is still incapable, so this is plumbing evidence rather than Gate 1/2. Listening revealed substantial leakage and slightly better FP32 drums/bass. Silent input produced exact zero, packed export contained 13 ternary weight tensors, and overlap-add reconstruction reached about 3.6e-7 maximum mixture error after fixing a long-window boundary bug. Checkpoints/audio remain outside Git.

## Verification in this environment

C++ configuration/build and scalar/AVX2 correctness tests pass. The full Python suite passes (58 tests), and Ruff passes. Smoke, Colab pilot, and matched long-run configurations pass local construction/configuration checks. Direct/mask shape, reconstruction, backward, frequency-padding, mixture-consistency, worker-independent sampling, legacy-checkpoint, scheduler/scaler-resume, diagnostic, comparison, persistence, distillation, and metadata tests pass. Python dependencies were installed in an isolated CPU-only distrobox environment.

## Open gates

Gate 0 is **not passed**. Against production FBGEMM INT8, the exact BitNet.cpp adapter ranges from 0.69x to 3.49x and misses the required 1.3x on four of five tested shapes; see `docs/GATE0_REPORT.md`. Selective BitNet dispatch remains plausible. Reduced music-data learning, ternary family sensitivity, and a matched selective-QAT continuation have now executed, but the FP separator quality is inadequate and no native W4/W8 runtime benchmark exists. Published-checkpoint reproduction, a capable FP baseline, W4/W8 sensitivity and training, matched fused epilogues, and ARM NEON measurements remain open. No quality or optimized latency values have been invented. Phases 1–4 remain experimental work, not completed deliverables.
