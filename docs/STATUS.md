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

## Vast RTX 5090 FP32 baseline

Deployment #1 is recorded under `results/remote/2026-07-23-vast-fp32-plateau/`. The clean commit `497d4f9aec2d01b9902aa53a428570b28936891c` trained the 632,208-parameter FP32 bounded-complex-mask model for 31 completed epochs / 310,000 dynamically remixed chunks. The best checkpoint was record epoch index 23 (the 24th completed epoch) at 5.4480 dB development `global_sdr`, with same-checkpoint vocals/drums/bass/other values of 6.1231/5.9805/5.4315/4.2568 dB versus a 1.2455 dB equal-share overall baseline. Training loss continued to decline while validation remained roughly 5.2–5.45 dB, so the operator stopped the run. The expected signal exit was recorded as pipeline failure status 130. No sensitivity or QAT ran.

The RTX 5090 sustained approximately 53–54 chunks/s, with about 5.39 GiB peak allocated and 6.42 GiB peak reserved. The run bundle and checkpoint hashes were independently copied to and content-checked on Filen; checkpoints remain outside Git. This result is not BSSEval and is 2.0520 dB below Gate 1.

Deployment #2 is locked in `docs/DEPLOYMENT_2.md` as an FP-only, $40-absolute-cap architecture screen. Local implementation now includes an architecture registry; the exact 2,398,783-parameter full-spectrum TernaryStem-v2 candidate; the pinned 10,578,768-parameter SCNet core and separately labeled canonical adapter; RNG-complete schema-v2 checkpoints; immutable-horizon rung stops; stable validation identities and float64 track energies; an FP-only hash allowlist; machine-readable gates; a hash-chained billing ledger; verified synchronization receipts; and PID/PGID identity supervision. The 31-test fake-controller matrix now covers transaction crash recovery, receipt-last immutable synchronization, concurrent-sync refusal, ledger/state gates, and process supervision. The implementation is committed and bound by the ignored `.deployment2-lock.json` source-lock artifact in the current checkout. Deployment remains `NO-GO` only because authorized local CUDA evidence is absent.

## Verification in this environment

C++ configuration/build and scalar/AVX2 correctness tests pass. In Generalbox with CPU PyTorch 2.13.0, the full Python suite passes (118 tests; six known multiprocessing/upstream rectangular-window warnings), and Ruff passes. Both deployment #2 candidates construct at their exact frozen parameter counts and their synthetic forward, backward, tiny-loss-decrease, source-order, mixture-consistency, provenance, inventory, and checkpoint tests pass. CUDA-required readiness remains deliberately open; CPU execution is not accepted as substitute evidence.

## Open gates

Gate 0 is **not passed**. Against production FBGEMM INT8, the exact BitNet.cpp adapter ranges from 0.69x to 3.49x and misses the required 1.3x on four of five tested shapes; see `docs/GATE0_REPORT.md`. Selective BitNet dispatch remains plausible. The 310,000-chunk FP baseline failed Gate 1 at 5.4480 dB, so no further QAT is authorized from that model. The existing sensitivity path must also gain explicit observer calibration, canonical energy-aggregated development diagnostics, and run-wide saturation aggregation before future QAT decisions. A capable FP baseline, complete native graph/runtime, native W4/W8 paths, matched fused epilogues, and ARM NEON measurements remain open. No quality or optimized latency values have been invented. Phases 1–4 remain experimental work, not completed deliverables.
