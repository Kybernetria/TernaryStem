# Vast RTX 5090 FP32 complex-mask baseline — 2026-07-23

## Scope

This is a stopped MUSDB18-HQ **development-split diagnostic**, not an official test-set evaluation. It used the frozen 86/14 split, seed `20250218`, 10,000 dynamically remixed six-second training chunks per completed epoch, 140 fixed development chunks, and the separately named energy-aggregated `global_sdr` diagnostic. No museval/BSSEval result or released-checkpoint quality claim follows from this run. The official test partition was not extracted on the training path.

The clean detached source checkout was commit `497d4f9aec2d01b9902aa53a428570b28936891c`. The run used PyTorch 2.11.0+cu128 on an NVIDIA GeForce RTX 5090 with 32 GB VRAM. The complete non-dataset run bundle and checkpoints were copied to independent Filen storage and content-checked after the stop. Checkpoints remain outside Git.

## Result

The 632,208-parameter FP32 bounded-complex-mask model completed 31 epochs (310,000 chunks) before an operator stop after the development metric plateaued. The pipeline exited with the expected signal status 130 and did not enter sensitivity or QAT.

The best checkpoint is record epoch index 23, the **24th completed epoch**:

| Metric | Epoch 1 | Best (completed epoch 24) | Final (completed epoch 31) |
|---|---:|---:|---:|
| development `global_sdr` | 3.3151 dB | **5.4480 dB** | 5.2606 dB |
| training loss | 0.28549 | 0.18168 | 0.17915 |
| equal-share development `global_sdr` | 1.2455 dB | 1.2455 dB | 1.2455 dB |

Best-checkpoint per-stem development `global_sdr`:

| Stem | SDR |
|---|---:|
| vocals | 6.1231 dB |
| drums | 5.9805 dB |
| bass | 5.4315 dB |
| other | 4.2568 dB |

The best score is 2.0520 dB below the configured 7.5 dB FP gate. Training loss continued to decline while validation remained roughly 5.2–5.45 dB. Stopping does not mathematically prove that the original 100-epoch cosine schedule could never improve later, but the observed result does not justify QAT.

## Performance

Sustained throughput was approximately 53–54 chunks/s. A representative completed epoch took about 186–191 seconds including development validation. PyTorch reported about 5.39 GiB peak allocated and 6.42 GiB peak reserved; observed process GPU memory was about 7.2 GB. The GPU generally ran around 85–97% utilization, 67–71°C, and 2,850 MHz without thermal slowdown.

## Artifact hashes

`checkpoint-sha256.txt` records the independently verified hashes of the uncommitted `best.pt` and `latest.pt`. `experiment.json` is the schema-v3 structured run record retrieved from the verified off-instance bundle; absolute runtime path values were replaced with explicit redaction markers before version control and the redacted fields are listed in `record_redactions`.

## Decision

Gate 1 failed at the observed budget. Do not run sensitivity or QAT from this checkpoint. Deployment #2 is an FP-only, cost-capped architecture screen defined in `docs/DEPLOYMENT_2.md`.
