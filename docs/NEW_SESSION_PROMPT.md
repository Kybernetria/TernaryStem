# New-session starter prompt

Copy the prompt below into a new coding-agent session opened at the TernaryStem repository root.

```text
We are continuing development of TernaryStem. Treat the repository as the source of truth; do not rely on assumptions from this prompt when they conflict with committed files.

Before proposing or changing anything:

1. Run `git status --short --branch` and `git log -3 --oneline`.
2. Read these files completely:
   - `README.md`
   - `docs/VAST_AI.md`
   - `docs/BENCHMARK_PROTOCOL.md`
   - `docs/STATUS.md`
   - `docs/PLAN.md`
3. Inspect the current training/Vast scripts and relevant configs before giving commands.
4. Report the current commit, dirty state, and any mismatch between this summary and the repository.

Main objective

Build one universal four-stem separator (vocals, drums, bass, other) that aims to approach `htdemucs_ft` quality while ultimately processing a three-minute track locally in under ten seconds on a modern CPU. This is a stretch target, not an established result.

Frozen technical direction

- Use a single model, not mixture-of-experts.
- Use bounded complex-mask output; matched smoke runs learned better than direct estimates.
- Keep STFT/iSTFT, losses, norms, projections, complex construction, mixture consistency, and reconstruction FP32.
- Train latent weights and optimizer state in FP32.
- The qualifying full baseline remains FP32 because equal full-run convergence with FP16 has not been proven. A five-epoch T4 pilot found FP16 about 27% faster, about 15% lower allocated memory, zero skipped updates, and only 0.01 dB behind FP32; this is short-run evidence only. BF16/FP16 may be calibrated on the rented GPU but must not silently replace the baseline.
- HTDemucs distillation is implemented but disabled for the expensive run because the matched reduced experiment changed SDR by only about 0.00012 dB at the tested cadence.
- Do not start expensive QAT unless the FP baseline passes the configured quality gate.
- Repeat layer-family sensitivity from the capable FP checkpoint. The provisional selective policy is ternary TDF Linear plus bottleneck convolution. Keep encoder/decoder/projections/norms/reconstruction FP32 unless fresh evidence selects otherwise; W4/W8 are fallbacks.
- Use schema-v3 energy-aggregated development global SDR for checkpoint selection. Mean-chunk SDR is diagnostic only. Neither is BSSEval.
- Keep the official MUSDB18-HQ test partition untouched until the recipe is frozen.

Validated engineering evidence

- Large shape: 632,208 parameters, six-second chunks, batch four, 4096-point STFT.
- T4 FP32 pilot: about 7.98 chunks/s, 5.39 GiB peak allocated, best 2.5480 dB energy-aggregated development global SDR versus 1.2470 dB equal-share after five short epochs.
- Large selective QAT pilot: 80.10% parameter coverage, approximately 40% zeros, maximum observed activation saturation about 0.10%, zero skipped updates, and -0.1245 dB versus the matched 56-chunk FP checkpoint after eight QAT updates.
- Listening showed genuine early isolation but substantial leakage; FP32 sounded slightly better on drums and bass. The FP model is not capable yet, so the QAT result is plumbing evidence, not Gate 1/2.
- Six-second 50%-overlap inference now reconstructs the mixture within about 3.6e-7 maximum error after a boundary fix.
- Silent input produces exact zero output.
- Packed export succeeded with 13 ternary weight tensors and 95 FP32 tensors; tensor count is not parameter coverage.
- Local distrobox suite passes 58 tests and Ruff at the documented checkpoint.

Artifact locations from the user-operated Colab run

The following are expected in private Google Drive storage and must never be committed:

- `TernaryStem/runs/colab-large-fp32-v3/`
- `TernaryStem/runs/colab-large-fp16-v3/`
- `TernaryStem/runs/colab-large-ternary-pilot/`
- `TernaryStem/listening/`
- `TernaryStem/COLAB_SHA256SUMS`
- `TernaryStem/colab-environment.txt`

Immediate next action

Guide me through a from-scratch Vast.ai run using the current `main` commit and `docs/VAST_AI.md`:

1. Select a reliable on-demand CUDA PyTorch instance (5090 preferred; reliable 4080S/4090 acceptable) with at least 150 GB disk.
2. Clone the repository and run `scripts/vast/setup.sh`.
3. Obtain MUSDB18-HQ through the official source under acceptable terms, verify MD5 `12d4f2ecd55245a4688754dd76363103`, extract only `train/`, and verify 100 tracks/400 required stems. Never commit or redistribute it.
4. Configure `.vast.env`, including an off-instance backup destination and `MUSDB_TERMS_ACCEPTED=1`. Do not expose credentials in chat, logs, Git, or images.
5. Start the detached pipeline. It must stop after smoke at `AWAITING_FULL_RUN_APPROVAL`.
6. Use measured throughput, VRAM, sync speed, offer price, and projected total cost to decide whether to set `FULL_RUN_APPROVED=1`.
7. Run the complex-mask FP32 baseline from scratch. If it fails the FP gate, stop and improve the FP model rather than forcing QAT.
8. If it passes, run fresh sensitivity, matched FP control, selective QAT, comparison, export, checksums, and verified off-instance restore before terminating compute.

Operational constraints

- Never invent quality, latency, hardware, or cost results.
- Never call development global SDR BSSEval.
- Never tune on the official test set.
- Never commit data, checkpoints, generated audio, credentials, or private keys.
- Treat completed-epoch atomic checkpoints as the supported recovery point; an interrupted partial epoch is repeated.
- Preserve `best.pt`, `latest.pt`, immutable periodic checkpoints, experiment JSON, generated configs, logs, environment metadata, and hashes.
- Do not destroy the Vast instance until the independent artifact copy has been checksum-verified and restored.
- Do not change configs during exact resume. Use warm-start with a fresh optimizer only when the procedure explicitly calls for a new branch.

Start by inspecting the repository and summarizing the exact commands I should run for the current stage. Ask me only for values that cannot be discovered locally, such as the selected Vast offer, hourly rate, SSH endpoint, dataset location, and backup destination. Do not ask me to paste private credentials.
```

## Notes

A future session should trust the current committed documentation over hard-coded figures in this prompt if later experiments update them. Keep this file updated whenever the frozen strategy, benchmark definition, quality gate, or cloud procedure changes.
