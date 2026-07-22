# Vast.ai training runbook

This runbook makes instance setup and completed-epoch recovery repeatable. It cannot guarantee that the model reaches the quality gates. A failed FP or ternary quality gate is a valid research result.

## 0. Frozen strategy before rental

- Use one universal four-stem TFC-TDF model; no mixture of experts.
- Use bounded complex masks. Matched Colab smoke runs learned materially faster than direct complex estimates.
- Keep STFT/iSTFT, losses, normalization, projections, complex construction, mixture consistency, and reconstruction FP32.
- Train the qualifying baseline in FP32, matching the historically conservative HTDemucs training precision. FP16 remains an acceleration candidate, not a proven full-run replacement; a five-epoch T4 pilot was 27% faster and 0.01 dB behind FP32, which does not rule out a later plateau. Benchmark BF16/FP16 on the rented GPU before changing the baseline config.
- Do not use HTDemucs distillation in the expensive run. A matched reduced experiment changed SDR by only 0.00012 dB at the tested cadence.
- Do not begin expensive QAT unless the FP baseline passes the configured quality gate.
- Run fresh layer-family sensitivity from the capable FP checkpoint. The provisional selective policy is ternary TDF Linear plus bottleneck convolution, with encoder, decoder, projections, norms, and reconstruction FP32. W4/W8 remain fallbacks.
- Treat energy-aggregated development global SDR as the checkpoint-selection diagnostic. Mean-chunk SDR is retained to expose silent-chunk behavior. Neither metric is BSSEval.
- Keep the official test partition untouched until the architecture, precision map, and inference recipe are frozen.

## 0.1 Colab evidence and limits

The operator-retained Colab artifacts (not committed because they include checkpoints/audio) established engineering feasibility on a 15 GiB T4:

- exact large shape: 632,208 parameters, six-second chunks, batch four, 4096-point STFT;
- FP32: about 7.98 chunks/s, 5.39 GiB peak allocated, best 2.5480 dB energy-aggregated development SDR versus a 1.2470 dB equal-share baseline;
- FP16 autocast: about 10.17 chunks/s, 4.59 GiB peak allocated, zero skipped updates, and -0.0096 dB versus FP32 after five short epochs;
- selective ternary QAT: 80.10% parameter coverage, approximately 40% zeros, at most about 0.10% observed activation saturation, and -0.1245 dB versus the matched FP checkpoint after only eight recovery updates;
- chunked inference: zero output for silent input and mixture reconstruction maximum error about 3.6e-7 after fixing long-Hann-window boundaries;
- deterministic export: 13 ternary weight tensors and 95 FP32 tensors (tensor count is not parameter coverage).

These are development diagnostics, not BSSEval, not a capable separator, and not evidence that FP16 will match FP32 after full convergence. Audible separation began, but both FP32 and ternary pilots retained substantial leakage, with FP32 slightly better on drums and bass.

## 1. Choose an instance

Start with an **on-demand, non-interruptible** single NVIDIA GPU offer for the long run. An RTX 5090 is preferred for throughput; a reliable 4080S/4090-class offer is acceptable. The exact large shape used only about 6.42 GiB reserved on a T4, but host/image/library differences still require preflight. Prefer:

- a CUDA-enabled PyTorch image (do not use a bare CUDA image);
- at least 150 GiB disk for the image, roughly 30 GiB dataset, run artifacts, and headroom;
- high host reliability and adequate verified download speed;
- direct SSH access;
- storage independent of the compute instance for checkpoint copies.

Do not pick the final offer solely by advertised TFLOPS. Host reliability, disk throughput, bandwidth, and $/completed epoch matter. Avoid interruptible instances unless losing up to the current epoch is acceptable.

## 2. Supply the dataset

Obtain MUSDB18-HQ legally and upload or mount it yourself. The repository intentionally does **not** download it. The expected path contains the official 100 training directories:

```text
/workspace/data/MUSDB18-HQ/train/<track>/{vocals,drums,bass,other}.wav
```

The preflight checks all 400 stems, the frozen 86/14 split, stereo 44.1 kHz format, matching stem lengths, and sampled decoding. It never opens the official test partition.

For a from-scratch instance, download the operator-authorized official archive manually after reviewing its educational/non-commercial terms:

```bash
mkdir -p /workspace/downloads /workspace/data/MUSDB18-HQ
cd /workspace/downloads
wget -c -O musdb18hq.zip \
  'https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1'
echo '12d4f2ecd55245a4688754dd76363103  musdb18hq.zip' | md5sum -c -
unzip -q musdb18hq.zip 'train/*' -d /workspace/data/MUSDB18-HQ
find /workspace/data/MUSDB18-HQ/train -mindepth 1 -maxdepth 1 -type d | wc -l
find /workspace/data/MUSDB18-HQ/train -type f \
  \( -name vocals.wav -o -name drums.wav -o -name bass.wav -o -name other.wav \) \
  | wc -l
```

The expected counts are 100 tracks and 400 required stems. Delete the verified ZIP only after extraction and preflight if disk space is needed. Do not extract or expose `test/` on the training path.

## 3. Clone and bootstrap

```bash
cd /workspace
git clone https://github.com/Kybernetria/TernaryStem.git
cd TernaryStem
git checkout <the-commit-you-intend-to-train>
bash scripts/vast/setup.sh
```

The setup creates `/workspace/venvs/ternarystem` with `--system-site-packages` so it reuses the image's CUDA PyTorch. It deliberately never installs or replaces Torch. It then runs Ruff, pytest, and a CUDA-required model dry run.

## 4. Configure durability and safeguards

Edit `.vast.env`:

```bash
DATA_ROOT=/workspace/data/MUSDB18-HQ/train
RUN_ROOT=/workspace/runs/ternarystem
MUSDB_TERMS_ACCEPTED=1
```

Configure at least one backup target:

- `RCLONE_REMOTE=remote:bucket/prefix` for storage independent of the instance; or
- `BACKUP_DIR=/mounted/persistent/storage` for a mounted destination.

`rclone` and its credentials are operator-managed and must not be committed. The pipeline refuses to start without a backup destination unless `ALLOW_EPHEMERAL=1` is explicitly set.

Leave `ALLOW_BELOW_FP_GATE=0` for a scientifically meaningful run. The default FP gate is 7.5 dB energy-aggregated development `global_sdr` and must also beat the equal-share baseline. This diagnostic is not BSSEval. Keep `BASELINE_CONFIG=configs/remote/fp32_complex_mask.yaml`; do not silently add AMP or distillation to the qualifying baseline.

## 5. Run smoke, approve cost, then continue

```bash
scripts/vast/start.sh
tail -f /workspace/runs/ternarystem/*/logs/pipeline.log
```

The first launch performs:

1. full data/environment preflight;
2. Ruff and pytest;
3. a data-backed three-epoch GPU smoke;
4. strict checkpoint resume loading;
5. off-instance sync.

It then sets `STATUS` to `AWAITING_FULL_RUN_APPROVAL`. Inspect elapsed time, GPU utilization, peak memory, disk, and sync throughput:

```bash
RUN_ID=$(cat /workspace/runs/ternarystem/.active-run-id)
RUN=/workspace/runs/ternarystem/$RUN_ID
cat "$RUN/STATUS"
tail -100 "$RUN/logs/smoke.log"
tail -100 "$RUN/logs/telemetry.log"
```

Estimate cost from observed seconds per training chunk and validation epoch. The T4 pilot implies roughly 35–40 hours for one million chunks on that GPU; do not extrapolate a 5090 rate without measuring it. Add margin for validation, checkpoint sync, host contention, and retries. If the offer and budget are acceptable, set `FULL_RUN_APPROVED=1` in `.vast.env` and continue the same run:

```bash
scripts/vast/start.sh
```

The detached process survives SSH disconnects and is capped by `MAX_PIPELINE_HOURS` (168 hours by default per launch). Do not terminate the Vast instance merely because your terminal disconnects.

## 6. What the full pipeline does

1. Train or resume the selected 100-epoch complex-mask FP32 baseline from scratch.
2. Run all-family ternary sensitivity from the exact best FP checkpoint.
3. Enforce the FP quality gate and selected-family sensitivity limit.
4. Generate matched FP-control and selective-QAT configs from the baseline's resolved config.
5. Warm-start both branches from the same checksum-pinned FP checkpoint with fresh optimizers.
6. Train the control and selective ternary branches for the configured QAT budget.
7. Compare records, export the best ternary checkpoint, hash the bundle, and sync it.

By default, QAT uses `tdf_linear,bottleneck_conv`, but only if fresh sensitivity shows each family loses no more than 0.5 dB immediately. This reproduces the historical candidate policy without blindly trusting the old incapable checkpoint.

Check progress with:

```bash
cat "$RUN/STATUS"                 # RUNNING, FAILED, or COMPLETE
tail -f "$RUN/logs/pipeline.log"
tail -f "$RUN/logs/ternary-qat.log"
```

To stop while preserving the prior completed epoch:

```bash
scripts/vast/stop.sh
```

Remove the run's `STOP` marker before resuming. The trainer publishes checkpoints and JSON via validated same-filesystem temporary files plus atomic rename. `latest.pt` is completed-epoch recovery; an interrupted partial epoch is intentionally repeated.

## 7. Before destroying the instance

A `COMPLETE` marker is not enough by itself. Verify the independent copy:

1. compare `SHA256SUMS` after downloading the bundle;
2. load FP, control, and QAT `best.pt` in a clean environment;
3. confirm `experiment.json`, sensitivity output, generated configs, logs, and export exist;
4. only then terminate compute and decide separately whether to retain paid storage.

Do not publish MUSDB-derived checkpoints until redistribution terms have been reviewed. Do not claim BSSEval or official test quality from these development diagnostics.
