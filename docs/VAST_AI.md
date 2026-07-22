# Vast.ai training runbook

This runbook makes instance setup and completed-epoch recovery repeatable. It cannot guarantee that the model reaches the quality gates. A failed FP or ternary quality gate is a valid research result.

## 1. Choose an instance

Start with an **on-demand, non-interruptible** single NVIDIA GPU offer for the long run. A 24 GiB RTX 3090/4090-class GPU is a reasonable first candidate, not a proven requirement; the preflight executes the exact configured batch/chunk and fails on OOM. Prefer:

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

Leave `ALLOW_BELOW_FP_GATE=0` for a scientifically meaningful run. The default FP gate is 7.5 dB development `global_sdr` and must also beat the equal-share baseline. This diagnostic is not BSSEval.

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

Estimate cost from observed seconds per training chunk and validation epoch. Add substantial margin for the much larger six-second/batch-four baseline. If the offer and budget are acceptable, set `FULL_RUN_APPROVED=1` in `.vast.env` and continue the same run:

```bash
scripts/vast/start.sh
```

The detached process survives SSH disconnects and is capped by `MAX_PIPELINE_HOURS` (168 hours by default per launch). Do not terminate the Vast instance merely because your terminal disconnects.

## 6. What the full pipeline does

1. Train or resume the selected 100-epoch FP32 baseline.
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
