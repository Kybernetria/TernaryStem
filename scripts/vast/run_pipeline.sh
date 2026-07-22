#!/usr/bin/env bash
# Idempotent, supervised single-GPU FP -> sensitivity -> selective ternary QAT pipeline.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="${VENV:-/workspace/venvs/ternarystem}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the manually supplied MUSDB18-HQ train directory}"
RUN_ROOT="${RUN_ROOT:-/workspace/runs/ternarystem}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/remote/fp32_complex_mask.yaml}"
WORKERS="${WORKERS:-4}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
MIN_FP_SDR="${MIN_FP_SDR:-7.5}"
QAT_EPOCHS="${QAT_EPOCHS:-30}"
QAT_LEARNING_RATE="${QAT_LEARNING_RATE:-0.0001}"
QAT_FAMILIES="${QAT_FAMILIES:-tdf_linear,bottleneck_conv}"
SENSITIVITY_MAX_SDR_DROP="${SENSITIVITY_MAX_SDR_DROP:-0.5}"
MAX_RETRIES="${MAX_RETRIES:-2}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-900}"
RUN_SMOKE="${RUN_SMOKE:-1}"
FULL_RUN_APPROVED="${FULL_RUN_APPROVED:-0}"
ALLOW_BELOW_FP_GATE="${ALLOW_BELOW_FP_GATE:-0}"
ALLOW_EPHEMERAL="${ALLOW_EPHEMERAL:-0}"
RCLONE_REMOTE="${RCLONE_REMOTE:-}"
BACKUP_DIR="${BACKUP_DIR:-}"
MUSDB_TERMS_ACCEPTED="${MUSDB_TERMS_ACCEPTED:-0}"

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
COMMIT="$(git rev-parse HEAD)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${COMMIT:0:12}}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
META_DIR="$RUN_DIR/meta"
mkdir -p "$LOG_DIR" "$META_DIR"
BOUND_COMMIT=""
if [[ -f "$META_DIR/source-commit" ]]; then BOUND_COMMIT="$(<"$META_DIR/source-commit")"; fi
if [[ -n "$BOUND_COMMIT" && "$BOUND_COMMIT" != "$COMMIT" ]]; then
  echo "run $RUN_ID is bound to commit $BOUND_COMMIT, not $COMMIT" >&2
  exit 1
fi
printf '%s\n' "$COMMIT" >"$META_DIR/source-commit"
exec 9>"$RUN_ROOT/.pipeline.lock"
flock -n 9 || { echo "another TernaryStem pipeline holds $RUN_ROOT/.pipeline.lock" >&2; exit 1; }

[[ "$MUSDB_TERMS_ACCEPTED" == 1 ]] || { echo "set MUSDB_TERMS_ACCEPTED=1 after reviewing dataset terms" >&2; exit 1; }
if [[ -z "$RCLONE_REMOTE" && -z "$BACKUP_DIR" && "$ALLOW_EPHEMERAL" != 1 ]]; then
  echo "configure RCLONE_REMOTE or BACKUP_DIR; otherwise explicitly set ALLOW_EPHEMERAL=1" >&2
  exit 1
fi
if [[ -n "$RCLONE_REMOTE" ]]; then
  command -v rclone >/dev/null || { echo "RCLONE_REMOTE set but rclone is unavailable" >&2; exit 1; }
fi

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG_DIR/pipeline.log"; }
write_status() { printf '%s\n' "$1" >"$RUN_DIR/STATUS"; }

sync_once() {
  if [[ -n "$BACKUP_DIR" ]]; then
    mkdir -p "$BACKUP_DIR/$RUN_ID"
    rsync -a --partial "$RUN_DIR/" "$BACKUP_DIR/$RUN_ID/"
  fi
  if [[ -n "$RCLONE_REMOTE" ]]; then
    rclone copy "$RUN_DIR" "${RCLONE_REMOTE%/}/$RUN_ID" --checksum --transfers 2
  fi
}

BACKGROUND_PIDS=()
cleanup() {
  local status=$?
  for pid in "${BACKGROUND_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  if [[ $status -ne 0 ]]; then
    write_status FAILED
    log "pipeline failed with exit code $status"
    sync_once || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

write_status RUNNING
{
  echo "run_id=$RUN_ID"
  echo "commit=$COMMIT"
  echo "baseline_config=$BASELINE_CONFIG"
  echo "data_root=$DATA_ROOT"
  echo "started_utc=$(date -u +%FT%TZ)"
  python --version
  python -m pip freeze
  nvidia-smi
} >"$META_DIR/environment.txt"

(
  while true; do
    printf '\n[%s]\n' "$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader || true
    df -h "$RUN_ROOT" || true
    sleep 60
  done
) >>"$LOG_DIR/telemetry.log" 2>&1 &
BACKGROUND_PIDS+=("$!")

if [[ -n "$RCLONE_REMOTE" || -n "$BACKUP_DIR" ]]; then
  (
    while true; do sleep "$SYNC_INTERVAL_SECONDS"; sync_once || true; done
  ) >>"$LOG_DIR/sync.log" 2>&1 &
  BACKGROUND_PIDS+=("$!")
fi

run_train() {
  local name=$1 config=$2 output=$3 init=${4:-}
  local attempt=0 status command_args
  mkdir -p "$(dirname "$output")"
  while true; do
    if [[ -f "$RUN_DIR/STOP" ]]; then
      log "STOP marker found; refusing to start $name"
      return 130
    fi
    command_args=(python scripts/train.py --config "$config" --data-root "$DATA_ROOT" \
      --output-dir "$output" --device cuda --require-cuda --workers "$WORKERS")
    if [[ -f "$output/latest.pt" ]]; then
      command_args+=(--resume "$output/latest.pt")
      log "$name: resuming validated epoch checkpoint (attempt $attempt)"
    elif [[ -n "$init" ]]; then
      command_args+=(--init-checkpoint "$init")
      log "$name: warm-starting from $init"
    else
      log "$name: starting from scratch"
    fi
    set +e
    "${command_args[@]}" 2>&1 | tee -a "$LOG_DIR/$name.log"
    status=${PIPESTATUS[0]}
    set -e
    if [[ $status -eq 0 ]]; then
      sync_once
      return 0
    fi
    attempt=$((attempt + 1))
    sync_once || true
    if (( attempt > MAX_RETRIES )); then
      log "$name: exhausted $MAX_RETRIES retries (last exit $status)"
      return "$status"
    fi
    [[ -f "$output/latest.pt" ]] || {
      log "$name: no completed epoch exists, so automatic retry is unsafe"
      return "$status"
    }
    log "$name: retrying from the last completed atomic checkpoint in 30 seconds"
    sleep 30
  done
}

log "preflight: validating all 400 stem files and a representative full-shape CUDA step"
python scripts/vast/preflight.py \
  --config "$BASELINE_CONFIG" --data-root "$DATA_ROOT" --output "$META_DIR/preflight.json" \
  --min-free-gb "$MIN_FREE_GB" --expected-commit "$COMMIT" --accept-musdb-terms

log "running correctness checks"
ruff check . | tee "$LOG_DIR/ruff.log"
pytest -q | tee "$LOG_DIR/pytest.log"

if [[ "$RUN_SMOKE" == 1 ]]; then
  run_train smoke configs/smoke/fp32.yaml "$RUN_DIR/smoke"
  # A no-op completed resume still verifies strict config/model/optimizer loading on this host.
  run_train smoke-resume configs/smoke/fp32.yaml "$RUN_DIR/smoke"
fi
if [[ "$FULL_RUN_APPROVED" != 1 ]]; then
  write_status AWAITING_FULL_RUN_APPROVAL
  sync_once
  log "smoke complete; inspect logs/telemetry, estimate cost, set FULL_RUN_APPROVED=1, then run start.sh again"
  trap - EXIT INT TERM
  for pid in "${BACKGROUND_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  exit 0
fi

BASELINE_DIR="$RUN_DIR/fp32-baseline"
run_train fp32-baseline "$BASELINE_CONFIG" "$BASELINE_DIR"
[[ -f "$BASELINE_DIR/best.pt" ]] || { log "baseline produced no best.pt"; exit 1; }

log "running fresh all-family ternary sensitivity from the exact best FP checkpoint"
python scripts/sensitivity.py "$BASELINE_DIR/best.pt" --data-root "$DATA_ROOT" \
  --output "$RUN_DIR/ternary-sensitivity.json" --precision ternary --workers "$WORKERS" \
  --device cuda 2>&1 | tee "$LOG_DIR/sensitivity.log"

IFS=',' read -r -a FAMILY_ARGS <<<"$QAT_FAMILIES"
PREPARE_ARGS=(python scripts/vast/prepare_qat.py \
  --experiment "$BASELINE_DIR/experiment.json" --checkpoint "$BASELINE_DIR/best.pt" \
  --sensitivity "$RUN_DIR/ternary-sensitivity.json" \
  --control-output "$META_DIR/fp32-control.yaml" --qat-output "$META_DIR/ternary-qat.yaml" \
  --families "${FAMILY_ARGS[@]}" --min-fp-sdr "$MIN_FP_SDR" \
  --max-sensitivity-sdr-drop "$SENSITIVITY_MAX_SDR_DROP" \
  --epochs "$QAT_EPOCHS" --learning-rate "$QAT_LEARNING_RATE")
if [[ "$ALLOW_BELOW_FP_GATE" == 1 ]]; then PREPARE_ARGS+=(--allow-below-fp-gate); fi
"${PREPARE_ARGS[@]}" | tee "$META_DIR/qat-preparation.json"

run_train fp32-control "$META_DIR/fp32-control.yaml" "$RUN_DIR/fp32-control" "$BASELINE_DIR/best.pt"
run_train ternary-qat "$META_DIR/ternary-qat.yaml" "$RUN_DIR/ternary-qat" "$BASELINE_DIR/best.pt"

python scripts/compare_runs.py "$RUN_DIR/fp32-control/experiment.json" \
  "$RUN_DIR/ternary-qat/experiment.json" | tee "$RUN_DIR/comparison.txt"
python scripts/export.py "$RUN_DIR/ternary-qat/best.pt" "$RUN_DIR/ternary-qat/best.npz" \
  2>&1 | tee "$LOG_DIR/export.log"

# Freeze background writers before creating the immutable-artifact checksum manifest.
for pid in "${BACKGROUND_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
for pid in "${BACKGROUND_PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
BACKGROUND_PIDS=()
write_status COMPLETE
(
  cd "$RUN_DIR"
  find . -type f ! -name SHA256SUMS ! -path './logs/*' ! -name STATUS \
    ! -name pipeline.pid -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
)
sync_once
log "pipeline COMPLETE; verify SHA256SUMS in the off-instance bundle before terminating compute"
trap - EXIT INT TERM
