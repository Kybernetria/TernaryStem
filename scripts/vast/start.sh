#!/usr/bin/env bash
# Start or resume the pipeline independently of the SSH session.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.vast.env}"
[[ -f "$ENV_FILE" ]] || { echo "missing $ENV_FILE; run scripts/vast/setup.sh first" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
RUN_ROOT="${RUN_ROOT:-/workspace/runs/ternarystem}"
mkdir -p "$RUN_ROOT"
exec 8>"$RUN_ROOT/.launcher.lock"
flock -n 8 || { echo "another launcher is changing run state" >&2; exit 1; }
ACTIVE_FILE="$RUN_ROOT/.active-run-id"

if [[ -f "$ACTIVE_FILE" ]]; then
  EXISTING_ID="$(<"$ACTIVE_FILE")"
  EXISTING_DIR="$RUN_ROOT/$EXISTING_ID"
  if [[ -f "$EXISTING_DIR/pipeline.pid" ]] && \
    kill -0 "$(<"$EXISTING_DIR/pipeline.pid")" 2>/dev/null; then
    echo "pipeline already running as PID $(<"$EXISTING_DIR/pipeline.pid")" >&2
    exit 1
  fi
  if [[ "${NEW_RUN:-0}" == 1 ]]; then
    rm -f "$ACTIVE_FILE"
  fi
fi
if [[ -f "$ACTIVE_FILE" ]]; then
  RUN_ID="$(<"$ACTIVE_FILE")"
  if [[ -f "$RUN_ROOT/$RUN_ID/STATUS" && "$(<"$RUN_ROOT/$RUN_ID/STATUS")" == COMPLETE ]]; then
    echo "run $RUN_ID is already COMPLETE; use NEW_RUN=1 for a new experiment" >&2
    exit 1
  fi
else
  cd "$REPO_ROOT"
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short=12 HEAD)"
  printf '%s\n' "$RUN_ID" >"$ACTIVE_FILE"
fi
export REPO_ROOT RUN_ROOT RUN_ID
RUN_DIR="$RUN_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR/logs"
setsid nohup timeout --signal=TERM --kill-after=120 \
  "${MAX_PIPELINE_HOURS:-168}h" bash "$REPO_ROOT/scripts/vast/run_pipeline.sh" \
  >>"$RUN_DIR/logs/launcher.log" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >"$RUN_DIR/pipeline.pid"
sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
  wait "$PID" || STATUS=$?
  echo "pipeline exited during startup (status ${STATUS:-0}); inspect $RUN_DIR/logs/launcher.log" >&2
  exit 1
fi
printf 'Started run %s as PID/process-group %s\n' "$RUN_ID" "$PID"
printf 'Status: cat %q\n' "$RUN_DIR/STATUS"
printf 'Logs:   tail -f %q\n' "$RUN_DIR/logs/pipeline.log"
