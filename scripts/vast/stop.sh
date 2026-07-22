#!/usr/bin/env bash
# Stop at the current batch boundary; the previous completed atomic epoch remains resumable.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.vast.env}"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
RUN_ROOT="${RUN_ROOT:-/workspace/runs/ternarystem}"
ACTIVE_FILE="$RUN_ROOT/.active-run-id"
[[ -f "$ACTIVE_FILE" ]] || { echo "no active run" >&2; exit 1; }
RUN_ID="$(<"$ACTIVE_FILE")"
RUN_DIR="$RUN_ROOT/$RUN_ID"
touch "$RUN_DIR/STOP"
if [[ -f "$RUN_DIR/pipeline.pid" ]]; then
  PID="$(<"$RUN_DIR/pipeline.pid")"
  if kill -0 "$PID" 2>/dev/null; then
    kill -TERM -- "-$PID"
    echo "sent TERM to run $RUN_ID process group $PID"
    exit 0
  fi
fi
echo "STOP marker written; no live pipeline PID found"
