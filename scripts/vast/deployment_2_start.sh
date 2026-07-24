#!/usr/bin/env bash
# Start the allowlisted deployment #2 controller with no inherited launcher lock.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.deployment2.env}"
[[ -f "$ENV_FILE" ]] || { echo "missing $ENV_FILE" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT}"
VENV="${VENV:?set VENV}"
CUDA_EVIDENCE="${CUDA_EVIDENCE:?set CUDA_EVIDENCE to authorized local-GPU evidence}"
mkdir -p "$RUN_ROOT"
exec 8>"$RUN_ROOT/.deployment2-launcher.lock"
flock -n 8 || { echo "another deployment #2 launcher is active" >&2; exit 1; }
IDENTITY="$RUN_ROOT/controller.identity.json"
if [[ -f "$IDENTITY" ]] && "$VENV/bin/python" scripts/vast/deployment_2_process.py verify "$IDENTITY" >/dev/null 2>&1; then
  echo "deployment #2 controller is already running" >&2
  exit 1
fi
rm -f "$IDENTITY"
# Close the lock before spawning; neither controller nor helpers may inherit it.
exec 8>&-
cd "$REPO_ROOT"
setsid nohup "$VENV/bin/python" "$REPO_ROOT/scripts/deployment_2.py" \
  --repository "$REPO_ROOT" readiness --cuda-evidence "$CUDA_EVIDENCE" \
  >"$RUN_ROOT/readiness.json" 2>"$RUN_ROOT/controller.stderr" </dev/null &
PID=$!
"$VENV/bin/python" scripts/vast/deployment_2_process.py record "$PID" "$IDENTITY" \
  --token "scripts/deployment_2.py" >/dev/null
printf 'Started deployment #2 readiness controller as PID/PGID %s\n' "$PID"
