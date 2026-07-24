#!/usr/bin/env bash
# Stop only the verified deployment #2 process group.
set -Eeuo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.deployment2.env}"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT}"
VENV="${VENV:?set VENV}"
IDENTITY="$RUN_ROOT/controller.identity.json"
"$VENV/bin/python" "$REPO_ROOT/scripts/vast/deployment_2_process.py" terminate "$IDENTITY" \
  --timeout "${STOP_TIMEOUT_SECONDS:-30}"
printf 'Verified deployment #2 process group stopped\n'
