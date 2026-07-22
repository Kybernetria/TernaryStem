#!/usr/bin/env bash
# Bootstrap TernaryStem on a CUDA-enabled Vast.ai PyTorch image.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORKSPACE="${WORKSPACE:-/workspace}"
VENV="${VENV:-$WORKSPACE/venvs/ternarystem}"

log() { printf '[setup] %s\n' "$*"; }
die() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -f "$REPO_ROOT/pyproject.toml" ]] || die "REPO_ROOT is not a TernaryStem checkout: $REPO_ROOT"
command -v python3 >/dev/null || die "python3 is missing"
command -v nvidia-smi >/dev/null || die "nvidia-smi is missing; select a CUDA/PyTorch Vast image"
nvidia-smi >/dev/null || die "the NVIDIA driver is not usable"

if command -v apt-get >/dev/null && [[ "$(id -u)" == 0 ]]; then
  log "installing small OS prerequisites (never installing/replacing CUDA)"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git libsndfile1 rsync ca-certificates util-linux python3-venv
  rm -rf /var/lib/apt/lists/*
fi

# Reuse the CUDA-enabled torch from the selected image. Installing torch from PyPI here
# can silently replace it with an incompatible build.
python3 - <<'PY' || die "base image does not contain a working CUDA-enabled PyTorch"
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
assert torch.version.cuda, "PyTorch has no CUDA runtime"
print(f"base torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
PY

log "creating venv with access to the image's CUDA PyTorch: $VENV"
python3 -m venv --system-site-packages "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  'numpy>=1.26,<3' 'soundfile>=0.12,<1' 'PyYAML>=6,<7' \
  'pytest>=8,<10' 'pytest-cov>=5,<8' 'ruff>=0.5,<1'
python -m pip install --no-deps -e "$REPO_ROOT"

python - <<'PY' || die "venv lost access to CUDA PyTorch"
import torch
assert torch.cuda.is_available()
print(f"venv torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
PY

cd "$REPO_ROOT"
log "running local correctness checks"
ruff check .
pytest -q
python scripts/train.py --config configs/remote/fp32_complex_mask.yaml --dry-run --require-cuda

ENV_FILE="$REPO_ROOT/.vast.env"
if [[ ! -e "$ENV_FILE" ]]; then
  cat >"$ENV_FILE" <<EOF
# Review every value, then: set -a; source .vast.env; set +a
DATA_ROOT=$WORKSPACE/data/MUSDB18-HQ/train
RUN_ROOT=$WORKSPACE/runs/ternarystem
VENV=$VENV
BASELINE_CONFIG=configs/remote/fp32_complex_mask.yaml
WORKERS=4
MIN_FREE_GB=20
MIN_FP_SDR=7.5
QAT_EPOCHS=30
QAT_LEARNING_RATE=0.0001
QAT_FAMILIES=tdf_linear,bottleneck_conv
SENSITIVITY_MAX_SDR_DROP=0.5
MAX_RETRIES=2
MAX_PIPELINE_HOURS=168
RUN_SMOKE=1
# The first launch stops after smoke so you can inspect throughput and approve cost.
FULL_RUN_APPROVED=0
# Required acknowledgment; setup never downloads MUSDB18-HQ.
MUSDB_TERMS_ACCEPTED=0
# Strongly recommended. Example: my-s3:ternarystem-backups
RCLONE_REMOTE=
# Set to 1 only if you consciously accept losing an ephemeral instance and its runs.
ALLOW_EPHEMERAL=0
EOF
  log "wrote $ENV_FILE"
fi

log "setup complete"
printf '\nNext steps:\n'
printf '  1. Legally upload/mount MUSDB18-HQ at DATA_ROOT (this script never downloads it).\n'
printf '  2. Configure off-instance rclone storage or explicitly set ALLOW_EPHEMERAL=1.\n'
printf '  3. Edit .vast.env and set MUSDB_TERMS_ACCEPTED=1.\n'
printf '  4. Start detached: scripts/vast/start.sh\n'
