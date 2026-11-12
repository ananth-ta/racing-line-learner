#!/usr/bin/env bash
# Run Python inside the project env on Colab, writing outputs to Google Drive.
#
#   bash scripts/colab_run.sh -m racing_line_learner.train --run_name cota_laptime_s0 --preset laptime
#   bash scripts/colab_run.sh render_ppo_lap.py --model ... --output ...
#
# Outputs go to $RLL_OUTPUT_DIR (default: /content/drive/MyDrive/F1Tenth when Drive
# is mounted). Re-running a training command after a disconnect resumes it.
set -eo pipefail

CONDA_DIR="${CONDA_DIR:-/content/miniforge}"
ENV_NAME="${ENV_NAME:-f1tenth}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if [ -z "${RLL_OUTPUT_DIR:-}" ] && [ -d /content/drive/MyDrive ]; then
    export RLL_OUTPUT_DIR=/content/drive/MyDrive/F1Tenth
fi
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_DIR"
echo "[run] outputs -> ${RLL_OUTPUT_DIR:-$REPO_DIR}"
exec python -u "$@"
