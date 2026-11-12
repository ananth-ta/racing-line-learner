#!/usr/bin/env bash
# One-time setup per Colab session: Miniforge, the pinned Python 3.8 env, dependencies.
# Safe to re-run: steps that are already done are skipped.
#
#   bash scripts/colab_setup.sh
#
# Override locations with CONDA_DIR / ENV_NAME if needed.
set -eo pipefail

CONDA_DIR="${CONDA_DIR:-/content/miniforge}"
ENV_NAME="${ENV_NAME:-f1tenth}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo "[setup] Installing Miniforge (avoids the Anaconda ToS prompt)..."
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR" > /dev/null
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "[setup] Creating Python 3.8 env '$ENV_NAME'..."
    conda create -y -q -n "$ENV_NAME" python=3.8 > /dev/null
fi
conda activate "$ENV_NAME"

cd "$REPO_DIR"
if ! python -c "import f110_gym, stable_baselines3" 2> /dev/null; then
    echo "[setup] Installing dependencies..."
    # pip < 24.1 is needed: newer pip rejects gym 0.19.0's metadata.
    pip install -q pip==24.0 setuptools==65.5.0 wheel==0.38.4
    pip install -q -r requirements.txt
fi

PYTHONPATH="$REPO_DIR/src" python -c "import f110_gym, stable_baselines3, racing_line_learner; print('[setup] OK')"
