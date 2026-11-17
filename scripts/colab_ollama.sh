#!/usr/bin/env bash
# Start an Ollama server on a Colab GPU and pull the models to benchmark or supervise with.
# Safe to re-run: install and pull are skipped when already done; the server is started only if not running.
#
#   bash scripts/colab_ollama.sh qwen3:8b [more models...]
#
# Set OLLAMA_MODELS to a Drive folder to keep downloaded weights between sessions.
set -eo pipefail

if ! command -v ollama > /dev/null; then
    echo "[ollama] Installing..."
    command -v zstd > /dev/null || (apt-get -qq update && apt-get -qq install -y zstd > /dev/null)
    curl -fsSL https://ollama.com/install.sh | sh > /dev/null
fi

if ! curl -s -m 2 localhost:11434/api/tags > /dev/null; then
    echo "[ollama] Starting server (log: /content/ollama.log)..."
    nohup ollama serve > /content/ollama.log 2>&1 &
    for _ in $(seq 60); do
        curl -s -m 2 localhost:11434/api/tags > /dev/null && break
        sleep 1
    done
fi
curl -s -m 2 localhost:11434/api/tags > /dev/null || { echo "[ollama] server did not start"; tail /content/ollama.log; exit 1; }

for model in "$@"; do
    echo "[ollama] Pulling $model..."
    ollama pull "$model" > /dev/null
done
ollama list
