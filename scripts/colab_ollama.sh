#!/usr/bin/env bash
# Start an Ollama server on a Colab GPU and pull the models to benchmark or supervise with.
# Safe to re-run: install and pull are skipped when already done; the server is started only if not running.
#
#   bash scripts/colab_ollama.sh qwen3:8b [more models...]
#
# Each model is loaded once with a one-token request, so a broken install (e.g. the `ollama` binary present but its
# llama-server runtime missing, which makes every chat request fail with HTTP 500) is caught here and repaired by
# reinstalling once, instead of surfacing as a benchmark full of errors.
#
# Set OLLAMA_MODELS to a Drive folder to keep downloaded weights between sessions.
set -eo pipefail

install_ollama() {
    echo "[ollama] Installing..."
    command -v zstd > /dev/null || (apt-get -qq update && apt-get -qq install -y zstd > /dev/null)
    curl -fsSL https://ollama.com/install.sh | sh > /dev/null
}

up() { curl -s -m 2 localhost:11434/api/tags > /dev/null; }

start_server() {
    if ! up; then
        echo "[ollama] Starting server (log: /content/ollama.log)..."
        nohup ollama serve > /content/ollama.log 2>&1 &
        for _ in $(seq 60); do up && break; sleep 1; done
    fi
    up || { echo "[ollama] server did not start"; tail /content/ollama.log; exit 1; }
}

# Load the model and generate one token; succeeds only if the model actually runs.
loads() {
    local reply
    reply=$(curl -s -m 900 localhost:11434/api/generate \
        -d "{\"model\": \"$1\", \"prompt\": \"hi\", \"stream\": false, \"options\": {\"num_predict\": 1}}") || return 1
    [[ "$reply" == *'"done":true'* ]]
}

command -v ollama > /dev/null || install_ollama
start_server

for model in "$@"; do
    echo "[ollama] Pulling $model..."
    ollama pull "$model" > /dev/null
    echo "[ollama] Loading $model..."
    if ! loads "$model"; then
        echo "[ollama] $model failed to load; last server log lines:"
        tail -3 /content/ollama.log
        echo "[ollama] Reinstalling Ollama and restarting the server..."
        pkill -f "ollama serve" || true
        sleep 2
        install_ollama
        start_server
        loads "$model" || { echo "[ollama] $model still fails to load"; tail -5 /content/ollama.log; exit 1; }
    fi
    echo "[ollama] $model loaded"
done
ollama list
curl -s localhost:11434/api/ps | head -c 300; echo
