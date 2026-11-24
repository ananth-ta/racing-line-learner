#!/usr/bin/env bash
# Run one experiment arm (3 seeds by default) on Colab, adapting to the GPU it got.
#
#   bash scripts/colab_arm.sh llm            # LLM-supervised arm, seeds 0 1 2
#   bash scripts/colab_arm.sh rules 1 2      # rule-supervised arm, seeds 1 and 2
#
# On a G4 (RTX PRO 6000, 96 GB) the seeds run in parallel and the LLM arm uses the 8-bit model ($LLM_MODEL_G4).
# Anywhere else the seeds run one after another and the LLM arm uses the model that fits a 24 GB L4 ($LLM_MODEL).
# Run names carry the model, so a run is never resumed with a different model. Re-running the same command after a
# disconnect resumes every unfinished seed; finished seeds are skipped.
#
# Settings (environment variables):
#   LLM_MODEL_G4  model on a G4              (default qwen3.8:27b-q8_0, the best Colab model in the benchmark)
#   LLM_MODEL     model on any other GPU     (default qwen3.8:27b-q4_K_M)
#   LLM_THINK     on / off / default / low / medium / high   (default on)
#   LLM_PROMPT    v1 / v2 / v3               (default v2)
#   LLM_URL       use an Ollama server that is already running (e.g. on the Windows host from WSL) instead of
#                 installing and starting one here
#   JUDGE         before_after / control  (default before_after; control trains a control branch for every change)
#   RUN_SUFFIX    appended to run names (e.g. _j2 for a second version of an arm), so earlier runs are not resumed
#   RLL_OUTPUT_DIR / CONDA_DIR   output folder and Miniforge location (for running outside Colab)
#   FORCE_PARALLEL=1 / FORCE_SEQUENTIAL=1    override the G4 check
#   ALLOW_SPLIT=1 start a seed with this GPU's model even if an unfinished run of it exists with another model
set -eo pipefail

ARM="${1:?usage: colab_arm.sh <llm|rules|stub> [seeds...]}"; shift
SEEDS="${*:-0 1 2}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RLL_OUTPUT_DIR:-/content/drive/MyDrive/F1Tenth}"
LLM_MODEL_G4="${LLM_MODEL_G4:-qwen3.8:27b-q8_0}"
LLM_MODEL="${LLM_MODEL:-qwen3.8:27b-q4_K_M}"
LLM_THINK="${LLM_THINK:-on}"
LLM_PROMPT="${LLM_PROMPT:-v2}"
case "$ARM" in llm|rules|stub) ;; *) echo "arm must be llm, rules or stub"; exit 1 ;; esac

# ---- hardware check -------------------------------------------------------------------------------------------
GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2> /dev/null | head -1 || true)"
IS_G4=0
if echo "$GPU" | grep -qi "RTX PRO 6000"; then IS_G4=1; fi
[ "${FORCE_PARALLEL:-0}" = 1 ] && IS_G4=1
[ "${FORCE_SEQUENTIAL:-0}" = 1 ] && IS_G4=0
NSEEDS=$(echo $SEEDS | wc -w)
echo "[arm] GPU: ${GPU:-none}; CPUs: $(nproc); G4 mode: $IS_G4; arm: $ARM; seeds: $SEEDS"
if [ "$IS_G4" = 1 ] && [ "$(nproc)" -lt $((8 * NSEEDS)) ]; then
    echo "[arm] note: $(nproc) CPUs for $NSEEDS x 8 envs; seeds will share cores and run slower each"
fi

MODEL=""
EXTRA=()
[ -n "${JUDGE:-}" ] && EXTRA+=(--judge "$JUDGE")
if [ "$ARM" = llm ]; then
    MODEL="$LLM_MODEL"; [ "$IS_G4" = 1 ] && MODEL="$LLM_MODEL_G4"
    TAG="$(echo "$MODEL" | tr ':/.' '___')"
    EXTRA+=(--llm_model "$MODEL" --llm_think "$LLM_THINK" --llm_prompt "$LLM_PROMPT" --llm_timeout 900)
    if [ -n "${LLM_URL:-}" ]; then
        curl -s -m 5 "$LLM_URL/api/tags" | grep -q "\"$MODEL\"" \
            || { echo "[arm] $LLM_URL is not answering or does not have $MODEL"; exit 1; }
        EXTRA+=(--llm_url "$LLM_URL")
    else
        export OLLAMA_NUM_PARALLEL="$NSEEDS" OLLAMA_KEEP_ALIVE=2h
        bash "$REPO_DIR/scripts/colab_ollama.sh" "$MODEL"
    fi
    echo "[arm] LLM: $MODEL (think $LLM_THINK, prompt $LLM_PROMPT)"
fi

run_name() {   # seed -> run name
    if [ "$ARM" = llm ]; then echo "cota_llm_${TAG}${RUN_SUFFIX:-}_s$1"; else echo "cota_${ARM}_arm${RUN_SUFFIX:-}_s$1"; fi
}

# ---- one seed -------------------------------------------------------------------------------------------------
run_seed() {
    local seed=$1 name; name=$(run_name "$seed")
    local dir="$OUT/runs/$name"
    if [ -f "$dir/DONE" ]; then echo "[arm] $name already finished"; return 0; fi
    if pgrep -f -- "--run_name $name " > /dev/null; then     # e.g. the launch cell was run twice
        echo "[arm] $name is already running; not starting a second copy"; return 0; fi
    if [ "$ARM" = llm ] && [ "${ALLOW_SPLIT:-0}" != 1 ]; then
        for other in "$OUT"/runs/cota_llm_*_s"$seed"; do
            [ -d "$other" ] && [ "$other" != "$dir" ] && [ ! -f "$other/DONE" ] && {
                echo "[arm] seed $seed has an unfinished run with another model: $other"
                echo "      resume it on matching hardware, or set ALLOW_SPLIT=1 to start $name anyway"; return 1; }
        done
    fi
    echo "[arm] start $name $(date +%T) (log: $OUT/runs/$name.log)"
    mkdir -p "$OUT/runs"
    bash "$REPO_DIR/scripts/colab_run.sh" -m racing_line_learner.train --run_name "$name" --track cota \
        --preset v2 --n_envs 8 --n_steps 512 --batch_size 256 --timesteps 1250000 --seed "$seed" \
        --supervisor "$ARM" --window_steps 250000 "${EXTRA[@]}" >> "$OUT/runs/$name.log" 2>&1 \
        && { touch "$dir/DONE"; echo "[arm] done $name $(date +%T)"; } \
        || { echo "[arm] FAILED $name; tail of log:"; tail -5 "$OUT/runs/$name.log"; return 1; }
}

status=0
if [ "$IS_G4" = 1 ]; then
    pids=()
    for s in $SEEDS; do run_seed "$s" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || status=1; done
else
    for s in $SEEDS; do run_seed "$s" || status=1; done
fi
echo "[arm] finished $(date +%T), status $status"
exit $status
