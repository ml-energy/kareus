#!/usr/bin/env bash
set -euo pipefail

# Launch PFO server for each freqs_pipeline_*.py and run Megatron GPT pretraining.
# Usage:
#   ./run_all_freqs.sh [freqs_dir] [host] [port]
# Defaults:
#   freqs_dir: /workspaces/nsdi/Kareus/tests/perseus/nemo_experiments/megatron_llama_3_2_1b/perseus_results
#   host: 0.0.0.0
#   port: 7787
# Notes:
# - Requires uvicorn and zeus PFO server available in PYTHONPATH.
# - Training entry: tests/perseus/megatron_gpt_pretraining.py (Hydra config enables Perseus).
# - MASTER_ADDR used by model code to infer PFO server URL; set to host.

FREQS_DIR=${1:-nemo_experiments/megatron_llama_3_2_1b/nanobatch_perseus_results}
# HOST=${2:-0.0.0.0}
PORT=${3:-7787}

# cd /workspaces/nsdi/Kareus/tests/perseus

LOG_DIR=logs_pfo_runs
mkdir -p "$LOG_DIR"

# Enumerate all freqs files sorted numerically by suffix if possible
mapfile -t FREQ_FILES < <(ls -1 "$FREQS_DIR"/freqs_pipeline_*.py 2>/dev/null | sort)

if [[ ${#FREQ_FILES[@]} -eq 0 ]]; then
  echo "No freqs_pipeline_*.py found under $FREQS_DIR" >&2
  exit 1
fi

TOTAL_FILES=${#FREQ_FILES[@]}
echo "Found ${TOTAL_FILES} frequency plans under $FREQS_DIR"

# Sample plans at a fixed stride from the largest number (default stride=200).
# Override with SAMPLE_STRIDE env. Always include the lowest-numbered plan.
SAMPLE_STRIDE=${SAMPLE_STRIDE:-100}
if (( SAMPLE_STRIDE > 1 )); then
  echo "Sampling from largest, every ${SAMPLE_STRIDE}th plan (total=${TOTAL_FILES})"
  declare -a STRIDED_FILES=()
  # Walk from last to first by stride
  for ((i=TOTAL_FILES-1; i>=0; i-=SAMPLE_STRIDE)); do
    STRIDED_FILES+=("${FREQ_FILES[$i]}")
  done
  # Ensure lowest-numbered (first) plan is included
  if [[ ${#STRIDED_FILES[@]} -eq 0 || "${STRIDED_FILES[-1]}" != "${FREQ_FILES[0]}" ]]; then
    STRIDED_FILES+=("${FREQ_FILES[0]}")
  fi
  FREQ_FILES=("${STRIDED_FILES[@]}")
fi

echo "Running ${#FREQ_FILES[@]} selected plans"

for f in "${FREQ_FILES[@]}"; do
  base=$(basename "$f")
  run_id=${base%.py}
  ts=$(date +%Y%m%d_%H%M%S)

  echo "=== Running plan: $base ==="

  # Start server in background
  export ZEUS_PFO_SCHEDULER=PointSolution3D
  export ZEUS_PFO_SCHEDULER_ARGS="{\"solution_path\": \"$f\"}"
  # export MASTER_ADDR=${HOST}

  server_log="$LOG_DIR/${run_id}_server_${ts}.log"
  # echo "Starting PFO server for $base on ${HOST}:${PORT} (log: $server_log)"
  echo "Starting PFO server for $base on ${PORT} (log: $server_log)"
  uvicorn zeus.optimizer.pipeline_frequency.server.router:app --port "$PORT" > "$server_log" 2>&1 &
  # uvicorn zeus.optimizer.pipeline_frequency.server.router:app --host "$HOST" --port "$PORT" > "$server_log" 2>&1 &
  server_pid=$!

  # Wait briefly for server to become ready
  sleep 3

  train_log="$LOG_DIR/${run_id}_train_${ts}.log"
  echo "Starting training (log: $train_log)"
  # Run training; hydra config within script controls toggles
  if ! python kareus_gpt_pretraining.py > "$train_log" 2>&1; then
    echo "Training failed for $base. See $train_log" >&2
  fi

  echo "Stopping server PID $server_pid"
  kill $server_pid >/dev/null 2>&1 || true
  wait $server_pid 2>/dev/null || true

  echo "Completed plan: $base"
  echo

done

mkdir -p nemo_experiments/megatron_llama_3_2_1b/frontier && setopt null_glob && mv nemo_experiments/megatron_llama_3_2_1b/2025* nemo_experiments/megatron_llama_3_2_1b/frontier/

echo "All runs completed. Logs in $LOG_DIR"

