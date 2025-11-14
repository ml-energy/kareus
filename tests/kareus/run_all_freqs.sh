#!/usr/bin/env bash
set -euo pipefail

########################################
# Run all frequency plans (Kareus)     #
########################################
#
# Usage:
#   ./run_all_freqs.sh <node_rank> [freqs_dir] [host] [port]
# Defaults:
#   freqs_dir: tests/kareus/<model_name>/<config>/perseus_results
#   host     : 0.0.0.0
#   port     : 7787
#
# This script:
#   - iterates over freqs_pipeline_*.py frequency plans
#   - launches a PFO server for each plan (as in run_one_config.sh)
#   - runs Kareus Megatron GPT pretraining for each plan
#   - finally moves frontier NeMo runs to the config-specific frontier dir:
#       nemo_experiments/<nemo_model_name>/<config>/frontier
########################################

########################################
# Node rank argument                   #
########################################

NODE_RANK="${1:-}"
if [[ -z "${NODE_RANK}" ]]; then
  echo "Usage: $0 <node_rank(0|1)> [freqs_dir] [host] [port]" >&2
  exit 1
fi

if [[ "${NODE_RANK}" != "0" && "${NODE_RANK}" != "1" ]]; then
  echo "ERROR: node_rank must be 0 or 1, got '${NODE_RANK}'" >&2
  exit 1
fi

########################################
# User configuration (edit as needed) #
########################################

model_name="llama3.2_3b"
config="cp2_tp4_bs16_seq4096"

# Nemo experiment name (directory under nemo_experiments/)
nemo_model_name="megatron_llama_3_2_3b"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

NEMO_ROOT="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"
NEMO_DIR="${NEMO_ROOT}/${config}"

# Default freqs directory is per-config perseus_results; can be overridden by arg2
DEFAULT_FREQS_DIR="${SCRIPT_DIR}/${model_name}/${config}/perseus_results}"
FREQS_DIR=${2:-$DEFAULT_FREQS_DIR}

HOST=${3:-0.0.0.0}
PORT=${4:-7787}

# Remote sync settings (used when NODE_RANK=1)
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/workspaces/Kareus/tests/kareus}"
# MASTER_ADDR should point to node 0; use same default as run.sh if not set
MASTER_ADDR="${MASTER_ADDR:-172.31.39.81}"

LOG_DIR="${SCRIPT_DIR}/logs_pfo_runs"
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
  # # Ensure lowest-numbered (first) plan is included
  # if [[ ${#STRIDED_FILES[@]} -eq 0 || "${STRIDED_FILES[-1]}" != "${FREQ_FILES[0]}" ]]; then
  #   STRIDED_FILES+=("${FREQ_FILES[0]}")
  # fi
  FREQ_FILES=("${STRIDED_FILES[@]}")
fi

echo "Using freqs directory: $FREQS_DIR"
echo "Running ${#FREQ_FILES[@]} selected plans"

for f in "${FREQ_FILES[@]}"; do
  base=$(basename "$f")
  run_id=${base%.py}
  ts=$(date +%Y%m%d_%H%M%S)

  echo "=== Running plan: $base ==="

  # Start PFO server in background (same style as run_one_config.sh)
  export ZEUS_PFO_SCHEDULER=PointSolution3D
  export ZEUS_PFO_SCHEDULER_ARGS="{\"solution_path\": \"$f\"}"
  export MASTER_ADDR="${HOST}"

  server_log="$LOG_DIR/${run_id}_server_${ts}.log"
  echo "Starting PFO server for $base on ${HOST}:${PORT} (log: $server_log)"
  uvicorn zeus.optimizer.pipeline_frequency.server.router:app \
    --host "$HOST" \
    --port "$PORT" \
    > "$server_log" 2>&1 &
  server_pid=$!

  # Wait briefly for server to become ready
  sleep 3

  train_log="$LOG_DIR/${run_id}_train_${ts}.log"
  echo "Starting training via run.sh (node_rank=${NODE_RANK}) (log: $train_log)"
  # Run training via unified launcher; hydra config within script controls toggles
  if ! bash "${SCRIPT_DIR}/run.sh" "${NODE_RANK}" > "$train_log" 2>&1; then
    echo "Training failed for $base. See $train_log" >&2
  fi

  echo "Stopping server PID $server_pid"
  kill $server_pid >/dev/null 2>&1 || true
  wait $server_pid 2>/dev/null || true

  # After training, organize NeMo outputs for this frequency plan
  # Extract numeric frequency_plan id from freqs_pipeline_<id>.py
  plan_id="${run_id#freqs_pipeline_}"

  if [[ "${NODE_RANK}" == "0" ]]; then
    mkdir -p "${NEMO_DIR}/${plan_id}/timers"
    mv "${NEMO_ROOT}"/2025* "${NEMO_DIR}/${plan_id}/" 2>/dev/null || true
    mv "${NEMO_ROOT}/timers/"* "${NEMO_DIR}/${plan_id}/timers" 2>/dev/null || true
    mv "${NEMO_ROOT}"/*.txt "${NEMO_DIR}/${plan_id}/" 2>/dev/null || true
  else
    mkdir -p "${NEMO_DIR}/${plan_id}"
    mv "${NEMO_ROOT}/timers" "${NEMO_DIR}/${plan_id}/" 2>/dev/null || true
    mv "${NEMO_ROOT}"/*.txt "${NEMO_DIR}/${plan_id}/" 2>/dev/null || true
  fi

  echo "Completed plan: $base (stored under ${NEMO_DIR}/${plan_id})"
  echo

done

if [[ "${NODE_RANK}" == "0" ]]; then
  FRONTIER_DIR="${NEMO_DIR}/frontier/node0"
else
  FRONTIER_DIR="${NEMO_DIR}/frontier/node1"
fi

mkdir -p "${FRONTIER_DIR}"

# Move all per-plan directories under this config into the frontier dir,
# keeping the per-plan structure (and leaving 'frontier' itself in place).
for d in "${NEMO_DIR}"/*; do
  base_d="$(basename "$d")"
  [[ "$base_d" == "frontier" ]] && continue
  mv "$d" "${FRONTIER_DIR}/" 2>/dev/null || true
done

echo "All runs completed on node${NODE_RANK}. Frontier runs saved under ${FRONTIER_DIR}"
echo "Logs in $LOG_DIR"

# When running on node 1, sync frontier results back to node 0
if [[ "${NODE_RANK}" == "1" ]]; then
  remote_dir="${REMOTE_BASE_DIR}/nemo_experiments/${nemo_model_name}/${config}/frontier/"
  echo "Syncing frontier results from node1 to ${REMOTE_USER}@${MASTER_ADDR}:${remote_dir}"

  scp -i "${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}" -r "${FRONTIER_DIR}/" "${REMOTE_USER}@${MASTER_ADDR}":"${remote_dir}/"
fi
