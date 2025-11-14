#!/usr/bin/env bash
set -euo pipefail

########################################
# Run all freqs+scheds plans (Kareus)  #
########################################
#
# Usage:
#   ./run_all_with_scheds.sh <node_rank> [results_dir] [host] [port]
# Defaults:
#   results_dir: tests/kareus/<model_name>/<config>/perseus_results
#   host     : 0.0.0.0
#   port     : 7787
# Env overrides:
#   SAMPLE_STRIDE      : stride when picking freqs (default 20)
#   SLEEP_BEFORE_TRAIN : seconds to wait for server readiness (default 3)
########################################

########################################
# Node rank argument                   #
########################################

NODE_RANK="${1:-}"
if [[ -z "${NODE_RANK}" ]]; then
  echo "Usage: $0 <node_rank(0|1)> [results_dir] [host] [port]" >&2
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

# Default results directory is per-config perseus_results; can be overridden by arg2
DEFAULT_RESULTS_DIR="${SCRIPT_DIR}/${model_name}/${config}/perseus_results"
RESULTS_DIR=${2:-$DEFAULT_RESULTS_DIR}

HOST=${3:-0.0.0.0}
PORT=${4:-7787}

# YAML config for this model in 2-node setting
YAML_FILE="${SCRIPT_DIR}/conf/${nemo_model_name}_config_2nodes.yaml"
if [[ ! -f "${YAML_FILE}" ]]; then
  echo "ERROR: YAML config not found: ${YAML_FILE}" >&2
  exit 1
fi

# Remote sync settings (used when NODE_RANK=1)
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/workspaces/Kareus/tests/kareus}"
# MASTER_ADDR should point to node 0; use same default as run.sh if not set
MASTER_ADDR="${MASTER_ADDR:-172.31.39.81}"

LOG_DIR="$SCRIPT_DIR/logs_pfo_runs"
mkdir -p "$LOG_DIR"

# Discover frequency plan files
mapfile -t FREQ_FILES < <(ls -1 "$RESULTS_DIR"/freqs_pipeline_*.py 2>/dev/null | sort)
if [[ ${#FREQ_FILES[@]} -eq 0 ]]; then
  echo "No freqs_pipeline_*.py found under $RESULTS_DIR" >&2
  exit 1
fi

TOTAL_FILES=${#FREQ_FILES[@]}
echo "Found ${TOTAL_FILES} frequency plans under $RESULTS_DIR"

# Optional sampling by stride from largest to smallest; include the smallest
SAMPLE_STRIDE=${SAMPLE_STRIDE:-100}
if (( SAMPLE_STRIDE > 1 )); then
  echo "Sampling from largest, every ${SAMPLE_STRIDE}th plan (total=${TOTAL_FILES})"
  declare -a STRIDED_FILES=()
  for ((i=TOTAL_FILES-1; i>=0; i-=SAMPLE_STRIDE)); do
    STRIDED_FILES+=("${FREQ_FILES[$i]}")
  done
  # if [[ ${#STRIDED_FILES[@]} -eq 0 || "${STRIDED_FILES[-1]}" != "${FREQ_FILES[0]}" ]]; then
  #   STRIDED_FILES+=("${FREQ_FILES[0]}")
  # fi
  FREQ_FILES=("${STRIDED_FILES[@]}")
fi

# # Optionally start from a specific plan (resume). Default to freqs_pipeline_02442.py.
# # Override with START_AT env var (set to empty to disable).
# START_AT=${START_AT:-freqs_pipeline_02442.py}
# if [[ -n "$START_AT" ]]; then
#   start_idx=-1
#   for i in "${!FREQ_FILES[@]}"; do
#     if [[ "$(basename "${FREQ_FILES[$i]}")" == "$START_AT" ]]; then
#       start_idx=$i; break
#     fi
#   done
#   if (( start_idx >= 0 )); then
#     echo "Resuming from $START_AT (index $start_idx of ${#FREQ_FILES[@]})"
#     FREQ_FILES=("${FREQ_FILES[@]:$start_idx}")
#   else
#     echo "START_AT '$START_AT' not found in selected plans; running full selection" >&2
#   fi
# fi

echo "Running ${#FREQ_FILES[@]} selected plans"

# # Backup YAML and restore on exit
# BACKUP_YAML="${YAML_FILE}.bak"
# cp "$YAML_FILE" "$BACKUP_YAML"
# restore_yaml() { mv -f "$BACKUP_YAML" "$YAML_FILE" || true; }
# trap restore_yaml EXIT

SLEEP_BEFORE_TRAIN=${SLEEP_BEFORE_TRAIN:-3}

for f in "${FREQ_FILES[@]}"; do
  base=$(basename "$f")
  iter_id=${base#freqs_pipeline_}
  iter_id=${iter_id%.py}
  run_id=${base%.py}
  ts=$(date +%Y%m%d_%H%M%S)

  sched_file="$RESULTS_DIR/scheds_pipeline_${iter_id}.py"
  if [[ ! -f "$sched_file" ]]; then
    echo "[skip] No matching scheds for $base at $sched_file" >&2
    continue
  fi

  echo "=== Running plan: $base with scheds: $(basename "$sched_file") ==="

  # Update YAML solution_path and parallelism/batch settings (as in run_one_config.sh)
  python - "$YAML_FILE" "$sched_file" "$config" <<'PY'
import re
import sys
from omegaconf import OmegaConf

yaml_path, sched_path, cfg_str = sys.argv[1], sys.argv[2], sys.argv[3]

cfg = OmegaConf.load(yaml_path)

if "model" not in cfg:
    raise SystemExit(f"'model' section not found in {yaml_path}")

ks = cfg.model.get("kareus_scheduler_kwargs")
if ks is None:
    ks = {}
    cfg.model.kareus_scheduler_kwargs = ks

cfg.model.kareus_scheduler_kwargs["solution_path"] = sched_path

# Parse config string: cp1_tp8_bs8_seq4096
m = re.match(r"^cp(\d+)_tp(\d+)_bs(\d+)_seq(\d+)$", cfg_str)
if not m:
    raise SystemExit(
        f"Config string '{cfg_str}' is not in expected format 'cp<cp>_tp<tp>_bs<mb>_seq<seq>'"
    )

cp, tp, mb, seq = map(int, m.groups())

cfg.model.context_parallel_size = cp
cfg.model.tensor_model_parallel_size = tp
cfg.model.micro_batch_size = mb
cfg.model.encoder_seq_length = seq

# global_batch_size = micro_batch_size * 8 (as requested)
cfg.model.global_batch_size = mb * 8

OmegaConf.save(cfg, yaml_path)
PY

  # Start server in background for this freqs plan (same style as run_all_freqs.sh)
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

  # Wait for server to become ready
  sleep "$SLEEP_BEFORE_TRAIN"

  train_log="$LOG_DIR/${run_id}_train_${ts}.log"
  echo "Starting training via run.sh (node_rank=${NODE_RANK}) (log: $train_log)"
  if ! bash "$SCRIPT_DIR/run.sh" "${NODE_RANK}" > "$train_log" 2>&1; then
    echo "Training failed for $base. See $train_log" >&2
  fi

  echo "Stopping server PID $server_pid"
  kill $server_pid >/dev/null 2>&1 || true
  wait $server_pid 2>/dev/null || true

  # After training, organize NeMo outputs for this freqs+scheds plan
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

# Create frontier directory per node and sync node1 results to node0
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
