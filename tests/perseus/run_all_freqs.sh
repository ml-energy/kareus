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
#   NUM_SAMPLES        : number of freqs plans to sample (default 10; use 0 or >= total to disable sampling)
#   SLEEP_BEFORE_TRAIN : seconds to wait for server readiness (default 3)
#   MASTER_ADDR        : node-0 address (used by both PFO server and training)
#   MASTER_PORT        : distributed training port (default 29500)
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

# Logical model/config identifiers used to locate Perseus frontier freqs
MODEL_NAME="${MODEL_NAME:-llama3.2_3b}"
config="${CONFIG:-cp2_tp4_bs16_seq4096}"

case "${MODEL_NAME}" in
  llama3.2_3b)
    model_name="llama3.2_3b"
    nemo_model_name="megatron_llama_3_2_3b"
    ;;
  qwen3_1.7b)
    model_name="qwen3_1.7b"
    nemo_model_name="megatron_qwen3_1p7b"
    ;;
  *)
    echo "ERROR: Unsupported MODEL_NAME='${MODEL_NAME}'. Supported: llama3.2_3b, qwen3_1.7b" >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

NEMO_ROOT="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"

# Default results directory is per-config frontier; can be overridden by arg2
DEFAULT_RESULTS_DIR="${SCRIPT_DIR}/${model_name}/${config}/frontier"
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
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-~/workspace/Kareus/tests/perseus}"

# Node-0 address and training port (can be overridden from environment)
MASTER_ADDR="${MASTER_ADDR:-172.31.35.92}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Kareus output root where per-plan NeMo outputs will be collected
KAREUS_ROOT="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}/${config}/frontier"
mkdir -p "${KAREUS_ROOT}"

# LOG_DIR="$SCRIPT_DIR/logs_pfo_runs"
# mkdir -p "$LOG_DIR"

# Export for downstream training processes
export MASTER_ADDR
export MASTER_PORT

# Discover frequency plan files in the frontier directory and run all of them
mapfile -t FREQ_FILES < <(ls -1 "$RESULTS_DIR"/freqs_pipeline_*.py 2>/dev/null | sort)
if [[ ${#FREQ_FILES[@]} -eq 0 ]]; then
  echo "No freqs_pipeline_*.py found under $RESULTS_DIR" >&2
  exit 1
fi

TOTAL_FILES=${#FREQ_FILES[@]}
echo "Found ${TOTAL_FILES} frequency plans under $RESULTS_DIR (running all plans; no sampling)"

## Old sampling logic (now disabled; kept for reference)
## Optional sampling: pick a fixed number of plans, compute stride automatically,
## sample from largest to smallest, and ensure we include the smallest plan.
# NUM_SAMPLES=${NUM_SAMPLES:-10}
# if (( NUM_SAMPLES > 0 && NUM_SAMPLES < TOTAL_FILES )); then
#   # Compute stride so that we pick at most NUM_SAMPLES plans
#   stride=$(( (TOTAL_FILES + NUM_SAMPLES - 1) / NUM_SAMPLES ))
#   (( stride < 1 )) && stride=1
#
#   echo "Sampling ${NUM_SAMPLES} plans from ${TOTAL_FILES} total (computed stride=${stride})"
#   declare -a STRIDED_FILES=()
#   for ((i=TOTAL_FILES-1; i>=0; i-=stride)); do
#     STRIDED_FILES+=("${FREQ_FILES[$i]}")
#   done
#
#   # # Ensure the smallest (first) freqs plan is included
#   # if [[ ${#STRIDED_FILES[@]} -eq 0 || "${STRIDED_FILES[-1]}" != "${FREQ_FILES[0]}" ]]; then
#   #   STRIDED_FILES+=("${FREQ_FILES[0]}")
#   # fi
#
#   FREQ_FILES=("${STRIDED_FILES[@]}")
# fi

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

SLEEP_BEFORE_TRAIN=${SLEEP_BEFORE_TRAIN:-5}

for f in "${FREQ_FILES[@]}"; do
  base=$(basename "$f")
  iter_id=${base#freqs_pipeline_}
  iter_id=${iter_id%.py}
  run_id=${base%.py}
  ts=$(date +%Y%m%d_%H%M%S)

  # Extract numeric plan id from freqs_pipeline_<id>.py and create per-plan
  # Kareus output directory up front (similar to run_one_config_kareus.sh).
  plan_id="${run_id#freqs_pipeline_}"
  plan_output_dir="${KAREUS_ROOT}/${plan_id}"
  mkdir -p "${plan_output_dir}"

  # Update YAML solution_path and parallelism/batch settings (as in run_one_config.sh)
  python - "$YAML_FILE" "$config" <<'PY'
import re
import sys
from omegaconf import OmegaConf

yaml_path, cfg_str = sys.argv[1], sys.argv[2]

cfg = OmegaConf.load(yaml_path)

if "model" not in cfg:
    raise SystemExit(f"'model' section not found in {yaml_path}")

# Parse config string: cp1_tp8_bs8_seq4096
m = re.match(r"^cp(\d+)_tp(\d+)_bs(\d+)_seq(\d+)$", cfg_str)
if not m:
    raise SystemExit(
        f"Config string '{cfg_str}' is not in expected format 'cp<cp>_tp<tp>_bs<mb>_seq<seq>'"
    )

cp, tp, mb, seq = map(int, m.groups())

# Trainer and monitoring settings (align with run_one_config.sh)
cfg.trainer.max_steps = 30
cfg.trainer.log_every_n_steps = 40
cfg.trainer.val_check_interval = 40

cfg.model.enable_megatron_timers = False
cfg.model.enable_zeus_monitor = True
cfg.model.enable_power_monitor = False
cfg.model.enable_perseus_optimizer = True
cfg.model.enable_kareus_scheduler = False

cfg.model.context_parallel_size = cp
cfg.model.tensor_model_parallel_size = tp
cfg.model.micro_batch_size = mb
cfg.model.encoder_seq_length = seq

# global_batch_size = micro_batch_size * 8 (as requested)
cfg.model.global_batch_size = mb * 8

OmegaConf.save(cfg, yaml_path)
PY

  # Configure PFO scheduler for this freqs plan. The PFO server itself should
  # only be started on node 0; node 1 simply waits for it and runs training.
  export ZEUS_PFO_SCHEDULER=PointSolution3D
  export ZEUS_PFO_SCHEDULER_ARGS="{\"solution_path\": \"$f\"}"

  if [[ "${NODE_RANK}" == "0" ]]; then
    # Node 0: start PFO server in background (same style as run_one_config_kareus.sh)
    server_log="${plan_output_dir}/pfo_server_${ts}.log"
    echo "Starting PFO server for $base on ${MASTER_ADDR}:${PORT} (log: $server_log)"
    uvicorn zeus.optimizer.pipeline_frequency.server.router:app \
      --host "${MASTER_ADDR}" \
      --port "$PORT" \
      > "$server_log" 2>&1 &
    server_pid=$!

    # Wait for server to become ready
    sleep "$SLEEP_BEFORE_TRAIN"
  else
    # Node 1: just wait for node 0's PFO server to be ready
    echo "Node 1 waiting ${SLEEP_BEFORE_TRAIN}s for PFO server on node 0..."
    sleep "$SLEEP_BEFORE_TRAIN"
  fi

  echo "MASTER_ADDR=${MASTER_ADDR}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "Starting training via run.sh (node_rank=${NODE_RANK})"
  # Do not capture training logs to a separate file; let them print to stdout/stderr.
  if ! bash "$SCRIPT_DIR/run.sh" "${NODE_RANK}"; then
    echo "Training failed for $base (node_rank=${NODE_RANK})" >&2
  fi

  if [[ "${NODE_RANK}" == "0" ]]; then
    echo "Stopping server PID $server_pid"
    kill $server_pid >/dev/null 2>&1 || true
    wait $server_pid 2>/dev/null || true
  fi

  if [[ "${NODE_RANK}" == "0" ]]; then
    chmod a+w "${plan_output_dir}"

    # Move time-stamped experiment directories (e.g., 20YY-*) contents, then delete source dirs
    if compgen -G "${NEMO_ROOT}/20*" > /dev/null; then
      shopt -s nullglob dotglob
      for d in "${NEMO_ROOT}"/20*; do
        if [[ -d "$d" ]]; then
          contents=("$d"/*)
          if (( ${#contents[@]} )); then
            mv "${contents[@]}" "${plan_output_dir}/"
          fi
          rm -rf "$d"
        fi
      done
      shopt -u nullglob dotglob
    fi

    # Move any text logs from the default experiments dir
    if compgen -G "${NEMO_ROOT}/*.txt" > /dev/null; then
      mv "${NEMO_ROOT}"/*.txt "${plan_output_dir}/" 2>/dev/null || true
    fi
  else
    # On node 1 we only expect text logs under the NeMo root
    if compgen -G "${NEMO_ROOT}/*.txt" > /dev/null; then
      mv "${NEMO_ROOT}"/*.txt "${plan_output_dir}/" 2>/dev/null || true
    fi

    # Sync node 1 results for this plan back to node 0, mirroring run_one_config_kareus.sh
    remote_dir="${REMOTE_BASE_DIR}/nemo_experiments/${nemo_model_name}/${config}/frontier/${plan_id}"
    echo "Syncing plan ${plan_id} results from node1 to ${REMOTE_USER}@${MASTER_ADDR}:${remote_dir}"

    # ssh -i "${SSH_KEY_PATH:-$HOME/.ssh/ruofanw.pem}" "${REMOTE_USER}@${MASTER_ADDR}" "mkdir -p '${remote_dir}'"
    sleep 5
    scp -i "${SSH_KEY_PATH:-$HOME/.ssh/ruofanw.pem}" -r "${plan_output_dir}/"* "${REMOTE_USER}@${MASTER_ADDR}":"${remote_dir}/"
  fi

  echo "Completed plan: $base (stored under ${plan_output_dir})"
  echo
done

echo "All runs completed on node${NODE_RANK}. Per-plan Kareus runs saved under ${KAREUS_ROOT}"
# echo "Logs in $LOG_DIR"
