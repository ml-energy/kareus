#!/usr/bin/env bash
set -euo pipefail

########################################
# Unified one-config runner            #
########################################
#
# Usage:
#   ./run_one_config.sh <node_rank>
# where <node_rank> is 0 or 1.
#
# Node 0:
#   - Locates freqs/scheds solutions
#   - Updates YAML (scheduler + cp/tp/mb/seq/global_batch)
#   - Starts PFO server
#   - Runs training via run.sh 0
#   - Collects NeMo outputs under nemo_experiments/.../config/kareus/<config_dir>
#
# Node 1:
#   - Updates YAML (same as node 0)
#   - Runs training via run.sh 1
#   - Collects logs and syncs back to node 0
########################################

NODE_RANK="${1:-}"
if [[ -z "${NODE_RANK}" ]]; then
  echo "Usage: $0 <node_rank(0|1)>" >&2
  exit 1
fi

if [[ "${NODE_RANK}" != "0" && "${NODE_RANK}" != "1" ]]; then
  echo "ERROR: node_rank must be 0 or 1, got '${NODE_RANK}'" >&2
  exit 1
fi

########################################
# User configuration (edit as needed) #
########################################

# Logical model/config identifiers used to locate Kareus solutions
# model_name="llama3.2_3b"
model_name="${MODEL_NAME:-qwen3_1.7b}"
config="cp1_tp8_bs8_seq4096"
config_dir="scale1.15"

# Nemo experiment name (directory under nemo_experiments/)
# For LLaMA 3.2 3B this is typically "megatron_llama_3_2_3b"
# nemo_model_name="megatron_llama_3_2_3b"
nemo_model_name="megatron_qwen3_1p7b"

# Node-0 address (IP or hostname) that all nodes use as MASTER_ADDR
# You MUST set this before running (can be overridden from environment)
MASTER_ADDR="${MASTER_ADDR:-172.31.35.92}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Remote (node 0) path to sync collected results into, from node 1
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-~/workspace/Kareus/tests/kareus}"

########################################
# Derived paths                        #
########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Directory where freqs/scheds solutions live, e.g.
#   tests/kareus/llama3.2_3b/cp1_tp8_bs8_seq4096/noscale
solution_root="${SCRIPT_DIR}/${model_name}/${config}/${config_dir}"

# YAML config for this model in 2-node setting, e.g.
#   tests/kareus/conf/megatron_llama3.2_3b_config_2nodes.yaml
yaml_file="${SCRIPT_DIR}/conf/${nemo_model_name}_config_2nodes.yaml"
if [[ ! -f "${yaml_file}" ]]; then
  echo "ERROR: YAML config not found: ${yaml_file}" >&2
  exit 1
fi

# Directory where we will collect NeMo outputs for this config
output_dir="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}/${config}/kareus/${config_dir}"
mkdir -p "${output_dir}"

########################################
# Locate Kareus solutions              #
########################################

# freqs needed only on node 0 for PFO server
freqs_solution_path=""
if [[ "${NODE_RANK}" == "0" ]]; then
  freqs_solution_path="$(ls "${solution_root}"/freqs_pipeline_*.py 2>/dev/null | head -n 1 || true)"
fi

scheds_solution_path="$(ls "${solution_root}"/scheds_pipeline_*.py 2>/dev/null | head -n 1 || true)"

if [[ "${NODE_RANK}" == "0" ]]; then
  if [[ -z "${freqs_solution_path}" || -z "${scheds_solution_path}" ]]; then
    echo "ERROR: Could not find freqs/scheds solution in '${solution_root}'." >&2
    echo "Expected files: freqs_pipeline_*.py and scheds_pipeline_*.py" >&2
    exit 1
  fi
  echo "Using freqs_solution_path = ${freqs_solution_path}"
  echo "Using scheds_solution_path = ${scheds_solution_path}"
else
  if [[ -z "${scheds_solution_path}" ]]; then
    echo "ERROR: Could not find scheds solution in '${solution_root}'." >&2
    echo "Expected file: scheds_pipeline_*.py" >&2
    exit 1
  fi
  echo "Using scheds_solution_path (node 1) = ${scheds_solution_path}"
fi

########################################
# Update YAML with scheduler + config  #
########################################

echo "Updating YAML ${yaml_file} with Kareus scheduler path and parallelism/batch settings from config='${config}'"

python - "${yaml_file}" "${scheds_solution_path}" "${config}" <<'PY'
import re
import sys
from omegaconf import OmegaConf

yaml_path, sched_path, cfg_str = sys.argv[1], sys.argv[2], sys.argv[3]

cfg = OmegaConf.load(yaml_path)

# 1) Set Kareus scheduler solution_path
if "model" not in cfg:
    raise SystemExit(f"'model' section not found in {yaml_path}")

ks = cfg.model.get("kareus_scheduler_kwargs")
if ks is None:
    ks = {}
    cfg.model.kareus_scheduler_kwargs = ks

cfg.model.kareus_scheduler_kwargs["solution_path"] = sched_path

# 2) Parse config string: cp1_tp8_bs8_seq4096
m = re.match(r"^cp(\d+)_tp(\d+)_bs(\d+)_seq(\d+)$", cfg_str)
if not m:
    raise SystemExit(
        f"Config string '{cfg_str}' is not in expected format 'cp<cp>_tp<tp>_bs<mb>_seq<seq>'"
    )

cp, tp, mb, seq = map(int, m.groups())

cfg.trainer.max_steps = 30
cfg.trainer.log_every_n_steps = 40
cfg.trainer.val_check_interval = 40

cfg.model.enable_megatron_timers = False
cfg.model.enable_zeus_monitor = True
cfg.model.enable_power_monitor = True
cfg.model.enable_perseus_optimizer = True
cfg.model.enable_kareus_scheduler = True

cfg.model.context_parallel_size = cp
cfg.model.tensor_model_parallel_size = tp
cfg.model.micro_batch_size = mb
cfg.model.encoder_seq_length = seq

# global_batch_size = micro_batch_size * 8 (as requested)
cfg.model.global_batch_size = mb * 8

OmegaConf.save(cfg, yaml_path)
PY

########################################
# Launch per-node workflow             #
########################################

export MASTER_ADDR
export MASTER_PORT

if [[ "${NODE_RANK}" == "0" ]]; then
  ######################################
  # Node 0: start PFO + run + collect  #
  ######################################

  server_log="${output_dir}/pfo_server_${config_dir}.log"

  echo "Starting PFO server for ${model_name} ${config} (${config_dir}) on ${MASTER_ADDR}:7787"
  ZEUS_PFO_SCHEDULER=PointSolution3D \
  ZEUS_PFO_SCHEDULER_ARGS="{\"solution_path\": \"${freqs_solution_path}\"}" \
  uvicorn zeus.optimizer.pipeline_frequency.server.router:app \
    --host "${MASTER_ADDR}" \
    --port 7787 \
    > "${server_log}" 2>&1 &

  PFO_PID=$!
  echo "PFO server PID: ${PFO_PID}"
  sleep 5

  echo "MASTER_ADDR=${MASTER_ADDR}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "Launching training via run.sh (node_rank=0)"

  bash "${SCRIPT_DIR}/run.sh" 0

  echo "Moving NeMo experiment outputs into ${output_dir}"

  chmod a+w "${output_dir}"

  # Move time-stamped experiment directories (e.g., 20YY-*) contents, then delete source dirs
  if compgen -G "${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}/20*" > /dev/null; then
    shopt -s nullglob dotglob
    for d in "${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"/20*; do
      if [[ -d "$d" ]]; then
        contents=("$d"/*)
        if (( ${#contents[@]} )); then
          mv "${contents[@]}" "${output_dir}/"
        fi
        rm -rf "$d"
      fi
    done
    shopt -u nullglob dotglob
  fi

  # Move any text logs from the default experiments dir
  if compgen -G "${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}/*.txt" > /dev/null; then
    mv "${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"/*.txt "${output_dir}/"
  fi

  # Stop PFO server
  if [[ -n "${PFO_PID:-}" ]]; then
    if ps -p "${PFO_PID}" > /dev/null 2>&1; then
      echo "Stopping PFO server PID ${PFO_PID}"
      kill "${PFO_PID}" || true
      wait "${PFO_PID}" 2>/dev/null || true
    fi
  fi

  echo "Node 0 run finished. Outputs are under: ${output_dir}"

else
  ######################################
  # Node 1: run + collect + sync       #
  ######################################

  echo "MASTER_ADDR=${MASTER_ADDR}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "Launching training via run.sh (node_rank=1)"

  bash "${SCRIPT_DIR}/run.sh" 1

  echo "Moving NeMo experiment text logs into ${output_dir}"

  if compgen -G "${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}/*.txt" > /dev/null; then
    mv "${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"/*.txt "${output_dir}/"
  fi

  remote_dir="${REMOTE_BASE_DIR}/nemo_experiments/${nemo_model_name}/${config}/kareus/${config_dir}"
  echo "Syncing results from node 1 to ${REMOTE_USER}@${MASTER_ADDR}:${remote_dir}"

  # Remote directory should exist; if not, attempt to create it
  ssh -i "${SSH_KEY_PATH:-$HOME/.ssh/ruofanw.pem}" "${REMOTE_USER}@${MASTER_ADDR}" "mkdir -p '${remote_dir}'"
  sleep 5
  scp -i "${SSH_KEY_PATH:-$HOME/.ssh/ruofanw.pem}" -r "${output_dir}/"* "${REMOTE_USER}@${MASTER_ADDR}":"${remote_dir}/"

  echo "Node 1 run finished. Outputs synced to node 0 under: ${remote_dir}"
fi



