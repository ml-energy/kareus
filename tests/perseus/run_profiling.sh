#!/usr/bin/env bash
set -euo pipefail

########################################
# Unified profiling runner             #
########################################
#
# Usage:
#   ./run_profiling.sh <node_rank>
# where <node_rank> is 0 or 1.
#
# This script:
#   - sweeps GPU frequency from 1410 down to 900 MHz
#   - calls ./run.sh <node_rank> at each frequency
#   - organizes NeMo outputs under nemo_experiments/megatron_llama_3_2_3b
#
# For node 1, it also scp's the final profiling directory back to node 0.
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

# Logical model/config identifiers
model_name="llama3.2_3b"
config="cp1_tp8_bs8_seq4096"

# Nemo experiment name (directory under nemo_experiments/)
nemo_model_name="megatron_llama_3_2_3b"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_DIR="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"

# YAML config for this model in 2-node setting
yaml_file="${SCRIPT_DIR}/conf/${nemo_model_name}_config_2nodes.yaml"
if [[ ! -f "${yaml_file}" ]]; then
  echo "ERROR: YAML config not found: ${yaml_file}" >&2
  exit 1
fi

# Remote sync settings (used when NODE_RANK=1)
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/workspaces/Kareus/tests/kareus}"
# MASTER_ADDR should point to node 0; use same default as run.sh if not set
MASTER_ADDR="${MASTER_ADDR:-172.31.39.81}"

########################################
# Update YAML with parsed config       #
########################################

echo "Updating YAML ${yaml_file} with parallelism/batch settings from config='${config}'"

python - "${yaml_file}" "${config}" <<'PY'
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

cfg.model.context_parallel_size = cp
cfg.model.tensor_model_parallel_size = tp
cfg.model.micro_batch_size = mb
cfg.model.encoder_seq_length = seq

# global_batch_size = micro_batch_size * 8 (as requested)
cfg.model.global_batch_size = mb * 8

OmegaConf.save(cfg, yaml_path)
PY

for frequency in $(seq 1410 -30 900); do
    echo "Setting GPU frequency to ${frequency} MHz on all GPUs"
    nvidia-smi -i 0,1,2,3,4,5,6,7 --lock-gpu-clocks=${frequency},${frequency}

    bash "${SCRIPT_DIR}/run.sh" "${NODE_RANK}"

    if [[ "${NODE_RANK}" == "0" ]]; then
        mkdir -p "${NEMO_DIR}/${frequency}/timers"
        mv "${NEMO_DIR}"/2025* "${NEMO_DIR}/${frequency}/" 2>/dev/null || true
        mv "${NEMO_DIR}/timers/"* "${NEMO_DIR}/${frequency}/timers" 2>/dev/null || true
        mv "${NEMO_DIR}"/*.txt "${NEMO_DIR}/${frequency}/" 2>/dev/null || true
    else
        mkdir -p "${NEMO_DIR}/${frequency}"
        mv "${NEMO_DIR}/timers" "${NEMO_DIR}/${frequency}/" 2>/dev/null || true
        mv "${NEMO_DIR}"/*.txt "${NEMO_DIR}/${frequency}/" 2>/dev/null || true
    fi
done

echo "Resetting GPU clocks"
nvidia-smi -i 0,1,2,3,4,5,6,7 --reset-gpu-clocks

if [[ "${NODE_RANK}" == "0" ]]; then
    target_dir="${NEMO_DIR}/${config}/profiling/node0"
else
    target_dir="${NEMO_DIR}/${config}/profiling/node1"
fi

mkdir -p "${target_dir}"
mv "${NEMO_DIR}/"* "${target_dir}/" 2>/dev/null || true

echo "Profiling complete for node${NODE_RANK}. Results under ${target_dir}"

# When running on node 1, sync profiling results back to node 0
if [[ "${NODE_RANK}" == "1" ]]; then
    remote_dir="${REMOTE_BASE_DIR}/nemo_experiments/${nemo_model_name}/${config}/profiling/"
    echo "Syncing profiling results from node1 to ${REMOTE_USER}@${MASTER_ADDR}:${remote_dir}"

    scp -i "${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}" -r "${target_dir}/" "${REMOTE_USER}@${MASTER_ADDR}":"${remote_dir}/"
fi
