#!/usr/bin/env bash
set -euo pipefail

###############################################
# Run run_all_with_scheds.sh for all configs  #
###############################################
#
# Usage:
#   ./run_all_with_scheds_all_configs.sh <node_rank> [host] [port]
# where <node_rank> is 0 or 1.
#
# This script iterates over a set of Kareus configs for LLaMA 3.2 3B and,
# for each config, calls run_all_with_scheds.sh to run all freqs+scheds
# frontier plans:
#   - cp1_tp8_bs8_seq4096   (tp8_8_4k)
#   - cp2_tp4_bs8_seq4096   (tp4cp2_8_4k)
#   - cp2_tp4_bs8_seq8192   (tp4cp2_8_8k)
#   - cp2_tp4_bs16_seq4096  (tp4cp2_16_4k)
###############################################

NODE_RANK="${1:-}"
if [[ -z "${NODE_RANK}" ]]; then
  echo "Usage: $0 <node_rank(0|1)> [host] [port]" >&2
  exit 1
fi

if [[ "${NODE_RANK}" != "0" && "${NODE_RANK}" != "1" ]]; then
  echo "ERROR: node_rank must be 0 or 1, got '${NODE_RANK}'" >&2
  exit 1
fi

HOST=${2:-0.0.0.0}
PORT=${3:-7787}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

model_name="${MODEL_NAME:-qwen3_1.7b}"

configs=(
  cp1_tp8_bs8_seq4096
  cp1_tp8_bs8_seq8192
  cp1_tp8_bs16_seq4096
  cp2_tp4_bs8_seq4096
  cp2_tp4_bs8_seq8192
  cp2_tp4_bs16_seq4096
)

for cfg in "${configs[@]}"; do
  results_dir="${SCRIPT_DIR}/${model_name}/${cfg}/kareus_frontier"

  if [[ ! -d "${results_dir}" ]]; then
    echo "[skip] results_dir not found for config '${cfg}': ${results_dir}" >&2
    continue
  fi

  echo "============================================"
  echo "Running all Kareus freqs+scheds for config: ${cfg}"
  echo "Using results_dir: ${results_dir}"
  echo "============================================"

  CONFIG="${cfg}" MODEL_NAME="${model_name}" \
    "${SCRIPT_DIR}/run_all_with_scheds.sh" "${NODE_RANK}" "${results_dir}" "${HOST}" "${PORT}"
done

echo "Completed run_all_with_scheds.sh for all configs on node${NODE_RANK}."


