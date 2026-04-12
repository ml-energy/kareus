#!/usr/bin/env bash
set -euo pipefail

###############################################
# Run run_profiling.sh for all configs        #
###############################################
#
# Usage:
#   ./run_profiling_all_configs.sh <node_rank>
# where <node_rank> is 0 or 1.
# 
# This script iterates over a set of Perseus configs for LLaMA 3.2 3B and,
# for each config, calls run_profiling.sh to sweep GPU frequencies and
# collect NeMo profiling data under nemo_experiments/megatron_llama_3_2_3b.
#
# Configs:
#   - cp1_tp8_bs8_seq4096   (tp8_8_4k)
#   - cp2_tp4_bs8_seq4096   (tp4cp2_8_4k)
#   - cp2_tp4_bs8_seq8192   (tp4cp2_8_8k)
#   - cp2_tp4_bs16_seq4096  (tp4cp2_16_4k)
###############################################

NODE_RANK="${1:-}"
if [[ -z "${NODE_RANK}" ]]; then
  echo "Usage: $0 <node_rank(0|1)>" >&2
  exit 1
fi

if [[ "${NODE_RANK}" != "0" && "${NODE_RANK}" != "1" ]]; then
  echo "ERROR: node_rank must be 0 or 1, got '${NODE_RANK}'" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

configs=(
  # cp1_tp8_bs8_seq4096
  # cp1_tp8_bs8_seq8192
  # cp1_tp8_bs16_seq4096
  cp2_tp4_bs8_seq4096
  cp2_tp4_bs8_seq8192
  cp2_tp4_bs16_seq4096
)

for cfg in "${configs[@]}"; do
  echo "============================================"
  echo "Running profiling sweep for config: ${cfg}"
  echo "============================================"

  CONFIG="${cfg}" bash "${SCRIPT_DIR}/run_profiling.sh" "${NODE_RANK}"
done

echo "Completed run_profiling.sh for all configs on node${NODE_RANK}."


