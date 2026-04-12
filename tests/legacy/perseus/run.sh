#!/usr/bin/env bash
set -euo pipefail

########################################
# Unified training launcher            #
########################################
#
# Usage:
#   ./run.sh <node_rank>
# where <node_rank> is 0 or 1.
#
# MASTER_ADDR:
#   Set the default address of node 0 here. You can override it from
#   the environment when invoking this script if needed.
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

# Set default MASTER_ADDR here; override by exporting MASTER_ADDR before calling if desired.
MASTER_ADDR="${MASTER_ADDR:-172.31.35.92}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "Running torchrun with NODE_RANK=${NODE_RANK}, MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  megatron_gpt_pretraining.py

