#!/usr/bin/env bash
# Artifact Nanobatch baseline on 16 GPUs (2 nodes x 8 GPUs A100), PP=2, microbatches=8.
#
# Uses Kareus MegatronGPTModel (nanobatching) WITHOUT any frequency
# optimization (no Perseus, no Kareus scheduler).  Counterpart of run_megatron.sh.
#
# Usage:
#   MASTER_ADDR=<node0_ip> bash run_nanobatch.sh <node_rank>
#
# Env vars: same as run_megatron.sh (CONFIG_MODE full|single).
set -euo pipefail

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
# shellcheck source=./env.sh
[[ -f "${SCRIPT_DIR}/env.sh" ]] && source "${SCRIPT_DIR}/env.sh"
cd "$SCRIPT_DIR"

if [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "ERROR: MASTER_ADDR must be set (in env.sh or via env var)" >&2
  exit 1
fi
MASTER_PORT="${MASTER_PORT:-6000}"
CONFIG_MODE="${CONFIG_MODE:-full}"

REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-$HOME/workspace/Kareus/tests/artifact}"
SSH_KEY_PATH="${SSH_KEY_PATH:-}"
SSH_KEY_OPTS=()
if [[ -n "${SSH_KEY_PATH}" ]]; then
    SSH_KEY_OPTS=(-i "${SSH_KEY_PATH}")
fi

export MASTER_ADDR MASTER_PORT REMOTE_USER REMOTE_BASE_DIR SSH_KEY_PATH

PP=2
NUM_MICROBATCHES=8

CONFIGS_FULL=(
    "megatron_llama3.2_3b_config  1  8  8   4096"
    "megatron_llama3.2_3b_config  2  4  8   4096"
    "megatron_llama3.2_3b_config  2  4  8   8192"
    "megatron_llama3.2_3b_config  2  4  16  4096"
    "megatron_qwen3_1.7b_config   1  8  8   4096"
    "megatron_qwen3_1.7b_config   1  8  8   8192"
    "megatron_qwen3_1.7b_config   1  8  16  4096"
    "megatron_qwen3_1.7b_config   2  4  8   4096"
    "megatron_qwen3_1.7b_config   2  4  8   8192"
    "megatron_qwen3_1.7b_config   2  4  16  4096"
)

case "${CONFIG_MODE}" in
    full)   CONFIGS=("${CONFIGS_FULL[@]}") ;;
    single) CONFIGS=("${CONFIGS_FULL[0]}") ;;
    *) echo "ERROR: CONFIG_MODE must be 'full' or 'single'" >&2; exit 1 ;;
esac

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "===== Artifact Nanobatch 16-GPU Tests (CONFIG_MODE=${CONFIG_MODE}) ====="
echo "Total configurations: ${#CONFIGS[@]}"
echo ""

for i in "${!CONFIGS[@]}"; do
    read -r CFG CP TP MBS SEQ <<< "${CONFIGS[$i]}"

    GBS=$(( MBS * NUM_MICROBATCHES ))

    nemo_model_name="${CFG%_config}"
    config_tag="cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}"
    OUTPUT_DIR="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}/${config_tag}/nanobatch"
    NEMO_DIR="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"
    LOG="${LOG_DIR}/${nemo_model_name}_${config_tag}_nanobatch.log"

    mkdir -p "${OUTPUT_DIR}"

    echo ">>> Test $((i+1))/${#CONFIGS[@]}: ${CFG} cp${CP}_tp${TP} MBS=${MBS} SEQ=${SEQ} GBS=${GBS}"

    torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank="${NODE_RANK}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        "$SCRIPT_DIR/kareus_gpt_pretraining.py" \
        --config-name="${CFG}" \
        model.tensor_model_parallel_size="${TP}" \
        model.context_parallel_size="${CP}" \
        model.micro_batch_size="${MBS}" \
        model.global_batch_size="${GBS}" \
        model.encoder_seq_length="${SEQ}" \
        model.enable_megatron_timers=False \
        model.enable_zeus_monitor=True \
        model.enable_power_monitor=False \
        model.enable_perseus_optimizer=False \
        model.enable_kareus_scheduler=False \
        2>&1 | tee "${LOG}"

    if [[ "${NODE_RANK}" == "0" ]]; then
      chmod a+w "${OUTPUT_DIR}"

      if compgen -G "${NEMO_DIR}/20*" > /dev/null; then
        shopt -s nullglob dotglob
        for d in "${NEMO_DIR}"/20*; do
          if [[ -d "$d" ]]; then
            contents=("$d"/*)
            if (( ${#contents[@]} )); then
              mv "${contents[@]}" "${OUTPUT_DIR}/"
            fi
            rm -rf "$d"
          fi
        done
        shopt -u nullglob dotglob
      fi

      if compgen -G "${NEMO_DIR}/*.txt" > /dev/null; then
        mv "${NEMO_DIR}"/*.txt "${OUTPUT_DIR}/"
      fi

      echo "    Node 0 done – outputs: ${OUTPUT_DIR}"
    else
      if compgen -G "${NEMO_DIR}/*.txt" > /dev/null; then
        mv "${NEMO_DIR}"/*.txt "${OUTPUT_DIR}/"
      fi

      remote_dir="${REMOTE_BASE_DIR}/nemo_experiments/${nemo_model_name}/${config_tag}/nanobatch"
      sleep 5
      scp "${SSH_KEY_OPTS[@]}" -r "${OUTPUT_DIR}/"* "${REMOTE_USER}@${MASTER_ADDR}":"${remote_dir}/"
      echo "    Node 1 done – synced to: ${remote_dir}"
    fi

    echo "    log: ${LOG}"
    echo ""
done

echo "All ${#CONFIGS[@]} nanobatch configurations completed (node_rank=${NODE_RANK})."
