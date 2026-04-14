#!/usr/bin/env bash
# Run 10 Megatron performance tests on 16 GPUs (2 nodes × 8 GPUs).
# PP=2, number of microbatches fixed at 8  →  GBS = MBS × 8.
#
# Usage:
#   MASTER_ADDR=<node0_ip> bash run_all_configs.sh <node_rank>
#
#   node_rank   0 or 1
#   MASTER_ADDR (required) IP/hostname of node 0
#   MASTER_PORT (optional, default 6000)
#   REMOTE_USER (optional, default ubuntu) – used by node 1 for scp
#   REMOTE_BASE_DIR (optional) – target dir on node 0 for synced results
#   SSH_KEY_PATH (optional, default ~/.ssh/ruofanw.pem) – key for scp
#
# Node 0: runs training, collects experiment outputs locally.
# Node 1: runs training, collects logs, then scp's results to node 0.
#
# Configurations:
#
#   Model           Parallelism  MBS  Seq   GBS
#   ─────────────── ─────────── ──── ───── ────
#   Llama 3.2 3B    TP8          8   4096   64
#   Llama 3.2 3B    CP2+TP4      8   4096   64
#   Llama 3.2 3B    CP2+TP4      8   8192   64
#   Llama 3.2 3B    CP2+TP4     16   4096  128
#   Qwen 3 1.7B     TP8          8   4096   64
#   Qwen 3 1.7B     TP8          8   8192   64
#   Qwen 3 1.7B     TP8         16   4096  128
#   Qwen 3 1.7B     CP2+TP4      8   4096   64
#   Qwen 3 1.7B     CP2+TP4      8   8192   64
#   Qwen 3 1.7B     CP2+TP4     16   4096  128
set -euo pipefail

########################################
# Argument: node_rank (0 or 1)         #
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

########################################
# Configuration                        #
########################################

if [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "ERROR: MASTER_ADDR must be set (IP or hostname of node 0)" >&2
  exit 1
fi
MASTER_PORT="${MASTER_PORT:-6000}"

REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-~/workspace/Kareus/tests/megatron}"

export MASTER_ADDR
export MASTER_PORT

PP=2
NUM_MICROBATCHES=8

# config_name  CP  TP  MBS  SEQ
CONFIGS=(
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

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "===== Megatron 16-GPU Performance Tests (PP=${PP}, #microbatches=${NUM_MICROBATCHES}) ====="
echo "Total configurations: ${#CONFIGS[@]}"
echo ""

for i in "${!CONFIGS[@]}"; do
    read -r CFG CP TP MBS SEQ <<< "${CONFIGS[$i]}"

    GBS=$(( MBS * NUM_MICROBATCHES ))

    PARA="cp${CP}_tp${TP}"
    TAG="${CFG}_${PARA}_mbs${MBS}_seq${SEQ}"
    LOG="${LOG_DIR}/${TAG}.log"

    nemo_model_name="${CFG%_config}"
    config_tag="cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}"
    output_dir="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}/${config_tag}/megatron"
    mkdir -p "${output_dir}"

    echo ">>> Test $((i+1))/${#CONFIGS[@]}: ${CFG} ${PARA} MBS=${MBS} SEQ=${SEQ} GBS=${GBS}"
    echo "    node_rank=${NODE_RANK}  output_dir=${output_dir}"

    torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank="${NODE_RANK}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        "$SCRIPT_DIR/megatron_gpt_pretraining.py" \
        --config-name="$CFG" \
        model.tensor_model_parallel_size="$TP" \
        model.context_parallel_size="$CP" \
        model.micro_batch_size="$MBS" \
        model.global_batch_size="$GBS" \
        model.encoder_seq_length="$SEQ" \
        2>&1 | tee "$LOG"

    ########################################
    # Per-node post-training collection    #
    ########################################

    if [[ "${NODE_RANK}" == "0" ]]; then
      echo "Moving NeMo experiment outputs into ${output_dir}"
      chmod a+w "${output_dir}"

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

      if compgen -G "${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}/*.txt" > /dev/null; then
        mv "${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"/*.txt "${output_dir}/"
      fi

      echo "    Node 0 done – outputs: ${output_dir}"

    else
      echo "Moving NeMo experiment text logs into ${output_dir}"

      if compgen -G "${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}/*.txt" > /dev/null; then
        mv "${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"/*.txt "${output_dir}/"
      fi

      remote_dir="${REMOTE_BASE_DIR}/nemo_experiments/${nemo_model_name}/${config_tag}/megatron"
      echo "Syncing results from node 1 to ${REMOTE_USER}@${MASTER_ADDR}:${remote_dir}"

      sleep 5
      scp -i "${SSH_KEY_PATH:-$HOME/.ssh/ruofanw.pem}" -r "${output_dir}/"* "${REMOTE_USER}@${MASTER_ADDR}":"${remote_dir}/"

      echo "    Node 1 done – synced to: ${remote_dir}"
    fi

    echo "    log: $LOG"
    echo ""
done

echo "All ${#CONFIGS[@]} configurations completed (node_rank=${NODE_RANK})."
