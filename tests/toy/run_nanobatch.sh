#!/usr/bin/env bash
# Toy Nanobatch performance test on 4 GPUs (1 node), PP=2, TP=2.
# Llama 3.2 3B, MBS=4, SEQ=2048, GBS=16.
#
# Uses Kareus MegatronGPTModel (nanobatching) WITHOUT any frequency
# optimization (no Perseus, no Kareus scheduler).  This is the
# baseline-nanobatch counterpart to run_megatron.sh.
#
# Usage:
#   bash run_nanobatch.sh [config_name]
#
#   config_name   (optional, default: megatron_llama3.2_3b_config)
#
# Environment variables (optional):
#   MASTER_PORT   (default 6000)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

########################################
# Configuration                        #
########################################

CFG="${1:-megatron_llama3.2_3b_config}"
MASTER_PORT="${MASTER_PORT:-6000}"

PP=2
TP=2
CP=1
MBS=4
NUM_MICROBATCHES=4
GBS=$(( MBS * NUM_MICROBATCHES ))
SEQ=2048

nemo_model_name="${CFG%_config}"
config_tag="cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}"

NEMO_DIR="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"
OUTPUT_DIR="${NEMO_DIR}/${config_tag}/nanobatch"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG="${LOG_DIR}/${nemo_model_name}_${config_tag}_nanobatch.log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "===== Toy Nanobatch 4-GPU Test (PP=${PP}, TP=${TP}, MBS=${MBS}, SEQ=${SEQ}, GBS=${GBS}) ====="
echo "Config: ${CFG}"

########################################
# Training                             #
########################################

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port="${MASTER_PORT}" \
    "$SCRIPT_DIR/kareus_gpt_pretraining.py" \
    --config-name="${CFG}" \
    model.tensor_model_parallel_size="${TP}" \
    model.micro_batch_size="${MBS}" \
    model.global_batch_size="${GBS}" \
    model.encoder_seq_length="${SEQ}" \
    model.enable_megatron_timers=False \
    model.enable_zeus_monitor=True \
    model.enable_power_monitor=False \
    model.enable_perseus_optimizer=False \
    model.enable_kareus_scheduler=False \
    2>&1 | tee "${LOG}"

########################################
# Post-training collection             #
########################################

echo "Moving NeMo experiment outputs into ${OUTPUT_DIR}"
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

echo "Done – outputs: ${OUTPUT_DIR}"
echo "Log: ${LOG}"
