#!/usr/bin/env bash
set -euo pipefail

########################################
# Frequency-sweep profiling for one    #
# config on 2 nodes × 8 GPUs.         #
########################################
#
# Usage (called by run_all_configs.sh):
#   bash run_profiling.sh <node_rank> <config_name> <CP> <TP> <MBS> <SEQ>
#
# Required env vars: MASTER_ADDR, MASTER_PORT
# Optional env vars: REMOTE_USER, REMOTE_BASE_DIR, SSH_KEY_PATH
#
# This script:
#   - Sweeps GPU frequency from 1410 down to 900 MHz (step -30)
#   - Runs torchrun at each frequency with enable_megatron_timers=True
#   - Organises timer/energy outputs into profiling/{freq}/timers/
#   - Groups everything under node0/ or node1/
#   - Node 1 SCPs its results back to node 0
#   - Touches .profiling_complete marker when done

NODE_RANK="${1:?Usage: $0 <node_rank> <config_name> <CP> <TP> <MBS> <SEQ>}"
CFG="${2:?}"
CP="${3:?}"
TP="${4:?}"
MBS="${5:?}"
SEQ="${6:?}"

if [[ "${NODE_RANK}" != "0" && "${NODE_RANK}" != "1" ]]; then
  echo "ERROR: node_rank must be 0 or 1, got '${NODE_RANK}'" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${MASTER_ADDR:?MASTER_ADDR must be set}"
: "${MASTER_PORT:?MASTER_PORT must be set}"
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-~/workspace/Kareus/tests/nanobatching_perseus}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/ruofanw.pem}"

NUM_MICROBATCHES=8
GBS=$(( MBS * NUM_MICROBATCHES ))

nemo_model_name="${CFG%_config}"
config_tag="cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}"
NEMO_DIR="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"
PROFILE_DIR="${NEMO_DIR}/${config_tag}/profiling"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${PROFILE_DIR}"

FREQ_START=1410
FREQ_END=900
FREQ_STEP=30

echo "===== Profiling: ${CFG} ${config_tag} node_rank=${NODE_RANK} ====="

########################################
# Frequency sweep                      #
########################################

for frequency in $(seq ${FREQ_START} -${FREQ_STEP} ${FREQ_END}); do
    echo "  Setting GPU frequency to ${frequency} MHz"
    nvidia-smi -i 0,1,2,3,4,5,6,7 --lock-gpu-clocks="${frequency},${frequency}"

    PROF_LOG="${LOG_DIR}/${nemo_model_name}_${config_tag}_prof_${frequency}.log"

    torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank="${NODE_RANK}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        "$SCRIPT_DIR/megatron_gpt_pretraining.py" \
        --config-name="${CFG}" \
        model.tensor_model_parallel_size="${TP}" \
        model.context_parallel_size="${CP}" \
        model.micro_batch_size="${MBS}" \
        model.global_batch_size="${GBS}" \
        model.encoder_seq_length="${SEQ}" \
        model.enable_megatron_timers=True \
        model.enable_zeus_monitor=False \
        2>&1 | tee "${PROF_LOG}"

    ########################################
    # Collect outputs per frequency        #
    ########################################

    if [[ "${NODE_RANK}" == "0" ]]; then
        freq_dir="${PROFILE_DIR}/node0/${frequency}"
        mkdir -p "${freq_dir}/timers"
        chmod a+w "${freq_dir}"
        shopt -s nullglob dotglob
        for d in "${NEMO_DIR}"/20*; do
            if [[ -d "$d" ]]; then
                contents=("$d"/*)
                if (( ${#contents[@]} )); then
                    mv "${contents[@]}" "${freq_dir}/"
                fi
                rm -rf "$d"
            fi
        done
        shopt -u nullglob dotglob
        mv "${NEMO_DIR}/timers/"* "${freq_dir}/timers" 2>/dev/null || true
        mv "${NEMO_DIR}"/*.txt "${freq_dir}/" 2>/dev/null || true
    else
        freq_dir="${PROFILE_DIR}/node1/${frequency}"
        mkdir -p "${freq_dir}"
        mv "${NEMO_DIR}/timers" "${freq_dir}/" 2>/dev/null || true
        mv "${NEMO_DIR}"/*.txt "${freq_dir}/" 2>/dev/null || true
    fi

    sleep 5
done

########################################
# Reset clocks                         #
########################################

echo "Resetting GPU clocks"
nvidia-smi -i 0,1,2,3,4,5,6,7 --reset-gpu-clocks

target_dir="${PROFILE_DIR}/node${NODE_RANK}"

if [[ "${NODE_RANK}" == "0" ]]; then
    node1_dir="${PROFILE_DIR}/node1"
    mkdir -p "${node1_dir}"
    chmod a+rwx "${node1_dir}"
fi

echo "Profiling complete for node${NODE_RANK}. Results under ${target_dir}"

########################################
# Node 1: sync to node 0              #
########################################

if [[ "${NODE_RANK}" == "1" ]]; then
    remote_dir="${REMOTE_BASE_DIR}/nemo_experiments/${nemo_model_name}/${config_tag}/profiling/"
    echo "Syncing profiling results from node1 to ${REMOTE_USER}@${MASTER_ADDR}:${remote_dir}"
    scp -i "${SSH_KEY_PATH}" -r "${target_dir}/." \
        "${REMOTE_USER}@${MASTER_ADDR}":"${remote_dir%/}/node1/"
fi

########################################
# Mark profiling done                  #
########################################

sleep 5
touch "${PROFILE_DIR}/.profiling_complete"
echo "Profiling marker created: ${PROFILE_DIR}/.profiling_complete"
