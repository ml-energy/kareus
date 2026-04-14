#!/usr/bin/env bash
set -euo pipefail

########################################
# Frequency-sweep profiling for one    #
# config on 1 node × 4 GPUs.          #
########################################
#
# Usage (called by run_nanobatch_perseus.sh):
#   bash nanobatch_perseus_run_profiling.sh <config_name> <TP> <MBS>
#
# Environment variables (optional):
#   MASTER_PORT   (default 6000)
#   FREQ_START    (default 1740)
#   FREQ_END      (default 900)
#   FREQ_STEP     (default 60)
#
# This script:
#   - Sweeps GPU frequency from FREQ_START down to FREQ_END (step -FREQ_STEP)
#   - Runs torchrun at each frequency with enable_megatron_timers=True
#   - Organises timer/energy outputs into profiling/node0/{freq}/timers/
#   - Touches .profiling_complete marker when done

CFG="${1:?Usage: $0 <config_name> <TP> <MBS>}"
TP="${2:?}"
MBS="${3:?}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MASTER_PORT="${MASTER_PORT:-6000}"

NUM_MICROBATCHES=4
GBS=$(( MBS * NUM_MICROBATCHES ))

nemo_model_name="${CFG%_config}"
NEMO_DIR="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"
PROFILE_DIR="${NEMO_DIR}/tp${TP}_mbs${MBS}_seq${SEQ}/profiling"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${PROFILE_DIR}"

FREQ_START="${FREQ_START:-1740}"
FREQ_END="${FREQ_END:-900}"
FREQ_STEP="${FREQ_STEP:-60}"

echo "===== Profiling: ${CFG} tp${TP}_mbs${MBS} (4 GPUs, 1 node) ====="
echo "Frequency range: ${FREQ_START} → ${FREQ_END} MHz (step ${FREQ_STEP})"

########################################
# Frequency sweep                      #
########################################

for frequency in $(seq ${FREQ_START} -${FREQ_STEP} ${FREQ_END}); do
    echo "  Setting GPU frequency to ${frequency} MHz"
    nvidia-smi -i 0,1,2,3 --lock-gpu-clocks="${frequency},${frequency}"

    PROF_LOG="${LOG_DIR}/${nemo_model_name}_tp${TP}_mbs${MBS}_prof_${frequency}.log"

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
        trainer.max_steps=30 \
        model.enable_megatron_timers=True \
        model.enable_zeus_monitor=False \
        2>&1 | tee "${PROF_LOG}"

    ########################################
    # Collect outputs per frequency        #
    ########################################

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

    sleep 5
done

########################################
# Reset clocks                         #
########################################

echo "Resetting GPU clocks"
nvidia-smi -i 0,1,2,3 --reset-gpu-clocks

echo "Profiling complete. Results under ${PROFILE_DIR}/node0/"

########################################
# Mark profiling done                  #
########################################

touch "${PROFILE_DIR}/.profiling_complete"
echo "Profiling marker created: ${PROFILE_DIR}/.profiling_complete"
