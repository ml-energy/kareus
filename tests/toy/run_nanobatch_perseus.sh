#!/usr/bin/env bash
# Toy Nanobatch+Perseus test: profiling → CSV → optimization → PFO + training
# for 4 GPUs (1 node), PP=2, TP=2.
#
# Uses Kareus MegatronGPTModel (nanobatching) with Perseus frequency optimization.
#
# Usage:
#   bash run_nanobatch_perseus.sh [config_name]
#
#   config_name   (optional, default: megatron_llama3.2_3b_config)
#
# Environment variables (all optional):
#   MASTER_PORT   (default 6000)
#   PFO_PORT      (default 7787)
#   FREQ_START    (default 1740)
#   FREQ_END      (default 900)
#   FREQ_STEP     (default 60)
#
# Per-config flow:
#   1. Profiling   – frequency sweep (skip if .profiling_complete exists)
#   2. CSV gen     – perseus_generate_profile_csv.py
#   3. Optimise    – perseus_run_optimization.py
#   4. PFO server  – uvicorn with largest freqs_pipeline_*.py
#   5. Training    – torchrun with Kareus model + enable_perseus_optimizer=True
#   6. Cleanup     – collect outputs, stop PFO server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

########################################
# Configuration                        #
########################################

CFG="${1:-megatron_llama3.2_3b_config}"
MASTER_PORT="${MASTER_PORT:-6000}"
PFO_PORT="${PFO_PORT:-7787}"

PP=2
TP=2
CP=1
NUM_MICROBATCHES=4
MBS=4
SEQ=2048
GBS=$(( MBS * NUM_MICROBATCHES ))

nemo_model_name="${CFG%_config}"

NANOBATCH_PERSEUS_DIR="${SCRIPT_DIR}/../nanobatch_perseus"

NEMO_DIR="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"
config_tag="cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}"
PROFILE_DIR="${NEMO_DIR}/${config_tag}/nanobatch_perseus/profiling"
RESULTS_DIR="${NEMO_DIR}/${config_tag}/nanobatch_perseus/lowtime"
OUTPUT_DIR="${NEMO_DIR}/${config_tag}/nanobatch_perseus"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${PROFILE_DIR}" "${RESULTS_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"

echo "===== Toy Nanobatch+Perseus 4-GPU Test (PP=${PP}, TP=${TP}, #microbatches=${NUM_MICROBATCHES}) ====="
echo "Config: ${CFG}"
echo ""

########################################
# Phase 1: Profiling (skip if done)    #
########################################

PROFILING_MARKER="${PROFILE_DIR}/.profiling_complete"

if [[ -f "${PROFILING_MARKER}" ]]; then
    echo "Profiling already done (${PROFILING_MARKER} exists) — skipping"
else
    echo "Starting frequency profiling..."
    bash "${SCRIPT_DIR}/nanobatch_perseus_run_profiling.sh" "${CFG}" "${CP}" "${TP}" "${MBS}" "${SEQ}"
fi

########################################
# Phase 2+3: CSV gen + Optimise       #
########################################

PROFILE_CSV="${PROFILE_DIR}/profile.csv"

if [[ ! -f "${PROFILE_CSV}" ]]; then
    echo "Generating profile CSV..."
    python "${NANOBATCH_PERSEUS_DIR}/generate_profile_csv.py" \
        --profile_dir="${PROFILE_DIR}" \
        --num_ranks_per_stage="${TP}" \
        --num_microbatches="${NUM_MICROBATCHES}" \
        --num_prof_iters=10 \
        --warmup_iters=5
else
    echo "Profile CSV exists: ${PROFILE_CSV}"
fi

if ! compgen -G "${RESULTS_DIR}/freqs_pipeline_*.py" > /dev/null 2>&1; then
    echo "Running optimisation..."
    python "${NANOBATCH_PERSEUS_DIR}/run_optimization.py" \
        --inst_profile="${PROFILE_CSV}" \
        --output_dir="${RESULTS_DIR}" \
        --num_mbs="${NUM_MICROBATCHES}" \
        --num_stages="${PP}" \
        --p2p_power=70.0
else
    echo "Optimisation results exist in: ${RESULTS_DIR}"
fi

########################################
# Phase 4: Find largest solution       #
# Phase 5: Start PFO server            #
########################################

freqs_path="$(ls "${RESULTS_DIR}"/freqs_pipeline_*.py 2>/dev/null | sort -V | tail -n 1 || true)"

if [[ -z "${freqs_path}" ]]; then
    echo "ERROR: No freqs_pipeline_*.py found in ${RESULTS_DIR}" >&2
    exit 1
fi

echo "Using solution: ${freqs_path}"

server_log="${OUTPUT_DIR}/pfo_server.log"
echo "Starting PFO server on localhost:${PFO_PORT}"
ZEUS_PFO_SCHEDULER=PointSolution3D \
ZEUS_PFO_SCHEDULER_ARGS="{\"solution_path\": \"${freqs_path}\"}" \
uvicorn zeus.optimizer.pipeline_frequency.server.router:app \
    --host localhost \
    --port "${PFO_PORT}" \
    > "${server_log}" 2>&1 &
PFO_PID=$!
echo "PFO server PID: ${PFO_PID}"

########################################
# Phase 6: Nanobatch+Perseus training #
########################################

TRAIN_LOG="${LOG_DIR}/${nemo_model_name}_${config_tag}_nanobatch_perseus.log"

echo "Running Nanobatch+Perseus training..."

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
    model.enable_megatron_timers=False \
    model.enable_zeus_monitor=True \
    model.enable_power_monitor=False \
    model.enable_perseus_optimizer=True \
    model.enable_kareus_scheduler=False \
    2>&1 | tee "${TRAIN_LOG}"

########################################
# Phase 7: Collect outputs + cleanup   #
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

if ps -p "${PFO_PID}" > /dev/null 2>&1; then
    echo "Stopping PFO server PID ${PFO_PID}"
    kill "${PFO_PID}" || true
    wait "${PFO_PID}" 2>/dev/null || true
fi

echo "Done — outputs: ${OUTPUT_DIR}"
echo "Log: ${TRAIN_LOG}"
