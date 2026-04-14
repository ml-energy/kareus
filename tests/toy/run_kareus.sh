#!/usr/bin/env bash
# Toy Kareus test: CSV gen → optimization → PFO + Kareus training
# for 4 GPUs (1 node), PP=2, TP=2.
#
# Prerequisites (run separately):
#   1. BO partition profiling + nonpartition prepost:
#        bash tests/toy/kareus_run_bayesian.sh
#
# Usage:
#   bash run_kareus.sh [config_name]
#
#   config_name   (optional, default: megatron_llama3.2_3b_config)
#
# Environment variables (all optional):
#   MASTER_PORT   (default 6000)
#   PFO_PORT      (default 7787)
#
# Per-config flow:
#   1. CSV gen     – generate_profile_csv.py  (from BO logs + prepost)
#   2. Optimise    – run_optimization.py      (Phillips-Dessouky)
#   3. PFO server  – uvicorn with largest freqs_pipeline_*.py
#   4. Training    – torchrun with Kareus model + enable_kareus_scheduler=True
#   5. Cleanup     – collect outputs, stop PFO server
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

MODEL_NAME="${CFG#megatron_}"   # e.g. llama3.2_3b_config
MODEL_NAME="${MODEL_NAME%_config}"  # e.g. llama3.2_3b

nemo_model_name="${CFG%_config}"

KAREUS_DIR="${SCRIPT_DIR}/../kareus"
BAYESIAN_DIR="${SCRIPT_DIR}/../bayesian"
PREPOST_DIR="${SCRIPT_DIR}/../bayesian/nonpartition"

NEMO_DIR="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"
config_tag="cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}"
RESULTS_DIR="${NEMO_DIR}/${config_tag}/lowtime"
OUTPUT_DIR="${NEMO_DIR}/${config_tag}/kareus"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${RESULTS_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"

echo "===== Toy Kareus 4-GPU Test (PP=${PP}, TP=${TP}, CP=${CP}, #microbatches=${NUM_MICROBATCHES}) ====="
echo "Config: ${CFG}  Model: ${MODEL_NAME}"
echo ""

########################################
# Phase 1: CSV generation              #
########################################

PROFILE_CSV="profile_${MODEL_NAME}_cp${CP}_tp${TP}_bs${MBS}_seq${SEQ}.csv"

if [[ ! -f "${PROFILE_CSV}" ]]; then
    echo "Generating profile CSV..."
    python "${KAREUS_DIR}/generate_profile_csv.py" \
        --bayesian_profile_dir="${BAYESIAN_DIR}" \
        --prepost_profile_dir="${PREPOST_DIR}" \
        --model_name="${MODEL_NAME}" \
        --context_parallel_size="${CP}" \
        --tensor_parallel_size="${TP}" \
        --pipeline_parallel_size="${PP}" \
        --batch_size="${MBS}" \
        --seq_len="${SEQ}"
else
    echo "Profile CSV exists: ${PROFILE_CSV}"
fi

########################################
# Phase 2: Optimization                #
########################################

if ! compgen -G "${RESULTS_DIR}/freqs_pipeline_*.py" > /dev/null 2>&1; then
    echo "Running optimisation..."
    python "${KAREUS_DIR}/run_optimization.py" \
        --inst_profile="${PROFILE_CSV}" \
        --output_dir="${RESULTS_DIR}" \
        --num_mbs="${NUM_MICROBATCHES}" \
        --num_stages="${PP}"
else
    echo "Optimisation results exist in: ${RESULTS_DIR}"
fi

########################################
# Phase 3: Find largest solution       #
# Phase 4: Start PFO server            #
########################################

freqs_path="$(ls "${RESULTS_DIR}"/freqs_pipeline_*.py 2>/dev/null | sort -V | tail -n 1 || true)"
scheds_path="$(ls "${RESULTS_DIR}"/scheds_pipeline_*.py 2>/dev/null | sort -V | tail -n 1 || true)"

if [[ -z "${freqs_path}" || -z "${scheds_path}" ]]; then
    echo "ERROR: No freqs/scheds_pipeline_*.py found in ${RESULTS_DIR}" >&2
    exit 1
fi

echo "Using freqs solution: ${freqs_path}"
echo "Using scheds solution: ${scheds_path}"

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
sleep 5

########################################
# Phase 5: Kareus training             #
########################################

TRAIN_LOG="${LOG_DIR}/${nemo_model_name}_${config_tag}_kareus.log"

echo "Running Kareus training..."

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
    model.enable_kareus_scheduler=True \
    "model.kareus_scheduler_kwargs.solution_path=${scheds_path}" \
    2>&1 | tee "${TRAIN_LOG}"

echo "Resetting GPU clocks"
nvidia-smi -i 0,1,2,3 --reset-gpu-clocks

########################################
# Phase 6: Collect outputs + cleanup   #
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
