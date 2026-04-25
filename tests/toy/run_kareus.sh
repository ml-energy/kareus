#!/usr/bin/env bash
# Toy Kareus test: BO profiling → CSV gen → optimization → PFO + Kareus training
# for 4 GPUs (1 node), PP=2, TP=2.
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
#   1. BO profiling – kareus_run_bayesian.sh (skip if results ready)
#   2. CSV gen      – generate_profile_csv.py  (from BO logs + prepost)
#   3. Optimise     – run_optimization.py      (Phillips-Dessouky)
#   4. Find largest solution + start PFO server (uvicorn)
#   5. Training     – torchrun with Kareus model + enable_kareus_scheduler=True
#   6. Cleanup      – collect outputs, stop PFO server
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
PREPOST_DIR="${SCRIPT_DIR}/../bayesian"

NEMO_DIR="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"
config_tag="cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}"
OUTPUT_DIR="${NEMO_DIR}/${config_tag}/kareus"
RESULTS_DIR="${OUTPUT_DIR}/lowtime"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${RESULTS_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"

echo "===== Toy Kareus 4-GPU Test (PP=${PP}, TP=${TP}, CP=${CP}, #microbatches=${NUM_MICROBATCHES}) ====="
echo "Config: ${CFG}  Model: ${MODEL_NAME}"
echo ""

########################################
# Phase 1: Bayesian profiling          #
########################################
#
# Skip if every expected BO artefact is already on disk:
#   - eval_results.jsonl for each of the 4 partitions (cp=1 toy)
#   - the nonpartition prepost CSVs (preprocess/postprocess/loss + backwards)
#     for at least one frequency.

BO_LOGS_DIR="${BAYESIAN_DIR}/logs/${MODEL_NAME}/cp${CP}-tp${TP}-bs${MBS}-seq${SEQ}"
BO_PARTITIONS=(fwd_attn fwd_mlp bwd_attn bwd_mlp)
BO_PREPOST_FILES=(
    preprocess_energy.csv
    postprocess_energy.csv
    loss_energy.csv
    preprocess_backward_energy.csv
    postprocess_backward_energy.csv
)

bo_ready=1
for part in "${BO_PARTITIONS[@]}"; do
    if [[ ! -f "${BO_LOGS_DIR}/${part}/eval_results.jsonl" ]]; then
        echo "BO missing: ${BO_LOGS_DIR}/${part}/eval_results.jsonl"
        bo_ready=0
        break
    fi
done

if (( bo_ready )); then
    np_dir="${BO_LOGS_DIR}/nonpartition"
    if [[ ! -d "${np_dir}" ]]; then
        echo "BO missing: ${np_dir}"
        bo_ready=0
    else
        np_freq_dir="$(ls -d "${np_dir}"/*/ 2>/dev/null | head -n 1 || true)"
        if [[ -z "${np_freq_dir}" ]]; then
            echo "BO missing: no frequency subdirs under ${np_dir}"
            bo_ready=0
        else
            for f in "${BO_PREPOST_FILES[@]}"; do
                if [[ ! -f "${np_freq_dir%/}/${f}" ]]; then
                    echo "BO missing: ${np_freq_dir%/}/${f}"
                    bo_ready=0
                    break
                fi
            done
        fi
    fi
fi

if (( bo_ready )); then
    echo "BO results already exist under ${BO_LOGS_DIR} — skipping Phase 1"
else
    echo "Running Bayesian profiling (kareus_run_bayesian.sh)..."
    bash "${SCRIPT_DIR}/kareus_run_bayesian.sh"
fi

########################################
# Phase 2: CSV generation              #
########################################

PROFILE_CSV="${OUTPUT_DIR}/profile.csv"

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
        --seq_len="${SEQ}" \
        --output_dir="${OUTPUT_DIR}"
else
    echo "Profile CSV exists: ${PROFILE_CSV}"
fi

########################################
# Phase 3: Optimization                #
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
# Phase 4: Find largest solution       #
#          Start PFO server            #
########################################

freqs_path="$(ls "${RESULTS_DIR}"/freqs_pipeline_*.py 2>/dev/null | sort -V | tail -n 1 || true)"
scheds_path="$(ls "${RESULTS_DIR}"/scheds_pipeline_*.py 2>/dev/null | sort -V | tail -n 1 || true)"

if [[ -z "${freqs_path}" || -z "${scheds_path}" ]]; then
    echo "ERROR: No freqs/scheds_pipeline_*.py found in ${RESULTS_DIR}" >&2
    exit 1
fi

echo "Using freqs solution: ${freqs_path}"
echo "Using scheds solution: ${scheds_path}"

cleanup() {
    echo "Resetting GPU clocks"
    nvidia-smi -i 0,1,2,3 --reset-gpu-clocks || true
    if [[ -n "${PFO_PID:-}" ]] && ps -p "${PFO_PID}" > /dev/null 2>&1; then
        echo "Stopping PFO server PID ${PFO_PID}"
        kill "${PFO_PID}" || true
        wait "${PFO_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

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
    2>&1 | tee "${TRAIN_LOG}" || true
TRAIN_EXIT=${PIPESTATUS[0]}

if [[ "${TRAIN_EXIT}" -ne 0 ]]; then
    echo "ERROR: Training failed with exit code ${TRAIN_EXIT}"
    exit "${TRAIN_EXIT}"
fi

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

echo "Done — outputs: ${OUTPUT_DIR}"
echo "Log: ${TRAIN_LOG}"
