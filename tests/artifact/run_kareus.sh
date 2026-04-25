#!/usr/bin/env bash
# Artifact Kareus on 16 GPUs (2 nodes x 8 GPUs A100), PP=2, microbatches=8.
# Per-config flow: BO profiling -> CSV -> optimization -> PFO + Kareus training.
#
# Prerequisites (run separately, unless SKIP_PROFILING=true):
#   1. BO partition profiling under tests/bayesian/
#   2. Nonpartition prepost profiling under tests/bayesian/nonpartition/
#
# Usage:
#   MASTER_ADDR=<node0_ip> bash run_kareus.sh <node_rank>
#
# Env vars:
#   MASTER_ADDR     (required) IP/hostname of node 0
#   MASTER_PORT     (default 6000)
#   PFO_PORT        (default 7787)
#   CONFIG_MODE     full | single   (default full)
#   SKIP_PROFILING  true | false    (default false)
#                   When true, skip CSV/optimization phases and load
#                   precomputed solutions from
#                   ../kareus/schedules/<model_name>/<config_tag>/{freqs,scheds}_pipeline_*.py
#   GPU_TYPE        (default A100, used for p2p_power lookup in generate_profile_csv.py)
#   REMOTE_USER, REMOTE_BASE_DIR, SSH_KEY_PATH (multi-node scp)
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
PFO_PORT="${PFO_PORT:-7787}"
CONFIG_MODE="${CONFIG_MODE:-full}"
SKIP_PROFILING="${SKIP_PROFILING:-false}"
GPU_TYPE="${GPU_TYPE:-A100}"

REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-$HOME/workspace/Kareus/tests/artifact}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/ruofanw.pem}"

export MASTER_ADDR MASTER_PORT REMOTE_USER REMOTE_BASE_DIR SSH_KEY_PATH

KAREUS_DIR="${SCRIPT_DIR}/../kareus"
BAYESIAN_DIR="${SCRIPT_DIR}/../bayesian"
PREPOST_DIR="${SCRIPT_DIR}/../bayesian"
SCHEDULES_DIR="${KAREUS_DIR}/schedules"

PP=2
NUM_MICROBATCHES=8

# config_name  CP  TP  MBS  SEQ  MODEL_NAME
CONFIGS_FULL=(
    "megatron_llama3.2_3b_config  1  8  8   4096  llama3.2_3b"
    "megatron_llama3.2_3b_config  2  4  8   4096  llama3.2_3b"
    "megatron_llama3.2_3b_config  2  4  8   8192  llama3.2_3b"
    "megatron_llama3.2_3b_config  2  4  16  4096  llama3.2_3b"
    "megatron_qwen3_1.7b_config   1  8  8   4096  qwen3_1.7b"
    "megatron_qwen3_1.7b_config   1  8  8   8192  qwen3_1.7b"
    "megatron_qwen3_1.7b_config   1  8  16  4096  qwen3_1.7b"
    "megatron_qwen3_1.7b_config   2  4  8   4096  qwen3_1.7b"
    "megatron_qwen3_1.7b_config   2  4  8   8192  qwen3_1.7b"
    "megatron_qwen3_1.7b_config   2  4  16  4096  qwen3_1.7b"
)

case "${CONFIG_MODE}" in
    full)   CONFIGS=("${CONFIGS_FULL[@]}") ;;
    single) CONFIGS=("${CONFIGS_FULL[0]}") ;;
    *) echo "ERROR: CONFIG_MODE must be 'full' or 'single'" >&2; exit 1 ;;
esac

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

PFO_PID=""
cleanup() {
    if [[ -n "${PFO_PID:-}" ]] && ps -p "${PFO_PID}" > /dev/null 2>&1; then
        echo "Stopping PFO server PID ${PFO_PID}"
        kill "${PFO_PID}" || true
        wait "${PFO_PID}" 2>/dev/null || true
    fi
    nvidia-smi -i 0,1,2,3,4,5,6,7 --reset-gpu-clocks || true
}
trap cleanup EXIT

echo "===== Artifact Kareus 16-GPU Tests (CONFIG_MODE=${CONFIG_MODE} SKIP_PROFILING=${SKIP_PROFILING}) ====="
echo "Total configurations: ${#CONFIGS[@]}"
echo ""

for i in "${!CONFIGS[@]}"; do
    read -r CFG CP TP MBS SEQ MODEL_NAME <<< "${CONFIGS[$i]}"

    GBS=$(( MBS * NUM_MICROBATCHES ))

    nemo_model_name="${CFG%_config}"
    config_tag="cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}"

    NEMO_DIR="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"
    OUTPUT_DIR="${NEMO_DIR}/${config_tag}/kareus"
    RESULTS_DIR="${NEMO_DIR}/${config_tag}/lowtime"

    mkdir -p "${RESULTS_DIR}" "${OUTPUT_DIR}"

    echo ">>> Config $((i+1))/${#CONFIGS[@]}: ${CFG} cp${CP}_tp${TP} MBS=${MBS} SEQ=${SEQ} GBS=${GBS}"

    if [[ "${SKIP_PROFILING}" == "true" ]]; then
        SCHED_DIR="${SCHEDULES_DIR}/${nemo_model_name}/${config_tag}"
        freqs_path="$(ls "${SCHED_DIR}"/freqs_pipeline_*.py 2>/dev/null | sort -V | tail -n 1 || true)"
        scheds_path="$(ls "${SCHED_DIR}"/scheds_pipeline_*.py 2>/dev/null | sort -V | tail -n 1 || true)"
        if [[ -z "${freqs_path}" || -z "${scheds_path}" ]]; then
            echo "    ERROR: SKIP_PROFILING=true but freqs_pipeline_*.py and/or scheds_pipeline_*.py not found in ${SCHED_DIR}" >&2
            exit 1
        fi
        echo "    Using precomputed freqs:  ${freqs_path}"
        echo "    Using precomputed scheds: ${scheds_path}"
    else
        ########################################
        # Phase 1: CSV gen (node 0)            #
        ########################################
        PROFILE_CSV="profile_${MODEL_NAME}_cp${CP}_tp${TP}_bs${MBS}_seq${SEQ}.csv"

        if [[ "${NODE_RANK}" == "0" ]]; then
            if [[ ! -f "${PROFILE_CSV}" ]]; then
                echo "    Generating profile CSV..."
                python "${KAREUS_DIR}/generate_profile_csv.py" \
                    --bayesian_profile_dir="${BAYESIAN_DIR}" \
                    --prepost_profile_dir="${PREPOST_DIR}" \
                    --model_name="${MODEL_NAME}" \
                    --context_parallel_size="${CP}" \
                    --tensor_parallel_size="${TP}" \
                    --pipeline_parallel_size="${PP}" \
                    --batch_size="${MBS}" \
                    --seq_len="${SEQ}" \
                    --gpu_type="${GPU_TYPE}"
            else
                echo "    Profile CSV exists: ${PROFILE_CSV}"
            fi

            ########################################
            # Phase 2: Optimization (node 0)       #
            ########################################
            if ! compgen -G "${RESULTS_DIR}/freqs_pipeline_*.py" > /dev/null 2>&1; then
                echo "    Running optimisation..."
                python "${KAREUS_DIR}/run_optimization.py" \
                    --inst_profile="${PROFILE_CSV}" \
                    --output_dir="${RESULTS_DIR}" \
                    --num_mbs="${NUM_MICROBATCHES}" \
                    --num_stages="${PP}"
            else
                echo "    Optimisation results exist in: ${RESULTS_DIR}"
            fi
        fi

        freqs_path=""
        scheds_path=""
        if [[ "${NODE_RANK}" == "0" ]]; then
            freqs_path="$(ls "${RESULTS_DIR}"/freqs_pipeline_*.py 2>/dev/null | sort -V | tail -n 1 || true)"
            scheds_path="$(ls "${RESULTS_DIR}"/scheds_pipeline_*.py 2>/dev/null | sort -V | tail -n 1 || true)"
            if [[ -z "${freqs_path}" || -z "${scheds_path}" ]]; then
                echo "    ERROR: No freqs/scheds_pipeline_*.py found in ${RESULTS_DIR}" >&2
                continue
            fi
            echo "    Using freqs solution:  ${freqs_path}"
            echo "    Using scheds solution: ${scheds_path}"
        fi
    fi

    ########################################
    # PFO server (node 0)                  #
    ########################################

    PFO_PID=""
    if [[ "${NODE_RANK}" == "0" ]]; then
        server_log="${OUTPUT_DIR}/pfo_server.log"
        echo "    Starting PFO server on ${MASTER_ADDR}:${PFO_PORT}"
        ZEUS_PFO_SCHEDULER=PointSolution3D \
        ZEUS_PFO_SCHEDULER_ARGS="{\"solution_path\": \"${freqs_path}\"}" \
        uvicorn zeus.optimizer.pipeline_frequency.server.router:app \
            --host "${MASTER_ADDR}" \
            --port "${PFO_PORT}" \
            > "${server_log}" 2>&1 &
        PFO_PID=$!
        echo "    PFO server PID: ${PFO_PID}"
        sleep 5
    fi

    ########################################
    # Training                             #
    ########################################

    TRAIN_LOG="${LOG_DIR}/${nemo_model_name}_${config_tag}_kareus.log"
    echo "    Running Kareus training (node_rank=${NODE_RANK})..."

    # Only rank 0 reads the schedule file; other ranks receive it via
    # torch.distributed broadcast during scheduler init.
    KAREUS_OVERRIDES=()
    if [[ "${NODE_RANK}" == "0" ]]; then
        KAREUS_OVERRIDES+=("model.kareus_scheduler_kwargs.solution_path=${scheds_path}")
    fi

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
        model.enable_perseus_optimizer=True \
        model.enable_kareus_scheduler=True \
        "${KAREUS_OVERRIDES[@]}" \
        2>&1 | tee "${TRAIN_LOG}"

    echo "    Resetting GPU clocks"
    nvidia-smi -i 0,1,2,3,4,5,6,7 --reset-gpu-clocks

    ########################################
    # Collection + cleanup                 #
    ########################################

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

        if [[ -n "${PFO_PID}" ]] && ps -p "${PFO_PID}" > /dev/null 2>&1; then
            echo "    Stopping PFO server PID ${PFO_PID}"
            kill "${PFO_PID}" || true
            wait "${PFO_PID}" 2>/dev/null || true
        fi

        echo "    Node 0 done – outputs: ${OUTPUT_DIR}"
    else
        if compgen -G "${NEMO_DIR}/*.txt" > /dev/null; then
            mv "${NEMO_DIR}"/*.txt "${OUTPUT_DIR}/"
        fi

        remote_dir="${REMOTE_BASE_DIR}/nemo_experiments/${nemo_model_name}/${config_tag}/kareus"
        echo "    Syncing results from node 1 to ${REMOTE_USER}@${MASTER_ADDR}:${remote_dir}"
        ssh -i "${SSH_KEY_PATH}" "${REMOTE_USER}@${MASTER_ADDR}" "mkdir -p '${remote_dir}'"
        sleep 5
        scp -i "${SSH_KEY_PATH}" -r "${OUTPUT_DIR}/"* \
            "${REMOTE_USER}@${MASTER_ADDR}":"${remote_dir}/"
        echo "    Node 1 done – synced to: ${remote_dir}"
    fi

    echo "    log: ${TRAIN_LOG}"
    echo ""
    sleep 5
done

echo "All ${#CONFIGS[@]} Kareus configurations completed (node_rank=${NODE_RANK})."
