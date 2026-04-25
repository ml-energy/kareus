#!/usr/bin/env bash
# Artifact Nanobatch+Perseus on 16 GPUs (2 nodes x 8 GPUs A100), PP=2, microbatches=8.
# Uses Kareus MegatronGPTModel (nanobatching) with Perseus frequency optimization.
#
# Usage:
#   MASTER_ADDR=<node0_ip> bash run_nanobatch_perseus.sh <node_rank>
#
# Env vars:
#   MASTER_ADDR     (required) IP/hostname of node 0
#   MASTER_PORT     (default 6000)
#   PFO_PORT        (default 7787)
#   CONFIG_MODE     full | single   (default full)
#   SKIP_PROFILING  true | false    (default false)
#                   When true, skip profiling/CSV/optimization and load
#                   precomputed solutions from
#                   ../perseus/schedules/<model_name>/<config_tag>/freqs_pipeline_*.py
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

REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-$HOME/workspace/Kareus/tests/artifact}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/ruofanw.pem}"

export MASTER_ADDR MASTER_PORT REMOTE_USER REMOTE_BASE_DIR SSH_KEY_PATH

NANOBATCH_PERSEUS_DIR="${SCRIPT_DIR}/../nanobatching_perseus"
SCHEDULES_DIR="${SCRIPT_DIR}/../perseus/schedules"

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

echo "===== Artifact Nanobatch+Perseus 16-GPU Tests (CONFIG_MODE=${CONFIG_MODE} SKIP_PROFILING=${SKIP_PROFILING}) ====="
echo "Total configurations: ${#CONFIGS[@]}"
echo ""

for i in "${!CONFIGS[@]}"; do
    read -r CFG CP TP MBS SEQ <<< "${CONFIGS[$i]}"

    GBS=$(( MBS * NUM_MICROBATCHES ))

    nemo_model_name="${CFG%_config}"
    config_tag="cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}"

    NEMO_DIR="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"
    PROFILE_DIR="${NEMO_DIR}/${config_tag}/nanobatch_perseus/profiling"
    RESULTS_DIR="${NEMO_DIR}/${config_tag}/nanobatch_perseus/lowtime"
    OUTPUT_DIR="${NEMO_DIR}/${config_tag}/nanobatch_perseus"

    mkdir -p "${PROFILE_DIR}" "${RESULTS_DIR}" "${OUTPUT_DIR}"

    echo ">>> Config $((i+1))/${#CONFIGS[@]}: ${CFG} cp${CP}_tp${TP} MBS=${MBS} SEQ=${SEQ} GBS=${GBS}"

    if [[ "${SKIP_PROFILING}" == "true" ]]; then
        SCHED_DIR="${SCHEDULES_DIR}/${nemo_model_name}/${config_tag}"
        freqs_path="$(ls "${SCHED_DIR}"/freqs_pipeline_*.py 2>/dev/null | sort -V | tail -n 1 || true)"
        if [[ -z "${freqs_path}" ]]; then
            echo "    ERROR: SKIP_PROFILING=true but no freqs_pipeline_*.py found in ${SCHED_DIR}" >&2
            exit 1
        fi
        echo "    Using precomputed solution: ${freqs_path}"
    else
        PROFILING_MARKER="${PROFILE_DIR}/.profiling_complete"
        if [[ -f "${PROFILING_MARKER}" ]]; then
            echo "    Profiling already done (${PROFILING_MARKER} exists) — skipping"
        else
            echo "    Starting frequency profiling..."
            bash "${SCRIPT_DIR}/nanobatch_perseus_run_profiling.sh" "${NODE_RANK}" "${CFG}" "${CP}" "${TP}" "${MBS}" "${SEQ}"
        fi

        PROFILE_CSV="${PROFILE_DIR}/profile.csv"
        if [[ "${NODE_RANK}" == "0" ]]; then
            if [[ ! -f "${PROFILE_CSV}" ]]; then
                echo "    Generating profile CSV..."
                NUM_RANKS_PER_STAGE=$(( CP * TP ))
                python "${NANOBATCH_PERSEUS_DIR}/generate_profile_csv.py" \
                    --profile_dir="${PROFILE_DIR}" \
                    --num_ranks_per_stage="${NUM_RANKS_PER_STAGE}" \
                    --num_microbatches="${NUM_MICROBATCHES}"
            else
                echo "    Profile CSV exists: ${PROFILE_CSV}"
            fi

            if ! compgen -G "${RESULTS_DIR}/freqs_pipeline_*.py" > /dev/null 2>&1; then
                echo "    Running optimisation..."
                python "${NANOBATCH_PERSEUS_DIR}/run_optimization.py" \
                    --inst_profile="${PROFILE_CSV}" \
                    --output_dir="${RESULTS_DIR}" \
                    --num_mbs="${NUM_MICROBATCHES}" \
                    --num_stages="${PP}" \
                    --p2p_power=85.0
            else
                echo "    Optimisation results exist in: ${RESULTS_DIR}"
            fi
        fi

        freqs_path=""
        if [[ "${NODE_RANK}" == "0" ]]; then
            freqs_path="$(ls "${RESULTS_DIR}"/freqs_pipeline_*.py 2>/dev/null | sort -V | tail -n 1 || true)"
            if [[ -z "${freqs_path}" ]]; then
                echo "    ERROR: No freqs_pipeline_*.py found in ${RESULTS_DIR}" >&2
                continue
            fi
            echo "    Using solution: ${freqs_path}"
        fi
    fi

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

    TRAIN_LOG="${LOG_DIR}/${nemo_model_name}_${config_tag}_nanobatch_perseus.log"
    echo "    Running Nanobatch+Perseus training (node_rank=${NODE_RANK})..."

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
        model.enable_kareus_scheduler=False \
        2>&1 | tee "${TRAIN_LOG}"

    echo "    Resetting GPU clocks"
    nvidia-smi -i 0,1,2,3,4,5,6,7 --reset-gpu-clocks

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

        remote_dir="${REMOTE_BASE_DIR}/nemo_experiments/${nemo_model_name}/${config_tag}/nanobatch_perseus"
        sleep 5
        scp -i "${SSH_KEY_PATH}" -r "${OUTPUT_DIR}/"* \
            "${REMOTE_USER}@${MASTER_ADDR}":"${remote_dir}/"
        echo "    Node 1 done – synced to: ${remote_dir}"
    fi

    echo "    log: ${TRAIN_LOG}"
    echo ""
    sleep 5
done

echo "All ${#CONFIGS[@]} Nanobatch+Perseus configurations completed (node_rank=${NODE_RANK})."
