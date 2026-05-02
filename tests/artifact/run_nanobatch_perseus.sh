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
#   FRONTIER        true | false    (default false)
#                   When true, run all 10 precomputed frontier schedules for
#                   each selected CONFIG_MODE config and write outputs under
#                   nemo_experiments/<model>/<config_tag>/nanobatch_perseus/frontier/.
#   FRONTIER_SYNC_RETRIES, FRONTIER_SYNC_SLEEP
#                   Node 1 retry settings when waiting for node 0 frontier files.
#   SKIP_PROFILING  true | false    (default false)
#                   When true, skip profiling/CSV/optimization and load
#                   precomputed solutions from
#                   ../nanobatch_perseus/schedules/<model_name>/<config_tag>/freqs_pipeline_*.py
#   GPU_TYPE        (default A100, used for P2P/static power lookup)
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
FRONTIER="${FRONTIER:-false}"
SKIP_PROFILING="${SKIP_PROFILING:-false}"
GPU_TYPE="${GPU_TYPE:-A100}"

REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-$HOME/workspace/Kareus/tests/artifact}"
SSH_KEY_PATH="${SSH_KEY_PATH:-}"
SSH_KEY_OPTS=()
if [[ -n "${SSH_KEY_PATH}" ]]; then
    SSH_KEY_OPTS=(-i "${SSH_KEY_PATH}")
fi

export MASTER_ADDR MASTER_PORT REMOTE_USER REMOTE_BASE_DIR SSH_KEY_PATH

NANOBATCH_PERSEUS_DIR="${SCRIPT_DIR}/../nanobatch_perseus"
SCHEDULES_DIR="${NANOBATCH_PERSEUS_DIR}/schedules"

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

case "${FRONTIER}" in
    true|false) ;;
    *) echo "ERROR: FRONTIER must be 'true' or 'false'" >&2; exit 1 ;;
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

sync_frontier_schedules_from_node0() {
    local remote_frontier_dir="$1"
    local local_frontier_dir="$2"
    local expected_freqs="$3"
    local attempts="${FRONTIER_SYNC_RETRIES:-120}"
    local sleep_secs="${FRONTIER_SYNC_SLEEP:-10}"
    local attempt
    local synced_freqs=()

    mkdir -p "${local_frontier_dir}"
    rm -f "${local_frontier_dir}"/*.py

    for ((attempt=1; attempt<=attempts; attempt++)); do
        if scp "${SSH_KEY_OPTS[@]}" "${REMOTE_USER}@${MASTER_ADDR}:${remote_frontier_dir}"/*.py "${local_frontier_dir}/"; then
            mapfile -t synced_freqs < <(ls "${local_frontier_dir}"/freqs_pipeline_*.py 2>/dev/null | sort -V || true)
            if (( ${#synced_freqs[@]} == expected_freqs )); then
                return 0
            fi
            echo "    Synced ${#synced_freqs[@]} frontier freqs; expected ${expected_freqs}. Retrying..."
            rm -f "${local_frontier_dir}"/*.py
        fi

        if (( attempt < attempts )); then
            echo "    Waiting for node 0 frontier schedules (${attempt}/${attempts})..."
            sleep "${sleep_secs}"
        fi
    done

    echo "    ERROR: Failed to sync ${expected_freqs} frontier schedules from ${REMOTE_USER}@${MASTER_ADDR}:${remote_frontier_dir}" >&2
    exit 1
}

echo "===== Artifact Nanobatch+Perseus 16-GPU Tests (GPU_TYPE=${GPU_TYPE} CONFIG_MODE=${CONFIG_MODE} FRONTIER=${FRONTIER} SKIP_PROFILING=${SKIP_PROFILING}) ====="
echo "Total configurations: ${#CONFIGS[@]}"
echo ""

for i in "${!CONFIGS[@]}"; do
    read -r CFG CP TP MBS SEQ <<< "${CONFIGS[$i]}"

    GBS=$(( MBS * NUM_MICROBATCHES ))

    nemo_model_name="${CFG%_config}"
    config_tag="cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}"

    NEMO_DIR="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"
    METHOD_DIR="${NEMO_DIR}/${config_tag}/nanobatch_perseus"
    PROFILE_DIR="${METHOD_DIR}/profiling"
    RESULTS_DIR="${METHOD_DIR}/lowtime"

    mkdir -p "${PROFILE_DIR}" "${RESULTS_DIR}" "${METHOD_DIR}"

    echo ">>> Config $((i+1))/${#CONFIGS[@]}: ${CFG} cp${CP}_tp${TP} MBS=${MBS} SEQ=${SEQ} GBS=${GBS}"

    RUN_FREQS_PATHS=()
    RUN_PLAN_TAGS=()

    if [[ "${FRONTIER}" == "true" ]]; then
        SCHED_DIR="${SCHEDULES_DIR}/${nemo_model_name}/${config_tag}/frontier"
        if [[ "${SKIP_PROFILING}" == "false" ]]; then
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
                        --gpu_type="${GPU_TYPE}"
                else
                    echo "    Optimisation results exist in: ${RESULTS_DIR}"
                fi

                echo "    Creating frontier schedules from optimisation results..."
                rm -f "${SCHED_DIR}"/*.py
                MODEL_NAME="${nemo_model_name}" CONFIG_TAG="${config_tag}" \
                    bash "${SCRIPT_DIR}/create_frontier.sh" nanobatch
            fi
        elif [[ "${SKIP_PROFILING}" != "true" ]]; then
            echo "    ERROR: SKIP_PROFILING must be 'true' or 'false'" >&2
            exit 1
        fi

        if [[ "${NODE_RANK}" == "1" ]]; then
            remote_frontier_dir="${REMOTE_BASE_DIR}/../nanobatch_perseus/schedules/${nemo_model_name}/${config_tag}/frontier"
            echo "    Syncing frontier schedules from ${REMOTE_USER}@${MASTER_ADDR}:${remote_frontier_dir}"
            sync_frontier_schedules_from_node0 "${remote_frontier_dir}" "${SCHED_DIR}" 10
        fi

        mapfile -t RUN_FREQS_PATHS < <(ls "${SCHED_DIR}"/freqs_pipeline_*.py 2>/dev/null | sort -V)
        if (( ${#RUN_FREQS_PATHS[@]} != 10 )); then
            echo "    ERROR: FRONTIER=true expected 10 freqs_pipeline_*.py files in ${SCHED_DIR}, found ${#RUN_FREQS_PATHS[@]}" >&2
            exit 1
        fi

        for freqs_path in "${RUN_FREQS_PATHS[@]}"; do
            freqs_file="$(basename "${freqs_path}")"
            suffix="${freqs_file#freqs_pipeline_}"
            suffix="${suffix%.py}"
            RUN_PLAN_TAGS+=("pipeline_${suffix}")
        done
        echo "    Frontier schedules: ${#RUN_FREQS_PATHS[@]} plans from ${SCHED_DIR}"
    elif [[ "${SKIP_PROFILING}" == "true" ]]; then
        SCHED_DIR="${SCHEDULES_DIR}/${nemo_model_name}/${config_tag}"
        freqs_path="$(ls "${SCHED_DIR}"/freqs_pipeline_*.py 2>/dev/null | sort -V | tail -n 1 || true)"
        if [[ -z "${freqs_path}" ]]; then
            echo "    ERROR: SKIP_PROFILING=true but no freqs_pipeline_*.py found in ${SCHED_DIR}" >&2
            exit 1
        fi
        echo "    Using precomputed solution: ${freqs_path}"
        RUN_FREQS_PATHS=("${freqs_path}")
        RUN_PLAN_TAGS=("default")
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
                    --gpu_type="${GPU_TYPE}"
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
        RUN_FREQS_PATHS=("${freqs_path}")
        RUN_PLAN_TAGS=("default")
    fi

    for run_idx in "${!RUN_FREQS_PATHS[@]}"; do
        freqs_path="${RUN_FREQS_PATHS[$run_idx]}"
        plan_tag="${RUN_PLAN_TAGS[$run_idx]}"

        if [[ "${FRONTIER}" == "true" ]]; then
            OUTPUT_DIR="${METHOD_DIR}/frontier/${plan_tag}"
            TRAIN_LOG="${LOG_DIR}/${nemo_model_name}_${config_tag}_nanobatch_perseus_${plan_tag}.log"
            echo "    >>> Frontier plan $((run_idx+1))/${#RUN_FREQS_PATHS[@]}: ${plan_tag}"
        else
            OUTPUT_DIR="${METHOD_DIR}"
            TRAIN_LOG="${LOG_DIR}/${nemo_model_name}_${config_tag}_nanobatch_perseus.log"
        fi

        mkdir -p "${OUTPUT_DIR}"

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
                PFO_PID=""
            fi

            echo "    Node 0 done – outputs: ${OUTPUT_DIR}"
        else
            if compgen -G "${NEMO_DIR}/*.txt" > /dev/null; then
                mv "${NEMO_DIR}"/*.txt "${OUTPUT_DIR}/"
            fi

            remote_dir="${REMOTE_BASE_DIR}/nemo_experiments/${nemo_model_name}/${config_tag}/nanobatch_perseus"
            if [[ "${FRONTIER}" == "true" ]]; then
                remote_dir="${remote_dir}/frontier/${plan_tag}"
            fi
            ssh "${SSH_KEY_OPTS[@]}" "${REMOTE_USER}@${MASTER_ADDR}" "mkdir -p '${remote_dir}'"
            sleep 5
            scp "${SSH_KEY_OPTS[@]}" -r "${OUTPUT_DIR}/"* \
                "${REMOTE_USER}@${MASTER_ADDR}":"${remote_dir}/"
            echo "    Node 1 done – synced to: ${remote_dir}"
        fi

        echo "    log: ${TRAIN_LOG}"
        echo ""
        sleep 5
    done
done

echo "All ${#CONFIGS[@]} Nanobatch+Perseus configurations completed (node_rank=${NODE_RANK})."
