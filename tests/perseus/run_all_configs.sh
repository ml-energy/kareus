#!/usr/bin/env bash
# Automated Perseus test: profiling → CSV → optimization → PFO + training
# for 10 configs on 16 GPUs (2 nodes × 8 GPUs), PP=2, microbatches=8.
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
# Per-config flow:
#   1. Profiling   – frequency sweep (skip if .profiling_complete exists)
#   2. CSV gen     – generate_profile_csv.py  (node 0 only)
#   3. Optimise    – run_optimization.py      (node 0 only)
#   4. PFO server  – uvicorn on node 0 with largest freqs_pipeline_*.py
#   5. Training    – torchrun with enable_perseus_optimizer=True
#   6. Collection  – move outputs, node 1 SCPs to node 0
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
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-~/workspace/Kareus/tests/perseus}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/ruofanw.pem}"

export MASTER_ADDR
export MASTER_PORT
export REMOTE_USER
export REMOTE_BASE_DIR
export SSH_KEY_PATH

PP=2
NUM_MICROBATCHES=8
PFO_PORT=7787

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

echo "===== Perseus 16-GPU Automated Tests (PP=${PP}, #microbatches=${NUM_MICROBATCHES}) ====="
echo "Total configurations: ${#CONFIGS[@]}"
echo ""

for i in "${!CONFIGS[@]}"; do
    read -r CFG CP TP MBS SEQ <<< "${CONFIGS[$i]}"

    GBS=$(( MBS * NUM_MICROBATCHES ))

    nemo_model_name="${CFG%_config}"
    config_tag="cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}"

    NEMO_DIR="${SCRIPT_DIR}/nemo_experiments/${nemo_model_name}"
    PROFILE_DIR="${NEMO_DIR}/${config_tag}/profiling"
    RESULTS_DIR="${NEMO_DIR}/${config_tag}/lowtime"
    OUTPUT_DIR="${NEMO_DIR}/${config_tag}/perseus"

    mkdir -p "${PROFILE_DIR}" "${RESULTS_DIR}" "${OUTPUT_DIR}"

    echo ">>> Config $((i+1))/${#CONFIGS[@]}: ${CFG} cp${CP}_tp${TP} MBS=${MBS} SEQ=${SEQ} GBS=${GBS}"

    ########################################
    # Phase 1: Profiling (skip if done)    #
    ########################################

    PROFILING_MARKER="${PROFILE_DIR}/.profiling_complete"

    if [[ -f "${PROFILING_MARKER}" ]]; then
        echo "    Profiling already done (${PROFILING_MARKER} exists) — skipping"
    else
        echo "    Starting frequency profiling..."
        bash "${SCRIPT_DIR}/run_profiling.sh" "${NODE_RANK}" "${CFG}" "${CP}" "${TP}" "${MBS}" "${SEQ}"
    fi

    ########################################
    # Phase 2+3: CSV gen + Optimise        #
    # (node 0 only, skip if already done)  #
    ########################################

    PROFILE_CSV="${PROFILE_DIR}/profile.csv"

    if [[ "${NODE_RANK}" == "0" ]]; then
        if [[ ! -f "${PROFILE_CSV}" ]]; then
            echo "    Generating profile CSV..."
            NUM_RANKS_PER_STAGE=$(( CP * TP ))
            python "${SCRIPT_DIR}/generate_profile_csv.py" \
                --profile_dir="${PROFILE_DIR}" \
                --num_ranks_per_stage="${NUM_RANKS_PER_STAGE}" \
                --num_microbatches="${NUM_MICROBATCHES}"
        else
            echo "    Profile CSV exists: ${PROFILE_CSV}"
        fi

        if ! compgen -G "${RESULTS_DIR}/freqs_pipeline_*.py" > /dev/null 2>&1; then
            echo "    Running optimisation..."
            python "${SCRIPT_DIR}/run_optimization.py" \
                --inst_profile="${PROFILE_CSV}" \
                --output_dir="${RESULTS_DIR}" \
                --num_mbs="${NUM_MICROBATCHES}" \
                --num_stages="${PP}"
        else
            echo "    Optimisation results exist in: ${RESULTS_DIR}"
        fi
    fi

    ########################################
    # Phase 4: Find largest solution       #
    # Phase 5: Start PFO server (node 0)   #
    ########################################

    PFO_PID=""

    if [[ "${NODE_RANK}" == "0" ]]; then
        freqs_path="$(ls "${RESULTS_DIR}"/freqs_pipeline_*.py 2>/dev/null | sort -V | tail -n 1 || true)"

        if [[ -z "${freqs_path}" ]]; then
            echo "    ERROR: No freqs_pipeline_*.py found in ${RESULTS_DIR}" >&2
            echo "    Skipping Perseus training for this config." >&2
            continue
        fi

        echo "    Using solution: ${freqs_path}"

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
    fi

    ########################################
    # Phase 6: Perseus training            #
    ########################################

    TRAIN_LOG="${LOG_DIR}/${nemo_model_name}_${config_tag}_perseus.log"

    echo "    Running Perseus training (node_rank=${NODE_RANK})..."

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
        model.enable_megatron_timers=False \
        model.enable_zeus_monitor=True \
        model.enable_power_monitor=False \
        model.enable_perseus_optimizer=True \
        model.enable_kareus_scheduler=False \
        2>&1 | tee "${TRAIN_LOG}"

    echo "    Resetting GPU clocks"
    nvidia-smi -i 0,1,2,3,4,5,6,7 --reset-gpu-clocks

    ########################################
    # Phase 7: Collect outputs + cleanup   #
    ########################################

    if [[ "${NODE_RANK}" == "0" ]]; then
        echo "    Moving NeMo experiment outputs into ${OUTPUT_DIR}"
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

        # Stop PFO server
        if [[ -n "${PFO_PID}" ]] && ps -p "${PFO_PID}" > /dev/null 2>&1; then
            echo "    Stopping PFO server PID ${PFO_PID}"
            kill "${PFO_PID}" || true
            wait "${PFO_PID}" 2>/dev/null || true
        fi

        echo "    Node 0 done – outputs: ${OUTPUT_DIR}"

    else
        echo "    Moving NeMo experiment text logs into ${OUTPUT_DIR}"

        if compgen -G "${NEMO_DIR}/*.txt" > /dev/null; then
            mv "${NEMO_DIR}"/*.txt "${OUTPUT_DIR}/"
        fi

        remote_dir="${REMOTE_BASE_DIR}/nemo_experiments/${nemo_model_name}/${config_tag}/perseus"
        echo "    Syncing results from node 1 to ${REMOTE_USER}@${MASTER_ADDR}:${remote_dir}"

        sleep 5
        scp -i "${SSH_KEY_PATH}" -r "${OUTPUT_DIR}/"* \
            "${REMOTE_USER}@${MASTER_ADDR}":"${remote_dir}/"

        echo "    Node 1 done – synced to: ${remote_dir}"
    fi

    echo "    log: ${TRAIN_LOG}"
    echo ""
    sleep 5
done

echo "All ${#CONFIGS[@]} Perseus configurations completed (node_rank=${NODE_RANK})."
