#!/usr/bin/env bash
# Prepare data and dataset indices for the artifact tests.
#
# Usage:
#   bash tests/data/prepare_data.sh
#   CONFIG_MODE=single bash tests/data/prepare_data.sh
#
# CONFIG_MODE (optional, default full):
#   full   - tokenize WikiText-103 with both Llama and Qwen tokenizers, then
#            generate dataset indices for all 10 configurations from
#            tests/artifact/README.md
#   single - tokenize WikiText-103 with the Llama tokenizer only, then
#            generate dataset indices for the single configuration
#            (Llama 3.2 3B, TP=8, MBS=8, SEQ=4096)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAREUS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_MODE="${CONFIG_MODE:-full}"
case "${CONFIG_MODE}" in
    full|single) ;;
    *) echo "ERROR: CONFIG_MODE must be 'full' or 'single', got '${CONFIG_MODE}'" >&2; exit 1 ;;
esac

echo "===== prepare_data.sh (CONFIG_MODE=${CONFIG_MODE}) ====="

# Step 1: tokenize WikiText-103 into Megatron mmap indexed datasets.
echo ""
echo ">>> Step 1: tokenize WikiText-103"
bash "${SCRIPT_DIR}/prepare_data_llama.sh"

if [[ "${CONFIG_MODE}" == "full" ]]; then
    bash "${SCRIPT_DIR}/prepare_data_qwen.sh"
fi

# Step 2: generate dataset indices by running a 1-step pretraining job per
# configuration. We use a tiny model (1 layer / hidden=128 / heads=4) so the
# job only does enough work to materialise the dataset shuffle/split index
# files under tests/data/.
echo ""
echo ">>> Step 2: generate dataset indices"

ARTIFACT_DIR="${KAREUS_DIR}/tests/artifact"
PRETRAIN_PY="${ARTIFACT_DIR}/megatron_gpt_pretraining.py"

# config_name  CP  TP  MBS  SEQ
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
esac

NUM_MICROBATCHES=8

for i in "${!CONFIGS[@]}"; do
    read -r CFG CP TP MBS SEQ <<< "${CONFIGS[$i]}"
    GBS=$(( MBS * NUM_MICROBATCHES ))

    echo ""
    echo ">>> Index $((i+1))/${#CONFIGS[@]}: ${CFG} cp${CP}_tp${TP} MBS=${MBS} SEQ=${SEQ} GBS=${GBS}"

    CUDA_VISIBLE_DEVICES=0 python "${PRETRAIN_PY}" \
        --config-name="${CFG}" \
        trainer.devices=1 \
        trainer.num_nodes=1 \
        model.tensor_model_parallel_size=1 \
        model.pipeline_model_parallel_size=1 \
        model.context_parallel_size=1 \
        model.num_layers_in_first_pipeline_stage=null \
        model.num_layers=1 \
        model.hidden_size=128 \
        model.ffn_hidden_size=512 \
        model.num_attention_heads=4 \
        model.num_query_groups=4 \
        model.micro_batch_size="${MBS}" \
        model.global_batch_size="${GBS}" \
        model.encoder_seq_length="${SEQ}"
done

echo ""
echo "All ${#CONFIGS[@]} dataset indices generated (CONFIG_MODE=${CONFIG_MODE})."
