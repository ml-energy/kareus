#!/bin/bash
set -euo pipefail

KAREUS_DIR="/workspaces/Kareus"
MEGATRON_DIR="${KAREUS_DIR}/3rdparty/Megatron-LM"
DATA_DIR="${KAREUS_DIR}/tests/data"
TOKENIZER="meta-llama/Llama-3.2-3B-Instruct"
export HF_TOKEN="${HF_TOKEN:?Please set HF_TOKEN environment variable}"

mkdir -p "${DATA_DIR}"

# Step 1: Download WikiText-103 and convert to JSONL
# Step 2: Tokenize and build Megatron mmap indexed dataset
# Output: ${DATA_DIR}/llama_dataset_text_document.{bin,idx}
python3 -c "
import sys, json, os
sys.path.insert(0, '${MEGATRON_DIR}')

from datasets import load_dataset
from transformers import AutoTokenizer
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder, DType

DATA_DIR = '${DATA_DIR}'
TOKENIZER = '${TOKENIZER}'

# Download WikiText-103
print('=== Downloading WikiText-103 ===')
ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')

# Load tokenizer
print('=== Loading tokenizer ===')
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
eod_id = tokenizer.eos_token_id

# Build indexed dataset
output_prefix = os.path.join(DATA_DIR, 'llama_dataset_text_document')
output_bin = output_prefix + '.bin'
output_idx = output_prefix + '.idx'

print('=== Tokenizing and building indexed dataset ===')
builder = IndexedDatasetBuilder(output_bin, dtype=DType.optimal_dtype(tokenizer.vocab_size))

doc_count = 0
for row in ds:
    text = row['text'].strip()
    if not text:
        continue
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if token_ids:
        token_ids.append(eod_id)
        builder.add_document(token_ids, [len(token_ids)])
        doc_count += 1
        if doc_count % 50000 == 0:
            print(f'  Processed {doc_count} documents...')

builder.finalize(output_idx)
print(f'=== Done: {doc_count} documents ===')
"

echo "Output files:"
ls -lh "${DATA_DIR}"/llama_dataset_text_document.*
