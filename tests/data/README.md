# tests/data/ — Dataset preparation

All artifact runs train on **WikiText-103** tokenized with the model's own
tokenizer and packed into Megatron's mmap indexed-dataset format. The scripts
here download, tokenize, and (optionally) materialise per-config dataset
shuffle/split index files.

## Prerequisites

Set your Hugging Face token before running anything in this directory — both
tokenizer downloads (`meta-llama/Llama-3.2-3B-Instruct`,
`Qwen/Qwen3-1.7B-Base`) are gated:

```bash
export HF_TOKEN=<your_huggingface_token>
```

The Hugging Face cache is mounted into the container at
`/root/.cache/huggingface` (see the `docker run` command in the top-level
[README.md](../../README.md)), so subsequent runs reuse downloads.

## Scripts

| Script                  | What it does                                                                |
| ----------------------- | --------------------------------------------------------------------------- |
| `prepare_data.sh`       | One-shot driver: tokenization (Step 1) + per-config index materialisation (Step 2). |
| `prepare_data_llama.sh` | Step 1 only, Llama 3.2 3B tokenizer → `llama_dataset_text_document.{bin,idx}`. |
| `prepare_data_qwen.sh`  | Step 1 only, Qwen 3 1.7B tokenizer → `qwen_dataset_text_document.{bin,idx}`. |
| `install_deps.sh`       | Builds the entire Python stack (Megatron / NeMo / mscclpp / zeus / TE patches). Already invoked by the Docker build; included here for reference. |

## CONFIG_MODE

`prepare_data.sh` honours the same `CONFIG_MODE` switch as the run scripts in
[tests/artifact/](../artifact/):

- `CONFIG_MODE=full` (default) — tokenize with **both** Llama and Qwen
  tokenizers, then generate the dataset-index files for all 10 configurations
  enumerated in [tests/artifact/README.md](../artifact/README.md).
- `CONFIG_MODE=single` — Llama tokenizer only, then generate the index files
  for the single Llama 3.2 3B / TP=8 / MBS=8 / SEQ=4096 configuration. This
  is the recommended starting point for reviewers.

```bash
CONFIG_MODE=single bash tests/data/prepare_data.sh
# or
bash tests/data/prepare_data.sh    # full sweep
```

## Outputs

After `prepare_data.sh` completes, `tests/data/` contains:

```
llama_dataset_text_document.bin    # tokenized WikiText-103 (Llama tokenizer)
llama_dataset_text_document.idx
qwen_dataset_text_document.bin     # only with CONFIG_MODE=full
qwen_dataset_text_document.idx
*.npy / *.idx (per-config)         # shuffle/split indices materialised by the
                                   # 1-step pretraining job in Step 2
```

The per-config indices are produced by running a tiny 1-layer pretraining
job (hidden=128, heads=4) per configuration just long enough for NeMo to
materialise the dataset-index files; nothing is trained.

## Notes

- The 1-step jobs in Step 2 only need 1 GPU each and run on `cuda:0`
  (`CUDA_VISIBLE_DEVICES=0`).
- All downstream run scripts ([tests/artifact/](../artifact/), [tests/toy/](../toy/))
  read the resulting `*.bin`/`*.idx` files directly; no further conversion is
  needed.
