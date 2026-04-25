# Toy 4-GPU Tests

A small, self-contained set of scripts for exercising the full
**Megatron / Perseus / Kareus** toolchain on a single node with
**4 × NVIDIA A40 GPUs**.

All scripts share the same fixed configuration:

| Item | Value |
|---|---|
| Model | Llama 3.2 3B (`megatron_llama3.2_3b_config`) |
| Pipeline parallel (PP) | 2 |
| Tensor parallel (TP) | 2 |
| Context parallel (CP) | 1 |
| Micro-batch size (MBS) | 4 |
| #microbatches | 4 |
| Global batch size (GBS) | 16 |
| Sequence length (SEQ) | 2048 |
| GPUs | 4 × A40 (1 node) |

## Layout of generated outputs

All experiment outputs are collected under

```
tests/toy/nemo_experiments/<model>/cp1_tp2_mbs4_seq2048/<method>/
```

where `<method>` is one of `megatron`, `nanobatch`, `perseus`,
`nanobatch_perseus`, or `kareus`. Per-run NeMo logs, Zeus monitor
files, profiling sweeps, and `lowtime` solver results all land in
this directory tree.

## Scripts

| Script | What it does | Frequency control |
|---|---|---|
| `run_megatron.sh` | Vanilla Megatron baseline (`megatron_gpt_pretraining.py`) | none |
| `run_nanobatch.sh` | Kareus model with nanobatching, no scheduler (`kareus_gpt_pretraining.py`) | none |
| `run_perseus.sh` | Megatron + Perseus pipeline-frequency optimizer | per-instruction freq via PFO server |
| `run_nanobatch_perseus.sh` | Nanobatching + Perseus | per-instruction freq via PFO server |
| `run_kareus.sh` | Full Kareus stack: Bayesian profiling → Phillips-Dessouky solver → Kareus scheduler + PFO | per-partition freq + schedule |
| `kareus_run_bayesian.sh` | Standalone BO partition profiling (auto-invoked by `run_kareus.sh` if results are missing) | n/a |
| `megatron_perseus_run_profiling.sh` | Frequency-sweep profiling helper for `run_perseus.sh` | n/a |
| `nanobatch_perseus_run_profiling.sh` | Frequency-sweep profiling helper for `run_nanobatch_perseus.sh` | n/a |

## Running

From the repo root, with the project's Python env activated:

```bash
bash tests/toy/run_megatron.sh
bash tests/toy/run_nanobatch.sh
bash tests/toy/run_perseus.sh
bash tests/toy/run_nanobatch_perseus.sh
bash tests/toy/run_kareus.sh
```

`run_perseus.sh` and `run_nanobatch_perseus.sh` perform a one-time
GPU-frequency sweep before training and skip it on subsequent runs
(via a `.profiling_complete` marker). `run_kareus.sh` performs a
one-time Bayesian-optimization sweep and skips it once the expected
artefacts under `tests/bayesian/logs/...` are in place.

### Common environment variables

| Variable | Default | Used by |
|---|---|---|
| `MASTER_PORT` | `6000` | all |
| `PFO_PORT` | `7787` | `run_perseus.sh`, `run_nanobatch_perseus.sh`, `run_kareus.sh` |
| `FREQ_START` | `1740` | profiling sweeps |
| `FREQ_END` | `900` | profiling sweeps |
| `FREQ_STEP` | `60` | profiling sweeps |

The frequency-sweep defaults (1740 → 900 MHz, step 60) and the
optimizer's `--p2p_power=70.0` correspond to the **NVIDIA A40**.
On other GPUs these need to be adjusted before running the
profiling/optimization scripts.

## Logs

Per-run training logs land in `tests/toy/logs/`, named
`<model>_<config_tag>_<method>.log` — e.g.
`megatron_llama3.2_3b_cp1_tp2_mbs4_seq2048_kareus.log`.
