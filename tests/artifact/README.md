# Artifact 16-GPU (2x8 A100) Tests

This directory contains the scripts used to reproduce the artifact evaluation
for **Megatron / Nanobatch / Perseus / Nanobatch+Perseus / Kareus** on
**2 nodes x 8 NVIDIA A100 GPUs** (16 GPUs total).

The shell scripts in this directory launch the training entry points and call
the per-method tools in sibling directories as needed:
`../kareus`, `../perseus`, `../nanobatch_perseus`, and `../bayesian`.

## Quick Start

1. Make sure both nodes have this repo, either on a shared filesystem or as
   matching local checkouts.
2. Edit `env.sh` once per cluster. At minimum, set `MASTER_ADDR` to the
   hostname or IP address of node 0.
3. Run each method script once on each node, passing the node rank as the only
   positional argument.
4. For `run_kareus.sh`, run `kareus_run_bayesian.sh` first on one 8-GPU node so
   that the Bayesian profiling logs exist.

Example two-node launch:

```bash
# On node 0 (master)
bash tests/artifact/run_megatron.sh 0

# On node 1
bash tests/artifact/run_megatron.sh 1
```

Single-config sanity check:

```bash
# On node 0
CONFIG_MODE=single bash tests/artifact/run_megatron.sh 0

# On node 1
CONFIG_MODE=single bash tests/artifact/run_megatron.sh 1
```

Override an `env.sh` value for one invocation:

```bash
MASTER_ADDR=10.0.0.42 bash tests/artifact/run_megatron.sh 0
```

## Scripts

| Script | What it does | Frequency control |
| ------ | ------------ | ----------------- |
| `run_megatron.sh` | Vanilla Megatron baseline (`megatron_gpt_pretraining.py`) | none |
| `run_nanobatch.sh` | Kareus model with nanobatching, no scheduler (`kareus_gpt_pretraining.py`) | none |
| `run_perseus.sh` | Megatron + Perseus pipeline-frequency optimizer | per-microbatch frequency via PFO server |
| `run_nanobatch_perseus.sh` | Nanobatching + Perseus | per-microbatch frequency via PFO server |
| `run_kareus.sh` | Full Kareus stack: Bayesian profiling, Phillips-Dessouky solver, Kareus scheduler, and PFO | per-partition frequency + schedule |
| `kareus_run_bayesian.sh` | Standalone Bayesian partition profiling driver; prerequisite for `run_kareus.sh` | n/a |
| `megatron_perseus_run_profiling.sh` | Frequency-sweep helper for `run_perseus.sh` | n/a |
| `nanobatch_perseus_run_profiling.sh` | Frequency-sweep helper for `run_nanobatch_perseus.sh` | n/a |

`run_perseus.sh` and `run_nanobatch_perseus.sh` perform a one-time per-config
GPU-frequency sweep before training and skip it on later runs via a
`.profiling_complete` marker.

## Evaluation Configurations

`CONFIG_MODE=full` is the default and sweeps these 10 configurations:

| Model | Parallelism | MBS | Seq | GBS |
| ----- | ----------- | --- | --- | --- |
| Llama 3.2 3B | TP=8 | 8 | 4096 | 64 |
| Llama 3.2 3B | CP=2 + TP=4 | 8 | 4096 | 64 |
| Llama 3.2 3B | CP=2 + TP=4 | 8 | 8192 | 64 |
| Llama 3.2 3B | CP=2 + TP=4 | 16 | 4096 | 128 |
| Qwen 3 1.7B | TP=8 | 8 | 4096 | 64 |
| Qwen 3 1.7B | TP=8 | 8 | 8192 | 64 |
| Qwen 3 1.7B | TP=8 | 16 | 4096 | 128 |
| Qwen 3 1.7B | CP=2 + TP=4 | 8 | 4096 | 64 |
| Qwen 3 1.7B | CP=2 + TP=4 | 8 | 8192 | 64 |
| Qwen 3 1.7B | CP=2 + TP=4 | 16 | 4096 | 128 |

All artifact configurations use `PP=2` and `NUM_MICROBATCHES=8`.
`CONFIG_MODE=single` runs only the first row: Llama 3.2 3B, TP=8, MBS=8,
SEQ=4096.

## Per-Cluster Setup

Edit `env.sh` once per cluster to set values that depend on your hardware,
network, and checkout location:

| Variable | Purpose |
| -------- | ------- |
| `MASTER_ADDR` | IP/hostname of node 0, used by torchrun and as the scp target from node 1 |
| `REMOTE_USER` | SSH user on node 0, used by node 1 for scp |
| `REMOTE_BASE_DIR` | Path to this checkout on node 0, used as the scp target |
| `SSH_KEY_PATH` | Optional SSH key node 1 uses to scp results back to node 0; leave unset to use the default ssh agent or `~/.ssh/config` |

Every run script and profiling helper sources `env.sh` automatically. Command
line overrides still win. If `env.sh` is removed, the in-script defaults take
over, but `MASTER_ADDR` must then be provided explicitly for each run.

### Common Environment Variables

| Variable | Default | Used by |
| -------- | ------- | ------- |
| `MASTER_ADDR` | from `env.sh` (required) | all run scripts |
| `MASTER_PORT` | `6000` | all distributed runs |
| `PFO_PORT` | `7787` | `run_perseus.sh`, `run_nanobatch_perseus.sh`, `run_kareus.sh` |
| `CONFIG_MODE` | `full` | all run scripts and `kareus_run_bayesian.sh` |
| `SKIP_PROFILING` | `false` | precomputed-schedule mode for `run_perseus.sh` and `run_nanobatch_perseus.sh` |
| `REMOTE_USER` | from `env.sh` (default `ubuntu`) | node-1 to node-0 scp |
| `REMOTE_BASE_DIR` | from `env.sh` (default `$HOME/workspace/Kareus/tests/artifact`) | node-1 to node-0 scp |
| `SSH_KEY_PATH` | from `env.sh` (optional) | node-1 to node-0 scp |

## Common GPU and Configuration Variables

The artifact defaults target **NVIDIA A100 SXM4 40GB GPUs**. For a different
GPU type, keep the GPU and frequency variables consistent across profiling and
training:

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `GPU_TYPE` | `A100` | Selects the GPU entry used by the optimizers and Bayesian profiling paths |
| `FREQ_START` | `1410` | Highest frequency used by profiling sweeps |
| `FREQ_END` | `900` | Lowest frequency used by profiling sweeps |
| `FREQ_STEP` | `30` | Frequency decrement between sweep points |

For a new GPU, first add an entry to `GPU_CONFIGS` in
`tests/bayesian/common/model_config.py` with both `p2p_power_w` and
`freq_range`. `p2p_power_w` is the measured power while the GPU is blocked on
P2P communication; Kareus uses it as static power when computing dynamic and
effective energy. You can measure it with `tests/data/profile_p2p.py`.

To change the evaluated parallelism dimensions, keep these constants and arrays
in sync:

| Setting | Where to edit |
| ------- | ------------- |
| Pipeline parallelism | `PP=` near the top of each `tests/artifact/run_*.sh` script |
| Number of microbatches | `NUM_MICROBATCHES=` near the top of each `tests/artifact/run_*.sh` script and `tests/data/prepare_data.sh` |
| Config sweep rows | The `CP/TP/MBS/SEQ` columns of `CONFIGS_FULL` in each `tests/artifact/run_*.sh` script |
| Data preparation rows | The matching `CONFIGS_FULL` array in `tests/data/prepare_data.sh` |
| Kareus Bayesian profiling rows | The matching `run_tp_only`, `run_cptp_*`, and `run_nonpartition` calls in `tests/artifact/kareus_run_bayesian.sh` |

`CONFIG_MODE=single` uses the first configuration row, so update that row first
when creating a single-config sanity check.

## Kareus Bayesian Profiling

`run_kareus.sh` reads Bayesian profiling results from:

```text
tests/bayesian/logs/<model>/cp${CP}-tp${TP}-bs${BS}-seq${SEQ}/
```

Generate those logs by running `kareus_run_bayesian.sh` once on a single
8-GPU node before launching `run_kareus.sh`. The script does not use
multi-node torchrun or `MASTER_ADDR`; it orchestrates partition and
nonpartition profilers across local GPU subsets.

```bash
# Full sweep
bash tests/artifact/kareus_run_bayesian.sh

# Single config, matching CONFIG_MODE=single in run_kareus.sh
CONFIG_MODE=single bash tests/artifact/kareus_run_bayesian.sh
```

## Precomputed Perseus Schedules

When `SKIP_PROFILING=true`, `run_perseus.sh` and
`run_nanobatch_perseus.sh` skip profiling and optimization. They read
precomputed frequency schedules instead:

| Script | Schedule directory | Files used |
| ------ | ------------------ | ---------- |
| `run_perseus.sh` | `tests/perseus/schedules/<model_name>/<config_tag>/` | `freqs_pipeline_*.py` |
| `run_nanobatch_perseus.sh` | `tests/nanobatch_perseus/schedules/<model_name>/<config_tag>/` | `freqs_pipeline_*.py` |

`<model_name>` is one of `megatron_llama3.2_3b` or `megatron_qwen3_1.7b`.
`<config_tag>` has the form `cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}`, for example
`cp1_tp8_mbs8_seq4096`. The scripts fail fast if the expected files are
missing.

```bash
SKIP_PROFILING=true bash tests/artifact/run_perseus.sh 0
SKIP_PROFILING=true bash tests/artifact/run_nanobatch_perseus.sh 0
```

## Outputs and Logs

Training outputs are organized as:

```text
tests/artifact/nemo_experiments/<model>/<config_tag>/<method>/
```

`<model>` is one of `megatron_llama3.2_3b` or `megatron_qwen3_1.7b`.
`<config_tag>` has the form `cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}`, for example
`cp1_tp8_mbs8_seq4096`.
`<method>` is one of `megatron`, `nanobatch`, `perseus`,
`nanobatch_perseus`, or `kareus`.

Per-run training logs land in `tests/artifact/logs/` and are named:

```text
<model>_<config_tag>_<method>.log
```

For example:

```text
megatron_llama3.2_3b_cp1_tp8_mbs8_seq4096_kareus.log
```
