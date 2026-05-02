# Kareus OSDI ’26 Artifact Evaluation

Artifact release for the paper **"Kareus: Joint Reduction of Dynamic and Static
Energy in Large Model Training"**.

This repository contains everything needed to reproduce the main
end-to-end results in the paper:

- Section 6.2.1 — Max-Throughput Comparison (Megatron-LM vs. Megatron+Perseus
vs. Nanobatching+Perseus vs. Kareus).
- Section 6.2.2 — Time/Energy Frontier Improvement (Megatron+Perseus vs.
Nanobatching+Perseus vs. Kareus, all evaluated as 10-point frontiers).

## Artifact organization

```
.
├── 3rdparty/              # Vendored dependencies (see 3rdparty/README.md)
│   ├── Megatron-LM/       #   Megatron-Core v0.12.1   with Perseus/Kareus support
│   ├── NeMo/              #   NeMo v2.3.1             with Perseus/Kareus support
│   ├── TransformerEngine/ #   TransformerEngine v2.4.0 with Perseus/Kareus support
│   ├── mscclpp/           #   MSCCL++ v0.7.0          with Perseus/Kareus support
│   └── zeus/              #   Zeus v0.15.1
│
├── kareus/                # Kareus runtime (see kareus/README.md)
│   ├── megatron/core/partitions/  # Core partition-overlap execution engine
│   ├── scheduler/         #   Per-microbatch communication scheduler
│   ├── nemo/              #   Kareus adaptations in the NeMo training-loop layer
│   ├── megatron/          #   Kareus adaptations in the Megatron-Core transformer layer
│   ├── transformer_engine/#   Kareus adaptations in the TransformerEngine attention/linear layer
│   ├── apex/              #   Kareus adaptations in the Apex transformer-utility layer
│   ├── flash_attn/        #   Kareus adaptations in the FlashAttention kernel layer
│   └── msccl/             #   Kareus adaptations in the MSCCL++ collective-comm layer
│
├── tests/
│   ├── artifact/          # 16-GPU (2x8 A100) reproduction scripts — main entry point
│   ├── toy/               # 4-GPU (1x4 A40) sanity scripts
│   ├── bayesian/          # Per-partition Bayesian-optimization profilers
│   ├── kareus/            # Utils for Kareus (Kareus-adapted Perseus optimizer)
│   ├── perseus/           # Utils for Megatron+Perseus
│   ├── nanobatch_perseus/ # Utils for Nanobatching+Perseus
│   └── data/              # Dataset preparation
│
├── Dockerfile             # Docker image for the entire stack (PyTorch 25.06)
└── videos/                # Screen recordings of each reproduction step
```

## Overall Kareus workflow

A Kareus end-to-end run is three phases:

1. **Bayesian optimization** of each partition produces a per-partition
  time–energy frontier. Driver: [tests/artifact/kareus_run_bayesian.sh](tests/artifact/kareus_run_bayesian.sh) (per-partition profilers in [tests/bayesian/partitions/](tests/bayesian/partitions/) and [tests/bayesian/nonpartition/](tests/bayesian/nonpartition/)). Outputs land in `tests/bayesian/logs/<model>/cp${CP}-tp${TP}-bs${BS}-seq${SEQ}/`.
2. **Compose** the per-partition frontiers and solve for an optimized
  iteration-level execution schedule. Driver: `generate_profile_csv.py` + `run_optimization.py` in [tests/kareus/](tests/kareus/), invoked by [tests/artifact/run_kareus.sh](tests/artifact/run_kareus.sh). Outputs are pairs of `freqs_pipeline_*.py` (per-microbatch GPU frequencies) and `scheds_pipeline_*.py` (per-microbatch communication schedules).
3. **Execute** either (a) the maximum-throughput plan, or (b) all 10 frontier
  plans, on the real cluster. Driver: [tests/artifact/run_kareus.sh](tests/artifact/run_kareus.sh) (set `FRONTIER=true` for the frontier sweep).

## Reproducing evaluation results

> [!NOTE]
> Hardware: experiments in the paper were generated on **2× AWS p4d.24xlarge instances (2x8 NVIDIA A100 SXM4 40GB)**; all defaults below assume 2x8 A100 GPUs.
>
> For other GPU types, adjust `FREQ_START`, `FREQ_END`, `FREQ_STEP`, and the optimizer's `--p2p_power` (see "Common environment variables" in [tests/artifact/README.md](tests/artifact/README.md)).
>
> To change the parallelism dimensions, edit the `PP=` and `NUM_MICROBATCHES=` constants and the `CP/TP/MBS/SEQ` columns of the `CONFIGS_FULL` array near the top of each `tests/artifact/run_*.sh` script (and the matching arrays in [tests/data/prepare_data.sh](tests/data/prepare_data.sh) and [tests/artifact/kareus_run_bayesian.sh](tests/artifact/kareus_run_bayesian.sh)).
>
> For example, [tests/toy/](tests/toy/) shows a small sanity check on 4× NVIDIA A40 GPUs with PP=2, TP=2 (see [tests/toy/README.md](tests/toy/README.md)).

### 0. Environment setup

**Clone the repository** on each node:

```bash
git clone -b osdi26ae https://github.com/ml-energy/kareus.git
cd kareus
```

**Build the image** (or pull the prebuilt one):

```bash
# Option A: pull
docker pull ruofanwu7/kareus-artifact:latest

# Option B: build locally (from the repo root, BuildKit required)
DOCKER_BUILDKIT=1 docker build -t kareus-artifact:latest .
```

**Launch the container** on each node, mounting the repository at
`/workspaces/Kareus`:

```bash
docker run -it \
    --gpus=all --ipc=host --network=host --privileged \
    --name=kareus-artifact \
    -v /dev/shm:/dev/shm \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd):/workspaces/Kareus \
    ruofanwu7/kareus-artifact:latest
```

**Configure the cluster** by editing [tests/artifact/env.sh](tests/artifact/env.sh)
once on each node:


| Variable          | Purpose                                                                |
| ----------------- | ---------------------------------------------------------------------- |
| `MASTER_ADDR`     | IP/hostname of node 0 (used by torchrun, and as scp target for node 1) |
| `REMOTE_USER`     | SSH user on node 0 (node 1 uses this to scp results back)              |
| `REMOTE_BASE_DIR` | Path to `tests/artifact/` on node 0                                    |
| `SSH_KEY_PATH`    | (Optional) SSH key node 1 uses to reach node 0                         |


**Export your Hugging Face token** (required by the tokenizer downloads):

```bash
export HF_TOKEN=<your_huggingface_token>
```

**Prepare the dataset.** Kareus evaluates 10 model configurations; we
recommend reviewers start with `CONFIG_MODE=single` (Llama 3.2 3B, TP=8,
MBS=8, SEQ=4096), which is the first row of the configuration table in evaluation. 
The same `CONFIG_MODE` threads through all run scripts.

```bash
# On each node (one-time; takes a few minutes)
CONFIG_MODE=single bash tests/data/prepare_data.sh
```

This produces `tests/data/llama_dataset_text_document.{bin,idx}` plus the 
per-config dataset-index files under `tests/data/`.

---

### 1. Section 6.2.1 — Max-Throughput Comparison

Each method below is run on **both** nodes. `0` and `1` are node ranks;
the script uses `MASTER_ADDR` from `env.sh` for torchrun rendezvous.

#### Running Kareus

**(1) Run Bayesian optimization** (one-time per config; only needs 8 GPUs,
so it runs on a single node):

```bash
CONFIG_MODE=single bash tests/artifact/kareus_run_bayesian.sh
```

This takes about **4 hours on a single 8× A100 node**, or about 2 hours if you
distribute the 4 partition groups across both nodes (run different
partitions on each node — see [tests/bayesian/README.md](tests/bayesian/README.md)).

Walkthrough video: [video](videos/1a_kareus_bayesian.mp4)

**(2) Run the maximum-throughput execution plan** (uses BO results from
step 1; runs on both nodes):

```bash
# On node 0
CONFIG_MODE=single bash tests/artifact/run_kareus.sh 0

# On node 1
CONFIG_MODE=single bash tests/artifact/run_kareus.sh 1
```

Outputs land in:

```
tests/artifact/nemo_experiments/megatron_llama3.2_3b/cp1_tp8_mbs8_seq4096/kareus/
```

Walkthrough video: [video](videos/1b_kareus_run.mp4)

#### Running Baselines

**(1) Megatron-LM** (no frequency control):

```bash
CONFIG_MODE=single bash tests/artifact/run_megatron.sh 0   # node 0
CONFIG_MODE=single bash tests/artifact/run_megatron.sh 1   # node 1
```

Walkthrough video: [video](videos/1c_baseline_megatron.mp4)

**(2) Megatron + Perseus**:

```bash
CONFIG_MODE=single bash tests/artifact/run_perseus.sh 0   # node 0
CONFIG_MODE=single bash tests/artifact/run_perseus.sh 1   # node 1
```

The first run does a per-config GPU-frequency sweep (~2 hours). We also
provide pre-profiled and precomputed results for Megatron+Perseus and
Nanobatching+Perseus (under `tests/perseus/schedules/` and
`tests/nanobatch_perseus/schedules/`) so you can skip the profiling phases:

```bash
SKIP_PROFILING=true CONFIG_MODE=single bash tests/artifact/run_perseus.sh 0
SKIP_PROFILING=true CONFIG_MODE=single bash tests/artifact/run_perseus.sh 1
```

Walkthrough video: [video](videos/1d_baseline_perseus.mp4)

**(3) Nanobatching + Perseus**:

```bash
SKIP_PROFILING=true CONFIG_MODE=single bash tests/artifact/run_nanobatch_perseus.sh 0
SKIP_PROFILING=true CONFIG_MODE=single bash tests/artifact/run_nanobatch_perseus.sh 1
```

Walkthrough video: [video](videos/1e_baseline_nanobatch_perseus.mp4)

#### Comparing results

After all four methods have been run, render the §6.2.1 table:

```bash
python tests/artifact/compare_method.py
```

This writes `tests/artifact/compare_methods_table.png` (and prints a text
table to stdout) showing per-config iteration-time and energy reductions of
Megatron+Perseus, Nanobatching+Perseus, and Kareus relative to the
Megatron-LM baseline.

Walkthrough video: [video](videos/1f_compare_methods.mp4)

---

### 2. Section 6.2.2 — Frontier Improvement

For the frontier comparison we run each method as **10 plans** spanning its
time–energy frontier. All commands below take the same node-rank argument
(`0` or `1`) as in §6.2.1.

#### Running Kareus (frontier)

If you have already run BO in §6.2.1, just flip `FRONTIER=true`:

```bash
# On node 0
FRONTIER=true CONFIG_MODE=single bash tests/artifact/run_kareus.sh 0

# On node 1
FRONTIER=true CONFIG_MODE=single bash tests/artifact/run_kareus.sh 1
```

Frontier outputs land in:

```
tests/artifact/nemo_experiments/megatron_llama3.2_3b/cp1_tp8_mbs8_seq4096/kareus/frontier/pipeline_*/
```

Walkthrough video: [video](videos/2a_kareus_frontier.mp4)

#### Running Baselines (frontier)

**(1) Megatron + Perseus**:

```bash
FRONTIER=true SKIP_PROFILING=true CONFIG_MODE=single bash tests/artifact/run_perseus.sh 0
FRONTIER=true SKIP_PROFILING=true CONFIG_MODE=single bash tests/artifact/run_perseus.sh 1
```

Walkthrough video: [video](videos/2b_perseus_frontier.mp4)

**(2) Nanobatching + Perseus**:

```bash
FRONTIER=true SKIP_PROFILING=true CONFIG_MODE=single bash tests/artifact/run_nanobatch_perseus.sh 0
FRONTIER=true SKIP_PROFILING=true CONFIG_MODE=single bash tests/artifact/run_nanobatch_perseus.sh 1
```

Walkthrough video: [video](videos/2c_nanobatch_perseus_frontier.mp4)

#### Comparing results

```bash
python tests/artifact/compare_method.py --frontier
```

This writes `tests/artifact/compare_methods_frontier_table.png` showing, for
each configuration, the iso-time energy reduction and iso-energy time
reduction of Nanobatching+Perseus and Kareus relative to the Perseus
frontier (10 points each).

Walkthrough video: [video](videos/2d_compare_frontier.mp4)

---

## Output and log conventions

Per-method run outputs (Zeus monitor files, NeMo logs, lowtime solver
artefacts) are collected under:

```
tests/artifact/nemo_experiments/<model>/<config_tag>/<method>/
```

with `<config_tag>` = `cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}` and `<method>` ∈
`{megatron, nanobatch, perseus, nanobatch_perseus, kareus}`. Frontier runs
add a `frontier/pipeline_*/` subdirectory per plan. Per-run training stdout
is duplicated to `tests/artifact/logs/<model>_<config_tag>_<method>.log`.

For the full configuration table (10 configs total), the complete list of
environment variables, and the `SKIP_PROFILING` schedule-directory layout per
method, see [tests/artifact/README.md](tests/artifact/README.md).