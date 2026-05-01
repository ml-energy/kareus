# Artifact 16-GPU (2x8 A100) Tests

A set of scripts for evaluating the
**Megatron / Nanobatch / Perseus / Nanobatch+Perseus / Kareus** stack on
**2 nodes x 8 NVIDIA A100 GPUs** (16 GPUs total).

All run scripts and entry-point training
files live in this directory; they reference the per-method python tools in
sibling directories (`../kareus`, `../perseus`, `../nanobatch_perseus`,
`../bayesian`) as needed.

## Configurations

`CONFIG_MODE=full` (default) sweeps these 10 configurations:

| Model         | Parallelism | MBS | Seq  | GBS |
|---------------|-------------|-----|------|-----|
| Llama 3.2 3B  | TP=8        | 8   | 4096 | 64  |
| Llama 3.2 3B  | CP=2 + TP=4 | 8   | 4096 | 64  |
| Llama 3.2 3B  | CP=2 + TP=4 | 8   | 8192 | 64  |
| Llama 3.2 3B  | CP=2 + TP=4 | 16  | 4096 | 128 |
| Qwen 3 1.7B   | TP=8        | 8   | 4096 | 64  |
| Qwen 3 1.7B   | TP=8        | 8   | 8192 | 64  |
| Qwen 3 1.7B   | TP=8        | 16  | 4096 | 128 |
| Qwen 3 1.7B   | CP=2 + TP=4 | 8   | 4096 | 64  |
| Qwen 3 1.7B   | CP=2 + TP=4 | 8   | 8192 | 64  |
| Qwen 3 1.7B   | CP=2 + TP=4 | 16  | 4096 | 128 |

Common: PP=2, #microbatches=8.

`CONFIG_MODE=single` runs only the first row (Llama 3.2 3B, TP=8, MBS=8, SEQ=4096).

## Layout of generated outputs

```
tests/artifact/nemo_experiments/<model>/<config_tag>/<method>/
```

- `<model>` ∈ {`megatron_llama3.2_3b`, `megatron_qwen3_1.7b`}
- `<config_tag>` = `cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}` (e.g. `cp1_tp8_mbs8_seq4096`)
- `<method>` ∈ {`megatron`, `nanobatch`, `perseus`, `nanobatch_perseus`, `kareus`}

Per-run training logs land in `tests/artifact/logs/`, named
`<model>_<config_tag>_<method>.log`.

## Scripts

| Script | What it does | Frequency control |
|---|---|---|
| `run_megatron.sh` | Vanilla Megatron baseline (`megatron_gpt_pretraining.py`) | none |
| `run_nanobatch.sh` | Kareus model with nanobatching, no scheduler (`kareus_gpt_pretraining.py`) | none |
| `run_perseus.sh` | Megatron + Perseus pipeline-frequency optimizer | per-instruction freq via PFO server |
| `run_nanobatch_perseus.sh` | Nanobatching + Perseus | per-instruction freq via PFO server |
| `run_kareus.sh` | Full Kareus stack: BO profiling + Phillips-Dessouky solver + Kareus scheduler + PFO | per-partition freq + schedule |
| `kareus_run_bayesian.sh` | Standalone BO partition profiling driver — prerequisite for `run_kareus.sh`; runs on a single 8-GPU node | n/a |
| `megatron_perseus_run_profiling.sh` | Frequency-sweep helper for `run_perseus.sh` | n/a |
| `nanobatch_perseus_run_profiling.sh` | Frequency-sweep helper for `run_nanobatch_perseus.sh` | n/a |

## Per-cluster setup: `env.sh`

Edit [env.sh](env.sh) once per cluster to set the variables that depend
on your hardware/setup:

| Variable | Purpose |
|---|---|
| `MASTER_ADDR`     | IP/hostname of node 0 (used by torchrun and as the scp target from node 1) |
| `REMOTE_USER`     | SSH user on node 0 (used by node 1 for scp) |
| `REMOTE_BASE_DIR` | Path to this checkout on node 0 (target of scp) |
| `SSH_KEY_PATH`    | (Optional) SSH key node 1 uses to scp results back to node 0. Leave unset to fall back on the default ssh agent / `~/.ssh/config`. |

Every run script and profiling helper sources `env.sh` automatically;
overriding any variable on the command line still wins.  If you delete
`env.sh`, the in-script defaults take over (you'd then have to set
`MASTER_ADDR` explicitly on every invocation).

## Running

Both nodes need this repo and the shared filesystem (or copies that match).
Each method script is invoked once per node:

After filling in `env.sh`:

```bash
# On node 0 (master)
bash tests/artifact/run_megatron.sh 0

# On node 1
bash tests/artifact/run_megatron.sh 1
```

Single-config sanity (Llama 3.2 3B TP=8 MBS=8 SEQ=4096):

```bash
CONFIG_MODE=single bash tests/artifact/run_megatron.sh 0
```

Use the precomputed schedules (skip profiling + optimization) — see
`SKIP_PROFILING` below:

```bash
SKIP_PROFILING=true bash tests/artifact/run_perseus.sh 0
SKIP_PROFILING=true bash tests/artifact/run_nanobatch_perseus.sh 0
SKIP_PROFILING=true bash tests/artifact/run_kareus.sh 0
```

Override an env.sh value just for one run:

```bash
MASTER_ADDR=10.0.0.42 bash tests/artifact/run_megatron.sh 0
```

`run_perseus.sh` and `run_nanobatch_perseus.sh` perform a one-time per-config
GPU-frequency sweep before training and skip it on subsequent runs (via a
`.profiling_complete` marker).  `run_kareus.sh` requires Bayesian profiling
results to already exist under `tests/bayesian/logs/` (run
`kareus_run_bayesian.sh` separately to produce them).

### Bayesian profiling (prerequisite for `run_kareus.sh`)

`kareus_run_bayesian.sh`: a 3-phase orchestration that runs all BO partition + nonpartition profilers
on a single 8-GPU node (no multi-node, no `MASTER_ADDR`).  Outputs land
under `tests/bayesian/logs/<model>/cp${CP}-tp${TP}-bs${BS}-seq${SEQ}/`,
which `run_kareus.sh` then reads via `../kareus/generate_profile_csv.py`.

```bash
# Full sweep:
bash tests/artifact/kareus_run_bayesian.sh

# Single config (matches CONFIG_MODE=single in run_kareus.sh):
CONFIG_MODE=single bash tests/artifact/kareus_run_bayesian.sh
```

Skip this step entirely when running `run_kareus.sh SKIP_PROFILING=true`
with precomputed solutions under `tests/kareus/schedules/`.

### Common environment variables

| Variable | Default | Used by |
|---|---|---|
| `MASTER_ADDR`     | from `env.sh` (required) | all run scripts |
| `MASTER_PORT`     | `6000` | all |
| `PFO_PORT`        | `7787` | `run_perseus.sh`, `run_nanobatch_perseus.sh`, `run_kareus.sh` |
| `CONFIG_MODE`     | `full` | all run scripts + `kareus_run_bayesian.sh` (`full` or `single`) |
| `SKIP_PROFILING`  | `false` | `run_perseus.sh`, `run_nanobatch_perseus.sh`, `run_kareus.sh` |
| `GPU_TYPE`        | `A100` | `run_kareus.sh`, `kareus_run_bayesian.sh` |
| `FREQ_START`      | `1410` | profiling sweeps |
| `FREQ_END`        | `900`  | profiling sweeps |
| `FREQ_STEP`       | `30`   | profiling sweeps |
| `REMOTE_USER`     | from `env.sh` (default `ubuntu`) | node-1 → node-0 scp |
| `REMOTE_BASE_DIR` | from `env.sh` (default `$HOME/workspace/Kareus/tests/artifact`) | node-1 → node-0 scp |
| `SSH_KEY_PATH`    | from `env.sh` (optional; unset → default ssh agent / `~/.ssh/config`) | node-1 → node-0 scp |

The frequency-sweep defaults (1410 → 900 MHz, step 30) and the optimizer's
`--p2p_power=85.0` correspond to the **NVIDIA A100 SXM4 40GB GPU**.  On other GPUs these
need to be adjusted before running the profiling/optimization scripts.

### `SKIP_PROFILING` mode

When `SKIP_PROFILING=true`, `run_perseus.sh`, `run_nanobatch_perseus.sh`,
and `run_kareus.sh` skip the profiling and optimization phases entirely
and instead read precomputed solutions from per-method schedule
directories:

| Script | Schedule directory | Files used |
|---|---|---|
| `run_perseus.sh`           | `tests/perseus/schedules/<model_name>/<config_tag>/`           | `freqs_pipeline_*.py` |
| `run_nanobatch_perseus.sh` | `tests/nanobatch_perseus/schedules/<model_name>/<config_tag>/` | `freqs_pipeline_*.py` |
| `run_kareus.sh`            | `tests/kareus/schedules/<model_name>/<config_tag>/`            | `freqs_pipeline_*.py`, `scheds_pipeline_*.py` |

Where `<model_name>` ∈ {`megatron_llama3.2_3b`, `megatron_qwen3_1.7b`} and
`<config_tag>` = `cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}` (e.g.
`cp1_tp8_mbs8_seq4096`).  If the expected files are missing the script
fails fast.

## Logs

Per-run training logs land in `tests/artifact/logs/`, named
`<model>_<config_tag>_<method>.log` — e.g.
`megatron_llama3.2_3b_cp1_tp8_mbs8_seq4096_kareus.log`.
