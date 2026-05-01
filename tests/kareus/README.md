# tests/kareus/ — Profile compose + Phillips-Dessouky solver + precomputed schedules

This directory holds Phase 2 of the Kareus workflow (composition + solving)
and the precomputed solutions used when reviewers want to skip BO entirely.

## Files

| File                       | What it does                                                                  |
| -------------------------- | ----------------------------------------------------------------------------- |
| `generate_profile_csv.py`  | Reads per-partition BO logs from [tests/bayesian/logs/](../bayesian/) plus the prepost (`nonpartition`) energy CSVs and emits a single per-config `profile_<model>_cp${CP}_tp${TP}_bs${MBS}_seq${SEQ}.csv` consumed by the solver. |
| `run_optimization.py`      | Phillips-Dessouky time-cost-tradeoff solver. Takes a profile CSV and writes one matched pair `freqs_pipeline_<iter>.py` (per-instruction GPU clocks for the Perseus PFO server) + `scheds_pipeline_<iter>.py` (Kareus pipeline-comm schedule) per Pareto point. |
| `schedules/`               | Shipped, precomputed solver outputs — used when `SKIP_PROFILING=true`.        |

Both scripts are invoked automatically by [tests/artifact/run_kareus.sh](../artifact/run_kareus.sh)
when `SKIP_PROFILING=false`. They can also be run by hand for debugging:

```bash
# 1) Compose BO logs + prepost CSVs into the solver input
python tests/kareus/generate_profile_csv.py \
    --bayesian_profile_dir=tests/bayesian \
    --prepost_profile_dir=tests/bayesian \
    --model_name=llama3.2_3b \
    --context_parallel_size=1 --tensor_parallel_size=8 --pipeline_parallel_size=2 \
    --batch_size=8 --seq_len=4096 --gpu_type=A100

# 2) Run the Phillips-Dessouky solver
python tests/kareus/run_optimization.py \
    --inst_profile=profile_llama3.2_3b_cp1_tp8_bs8_seq4096.csv \
    --output_dir=lowtime_out \
    --num_mbs=8 --num_stages=2
```

The `--p2p_power` value used by the solver depends on the GPU; the default
(85 W) is for A100 SXM4 40GB. Other GPUs require a different value — see
[tests/artifact/README.md](../artifact/README.md) for the A100 vs. A40 settings.

## Precomputed `schedules/` layout

The solver outputs are reproduced and shipped under `schedules/` so reviewers
can run Kareus end-to-end without ever doing BO or solving:

```
tests/kareus/schedules/<model_name>/<config_tag>/
    freqs_pipeline_<iter>.py        # max-throughput plan (used when FRONTIER=false)
    scheds_pipeline_<iter>.py
    frontier/
        freqs_pipeline_<iter>.py    # 10 plans spanning the time-energy frontier
        scheds_pipeline_<iter>.py   # (used when FRONTIER=true)
        ...
```

with `<model_name>` ∈ `{megatron_llama3.2_3b, megatron_qwen3_1.7b}` and
`<config_tag>` = `cp${CP}_tp${TP}_mbs${MBS}_seq${SEQ}`. Each frontier
directory ships exactly **10** matched (`freqs`, `scheds`) pairs; the
`<iter>` suffix is the solver iteration index (also a rough proxy for how
much energy was spent vs. throughput sacrificed — lower iters are
higher-throughput, higher iters are lower-energy).

[run_kareus.sh](../artifact/run_kareus.sh) selects:

- the **highest-iter** matched pair from `<config_tag>/` when `SKIP_PROFILING=true` and `FRONTIER=false`;
- **all 10 pairs** under `<config_tag>/frontier/` when `SKIP_PROFILING=true` and `FRONTIER=true`.

Per-method schedule directories for the other methods live in sibling
locations: [tests/perseus/schedules/](../perseus/schedules/) and
[tests/nanobatch_perseus/schedules/](../nanobatch_perseus/schedules/). They
follow the same `<model_name>/<config_tag>/[frontier/]` layout but contain
only `freqs_pipeline_*.py` (no `scheds_pipeline_*.py`, since those baselines
do not run the Kareus pipeline-comm scheduler).
