# tests/bayesian/ — Per-partition Bayesian-optimization profilers

This directory holds the Bayesian-optimization (BO) profilers that produce
the per-partition time–energy frontiers consumed by Kareus. Each partition
identified in [`kareus/megatron/core/partitions/`](../../kareus/megatron/core/partitions/)
has its own profiler that sweeps GPU clocks (and a small number of other
knobs) under BO to map out the Pareto frontier of latency vs. energy for that
partition only.

The profilers are driven by [tests/artifact/kareus_run_bayesian.sh](../artifact/kareus_run_bayesian.sh)
(or, for the toy 4-GPU setting, by `tests/toy/kareus_run_bayesian.sh`).
Reviewers normally do **not** invoke the per-partition scripts directly;
they invoke the orchestration script and let it dispatch partitions onto the
appropriate GPU subsets.

## Layout

```
tests/bayesian/
├── common/              # Shared BO machinery
│   ├── runner.py            # BO loop (botorch surrogate + acquisition)
│   ├── surrogates.py        # Time/energy GP surrogates
│   ├── partition_executor.py # Replays the partition kernel + measures with Zeus
│   ├── model_config.py      # Llama 3.2 3B / Qwen 3 1.7B configs
│   ├── hardware.py, encoding.py, orchestration.py
│   └── __init__.py
├── partitions/          # One subdirectory per partition; each contains bo_search.py
│   ├── fwd_qkv_ag/, fwd_qkv_ar/, fwd_attn/, fwd_ao_ag/, fwd_ao_ar/, fwd_mlp/
│   └── bwd_qkv_rs/, bwd_qkv_ar/, bwd_attn/, bwd_a_rs/, bwd_a_ag/, bwd_o_ag/, bwd_o_ar/, bwd_mlp/
├── nonpartition/
│   └── profile_nonpartition.py  # Pre/post (non-overlapped) phase profiler
├── test_all_configs.sh  # Standalone runner for the toy 4-GPU configs
└── logs/                # Output (created on first run)
```

## 3-phase orchestration

[tests/artifact/kareus_run_bayesian.sh](../artifact/kareus_run_bayesian.sh)
fans the per-partition jobs out across an 8-GPU node according to the
parallelism requirements of each partition:

1. **Phase 1 — TP-only** (8 GPUs, sequential). Profiles `fwd_attn`, `fwd_mlp`
   for `TP=8` configs (Llama 3.2 3B TP=8 SEQ=4096; Qwen 3 1.7B TP=8 at all
   three batch/sequence settings) plus the `nonpartition` prepost profile.
2. **Phase 2 — CP+TP TP-side partitions** (4 GPUs each, 2 in parallel).
   Profiles `fwd_qkv_ar`, `fwd_ao_ar`, `fwd_mlp`, `bwd_qkv_ar`, `bwd_o_ar`,
   `bwd_mlp` for the `CP=2 + TP=4` configs.
3. **Phase 3 — CP+TP CP-side partitions** (2 GPUs each, 4 in parallel).
   Profiles `fwd_qkv_ag`, `fwd_ao_ag`, `bwd_qkv_rs`, `bwd_a_rs`, `bwd_a_ag`,
   `bwd_o_ag` for the `CP=2 + TP=4` configs.

A single 8-GPU node completes the full 3-phase sweep in roughly **4 hours**
on A100 SXM4 40GB. Splitting the per-partition jobs across both nodes
(by editing the script to pin different partitions to different nodes) cuts
this to about 2 hours.

`CONFIG_MODE=single` skips Phases 2/3 and only profiles the
`nonpartition` step for Llama 3.2 3B / TP=8 / MBS=8 / SEQ=4096 (the BO
results for the partitions themselves are shipped under
[`tests/kareus/schedules/`](../kareus/schedules/) so you can use
`SKIP_PROFILING=true` instead).

## Output layout

Each per-partition profiler writes its tee'd stdout under:

```
tests/bayesian/logs/<model>/cp${CP}-tp${TP}-bs${BS}-seq${SEQ}/<partition>/bo_<tag>.log
```

For example, after `CONFIG_MODE=full bash tests/artifact/kareus_run_bayesian.sh`:

```
tests/bayesian/logs/llama3.2_3b/cp1-tp8-bs8-seq4096/fwd_attn/bo_llama3.2_3b_tp8_bs8_seq4096.log
tests/bayesian/logs/llama3.2_3b/cp1-tp8-bs8-seq4096/nonpartition/nonpartition_llama3.2_3b_cp1_tp8_bs8_seq4096.log
...
```

These logs (one per BO run) are exactly what
[tests/kareus/generate_profile_csv.py](../kareus/generate_profile_csv.py)
parses to build the Phillips-Dessouky solver's input CSV. After parsing, the
solver in [tests/kareus/run_optimization.py](../kareus/run_optimization.py)
emits the matched pair of `freqs_pipeline_*.py` (per-instruction frequencies)
and `scheds_pipeline_*.py` (kareus pipeline-comm schedule) consumed by
[tests/artifact/run_kareus.sh](../artifact/run_kareus.sh).

## Skipping BO entirely

Reviewers who only want to verify the end-to-end run can skip this entire
directory by passing `SKIP_PROFILING=true` to `run_kareus.sh`; the run script
will then load the precomputed schedules from
[`tests/kareus/schedules/`](../kareus/schedules/) instead. See the top-level
[README.md](../../README.md) for details.
