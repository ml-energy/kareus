# tests/bayesian/ — Per-partition Bayesian-optimization profilers

This directory contains the Bayesian-optimization (BO) profilers that generate
the per-partition time-energy frontiers used by Kareus. Each partition defined
in [`kareus/megatron/core/partitions/`](../../kareus/megatron/core/partitions/)
has a dedicated profiler that explores GPU frequency, SM allocation, and launch
timing to approximate that partition's time-energy Pareto frontier.

## Layout

| Area | Purpose |
| ---- | ------- |
| `common/` | Shared BO runtime: model shapes, search-space encoding, surrogate/acquisition logic, hardware measurement, and the bridge into Kareus partition execution. |
| `partitions/` | Per-partition profilers aligned with the Kareus partition names. Each profiler defines the workload and search space for one forward or backward partition and records its measured frontier. |
| `nonpartition/` | Profiles transformer-block work outside the overlap scheduler so the end-to-end optimizer also has costs for the non-overlapped phases. |
| `logs/` | Stores BO and nonpartition run artifacts consumed by CSV generation and schedule optimization. |

## 3-phase orchestration

[tests/artifact/kareus_run_bayesian.sh](../artifact/kareus_run_bayesian.sh)
launches the per-partition profilers across an 8-GPU node, grouping jobs by
their parallelism requirements:

1. **Phase 1 — TP-only** runs sequential 8-GPU jobs for `fwd_attn` and
   `fwd_mlp` at `TP=8`, plus the `nonpartition` pre/post profile.
2. **Phase 2 — CP+TP TP-side partitions** runs two 4-GPU jobs at a time for
   `fwd_qkv_ar`, `fwd_ao_ar`, `fwd_mlp`, `bwd_qkv_ar`, `bwd_o_ar`, and
   `bwd_mlp` under `CP=2 + TP=4`.
3. **Phase 3 — CP+TP CP-side partitions** runs four 2-GPU jobs at a time for
   `fwd_qkv_ag`, `fwd_ao_ag`, `bwd_qkv_rs`, `bwd_a_rs`, `bwd_a_ag`, and
   `bwd_o_ag` under `CP=2 + TP=4`.

On A100 SXM4 40GB, a single 8-GPU node completes the full three-phase sweep in
roughly **4 hours**. Splitting the partition jobs across two nodes, with each
node pinned to a different subset of partitions, reduces the runtime to about
2 hours.

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
