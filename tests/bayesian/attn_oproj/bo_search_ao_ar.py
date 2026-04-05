#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bayesian optimization for Attention OProj fuser (AO-AR, forward) overlap-window
and communication configs.
"""

import os
import sys
import argparse

CUR_DIR = os.path.dirname(__file__)
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)
BO_UTILS_DIR = os.path.join(CUR_DIR, '..')
if BO_UTILS_DIR not in sys.path:
    sys.path.append(BO_UTILS_DIR)
FUSER_DIR = os.path.join(CUR_DIR, '..', '..', 'fuser')
if FUSER_DIR not in sys.path:
    sys.path.append(FUSER_DIR)

from overlap_test_ao_ar import AttentionFuserTest  # noqa: E402
from kareus.megatron.core.extensions.fusers.attn_oproj_fuser import AttnOprojPartitionFuser as PartitionFuser  # noqa: E402
from common import SearchSpace, BOSearchConfig, run_bo_search  # noqa: E402


SEARCH_SPACE = SearchSpace(
    overlap_windows=[(0, 2)],
    sm_values=list(range(3, 31, 3)),
    n_init=36,
    batches=3,
    acq_batch=16,
    master_port=9301,
    explore_fraction=0.2,
    time_fraction=0.2,
)

BO_CONFIG = BOSearchConfig(
    banner="Attention OProj Fuser (AO-AR forward)",
    logs_dir_fn=lambda args: f"logs/ao_ar/{args.model_name}/cp{args.context_parallel_size}-tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}",
    eval_log_filename="eval_results.jsonl",
    world_size_default="tp",
    timing_csv="fwd",
)


class PartitionTestRunner:
    """
    Initializes tensors and AO-AR fuser for forward pass.
    test_config(overlap_window, sm_configs) executes one forward step.
    """
    def __init__(self, args: argparse.Namespace, rank: int, world_size: int) -> None:
        self.args = args
        self.rank = rank
        self.world_size = world_size

        self.test = AttentionFuserTest(args, rank=rank, world_size=world_size)
        (
            self.query_1,
            self.query_2,
            self.allreduce_inputs,
        ) = self.test.create_test_tensors()

        operations = self.test.create_operations(self.allreduce_inputs)
        comp_ops = operations[:-1]
        comm_op = operations[-1]

        self.attention_fuser = PartitionFuser(
            ops=comp_ops,
            comm_ops_fwd=[None, comm_op],
            fuse_ops=False,
            profile_ao_ar=True,
        )

        self.group = self.test.tp_group
        self.FREQ_VALUES = args.freq_values
        self.SM_VALUES = args.sm_values
        self.OVERLAP_WINDOWS = args.overlap_windows

    @property
    def tp_group(self):
        return self.test.tp_group

    def test_config(self, overlap_window, sm_configs):
        return self.attention_fuser(
            query_1=self.query_1,
            query_2=self.query_2,
            comm_key=self.allreduce_inputs,
            comm_value=None,
            comm_overlap_window_ao_ar=overlap_window,
            comm_sm_configs_ao_ar=sm_configs,
        )


if __name__ == "__main__":
    run_bo_search(SEARCH_SPACE, BO_CONFIG, PartitionTestRunner)
