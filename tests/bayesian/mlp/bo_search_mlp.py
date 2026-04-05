#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bayesian optimization for MLP fuser (forward) overlap-window and communication configs.
"""

import os
import sys
import argparse

CUR_DIR = os.path.dirname(__file__)
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)
FUSER_DIR = os.path.join(CUR_DIR, '..', '..', 'fuser')
if FUSER_DIR not in sys.path:
    sys.path.append(FUSER_DIR)
BO_UTILS_DIR = os.path.join(CUR_DIR, '..')
if BO_UTILS_DIR not in sys.path:
    sys.path.append(BO_UTILS_DIR)

from overlap_test_mlp import MLPFuserTest  # noqa: E402
from kareus.megatron.core.extensions.fusers.partition_fuser import PartitionFuser  # noqa: E402
from common import SearchSpace, BOSearchConfig, run_bo_search  # noqa: E402


SEARCH_SPACE = SearchSpace(
    overlap_windows=[(0, 6), (2, 6), (4, 6)],
    sm_values=list(range(3, 31, 3)),
    n_init=96,
    batches=4,
    acq_batch=32,
    master_port=9010,
    explore_fraction=0.25,
    time_fraction=0.2,
)

BO_CONFIG = BOSearchConfig(
    banner="MLP Fuser (forward)",
    logs_dir_fn=lambda args: f"logs/{args.model_name}/cp{args.context_parallel_size}-tp{args.tensor_parallel_size}-bs{args.batch_size}-seq{args.seq_len}/forward",
    eval_log_filename="eval_results.jsonl",
    world_size_default="tp",
    timing_csv="fwd",
)


class PartitionTestRunner:
    """
    Initializes tensors and fuser for forward pass.
    test_config(overlap_window, sm_configs) executes one forward step.
    """
    def __init__(self, args: argparse.Namespace, rank: int, world_size: int) -> None:
        self.args = args
        self.rank = rank
        self.world_size = world_size

        self.test = MLPFuserTest(args, rank=rank, world_size=world_size)
        (
            self.hidden_states,
            self.bias,
            self.residual,
            self.allreduce_inputs,
        ) = self.test.create_test_tensors()

        operations = self.test.create_operations(self.allreduce_inputs)
        comp_ops = operations[:-1]
        allreduce_comm_op = operations[-1]

        self.mlp_fuser = PartitionFuser(
            ops=comp_ops,
            comm_op_fwd=allreduce_comm_op,
            fuse_ops=False,
        )

        self.group = self.test.tp_group
        self.FREQ_VALUES = args.freq_values
        self.SM_VALUES = args.sm_values
        self.OVERLAP_WINDOWS = args.overlap_windows

    @property
    def tp_group(self):
        return self.test.tp_group

    def test_config(self, overlap_window, sm_configs):
        return self.mlp_fuser(
            hidden_states=self.hidden_states,
            bias=self.bias,
            residual=self.residual,
            comm_input=self.allreduce_inputs,
            comm_overlap_window=overlap_window,
            comm_sm_configs=sm_configs,
        )


if __name__ == "__main__":
    run_bo_search(SEARCH_SPACE, BO_CONFIG, PartitionTestRunner)
