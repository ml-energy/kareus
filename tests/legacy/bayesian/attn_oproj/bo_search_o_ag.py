#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bayesian optimization for Attention OProj fuser (O-AG, backward) overlap-window
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

from overlap_test_o_ag import AttentionFuserTest  # noqa: E402
from kareus.megatron.core.extensions.fusers.attn_oproj_fuser import AttnOprojPartitionFuser as PartitionFuser  # noqa: E402
from kareus.megatron.core.extensions.fusers.attn_oproj_fuser import _AttnOprojFuserAutogradFunction as AttnOprojAutogradFunction  # noqa: E402
from common import SearchSpace, BOSearchConfig, run_bo_search  # noqa: E402


SEARCH_SPACE = SearchSpace(
    overlap_windows=[(0, 1)],
    sm_values=list(range(1, 21)),
    n_init=36,
    batches=3,
    acq_batch=16,
    master_port=9305,
    explore_fraction=0.2,
    time_fraction=0.2,
)

BO_CONFIG = BOSearchConfig(
    banner="Attention OProj Fuser (O-AG backward)",
    logs_dir_fn=lambda args: f"logs/o_ag/{args.model_name}/cp{args.context_parallel_size}-tp{args.tensor_parallel_size}-bs{args.batch_size}-seq{args.seq_len}",
    eval_log_filename="eval_results_bwd.jsonl",
    world_size_default="cp",
    timing_csv="bwd",
)


class PartitionTestRunner:
    """
    Initializes tensors and O-AG fuser for backward pass.
    test_config(overlap_window, sm_configs) executes one backward step.
    """
    def __init__(self, args: argparse.Namespace, rank: int, world_size: int) -> None:
        self.args = args
        self.rank = rank
        self.world_size = world_size

        self.test = AttentionFuserTest(args, rank=rank, world_size=world_size)
        (
            self.query_1,
            self.query_2,
            self.allgather_key,
            self.allgather_value,
            self.allreduce_inputs,
        ) = self.test.create_test_tensors()

        (
            self.output_grad_1,
            self.output_grad_2,
            self.bias_grad_1,
            self.bias_grad_2,
        ) = self.test.create_gradient_tensors()

        operations = self.test.create_operations()
        comp_ops = operations[:-1]
        comm_op = operations[-1]

        self.attention_fuser = PartitionFuser(
            ops=comp_ops,
            comm_ops_fwd=[None, None],
            comm_ops_bwd=[None, None, comm_op, None],
            fuse_ops=False,
            profile_o_ag=True,
        )

        self.out_1 = None
        self.out_2 = None
        self.bias_1 = None
        self.bias_2 = None
        self.func_ctx = None

        self.group = self.test.cp_group
        self.FREQ_VALUES = args.freq_values
        self.SM_VALUES = args.sm_values
        self.OVERLAP_WINDOWS = args.overlap_windows

    @property
    def tp_group(self):
        return self.group

    def test_config(self, overlap_window, sm_configs):
        if self.func_ctx is None:
            self.out_1, self.out_2, self.bias_1, self.bias_2, self.func_ctx = self.attention_fuser(
                query_1=self.query_1,
                query_2=self.query_2,
                comm_key=self.allgather_key,
                comm_value=self.allgather_value,
                comm_overlap_window_o_ag=overlap_window,
                comm_sm_configs_o_ag=sm_configs,
            )
        AttnOprojAutogradFunction.backward(
            self.func_ctx,
            self.output_grad_1,
            self.output_grad_2,
            self.bias_grad_1,
            self.bias_grad_2,
        )

    def clean(self):
        self.out_1 = None
        self.out_2 = None
        self.bias_1 = None
        self.bias_2 = None
        self.func_ctx = None


if __name__ == "__main__":
    run_bo_search(SEARCH_SPACE, BO_CONFIG, PartitionTestRunner)
