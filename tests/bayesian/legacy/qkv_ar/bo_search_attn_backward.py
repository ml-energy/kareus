#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bayesian optimization for QKV attention fuser (backward) overlap-window and communication configs.
"""

import os
import sys
import argparse
import torch

CUR_DIR = os.path.dirname(__file__)
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)
FUSER_DIR = os.path.join(CUR_DIR, '..', '..', 'fuser')
if FUSER_DIR not in sys.path:
    sys.path.append(FUSER_DIR)
BO_UTILS_DIR = os.path.join(CUR_DIR, '..')
if BO_UTILS_DIR not in sys.path:
    sys.path.append(BO_UTILS_DIR)

from overlap_test_qkv_ar_backward import AttentionFuserTest  # noqa: E402
from kareus.megatron.core.extensions.fusers.qkv_fuser import QKVPartitionFuser as PartitionFuser  # noqa: E402
from common import SearchSpace, BOSearchConfig, run_bo_search  # noqa: E402


SEARCH_SPACE = SearchSpace(
    overlap_windows=[(0, 5), (2, 5)],
    sm_values=list(range(3, 31, 3)),
    n_init=48,
    batches=4,
    acq_batch=16,
    master_port=9005,
    explore_fraction=0.2,
    time_fraction=0.2,
)

BO_CONFIG = BOSearchConfig(
    banner="QKV-AR Fuser (backward)",
    logs_dir_fn=lambda args: f"logs/{args.model_name}/cp{args.context_parallel_size}-tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/backward",
    eval_log_filename="eval_results_bwd.jsonl",
    world_size_default="tp",
    timing_csv="bwd",
)


class PartitionTestRunner:
    """
    Initializes tensors and QKV fuser for backward pass.
    test_config(overlap_window, sm_configs) performs one backward step.
    """
    def __init__(self, args: argparse.Namespace, rank: int, world_size: int) -> None:
        self.args = args
        self.rank = rank
        self.world_size = world_size

        self.test = AttentionFuserTest(args, rank=rank, world_size=world_size)
        (
            self.hidden_states,
            self.bias,
            self.residual,
            self.rotary_pos_emb,
            self.attention_mask,
            self.allreduce_inputs,
        ) = self.test.create_test_tensors()

        operations = self.test.create_operations(self.allreduce_inputs)
        comp_ops = operations[:-1]
        comm_op = operations[-1]

        self.attention_fuser = PartitionFuser(
            ops=comp_ops,
            comm_op_bwd=comm_op,
            fuse_ops=False,
        )

        self.query_grad, self.key_grad, self.value_grad, self.residual_grad, self.allreduce_input_grad = (
            self.test.create_gradient_tensors()
        )

        self.query = None
        self.key = None
        self.value = None
        self.residual_out = None
        self.allreduce_output = None

        self.group = self.test.tp_group
        self.FREQ_VALUES = args.freq_values
        self.SM_VALUES = args.sm_values
        self.OVERLAP_WINDOWS = args.overlap_windows

    @property
    def tp_group(self):
        return self.test.tp_group

    def test_config(self, overlap_window, sm_configs):
        if self.query is None:
            self.query, self.key, self.value, self.residual_out, self.allreduce_output = self.attention_fuser(
                hidden_states=self.hidden_states,
                bias=self.bias,
                residual=self.residual,
                rotary_pos_emb=self.rotary_pos_emb,
                attention_mask=self.attention_mask,
                comm_input=self.allreduce_inputs,
                comm_overlap_window_backward=overlap_window,
                comm_sm_configs_backward=sm_configs,
            )
        _ = torch.autograd.grad(
            outputs=[self.query, self.key, self.value, self.residual_out, self.allreduce_output],
            inputs=[self.hidden_states, self.residual, self.allreduce_inputs],
            grad_outputs=[self.query_grad, self.key_grad, self.value_grad, self.residual_grad, self.allreduce_input_grad],
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )

    def clean(self):
        self.query = None
        self.key = None
        self.value = None
        self.residual_out = None
        self.allreduce_output = None


if __name__ == "__main__":
    run_bo_search(SEARCH_SPACE, BO_CONFIG, PartitionTestRunner)
