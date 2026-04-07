#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bayesian optimization for Forward AO-AR Partition (TP, ALL_REDUCE)."""

import os
import sys
import argparse

CUR_DIR = os.path.dirname(__file__)
BO_UTILS_DIR = os.path.join(CUR_DIR, '../..')
if BO_UTILS_DIR not in sys.path:
    sys.path.append(BO_UTILS_DIR)

from overlap_test import PartitionTest  # noqa: E402
from common import SearchSpace, BOSearchConfig, run_bo_search  # noqa: E402


SEARCH_SPACE = SearchSpace(
    overlap_windows=[(0, 2)],
    sm_values=list(range(3, 31, 3)),
    n_init=36,
    batches=3,
    acq_batch=16,
    master_port=9304,
    real_fraction=0.4,
    dynamic_fraction=0.2,
    time_fraction=0.2,
)

BO_CONFIG = BOSearchConfig(
    banner="Forward AO-AR Partition (TP, ALL_REDUCE)",
    logs_dir_fn=lambda args: (
        f"logs/{args.model_name}/cp{args.context_parallel_size}"
        f"-tp{args.tensor_parallel_size}-bs{args.batch_size}-seq{args.seq_len}/fwd_ao_ar"
    ),
    eval_log_filename="eval_results.jsonl",
    world_size_default="tp",
    timing_csv="fwd",
)


class PartitionTestRunner:
    def __init__(self, args: argparse.Namespace, rank: int, world_size: int) -> None:
        self.test = PartitionTest(args, rank=rank, world_size=world_size)
        self.group = self.test.tp_group
        self.FREQ_VALUES = args.freq_values
        self.SM_VALUES = args.sm_values
        self.OVERLAP_WINDOWS = args.overlap_windows

    @property
    def tp_group(self):
        return self.test.tp_group

    def test_config(self, overlap_window, sm_configs):
        self.test.test_config(overlap_window, sm_configs)


if __name__ == "__main__":
    run_bo_search(SEARCH_SPACE, BO_CONFIG, PartitionTestRunner)
