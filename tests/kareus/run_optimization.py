"""Example script of running pipeline frequency optimization."""

from __future__ import annotations

import time
import itertools
import logging
from pathlib import Path
from typing import Type
from collections import defaultdict
from dataclasses import dataclass
import sys
import os
FUSER_DIR = os.path.join(os.path.dirname(__file__), '..', 'fuser')
if FUSER_DIR not in sys.path:
    sys.path.append(FUSER_DIR)
from common_config import FuserTestConfig  # noqa: E402

import tyro
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from lowtime.operation import (
    CandidateExecutionOptions,
    OperationSpec,
    ExecutionOption,
)
from lowtime.cost_model import ExponentialModel
from lowtime.perseus.instruction import (
    Instruction,
    Forward,
    Backward,
    forward_dep,
    backward_dep,
)
from lowtime.solver import PhillipsDessouky
from lowtime.graph_utils import add_source_node, add_sink_node, DependencyResolver
from lowtime.perseus.schedule import Synchronous1F1B
from lowtime.perseus.visualizer import PipelineVisualizer, ANNOTATE_ARGS, LINE_ARGS

logger = logging.getLogger()


@dataclass
class Args:
    # Path to instruction profile results
    inst_profile: str = "profile_llama3.2_3b_cp1_tp8_bs8_seq4096.csv"
    # Directory to output results
    output_dir: str = "llama3.2_3b/cp1_tp8_bs8_seq4096/perseus_results2"
    # Number of microbatchs
    num_mbs: int = FuserTestConfig.DEFAULT_NUM_MICROBATCHES
    # Number of stages
    num_stages: int = FuserTestConfig.DEFAULT_STAGES
    # GPU power consumption while blocking on P2P communication, in Watts
    p2p_power: float = FuserTestConfig.get_p2p_power(FuserTestConfig.GPU_TYPE)
    # Interval to draw the state of the pipeline
    plot_interval: int = 100
    # The unit of reduction for each iteration, in seconds
    unit_time: float = 0.001
    # Noise factor for soft Pareto frontier filtering
    noise_factor: float = 0.95
    # Whether to use activation checkpointing parsing (backward has recompute-forward configs)
    use_activation_checkpointing: bool = True


def main(args: Args) -> None:
    """Perseus time-cost tradeoff optimization."""
    # Setup logging and output.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "job.log"

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
    )
    logger.info("Arguments: %s", args)

    # Instruction offline profiling results.
    inst_df = pd.read_csv(args.inst_profile)

    ####################
    # Execution Option #
    ####################
    # Construct the OperationSpec object of each pipeline instruction in each stage.
    op_spec_map: dict[int, dict[Type[Instruction], OperationSpec]] = defaultdict(dict)
    for instruction in [Forward, Backward]:
        inst_name = instruction.__name__
        for stage_id in range(args.num_stages):
            logger.info("Processing %s stage %d", inst_name, stage_id)
            options = []
            _df = inst_df.query(
                f"stage == {stage_id} and instruction == '{inst_name.lower()}'"
            )
            for _, row in _df.iterrows():
                row = row.to_dict()
                if instruction is Backward and args.use_activation_checkpointing:
                    # Backward knob includes recompute-forward and backward configs due to activation checkpointing
                    options.append(
                        ExecutionOption[tuple[int, str, str, str, str]](
                            real_time=row["time"],
                            unit_time=args.unit_time,
                            cost=row["energy"],
                            knob=(
                                int(row["frequency"]),
                                str(row.get("recompute_attention_configs", "")),
                                str(row.get("recompute_mlp_configs", "")),
                                str(row.get("attention_configs", "")),
                                str(row.get("mlp_configs", "")),
                            ),
                        )
                    )
                else:
                    options.append(
                        ExecutionOption[tuple[int, str, str]](
                            real_time=row["time"],
                            unit_time=args.unit_time,
                            cost=row["energy"],
                            knob=(
                                int(row["frequency"]),
                                str(row.get("attention_configs", "")),
                                str(row.get("mlp_configs", "")),
                            ),
                        )
                    )

            # Map the cost to be effective computation energy.
            # Everything from now on is in terms of effective energy.
            for option in options:
                option.cost -= args.p2p_power * option.quant_time * option.unit_time

            # Get the Pareto frontier, quantize time, and deduplicate time.
            if instruction is Backward and args.use_activation_checkpointing:
                cand_options = CandidateExecutionOptions[tuple[int, str, str, str, str]](options=options)
                if len(cand_options.options) <= 3:
                    cand_options = CandidateExecutionOptions[tuple[int, str, str, str, str]](
                        options=options, noise_factor=args.noise_factor
                    )
            else:
                cand_options = CandidateExecutionOptions[tuple[int, str, str]](options=options)
                if len(cand_options.options) <= 3:
                    cand_options = CandidateExecutionOptions[tuple[int, str, str]](
                        options=options, noise_factor=args.noise_factor
                    )

            # Fit the cost model.
            model = ExponentialModel(cand_options, search_strategy="best")

            # Draw the cost model.
            fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
            model.draw(ax, cand_options)
            fig.savefig(f"{output_dir}/{inst_name.lower()}_{stage_id}.png")

            # Initialize the operation spec.
            if instruction is Backward and args.use_activation_checkpointing:
                op_spec = OperationSpec[tuple[int, str, str, str, str]](options=cand_options, cost_model=model)
            else:
                op_spec = OperationSpec[tuple[int, str, str]](options=cand_options, cost_model=model)
            op_spec_map[stage_id][instruction] = op_spec

    ####################
    # DAG construction #
    ####################
    dag = nx.DiGraph()

    # Generate and add all instructions to the DAG.
    # Reserve 0 for dummy source and 1 for dummy sink.
    node_id = 2
    instructions: list[list[Instruction]] = []
    for stage_id in range(args.num_stages):
        # Generate instructions for each stage.
        stage_insts: list[Instruction] = []
        stage_node_ids: list[int] = []
        for inst in Synchronous1F1B(
            num_stages=args.num_stages,
            num_micro_batches=args.num_mbs,
            stage_id=stage_id,
            operation_spec_map=op_spec_map[stage_id],
        ):
            dag.add_node(node_id, op=inst)
            stage_insts.append(inst)
            stage_node_ids.append(node_id)
            node_id += 1
        instructions.append(stage_insts)

        # Add dependencies between adjacent instructions in the same stage.
        for node_id1, node_id2 in zip(stage_node_ids, stage_node_ids[1:]):
            dag.add_edge(node_id1, node_id2)

    # Add dependencies between dependent pipeline instructions.
    insts = dag.nodes(data=True)
    resolver = DependencyResolver(
        dependency_rules=[forward_dep, backward_dep],
        node_type=Instruction,
    )
    for (id1, data1), (id2, data2) in itertools.product(insts, insts):
        if resolver.is_dependent(data1["op"], data2["op"]):
            dag.add_edge(id1, id2)

    # Add source and sink nodes.
    add_source_node(dag, 0)
    add_sink_node(dag, 1)
    dag.graph["source_node"] = 0
    dag.graph["sink_node"] = 1

    ###################################
    # Time-cost tradeoff optimization #
    ###################################
    def annotation_hook(inst: Instruction) -> str:
        return f"{type(inst).__name__[0]}\n{inst.micro_batch_id}"

    def draw(dag: nx.DiGraph, iteration: int, xlim: int) -> None:
        ANNOTATE_ARGS[Forward]["fontsize"] = 11.0
        ANNOTATE_ARGS[Backward]["fontsize"] = 11.0
        ANNOTATE_ARGS[Forward]["color"] = "black"
        ANNOTATE_ARGS[Backward]["color"] = "black"
        LINE_ARGS["linewidth"] = 3.0

        fig, ax = plt.subplots(figsize=(args.num_mbs, 4), tight_layout=True)

        vis = PipelineVisualizer(dag)
        vis.draw(
            ax,
            draw_time_axis=True,
            p2p_power=args.p2p_power,
            annotation_hook=annotation_hook,
            power_color="RdBu_r",
            normalizer_range=(-200, 550),
        )
        vis.draw_critical_path(ax)

        # Fix xlim so that we can visually see the pipeline width shrink.
        ax.set_xlim(0.0, xlim)
        ax.set_title(f"Iteration {iteration:4d}")
        fig.savefig(f"{output_dir}/pipeline_{iteration:05d}.png")
        plt.close(fig)

    solver = PhillipsDessouky(dag)

    draw_xlim = None
    iteration = 0
    for iteration, result in enumerate(solver.run()):
        # Maybe draw the pipeline.
        if iteration % args.plot_interval == 0:
            if draw_xlim is None:
                draw_xlim = int(result.real_time) + 1
            draw(dag, iteration, draw_xlim)

        # Write frequency assignments and separate configs per iteration.
        f_freqs = open(output_dir / f"freqs_pipeline_{iteration:05d}.py", "w")
        f_cfgs = open(output_dir / f"scheds_pipeline_{iteration:05d}.py", "w")

        # Write top-level lists
        f_freqs.write("[\n")
        f_cfgs.write("[\n")

        for stage_id, stage_insts in enumerate(instructions):
            stage_freq = []
            # For configs, group by microbatch id and store (forward_tuple, backward_tuple)
            # forward tuple: ("forward", attn_cfg, mlp_cfg)
            # backward tuple: if AC on -> ("backward", rec_attn_cfg, rec_mlp_cfg, bwd_attn_cfg, bwd_mlp_cfg)
            #                 else     -> ("backward", bwd_attn_cfg, bwd_mlp_cfg)
            if args.use_activation_checkpointing:
                # Collect configs directly in 1F1B order (per instruction)
                stage_cfgs_1f1b: list[tuple] = []
            else:
                stage_cfgs_by_mb: dict[int, dict[str, tuple]] = defaultdict(dict)
            for inst in stage_insts:
                inst_name = type(inst).__name__.lower()
                knob = inst.assigned_knob
                # Expect knob to be a tuple: forward => (frequency, attention_configs, mlp_configs)
                # backward => (frequency, rec_attn_cfg, rec_mlp_cfg, bwd_attn_cfg, bwd_mlp_cfg)
                if args.use_activation_checkpointing:
                    if isinstance(knob, tuple):
                        freq = knob[0] if len(knob) > 0 else None
                        if inst_name == "backward" and len(knob) >= 5:
                            rec_attn_cfg = knob[1]
                            rec_mlp_cfg = knob[2]
                            bwd_attn_cfg = knob[3]
                            bwd_mlp_cfg = knob[4]
                            stage_cfgs_1f1b.append(
                                (
                                    inst_name,
                                    rec_attn_cfg,
                                    rec_mlp_cfg,
                                    bwd_attn_cfg,
                                    bwd_mlp_cfg,
                                )
                            )
                        else:
                            attn_cfg = knob[1] if len(knob) > 1 else ""
                            mlp_cfg = knob[2] if len(knob) > 2 else ""
                            stage_cfgs_1f1b.append(
                                (
                                    inst_name,
                                    attn_cfg,
                                    mlp_cfg,
                                )
                            )
                    else:
                        raise ValueError(f"Invalid knob type: {type(knob)}")
                else:
                    if isinstance(knob, tuple):
                        freq = knob[0] if len(knob) > 0 else None
                        if inst_name == "backward":
                            bwd_attn_cfg = knob[1] if len(knob) > 1 else ""
                            bwd_mlp_cfg = knob[2] if len(knob) > 2 else ""
                            stage_cfgs_by_mb[inst.micro_batch_id][inst_name] = (
                                inst_name,
                                bwd_attn_cfg,
                                bwd_mlp_cfg,
                            )
                        else:
                            attn_cfg = knob[1] if len(knob) > 1 else ""
                            mlp_cfg = knob[2] if len(knob) > 2 else ""
                            stage_cfgs_by_mb[inst.micro_batch_id][inst_name] = (
                                inst_name,
                                attn_cfg,
                                mlp_cfg,
                            )
                    else:
                        freq = knob
                        if inst_name == "backward":
                            stage_cfgs_by_mb[inst.micro_batch_id][inst_name] = (
                                inst_name,
                                "",
                                "",
                            )
                        else:
                            stage_cfgs_by_mb[inst.micro_batch_id][inst_name] = (
                                inst_name,
                                "",
                                "",
                            )
                stage_freq.append((inst_name, freq))

            f_freqs.write(f"{stage_freq},\n")
            if args.use_activation_checkpointing:
                # When using activation checkpointing, also save configs in 1F1B order
                f_cfgs.write(f"{stage_cfgs_1f1b},\n")
            else:
                # Build a list ordered by microbatch id, each entry is (forward_tuple, backward_tuple)
                stage_mb_cfgs: list[tuple[tuple, tuple]] = []
                for mb_id in range(args.num_mbs):
                    mb_cfgs = stage_cfgs_by_mb.get(mb_id, {})
                    fwd_tuple = mb_cfgs.get("forward", ("forward", "", ""))
                    if args.use_activation_checkpointing:
                        bwd_tuple = mb_cfgs.get("backward", ("backward", "", "", "", ""))
                    else:
                        bwd_tuple = mb_cfgs.get("backward", ("backward", "", ""))
                    stage_mb_cfgs.append((fwd_tuple, bwd_tuple))
                f_cfgs.write(f"{stage_mb_cfgs},\n")

        f_freqs.write("]\n")
        f_cfgs.write("]\n")

        # Append iteration cost info to the configs file
        iter_str = f"# Iteration {iteration} "
        real_cost = result.cost + args.num_stages * result.real_time * args.p2p_power
        f_cfgs.write(iter_str + f"cost change: {result.cost_change}\n")
        f_cfgs.write(iter_str + f"total cost: {result.cost}\n")
        f_cfgs.write(iter_str + f"total cost with P2P: {real_cost}\n")
        f_freqs.write(iter_str + f"cost change: {result.cost_change}\n")
        f_freqs.write(iter_str + f"total cost: {result.cost}\n")
        f_freqs.write(iter_str + f"total cost with P2P: {real_cost}\n")

        # Close files
        f_freqs.close()
        f_cfgs.close()

    assert draw_xlim is not None
    draw(dag, iteration, draw_xlim)


if __name__ == "__main__":
    args = tyro.cli(Args)

    start_time = time.time()
    main(args)
    logger.info("Total time: %.2fs", time.time() - start_time)