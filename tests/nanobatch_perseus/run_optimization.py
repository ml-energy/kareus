"""Pipeline frequency optimization using the Perseus solver.

Finds the Pareto-optimal trade-off between iteration time and energy
consumption by assigning per-operation GPU frequencies to a pipeline-parallel
training schedule.

The script operates in three phases:

1. **Execution options** -- Reads a profile.csv (produced by
   generate_profile_csv.py) containing per-stage, per-frequency time/energy
   measurements.  For each (stage, instruction) pair an ExponentialModel cost
   model is fitted to the candidate execution options.

2. **DAG construction** -- Builds a Synchronous1F1B pipeline schedule as a
   NetworkX DAG with intra-stage sequential edges and inter-stage
   forward/backward data-dependency edges, plus virtual source/sink nodes.

3. **Phillips-Dessouky optimisation** -- Iteratively finds the minimum-cost
   way to shorten the critical path by one time unit, sweeping out the full
   Pareto frontier.  Each iteration emits a ``freqs_pipeline_{iter:05d}.py``
   file with the chosen GPU frequency for every forward/backward operation on
   every stage, and periodically saves pipeline schedule visualisations.

Usage:
    python run_optimization.py --inst_profile <csv> --output_dir <dir> [options]
"""

from __future__ import annotations

import itertools
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Type

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import tyro

from lowtime.operation import (
    CandidateExecutionOptions,
    ExecutionOption,
    OperationSpec,
)
from lowtime.cost_model import ExponentialModel
from lowtime.perseus.instruction import (
    Backward,
    Forward,
    Instruction,
    backward_dep,
    forward_dep,
)
from lowtime.solver import PhillipsDessouky
from lowtime.graph_utils import DependencyResolver, add_sink_node, add_source_node
from lowtime.perseus.schedule import Synchronous1F1B
from lowtime.perseus.visualizer import ANNOTATE_ARGS, LINE_ARGS, PipelineVisualizer

logger = logging.getLogger()


@dataclass
class Args:
    # Path to profile.csv from generate_profile_csv.py
    inst_profile: str
    # Directory to write freqs_pipeline_*.py solutions and plots
    output_dir: str
    # Number of microbatches
    num_mbs: int = 8
    # Number of pipeline stages
    num_stages: int = 2
    # GPU power (W) while blocking on P2P communication
    p2p_power: float = 85.0
    # Draw pipeline state every N iterations
    plot_interval: int = 100
    # Unit of time reduction per solver iteration (seconds)
    unit_time: float = 0.001
    # Noise factor for soft Pareto frontier filtering
    noise_factor: float = 0.95


def main(args: Args) -> None:
    """Perseus time-cost tradeoff optimisation."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "job.log"

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
    )
    logger.info("Arguments: %s", args)

    inst_df = pd.read_csv(args.inst_profile)

    ####################
    # Execution Option #
    ####################
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
                options.append(
                    ExecutionOption[int](
                        real_time=row["time"],
                        unit_time=args.unit_time,
                        cost=row["energy"],
                        knob=int(row["frequency"]),
                    )
                )

            for option in options:
                option.cost -= args.p2p_power * option.quant_time * option.unit_time

            cand_options = CandidateExecutionOptions[int](options=options)
            if len(cand_options.options) <= 3:
                cand_options = CandidateExecutionOptions[int](
                    options=options, noise_factor=args.noise_factor
                )

            model = ExponentialModel(cand_options, search_strategy="best")

            fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
            model.draw(ax, cand_options)
            fig.savefig(f"{output_dir}/{inst_name.lower()}_{stage_id}.png")
            plt.close(fig)

            op_spec = OperationSpec[int](options=cand_options, cost_model=model)
            op_spec_map[stage_id][instruction] = op_spec

    ####################
    # DAG construction #
    ####################
    dag = nx.DiGraph()

    node_id = 2  # 0 = source, 1 = sink
    instructions: list[list[Instruction]] = []
    for stage_id in range(args.num_stages):
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

        for n1, n2 in zip(stage_node_ids, stage_node_ids[1:]):
            dag.add_edge(n1, n2)

    insts = dag.nodes(data=True)
    resolver = DependencyResolver(
        dependency_rules=[forward_dep, backward_dep],
        node_type=Instruction,
    )
    for (id1, data1), (id2, data2) in itertools.product(insts, insts):
        if resolver.is_dependent(data1["op"], data2["op"]):
            dag.add_edge(id1, id2)

    add_source_node(dag, 0)
    add_sink_node(dag, 1)
    dag.graph["source_node"] = 0
    dag.graph["sink_node"] = 1

    ###################################
    # Time-cost tradeoff optimisation #
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

        ax.set_xlim(0.0, xlim)
        ax.set_title(f"Iteration {iteration:4d}")
        fig.savefig(f"{output_dir}/pipeline_{iteration:05d}.png")
        plt.close(fig)

    solver = PhillipsDessouky(dag)

    draw_xlim = None
    iteration = 0
    for iteration, result in enumerate(solver.run()):
        if iteration % args.plot_interval == 0:
            if draw_xlim is None:
                draw_xlim = int(result.real_time) + 1
            draw(dag, iteration, draw_xlim)

        with open(output_dir / f"freqs_pipeline_{iteration:05d}.py", "w") as f:
            f.write("[\n")
            for stage_id, stage_insts in enumerate(instructions):
                stage_freq: list[tuple[str, int]] = []
                for inst in stage_insts:
                    stage_freq.append((type(inst).__name__.lower(), inst.assigned_knob))
                f.write(f"{stage_freq},\n")
            f.write("]\n")

            iter_str = f"# Iteration {iteration} "
            real_cost = result.cost + args.num_stages * result.real_time * args.p2p_power
            f.write(iter_str + f"cost change: {result.cost_change}\n")
            f.write(iter_str + f"total cost: {result.cost}\n")
            f.write(iter_str + f"total cost with P2P: {real_cost}\n")

    assert draw_xlim is not None
    draw(dag, iteration, draw_xlim)


if __name__ == "__main__":
    args = tyro.cli(Args)

    start_time = time.time()
    main(args)
    logger.info("Total time: %.2fs", time.time() - start_time)
