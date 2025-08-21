"""Draw a training pipeline figure from a profile CSV (no optimization).

Input CSV is expected to have columns like those produced by
`generate_profile_csv_single_dir.py` group output:

- rank: integer stage id (0-indexed)
- name: 'forward' or 'backward'
- avg_time: seconds
- avg_energy: joules (for the interval)

We construct a resource- and dependency-constrained schedule approximating
1F1B by scheduling all forward and backward tasks with precedence constraints
and per-stage serialization. The figure is a Gantt chart with color encoding
by average power (avg_energy / avg_time).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class TaskKey:
    stage: int
    microbatch: int
    kind: str  # 'F' or 'B'


@dataclass
class Task:
    key: TaskKey
    duration: float  # seconds
    energy: float  # joules for this task
    preds: List[TaskKey]
    succs: List[TaskKey]
    earliest_pred_finish: float = 0.0
    start: float = 0.0
    finish: float = 0.0


def _build_tasks(
    num_stages: int,
    num_mbs: int,
    f_time_by_stage: Dict[int, float],
    b_time_by_stage: Dict[int, float],
    f_energy_by_stage: Dict[int, float],
    b_energy_by_stage: Dict[int, float],
) -> Dict[TaskKey, Task]:
    tasks: Dict[TaskKey, Task] = {}

    def add_task(stage: int, mb: int, kind: str) -> TaskKey:
        if kind == 'F':
            dur = f_time_by_stage[stage]
            en = f_energy_by_stage[stage]
        else:
            dur = b_time_by_stage[stage]
            en = b_energy_by_stage[stage]
        key = TaskKey(stage, mb, kind)
        tasks[key] = Task(key=key, duration=dur, energy=en, preds=[], succs=[])
        return key

    # Create all tasks
    for s in range(num_stages):
        for m in range(num_mbs):
            add_task(s, m, 'F')
            add_task(s, m, 'B')

    # Add precedence constraints
    for s in range(num_stages):
        for m in range(num_mbs):
            f_key = TaskKey(s, m, 'F')
            b_key = TaskKey(s, m, 'B')

            # Within-stage: B depends on F for the same microbatch
            tasks[b_key].preds.append(f_key)
            tasks[f_key].succs.append(b_key)

            # Forward pipeline dependency from previous stage
            if s > 0:
                prev_f = TaskKey(s - 1, m, 'F')
                tasks[f_key].preds.append(prev_f)
                tasks[prev_f].succs.append(f_key)

            # Backward pipeline dependency to next stage
            if s < num_stages - 1:
                next_b = TaskKey(s + 1, m, 'B')
                tasks[b_key].preds.append(next_b)
                tasks[next_b].succs.append(b_key)

    # Note: We no longer add per-stage chaining edges here to avoid accidental
    # dependency cycles. The synchronous 1F1B order is enforced in the scheduler
    # by allowing only the head-of-queue task of each stage-specific sequence.

    return tasks


def _schedule(tasks: Dict[TaskKey, Task], num_stages: int) -> None:
    # Build synchronous 1F1B per-stage queues to enforce local order without
    # adding edges that could create cycles across stages.
    per_stage_queues: List[List[TaskKey]] = []
    for s in range(num_stages):
        # W forwards for warmup at stage s
        W = max(0, num_stages - 1 - s)
        seq: List[TaskKey] = []
        for i in range(W):
            seq.append(TaskKey(s, i, 'F'))
        for i in range(W, max(W, len({k.microbatch for k in tasks.keys() if k.stage == s and k.kind == 'F'}))):
            # ensure we only iterate for available microbatches
            if TaskKey(s, i, 'F') in tasks:
                seq.append(TaskKey(s, i, 'F'))
            bi = i - W
            if bi >= 0 and TaskKey(s, bi, 'B') in tasks:
                seq.append(TaskKey(s, bi, 'B'))
        # Drain
        num_mbs_stage = max((k.microbatch for k in tasks.keys() if k.stage == s), default=-1) + 1
        for i in range(num_mbs_stage - W, num_mbs_stage):
            if TaskKey(s, i, 'B') in tasks:
                seq.append(TaskKey(s, i, 'B'))

        # Dedup preserving order
        seen = set()
        ordered: List[TaskKey] = []
        for k in seq:
            if k in tasks and k not in seen:
                ordered.append(k)
                seen.add(k)
        per_stage_queues.append(ordered)

    # Kahn-like list scheduling, but a task can only be picked if it is the
    # head of its stage queue in addition to having no unmet predecessors.
    indegree: Dict[TaskKey, int] = {}
    for t in tasks.values():
        indegree[t.key] = len(t.preds)

    # Map from stage to current queue index
    stage_idx = [0 for _ in range(num_stages)]

    # Track stage availability times
    stage_available: List[float] = [0.0 for _ in range(num_stages)]

    # For each task, track the max predecessor finish time
    for t in tasks.values():
        t.earliest_pred_finish = 0.0

    scheduled = 0
    total = len(tasks)
    while scheduled < total:
        best_key = None
        best_start = None

        # Consider at most one candidate per stage: the head-of-queue that is ready
        for s in range(num_stages):
            q = per_stage_queues[s]
            idx = stage_idx[s]
            if idx >= len(q):
                continue
            k = q[idx]
            if indegree[k] != 0:
                continue
            t = tasks[k]
            candidate_start = max(stage_available[s], t.earliest_pred_finish)
            if best_start is None or candidate_start < best_start or (
                candidate_start == best_start and (k.kind, k.stage, k.microbatch) < (best_key.kind, best_key.stage, best_key.microbatch)  # type: ignore[arg-type]
            ):
                best_key = k
                best_start = candidate_start

        if best_key is None:
            # No stage head is currently ready; advance dependency readiness by
            # relaxing to any zero-indegree task (should not happen often).
            zero_indegree = [k for k, d in indegree.items() if d == 0]
            if not zero_indegree:
                raise RuntimeError("Deadlock: no schedulable tasks and no zero-indegree tasks.")
            # Pick the earliest-start among zero-indegree tasks
            for k in zero_indegree:
                t = tasks[k]
                candidate_start = max(stage_available[k.stage], t.earliest_pred_finish)
                if best_start is None or candidate_start < best_start:
                    best_key = k
                    best_start = candidate_start

        assert best_key is not None and best_start is not None
        t = tasks[best_key]
        t.start = best_start
        t.finish = t.start + t.duration
        stage_available[best_key.stage] = t.finish
        scheduled += 1

        # If it was the head of its stage queue, advance the pointer
        s = best_key.stage
        if stage_idx[s] < len(per_stage_queues[s]) and per_stage_queues[s][stage_idx[s]] == best_key:
            stage_idx[s] += 1

        # Update successors
        for succ_key in t.succs:
            succ = tasks[succ_key]
            if t.finish > succ.earliest_pred_finish:
                succ.earliest_pred_finish = t.finish
            indegree[succ_key] -= 1

    if scheduled != total:
        raise RuntimeError(f"Failed to schedule all tasks: scheduled={scheduled}, total={total}")


def _draw(
    tasks: Dict[TaskKey, Task],
    num_stages: int,
    num_mbs: int,
    output_path: Path,
    figsize_scale: float = 1.0,
) -> None:
    # Configure fonts to use DejaVu Sans and embed TrueType in PDFs
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans"],
            "pdf.fonttype": 42,
        }
    )
    # Compute power for color mapping; avoid division by zero.
    powers = [t.energy / t.duration if t.duration > 0 else 0.0 for t in tasks.values()]

    # Set colorbar/power normalization range to [0, 300] W
    vmin = 0.0
    vmax = 300.0

    # Use a truncated, lighter section of RdBu_r to avoid very dark reds/blues
    _orig_cmap = mpl.colormaps.get_cmap('RdBu_r')
    _colors = _orig_cmap(np.linspace(0.2, 0.8, 256))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('RdBu_r_trunc', _colors)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Figure size: make it flatter and reduce vertical space per row
    fig, ax = plt.subplots(
        figsize=(
            max(6.0, num_mbs * 0.8 * figsize_scale),
            max(3.0, 0.6 * num_stages + 1.3),
        ),
        tight_layout=True,
    )

    # Remove space between rows by filling each lane height
    bar_height = 1.0

    def _lighten_rgba(rgba: tuple[float, float, float, float], amount: float = 0.15) -> tuple[float, float, float, float]:
        r, g, b, a = rgba
        r = r + (1.0 - r) * amount
        g = g + (1.0 - g) * amount
        b = b + (1.0 - b) * amount
        return (r, g, b, 1.0)

    for t in tasks.values():
        y = t.key.stage
        power = t.energy / t.duration if t.duration > 0 else 0.0
        base = cmap(norm(power))
        # Lighten color to avoid very dark extremes
        # color = _lighten_rgba(base, amount=0)
        color = base
        ax.add_patch(
            plt.Rectangle(
                (t.start, y - bar_height / 2.0),
                t.duration,
                bar_height,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
            )
        )
        # Annotate with kind and microbatch id for readability
        ax.text(
            t.start + t.duration / 2.0,
            y,
            f"{t.key.kind}{t.key.microbatch}",
            ha='center',
            va='center',
            fontsize=8,
            color='black',
        )

    # Set background color to the color corresponding to power=90 W
    base = cmap(norm(90.0))
    # ax.set_facecolor(_lighten_rgba(base, amount=0.0))
    ax.set_facecolor(base)

    # Axes formatting
    ax.set_xlabel('Time (s)')
    # Remove y-axis label and show tick labels as S1..S{num_stages}
    ax.set_ylabel('')
    ax.set_yticks(range(num_stages))
    ax.set_yticklabels([f"S{s+1}" for s in range(num_stages)])
    # No title per request

    # Colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Move colorbar to the top
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation='horizontal',
        location='top',
        pad=0.05,
        fraction=0.05,
        aspect=40,
    )
    cbar.set_label('Power (W)')

    # Fixed time axis range 0..10
    ax.set_xlim(0.0, 9.0)
    # Invert Y so that rank 0 is at the top
    ax.set_ylim(num_stages - 0.5, -0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def draw_pipeline_from_csv(csv_path: str, output: str | None = None, num_microbatches: int = 8) -> Path:
    df = pd.read_csv(csv_path)
    required_cols = {'rank', 'name', 'avg_time', 'avg_energy'}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns {required_cols}, got {set(df.columns)}")

    # Stages
    stage_ids = sorted(set(int(x) for x in df['rank'].unique()))
    if not stage_ids:
        raise ValueError('No stages found in CSV (rank column empty).')
    num_stages = stage_ids[-1] + 1

    # Extract per-stage forward/backward time and energy
    f_time_by_stage: Dict[int, float] = {}
    b_time_by_stage: Dict[int, float] = {}
    f_energy_by_stage: Dict[int, float] = {}
    b_energy_by_stage: Dict[int, float] = {}

    for s in stage_ids:
        row_f = df[(df['rank'] == s) & (df['name'] == 'forward')]
        row_b = df[(df['rank'] == s) & (df['name'] == 'backward')]
        if row_f.empty or row_b.empty:
            raise ValueError(f"Missing 'forward' or 'backward' row for stage {s}.")
        f_time_by_stage[s] = float(row_f.iloc[0]['avg_time'])
        b_time_by_stage[s] = float(row_b.iloc[0]['avg_time'])
        f_energy_by_stage[s] = float(row_f.iloc[0]['avg_energy'])
        b_energy_by_stage[s] = float(row_b.iloc[0]['avg_energy'])

    tasks = _build_tasks(
        num_stages=num_stages,
        num_mbs=num_microbatches,
        f_time_by_stage=f_time_by_stage,
        b_time_by_stage=b_time_by_stage,
        f_energy_by_stage=f_energy_by_stage,
        b_energy_by_stage=b_energy_by_stage,
    )
    _schedule(tasks, num_stages=num_stages)

    # Determine output path (default to PDF)
    out_path = (
        Path(output)
        if output is not None
        else (Path(csv_path).parent / 'pipeline.pdf')
    )
    _draw(tasks, num_stages=num_stages, num_mbs=num_microbatches, output_path=out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Draw training pipeline from profile CSV (color by energy).')
    parser.add_argument('--csv', required=True, help='Path to profile CSV, e.g., profile_by_groups.csv')
    parser.add_argument('--output', default=None, help='Output image path (PDF). Default: sibling of CSV named pipeline.pdf')
    parser.add_argument('--num_microbatches', type=int, default=8, help='Number of microbatches to visualize.')
    args = parser.parse_args()

    out = draw_pipeline_from_csv(args.csv, args.output, args.num_microbatches)
    print(f"Saved pipeline figure to {out}")


if __name__ == '__main__':
    main()


