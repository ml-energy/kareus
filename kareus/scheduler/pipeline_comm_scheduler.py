from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Optional, Union

from kareus.scheduler.parser import ScheduleItem, ScheduleItemCP, parse_schedule_file


__all__ = ["PipelineCommScheduler"]


class PipelineCommScheduler:
    """Simple scheduler that iterates a precomputed per-PP-rank schedule.

    The schedule is read from a config file that contains a literal Python list
    of lists of 3-tuples: (name, attn_configs, mlp_configs), one inner list per
    pipeline-parallel rank.

    Parameters
    ----------
    configs_pipeline:
        Path to a Python file containing the schedules list. Must point to an
        existing .py file that holds a top-level list literal.
    pp_rank:
        The pipeline parallelism rank whose schedule will be iterated.
    """

    def __init__(self,
        configs_pipeline: Union[str, os.PathLike],
        pp_rank: int,
        num_microbatches: int,
        use_activation_checkpointing: bool = False,
        context_parallel: bool = False
    ) -> None:
        p = Path(str(configs_pipeline))
        if p.suffix != ".py" or not p.exists():
            raise FileNotFoundError(
                f"configs_pipeline must be an existing .py file, got: {p}"
            )
        self._config_path = p.resolve()
        parsed = parse_schedule_file(self._config_path, use_activation_checkpointing, context_parallel)
        self._init_from_parsed(parsed, pp_rank, num_microbatches, use_activation_checkpointing, context_parallel)

    @classmethod
    def from_parsed(
        cls,
        parsed: List[List["ScheduleItem | ScheduleItemCP"]],
        pp_rank: int,
        num_microbatches: int,
        use_activation_checkpointing: bool = False,
        context_parallel: bool = False,
    ) -> "PipelineCommScheduler":
        """Build a scheduler from an already-parsed schedule list.

        Use when the schedule has been read on one rank and broadcast to others,
        avoiding redundant filesystem reads and shared-storage requirements.
        """
        self = cls.__new__(cls)
        self._config_path = None
        self._init_from_parsed(parsed, pp_rank, num_microbatches, use_activation_checkpointing, context_parallel)
        return self

    def _init_from_parsed(
        self,
        parsed: List[List["ScheduleItem | ScheduleItemCP"]],
        pp_rank: int,
        num_microbatches: int,
        use_activation_checkpointing: bool,
        context_parallel: bool,
    ) -> None:
        self._ac_mode: bool = use_activation_checkpointing
        self._context_parallel: bool = context_parallel

        if pp_rank < 0 or pp_rank >= len(parsed):
            raise IndexError(
                f"pp_rank {pp_rank} is out of range for schedule list of length {len(parsed)}"
            )

        self._pp_rank: int = pp_rank
        # Parsed is List[List[ScheduleItem]] in both modes now
        self._schedule_items: List[ScheduleItem | ScheduleItemCP] = list(parsed[pp_rank])  # type: ignore[index]
        if not self._ac_mode:
            if num_microbatches != len(self._schedule_items):
                raise ValueError(
                    f"Number of microbatches in schedule ({len(self._schedule_items)}) does not match num_microbatches ({num_microbatches})"
                )
        # Iterator: AC iterates per instruction, non-AC advances on forward only
        self._iter_items: Iterator[ScheduleItem | ScheduleItemCP] = iter(self._schedule_items)
        self.current_schedule: Optional[ScheduleItem | ScheduleItemCP] = None

    def _reset(self) -> None:
        """Reset iterator and counters for the next step."""
        self._iter_items = iter(self._schedule_items)
        self.current_schedule = None

    def on_instruction_begin(self, name) -> None:
        """Return the next scheduled item for this PP rank.

        On a 'forward' call, advance the microbatch pair iterator,
        store the pair, and return the pair so callers can use both configs.
        On a 'backward' call, return the backward config from the stored pair.
        """
        if self._ac_mode:
            try:
                item = next(self._iter_items)
            except StopIteration as exc:
                raise RuntimeError("No more scheduled instructions available for this step") from exc
            expected = "forward" if (item.bwd_mlp is None) else "backward"
            if expected != name:
                raise RuntimeError(
                    f"Schedule/item mismatch: expected '{expected}', got '{name}'"
                )
            self.current_schedule = item
        else:
            if name == "forward":
                try:
                    item = next(self._iter_items)
                except StopIteration as exc:
                    raise RuntimeError("No more forward microbatches available for this step") from exc
                self.current_schedule = item

    def on_instruction_end(self, name: Optional[str] = None) -> None:
        """Mark the end of an instruction, like forward or backward."""
        pass

    def on_step_begin(self) -> None:
        """Mark the beginning of a step."""
        pass

    def on_step_end(self) -> None:
        """Mark the end of a step and reset the schedule iterator.

        Ensures all forward microbatches have been dispatched and all pending
        microbatches have been drained via backward before resetting.
        """
        # Drain should complete exactly at step end as well
        extra = next(self._iter_items, None)
        if extra is not None:
            raise RuntimeError(
                "Schedule returned more items than expected at step end. "
                f"Next item: {extra}"
            )
        self._reset()
