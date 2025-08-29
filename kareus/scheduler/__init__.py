from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union
from dataclasses import dataclass


__all__ = ["PipelineCommScheduler"]


OverlapWindow = Tuple[int, int]
ResourceShape = Tuple[int, int]

@dataclass(frozen=True)
class CommConfig:
    overlap_window: OverlapWindow
    resource_shape: ResourceShape


@dataclass(frozen=True)
class ScheduleItem:
    name: str
    attn: CommConfig
    mlp: CommConfig

MicrobatchPair = Tuple[ScheduleItem, ScheduleItem]


def _parse_config_string(cfg: str) -> CommConfig:
    parts = cfg.split("-")
    if len(parts) != 4:
        raise ValueError(
            f"Config string must have 4 dash-separated integers (overlap_start-overlap_end-sm_num-block_size), got: {cfg}"
        )
    try:
        overlap_start = int(parts[0])
        overlap_end = int(parts[1])
        sm_num = int(parts[2])
        block_size = int(parts[3])
    except ValueError as exc:
        raise ValueError(f"Non-integer values in config string: {cfg}") from exc
    return CommConfig((overlap_start, overlap_end), (sm_num, block_size))


def _parse_config_file(path: Path) -> List[List[MicrobatchPair]]:
    with path.open("r", encoding="utf-8") as f:
        parsed = eval(f.read())

    if not isinstance(parsed, list):
        raise TypeError("Top-level config must be a list")

    # Ensure nested structure and parse strings into structured configs
    normalized: List[List[MicrobatchPair]] = []
    for idx, per_rank in enumerate(parsed):
        if not isinstance(per_rank, list):
            raise TypeError(
                f"Entry {idx} must be a list of microbatch pairs ((forward_tuple),(backward_tuple))"
            )
        rank_list: List[MicrobatchPair] = []
        for item in per_rank:
            # New format: ((name, attn, mlp), (name, attn, mlp)) per microbatch
            if not (isinstance(item, tuple) and len(item) == 2):
                raise TypeError(
                    "Each microbatch entry must be a tuple of two tuples: (forward, backward)"
                )
            fwd_raw, bwd_raw = item
            if not (
                isinstance(fwd_raw, tuple)
                and len(fwd_raw) == 3
                and isinstance(fwd_raw[0], str)
                and isinstance(fwd_raw[1], str)
                and isinstance(fwd_raw[2], str)
            ):
                raise TypeError("Forward entry must be (name:str, attn:str, mlp:str)")
            if not (
                isinstance(bwd_raw, tuple)
                and len(bwd_raw) == 3
                and isinstance(bwd_raw[0], str)
                and isinstance(bwd_raw[1], str)
                and isinstance(bwd_raw[2], str)
            ):
                raise TypeError("Backward entry must be (name:str, attn:str, mlp:str)")

            fwd_name: str = fwd_raw[0]
            bwd_name: str = bwd_raw[0]
            if fwd_name != "forward" or bwd_name != "backward":
                raise ValueError(
                    f"Expected ('forward', ...), ('backward', ...), got: {fwd_name}, {bwd_name}"
                )

            fwd_item: ScheduleItem = ScheduleItem(
                name=fwd_name,
                attn=_parse_config_string(fwd_raw[1]),
                mlp=_parse_config_string(fwd_raw[2]),
            )
            bwd_item: ScheduleItem = ScheduleItem(
                name=bwd_name,
                attn=_parse_config_string(bwd_raw[1]),
                mlp=_parse_config_string(bwd_raw[2]),
            )
            rank_list.append((fwd_item, bwd_item))
        normalized.append(rank_list)

    return normalized



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

    def __init__(self, configs_pipeline: Union[str, os.PathLike], pp_rank: int, num_microbatches: int) -> None:
        p = Path(str(configs_pipeline))
        if p.suffix != ".py" or not p.exists():
            raise FileNotFoundError(
                f"configs_pipeline must be an existing .py file, got: {p}"
            )
        self._config_path = p.resolve()
        self._all_schedules: List[List[MicrobatchPair]] = _parse_config_file(self._config_path)

        if pp_rank < 0 or pp_rank >= len(self._all_schedules):
            raise IndexError(
                f"pp_rank {pp_rank} is out of range for schedule list of length {len(self._all_schedules)}"
            )

        self._pp_rank: int = pp_rank
        self._schedule: List[MicrobatchPair] = list(self._all_schedules[pp_rank])

        if num_microbatches != len(self._schedule):
            raise ValueError(
                f"Number of microbatches in schedule ({len(self._schedule)}) does not match num_microbatches ({num_microbatches})"
            )

        self._iter_pairs: Iterator[MicrobatchPair] = iter(self._schedule)

        self.current_schedule: Optional[MicrobatchPair] = None

    def _reset(self) -> None:
        """Reset iterator and counters for the next step."""
        self._iter_pairs = iter(self._schedule)
        self.current_schedule = None

    def on_instruction_begin(self, name) -> None:
        """Return the next scheduled item for this PP rank.

        On a 'forward' call, advance the microbatch pair iterator,
        store the pair, and return the pair so callers can use both configs.
        On a 'backward' call, return the backward config from the stored pair.
        """
        if name == "forward":
            try:
                pair = next(self._iter_pairs)
            except StopIteration as exc:
                raise RuntimeError("No more forward microbatches available for this step") from exc
            self.current_schedule = pair
    
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
        item = next(self._iter_pairs, None)
        if item is not None:
            raise RuntimeError(
                "Schedule returned more items than expected at step end. "
                f"Next item: {item}"
            )
        self._reset()
