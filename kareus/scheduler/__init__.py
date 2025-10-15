from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union
import re
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
    fwd_attn: Optional[CommConfig]
    fwd_mlp: Optional[CommConfig]
    bwd_attn: Optional[CommConfig]
    bwd_mlp: Optional[CommConfig]

@dataclass(frozen=True)
class ScheduleItemCP:
    fwd_qkv_ar: Optional[CommConfig]
    fwd_qkv_ag: Optional[CommConfig]
    fwd_ao_ag: Optional[CommConfig]
    fwd_ao_ar: Optional[CommConfig]
    fwd_mlp: Optional[CommConfig]

    bwd_qkv_ar: Optional[CommConfig]
    bwd_qkv_rs: Optional[CommConfig]
    bwd_a_rs: Optional[CommConfig]
    bwd_a_ag: Optional[CommConfig]
    bwd_o_ag: Optional[CommConfig]
    bwd_o_ar: Optional[CommConfig]
    bwd_mlp: Optional[CommConfig]

MicrobatchPair = Tuple[ScheduleItem, ScheduleItem]


def _parse_config_string(cfg: str) -> CommConfig:
    match = re.match(r"^(-?\d+)-(-?\d+)-(-?\d+)-(-?\d+)$", cfg)
    if not match:
        raise ValueError(
            f"Config string must match 'overlap_start-overlap_end-sm_num-block_size' with integers, got: {cfg}"
        )
    overlap_start = int(match.group(1))
    overlap_end = int(match.group(2))
    sm_num = int(match.group(3))
    block_size = int(match.group(4))
    return CommConfig((overlap_start, overlap_end), (sm_num, block_size))


def _parse_config_file(
    path: Path, 
    use_activation_checkpointing: bool,
    context_parallel: bool
) -> List[List[ScheduleItem | ScheduleItemCP]]:
    with path.open("r", encoding="utf-8") as f:
        parsed = eval(f.read())

    if not isinstance(parsed, list):
        raise TypeError("Top-level config must be a list")

    if use_activation_checkpointing:
        if not context_parallel:
            # Parse per-instruction 1F1B list: entries are tuples
            # forward: ("forward", attn:str, mlp:str)
            # backward: ("backward", rec_attn:str, rec_mlp:str, attn:str, mlp:str)
            normalized_items: List[List[ScheduleItem]] = []
            for idx, per_rank in enumerate(parsed):
                if not isinstance(per_rank, list):
                    raise TypeError(
                        f"Entry {idx} must be a list of instruction tuples in 1F1B order"
                    )
                rank_items: List[ScheduleItem] = []
                for tup in per_rank:
                    if not isinstance(tup, tuple) or len(tup) < 3:
                        raise TypeError("Instruction tuple must be ('name', ...) with at least 3 elements")
                    name = tup[0]
                    if name == "forward":
                        if len(tup) != 3:
                            raise TypeError("Forward tuple must be ('forward', attn:str, mlp:str)")
                        rank_items.append(
                            ScheduleItem(
                                fwd_attn=_parse_config_string(tup[1]),
                                fwd_mlp=_parse_config_string(tup[2]),
                                bwd_attn=None,
                                bwd_mlp=None,
                            )
                        )
                    elif name == "backward":
                        if len(tup) != 5:
                            raise TypeError(
                                "Backward tuple must be ('backward', rec_attn:str, rec_mlp:str, attn:str, mlp:str)"
                            )
                        rank_items.append(
                            ScheduleItem(
                                fwd_attn=_parse_config_string(tup[1]),
                                fwd_mlp=_parse_config_string(tup[2]),
                                bwd_attn=_parse_config_string(tup[3]),
                                bwd_mlp=_parse_config_string(tup[4]),
                            )
                        )
                    else:
                        raise ValueError(f"Unknown instruction name: {name}")
                normalized_items.append(rank_items)
            return normalized_items
        else:
            # Parse per-instruction 1F1B list: entries are tuples
            # forward: ("forward", qkv_ar:str, qkv_ag:str, ao_ag:str, ao_ar:str, mlp:str)
            # backward: ("backward", qkv_ar:str, qkv_rs:str, a_rs:str, a_ag:str, o_ag:str, o_ar:str, mlp:str)
            normalized_items: List[List[ScheduleItemCP]] = []
            for idx, per_rank in enumerate(parsed):
                if not isinstance(per_rank, list):
                    raise TypeError(
                        f"Entry {idx} must be a list of instruction tuples in 1F1B order"
                    )
                rank_items: List[ScheduleItemCP] = []
                for tup in per_rank:
                    if not isinstance(tup, tuple) or len(tup) < 2:
                        raise TypeError("Instruction tuple must be ('name', ...) with at least 2 elements")
                    name = tup[0]
                    if name == "forward":
                        if len(tup) != 6:
                            raise TypeError(
                                "Forward tuple must be ('forward', qkv_ar:str, qkv_ag:str, ao_ag:str, ao_ar:str, mlp:str)"
                            )
                        rank_items.append(
                            ScheduleItemCP(
                                fwd_qkv_ar=_parse_config_string(tup[1]),
                                fwd_qkv_ag=_parse_config_string(tup[2]),
                                fwd_ao_ag=_parse_config_string(tup[3]),
                                fwd_ao_ar=_parse_config_string(tup[4]),
                                fwd_mlp=_parse_config_string(tup[5]),
                                bwd_qkv_ar=None,
                                bwd_qkv_rs=None,
                                bwd_a_rs=None,
                                bwd_a_ag=None,
                                bwd_o_ag=None,
                                bwd_o_ar=None,
                                bwd_mlp=None,
                            )
                        )
                    elif name == "backward":
                        if len(tup) != 13:
                            raise TypeError(
                                "Backward tuple must be ('backward', qkv_ar:str, qkv_rs:str, a_rs:str, a_ag:str, o_ag:str, o_ar:str, mlp:str)"
                            )
                        rank_items.append(
                            ScheduleItemCP(
                                fwd_qkv_ar=_parse_config_string(tup[1]),
                                fwd_qkv_ag=_parse_config_string(tup[2]),
                                fwd_ao_ag=_parse_config_string(tup[3]),
                                fwd_ao_ar=_parse_config_string(tup[4]),
                                fwd_mlp=_parse_config_string(tup[5]),
                                bwd_qkv_ar=_parse_config_string(tup[6]),
                                bwd_qkv_rs=_parse_config_string(tup[7]),
                                bwd_a_rs=_parse_config_string(tup[8]),
                                bwd_a_ag=_parse_config_string(tup[9]),
                                bwd_o_ag=_parse_config_string(tup[10]),
                                bwd_o_ar=_parse_config_string(tup[11]),
                                bwd_mlp=_parse_config_string(tup[12]),
                            )
                        )
                    else:
                        raise ValueError(f"Unknown instruction name: {name}")
                normalized_items.append(rank_items)
            return normalized_items
            
    else:
        # Legacy per-microbatch pair format, collapse into one ScheduleItem per microbatch
        if not context_parallel:
            normalized_items: List[List[ScheduleItem]] = []
            for idx, per_rank in enumerate(parsed):
                if not isinstance(per_rank, list):
                    raise TypeError(
                        f"Entry {idx} must be a list of microbatch pairs ((forward_tuple),(backward_tuple))"
                    )
                rank_list: List[ScheduleItem] = []
                for item in per_rank:
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
                    item_both: ScheduleItem = ScheduleItem(
                        fwd_attn=_parse_config_string(fwd_raw[1]),
                        fwd_mlp=_parse_config_string(fwd_raw[2]),
                        bwd_attn=_parse_config_string(bwd_raw[1]),
                        bwd_mlp=_parse_config_string(bwd_raw[2]),
                    )
                    rank_list.append(item_both)
                normalized_items.append(rank_list)
        else:
            normalized_items: List[List[ScheduleItemCP]] = []
            for idx, per_rank in enumerate(parsed):
                if not isinstance(per_rank, list):
                    raise TypeError(
                        f"Entry {idx} must be a list of microbatch pairs ((forward_tuple),(backward_tuple))"
                    )
                rank_list: List[ScheduleItemCP] = []
                for item in per_rank:
                    if not (isinstance(item, tuple) and len(item) == 2):
                        raise TypeError(
                            "Each microbatch entry must be a tuple of two tuples: (forward, backward)"
                        )
                    fwd_raw, bwd_raw = item
                    if not len(fwd_raw) == 6:
                        raise TypeError("Forward entry must be (name:str, qkv_ar:str, qkv_ag:str, ao_ag:str, ao_ar:str, mlp:str)")
                    if not len(bwd_raw) == 8:
                        raise TypeError("Backward entry must be (name:str, qkv_ar:str, qkv_rs:str, a_rs:str, a_ag:str, o_ag:str, o_ar:str, mlp:str)")

                    fwd_name: str = fwd_raw[0]
                    bwd_name: str = bwd_raw[0]
                    if fwd_name != "forward" or bwd_name != "backward":
                        raise ValueError(
                            f"Expected ('forward', ...), ('backward', ...), got: {fwd_name}, {bwd_name}"
                        )
                    item_both: ScheduleItemCP = ScheduleItemCP(
                        fwd_qkv_ar=_parse_config_string(fwd_raw[1]),
                        fwd_qkv_ag=_parse_config_string(fwd_raw[2]),
                        fwd_ao_ag=_parse_config_string(fwd_raw[3]),
                        fwd_ao_ar=_parse_config_string(fwd_raw[4]),
                        fwd_mlp=_parse_config_string(fwd_raw[5]),

                        bwd_qkv_ar=_parse_config_string(bwd_raw[1]),
                        bwd_qkv_rs=_parse_config_string(bwd_raw[2]),
                        bwd_a_rs=_parse_config_string(bwd_raw[3]),
                        bwd_a_ag=_parse_config_string(bwd_raw[4]),
                        bwd_o_ag=_parse_config_string(bwd_raw[5]),
                        bwd_o_ar=_parse_config_string(bwd_raw[6]),
                        bwd_mlp=_parse_config_string(bwd_raw[7]),
                    )
                    rank_list.append(item_both)
                normalized_items.append(rank_list)

        return normalized_items



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
        self._ac_mode: bool = use_activation_checkpointing
        self._context_parallel: bool = context_parallel
        parsed = _parse_config_file(self._config_path, self._ac_mode, self._context_parallel)

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
        self._iter_items: Iterator[ScheduleItem ｜ ScheduleItemCP] = iter(self._schedule_items)
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
