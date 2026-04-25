from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union


__all__ = [
    "CommConfig",
    "ScheduleItem",
    "ScheduleItemCP",
    "MicrobatchPair",
    "OverlapWindow",
    "ResourceShape",
    "parse_schedule_file",
]


OverlapWindow = Tuple[Union[int, str], Union[int, str]]
ResourceShape = Tuple[int, int]


@dataclass(frozen=True)
class CommConfig:
    overlap_window: OverlapWindow
    resource_shape: ResourceShape


_TRAIL_CONFIG = CommConfig(overlap_window=("norm_0", "none"), resource_shape=(20, 1024))


@dataclass(frozen=True)
class ScheduleItem:
    fwd_attn: Optional[CommConfig]
    fwd_mlp: Optional[CommConfig]
    bwd_attn: Optional[CommConfig]
    bwd_mlp: Optional[CommConfig]
    fwd_trail: Optional[CommConfig] = _TRAIL_CONFIG
    bwd_trail: Optional[CommConfig] = _TRAIL_CONFIG


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

    fwd_trail: Optional[CommConfig] = _TRAIL_CONFIG
    bwd_trail: Optional[CommConfig] = _TRAIL_CONFIG


MicrobatchPair = Tuple[ScheduleItem, ScheduleItem]


def _parse_config_string(cfg: str) -> CommConfig:
    """Parse a comm config string.

    Supports two formats for the overlap window:
      - Legacy integer: ``"0-8-16-1024"`` (possibly with negatives: ``"-1--1-16-1024"``)
      - Slot names: ``"norm_0-linear_1-16-1024"`` or ``"none-none-16-1024"``
    """
    # Try legacy all-integer format first (allows negative numbers).
    match = re.match(r"^(-?\d+)-(-?\d+)-(-?\d+)-(-?\d+)$", cfg)
    if match:
        return CommConfig(
            (int(match.group(1)), int(match.group(2))),
            (int(match.group(3)), int(match.group(4))),
        )

    # Slot-name format: split off the last two numeric fields (sm, block).
    parts = cfg.rsplit("-", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Config string must have format 'overlap_start-overlap_end-sm_num-block_size', got: {cfg}"
        )
    overlap_part, sm_str, block_str = parts
    sm_num = int(sm_str)
    block_size = int(block_str)

    # Split overlap_part into (start, end) slot names. Slot names never
    # contain '-' (they use '_' for numbering), so split on the first '-'.
    sep_idx = overlap_part.find("-")
    if sep_idx == -1:
        raise ValueError(
            f"Overlap part must contain 'start-end', got: {overlap_part!r}"
        )
    overlap_start = overlap_part[:sep_idx]
    overlap_end = overlap_part[sep_idx + 1:]

    return CommConfig((overlap_start, overlap_end), (sm_num, block_size))


def parse_schedule_file(
    path: Union[str, Path],
    use_activation_checkpointing: bool,
    context_parallel: bool,
) -> List[List["ScheduleItem | ScheduleItemCP"]]:
    """Read and normalize a schedule .py file into per-pp-rank lists.

    The file is a top-level Python list literal, parsed via ``eval``. Output is
    a list of per-pp-rank lists of ``ScheduleItem`` (or ``ScheduleItemCP`` when
    ``context_parallel=True``).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        parsed = eval(f.read())

    if not isinstance(parsed, list):
        raise TypeError("Top-level config must be a list")

    if use_activation_checkpointing:
        if not context_parallel:
            # Per-instruction 1F1B list:
            # forward:  ("forward", attn:str, mlp:str)
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
            # Per-instruction 1F1B list with context parallel:
            # forward:  ("forward", qkv_ar, qkv_ag, ao_ag, ao_ar, mlp)
            # backward: ("backward", qkv_ar, qkv_rs, a_rs, a_ag, o_ag, o_ar, mlp) plus recompute fields
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
