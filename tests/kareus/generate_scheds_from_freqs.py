"""Generate scheds_pipeline from a freqs_pipeline and instruction profile.

Given an input freqs_pipeline_XXXXX.py file (a Python list literal of per-stage
lists of (instruction, frequency) tuples) and a profiling CSV, this script
selects the energy-optimal configuration per (stage, instruction, frequency)
and produces a matching scheds_pipeline_XXXXX.py file in 1F1B order.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import tyro

FUSER_DIR = os.path.join(os.path.dirname(__file__), '..', 'fuser')
if FUSER_DIR not in sys.path:
    sys.path.append(FUSER_DIR)
try:
    from common_config import FuserTestConfig  # type: ignore
except Exception:
    FuserTestConfig = None  # Fallback if not available; CLI arg can still override


InstructionTuple = Tuple[str, int | None]
StageFreqs = List[InstructionTuple]
FreqsPipeline = List[StageFreqs]


@dataclass
class Args:
    # Path to instruction profile results (CSV)
    inst_profile: str = "profile.csv"
    # Path to input freqs pipeline file (Python list literal)
    freqs_file: str = "/workspaces/Kareus/tests/perseus/nemo_experiments/megatron_llama_3_2_1b/perseus_results/freqs_pipeline_01946.py"
    # Where to write the generated scheds pipeline file
    output_file: str | None = None
    # Whether to assume activation checkpointing (AC) configs are present
    # If True and the CSV lacks recompute_* columns, this will auto-fallback to False
    use_activation_checkpointing: bool = True
    # GPU power consumption while blocking on P2P communication, in Watts
    p2p_power: float = (
        float(FuserTestConfig.get_p2p_power('A100'))
    )
    # The unit of reduction for each iteration, in seconds (for time quantization)
    unit_time: float = 0.001


def _read_top_level_list_literal(file_path: Path) -> Any:
    """Read the first top-level Python list literal in a file.

    The freqs/scheds files are top-level list literals followed by comments.
    This parser extracts the bracket-balanced list starting at the first '['
    and uses ast.literal_eval to safely construct the Python object.
    """
    text = file_path.read_text()

    # Find first '[' and parse until the matching closing ']' at depth 0
    start_idx = text.find("[")
    if start_idx == -1:
        raise ValueError(f"No list literal found in {file_path}")

    depth = 0
    end_idx = -1
    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end_idx = idx + 1
                break
    if end_idx == -1:
        raise ValueError(f"Unbalanced brackets in {file_path}")

    list_text = text[start_idx:end_idx]
    return ast.literal_eval(list_text)


def _detect_ac(inst_df: pd.DataFrame, requested: bool) -> bool:
    """Detect activation checkpointing support based on CSV columns and flag."""
    if not requested:
        return False
    required_cols = {"recompute_attention_configs", "recompute_mlp_configs"}
    if required_cols.issubset(set(inst_df.columns)):
        return True
    # Auto-fallback with a gentle warning
    print(
        "[generate_scheds_from_freqs] AC requested but recompute_* columns not found; "
        "falling back to non-AC configs."
    )
    return False


def _build_best_config_map(
    inst_df: pd.DataFrame, use_ac: bool, p2p_power: float, unit_time: float
) -> Dict[Tuple[int, str, int], Tuple[str, ...]]:
    """Build a mapping (stage, instruction, frequency) -> config tuple.

    - For forward (both AC and non-AC): ("forward", attn_cfg, mlp_cfg)
    - For backward with AC: ("backward", rec_attn_cfg, rec_mlp_cfg, bwd_attn_cfg, bwd_mlp_cfg)
    - For backward without AC: ("backward", bwd_attn_cfg, bwd_mlp_cfg)
    """
    # Normalize and enforce expected dtypes
    df = inst_df.copy()
    if "instruction" in df.columns:
        df["instruction"] = df["instruction"].str.lower()
    if "frequency" in df.columns:
        df["frequency"] = df["frequency"].astype(int)

    # Keep only relevant columns to avoid surprises
    required = {"stage", "instruction", "frequency", "energy", "time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Profile CSV is missing required columns: {sorted(missing)}")

    # Group and pick row with minimum energy
    best_map: Dict[Tuple[int, str, int], Tuple[str, ...]] = {}
    group_cols = ["stage", "instruction", "frequency"]
    grouped = df.groupby(group_cols, dropna=False, sort=False)
    for (stage, instruction, freq), g in grouped:
        # Choose metric based on frequency: effective energy if freq < 1000 else real energy
        if int(freq) < 1000:
            # effective_energy = energy - p2p_power * quant_time * unit_time
            # quant_time approximated as ceil(time / unit_time)
            metrics = g.apply(
                lambda r: r["energy"]
                - p2p_power * math.ceil(float(r["time"]) / float(unit_time)) * float(unit_time),
                axis=1,
            )
        else:
            metrics = g["energy"]

        idx = metrics.idxmin()
        row = df.loc[idx]
        inst = str(instruction)
        stg = int(stage)
        frq = int(freq)

        if inst == "forward":
            attn_cfg = str(row.get("attention_configs", ""))
            mlp_cfg = str(row.get("mlp_configs", ""))
            best_map[(stg, inst, frq)] = ("forward", attn_cfg, mlp_cfg)
        elif inst == "backward":
            if use_ac:
                rec_attn_cfg = str(row.get("recompute_attention_configs", ""))
                rec_mlp_cfg = str(row.get("recompute_mlp_configs", ""))
                bwd_attn_cfg = str(row.get("attention_configs", ""))
                bwd_mlp_cfg = str(row.get("mlp_configs", ""))
                best_map[(stg, inst, frq)] = (
                    "backward",
                    rec_attn_cfg,
                    rec_mlp_cfg,
                    bwd_attn_cfg,
                    bwd_mlp_cfg,
                )
            else:
                bwd_attn_cfg = str(row.get("attention_configs", ""))
                bwd_mlp_cfg = str(row.get("mlp_configs", ""))
                best_map[(stg, inst, frq)] = ("backward", bwd_attn_cfg, bwd_mlp_cfg)
        else:
            # Unknown instruction type; skip
            continue

    return best_map


def _generate_scheds_from_freqs(
    freqs: FreqsPipeline,
    best_map: Dict[Tuple[int, str, int], Tuple[str, ...]],
    use_ac: bool,
) -> List[List[Tuple[str, ...]]]:
    """Create scheds pipeline in 1F1B order to match the freqs pipeline."""
    scheds: List[List[Tuple[str, ...]]] = []
    for stage_id, stage_freqs in enumerate(freqs):
        stage_cfgs: List[Tuple[str, ...]] = []
        for inst_name, freq in stage_freqs:
            if freq is None:
                raise ValueError(
                    f"Missing frequency for stage {stage_id} instruction {inst_name}"
                )
            key = (int(stage_id), str(inst_name).lower(), int(freq))
            cfg = best_map.get(key)
            if cfg is None:
                # Fallback: try the next higher available frequency for this stage+instruction
                stage_inst = (int(stage_id), str(inst_name).lower())
                available_freqs = sorted(
                    f for (stg, ins, f) in best_map.keys() if (stg, ins) == stage_inst
                )
                higher_or_equal = [f for f in available_freqs if f >= int(freq)]
                if higher_or_equal:
                    fallback_freq = min(higher_or_equal)
                    fallback_key = (stage_inst[0], stage_inst[1], fallback_freq)
                    cfg = best_map.get(fallback_key)
                    print(
                        f"[generate_scheds_from_freqs] No exact match for (stage={stage_id}, "
                        f"instruction={inst_name}, frequency={freq}); using higher frequency {fallback_freq}."
                    )
                if cfg is None:
                    # If no higher option, try the closest lower-or-equal frequency
                    lower_or_equal = [f for f in available_freqs if f <= int(freq)]
                    if lower_or_equal:
                        fallback_freq = max(lower_or_equal)
                        fallback_key = (stage_inst[0], stage_inst[1], fallback_freq)
                        cfg = best_map.get(fallback_key)
                        print(
                            f"[generate_scheds_from_freqs] No higher-or-equal match for (stage={stage_id}, "
                            f"instruction={inst_name}, frequency={freq}); using lower frequency {fallback_freq}."
                        )
                if cfg is None:
                    raise KeyError(
                        "No profile match for (stage={}, instruction={}, frequency={}); "
                        "available frequencies: {}".format(
                            stage_id, inst_name, freq, available_freqs
                        )
                    )
            # Sanity: shape should match instruction and AC mode
            if inst_name == "forward":
                if len(cfg) != 3 or cfg[0] != "forward":
                    raise ValueError(
                        f"Invalid forward cfg shape for key {key}: got {cfg}"
                    )
            elif inst_name == "backward":
                if use_ac and (len(cfg) != 5 or cfg[0] != "backward"):
                    raise ValueError(
                        f"Invalid backward (AC) cfg shape for key {key}: got {cfg}"
                    )
                if not use_ac and (len(cfg) != 3 or cfg[0] != "backward"):
                    raise ValueError(
                        f"Invalid backward cfg shape for key {key}: got {cfg}"
                    )
            stage_cfgs.append(cfg)
        scheds.append(stage_cfgs)
    return scheds


def _write_scheds_file(output_path: Path, scheds: List[List[Tuple[str, ...]]]) -> None:
    """Write scheds list-of-lists to a .py file compatible with existing format."""
    with output_path.open("w") as f:
        f.write("[\n")
        for stage_cfgs in scheds:
            f.write(f"{stage_cfgs},\n")
        f.write("]\n")


def main(args: Args) -> None:
    inst_profile_path = Path(args.inst_profile)
    freqs_path = Path(args.freqs_file)
    if args.output_file is not None:
        output_path = Path(args.output_file)
    else:
        # Default: same dir/name pattern replacing freqs_ with scheds_
        name = freqs_path.name.replace("freqs_pipeline", "scheds_pipeline")
        output_path = freqs_path.with_name(name)

    # Load data
    inst_df = pd.read_csv(inst_profile_path)
    use_ac = _detect_ac(inst_df, args.use_activation_checkpointing)
    freqs: FreqsPipeline = _read_top_level_list_literal(freqs_path)

    # Build mapping and generate scheds
    best_map = _build_best_config_map(inst_df, use_ac, args.p2p_power, args.unit_time)
    scheds = _generate_scheds_from_freqs(freqs, best_map, use_ac)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_scheds_file(output_path, scheds)
    print(f"Wrote scheds pipeline to: {output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))


