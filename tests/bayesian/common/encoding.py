"""Configuration encoding, decoding, and candidate generation."""

from __future__ import annotations

from typing import List, Dict

import numpy as np


FREQ_IDX = 0
SM_IDX = 1
OVERLAP_IDX = 2
BLOCK_SIZE = 1024


def encode_cfg(partition_test, cfg: Dict[str, int]) -> np.ndarray:
    """
    Encode configuration to an index vector [freq_idx, sm_idx, overlap_idx].

    cfg keys:
      - freq: actual GPU core frequency (per partition_test.FREQ_VALUES)
      - sm: actual SM count (1..20)
      - block: CUDA block size (512 or 1024)
      - overlap: overlap window tuple from partition_test.OVERLAP_WINDOWS
    """
    freq_idx = partition_test.FREQ_VALUES.index(cfg["freq"])
    sm_idx = partition_test.SM_VALUES.index(cfg["sm"])
    overlap_idx = partition_test.OVERLAP_WINDOWS.index(cfg["overlap"])
    return np.array([freq_idx, sm_idx, overlap_idx], dtype=np.int64)


def one_hot_encode(partition_test, x: np.ndarray) -> np.ndarray:
    """
    One-hot encode categorical features (overlap) and keep freq/SM as numeric.

    x: [freq_idx, sm_idx, overlap_idx]
    """
    freq_mhz = float(partition_test.FREQ_VALUES[int(x[FREQ_IDX])])
    numeric = np.array([freq_mhz, x[SM_IDX]], dtype=np.float32)
    overlap_one_hot = np.zeros(len(partition_test.OVERLAP_WINDOWS), dtype=np.float32)
    overlap_one_hot[int(x[OVERLAP_IDX])] = 1.0
    return np.concatenate([numeric, overlap_one_hot], axis=0)


def decode_vec(partition_test, x: np.ndarray) -> Dict[str, int]:
    """Decode an index vector [freq_idx, sm_idx, overlap_idx] back to a config dict."""
    freq_idx = int(np.clip(round(float(x[FREQ_IDX])), 0, len(partition_test.FREQ_VALUES) - 1))
    sm_idx = int(np.clip(round(float(x[SM_IDX])), 0, len(partition_test.SM_VALUES) - 1))
    overlap_idx = int(np.clip(round(float(x[OVERLAP_IDX])), 0, len(partition_test.OVERLAP_WINDOWS) - 1))
    return {
        "freq": partition_test.FREQ_VALUES[freq_idx],
        "sm": partition_test.SM_VALUES[sm_idx],
        "block": BLOCK_SIZE,
        "overlap": partition_test.OVERLAP_WINDOWS[overlap_idx],
    }


def generate_all_configurations(partition_test) -> List[np.ndarray]:
    configs: List[np.ndarray] = []
    for freq_idx in range(len(partition_test.FREQ_VALUES)):
        for sm_idx in range(len(partition_test.SM_VALUES)):
            for overlap_idx in range(len(partition_test.OVERLAP_WINDOWS)):
                configs.append(
                    np.array([freq_idx, sm_idx, overlap_idx], dtype=np.int64)
                )
    return configs


def is_config_in_dataset(config: np.ndarray, dataset: np.ndarray) -> bool:
    return any(np.array_equal(config, x) for x in dataset)


def get_unevaluated_configs(all_configs: List[np.ndarray], X_train: np.ndarray) -> List[np.ndarray]:
    """Return configs from all_configs that are not in X_train. O(n+m) via set lookup."""
    seen = set(map(tuple, X_train))
    return [c for c in all_configs if tuple(c) not in seen]
