"""Shared raw-frame sampling protocol for Test3 cache/target generation."""
from __future__ import annotations
import numpy as np

RAW_FRAMES = 32
RAW_STEP = 4
CONTEXT_RAW_FRAMES = 24

def sampled_indices(window_start: int) -> tuple[np.ndarray, np.ndarray]:
    values = int(window_start) + np.arange(RAW_FRAMES, dtype=np.int64) * RAW_STEP
    return values[:CONTEXT_RAW_FRAMES], values[CONTEXT_RAW_FRAMES:]
