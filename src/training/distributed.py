"""Shared distributed data-loading utilities for recipe code."""

from __future__ import annotations

import torch


class ExactDistributedSampler(torch.utils.data.Sampler):
    """Shard deterministic evaluation without padding or duplicate samples."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank must be in [0, {self.world_size}), got {self.rank}"
            )

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.world_size - 1) // self.world_size

    def set_epoch(self, epoch):
        return None
