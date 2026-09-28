"""Local label utilities; EK100 sparse ids become contiguous classifier ids."""
from __future__ import annotations
import torch


def encode_labels(labels: torch.Tensor, vocabulary: list[int]) -> torch.Tensor:
    mapping = {int(raw): i for i, raw in enumerate(vocabulary)}
    return torch.tensor([mapping.get(int(raw), -1) for raw in labels.tolist()], dtype=torch.long)
