from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def flatten_feature(x: torch.Tensor) -> torch.Tensor:
    """Pool a token tensor into a compact, probe-friendly vector."""
    x = x.float()
    if x.ndim == 1:
        return x
    dims = tuple(range(1, x.ndim - 1))
    if not dims:
        return x
    return torch.cat((x.mean(dims), x.amax(dims), x.std(dims).nan_to_num()), dim=-1)


def load_state(path: str | Path, key: str = "model") -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if key in payload and isinstance(payload[key], dict):
        return payload[key]
    return payload

