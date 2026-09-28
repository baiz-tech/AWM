"""Load recipe-local world models for shared Physion++ evaluations."""

from __future__ import annotations

from src.core.world_model_loader import load_model_from_adapter


def load_world_model(config, checkpoint, device, adapter_name=None):
    """Load and freeze a world model through its recipe-local adapter."""
    if adapter_name is None:
        adapter_name = config.get("evaluation", {}).get("world_model_adapter")
    if not adapter_name:
        raise ValueError(
            "evaluation.world_model_adapter must name a recipe-local world-model adapter"
        )
    return load_model_from_adapter(config, checkpoint, device, adapter_name)
