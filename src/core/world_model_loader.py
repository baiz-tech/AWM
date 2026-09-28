"""Unified world-model contract used by shared evaluations."""

from __future__ import annotations

import importlib


REQUIRED_METHODS = ("encode_current", "predict_next", "encode_target")
REQUIRED_ATTRIBUTES = (
    "clip_frames",
    "temporal_steps",
    "spatial_tokens",
    "tubelet_size",
    "crop_size",
    "patch_size",
)


def prepare_world_model(model, adapter_name):
    """Validate, freeze, and return a model implementing the shared contract."""
    for name in REQUIRED_METHODS:
        method = getattr(model, name, None)
        if method is None or not callable(method):
            raise AttributeError(
                f"world-model adapter {adapter_name!r} returned a model without callable {name}"
            )
    for name in REQUIRED_ATTRIBUTES:
        value = getattr(model, name, None)
        if not isinstance(value, int) or value <= 0:
            raise AttributeError(
                f"world-model adapter {adapter_name!r} returned a model without positive int {name}"
            )
    expected_tokens = model.temporal_steps * model.spatial_tokens
    tokens_per_chunk = getattr(model, "tokens_per_chunk", expected_tokens)
    if int(tokens_per_chunk) != expected_tokens:
        raise ValueError(
            f"tokens_per_chunk={tokens_per_chunk} != temporal_steps*spatial_tokens={expected_tokens}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def load_model_from_adapter(config, checkpoint, device, adapter_name):
    """Load a model through the single shared adapter entrypoint."""
    adapter = importlib.import_module(str(adapter_name))
    loader = getattr(adapter, "load_world_model", None)
    if loader is None or not callable(loader):
        raise AttributeError(
            f"world-model adapter {adapter_name!r} must define callable load_world_model"
        )
    loaded = loader(config, checkpoint, device)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise TypeError(
            f"world-model adapter {adapter_name!r} must return (model, metadata)"
        )
    model, metadata = loaded
    if not isinstance(metadata, dict):
        raise TypeError(f"world-model adapter {adapter_name!r} metadata must be a dict")
    return prepare_world_model(model, adapter_name), metadata


def configured_model_interface(config, *, required=True):
    """Return the recipe-neutral model interface declared by a config.

    CLEVRER used to keep this value under ``qa.world_model_adapter``.  The
    interface is a property of the world model, not of a QA task, so new
    configs declare it under ``evaluation.world_model_adapter``.  The old
    location remains a read-only compatibility fallback for existing runs.
    """
    evaluation = config.get("evaluation") or {}
    name = evaluation.get("world_model_adapter")
    if not name:
        name = (config.get("qa") or {}).get("world_model_adapter")
    if required and not name:
        raise ValueError(
            "evaluation.world_model_adapter must name the recipe's generic model interface"
        )
    return name


def load_configured_model(config, checkpoint, device):
    """Load the generic model interface declared by ``evaluation``."""
    adapter_name = configured_model_interface(config)
    return load_model_from_adapter(config, checkpoint, device, adapter_name)
