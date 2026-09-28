"""Shared-evaluation world-model adapter for ``vjepa2_naive``."""

from __future__ import annotations

from pathlib import Path

import torch

from src.experiments.vjepa2_clevrer_naive_world_model_seed239.model import build_model


def load_world_model(config, checkpoint, device):
    """Build and load the naive predictor for shared evaluation code."""
    meta = config.get("meta", {})
    model, source_epoch, load_message = build_model(
        device=device,
        cfg_model=config.get("model", {}),
        cfg_data=config.get("data", {}),
        checkpoint=meta["pretrain_checkpoint"],
        checkpoint_key=meta.get("encoder_checkpoint_key", "target_encoder"),
    )

    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_protocol = config.get("protocol", {}).get("name")
    if expected_protocol is not None and payload.get("protocol") != expected_protocol:
        raise ValueError(
            f"checkpoint protocol {payload.get('protocol')!r} != expected {expected_protocol!r}"
        )
    if "predictor" not in payload:
        raise KeyError("vjepa2_naive checkpoint must contain 'predictor'")
    model.predictor.load_state_dict(payload["predictor"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    metadata = {
        "interface_version": "world_model_v1",
        "token_layout": "frame_tokens",
        "feature_dim": int(model.predictor.embed_dim) if hasattr(model.predictor, "embed_dim") else None,
        "supports": ["encode", "predict", "rollout"],
        "source_epoch": int(source_epoch),
        "load_message": str(load_message),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_protocol": payload.get("protocol"),
        "model_label": "Naive V-JEPA 2 predicted future",
        "rollout_adapter": "identity (naive checkpoint has no learned multi-Future adapter)",
    }
    return model, metadata


def load_qa_world_model(config, checkpoint, device):
    """Compatibility entrypoint used by CLEVRER QA export."""
    return load_world_model(config, checkpoint, device)
