"""Load the frozen structured probe for CLEVRER v2 trajectory export."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .model import FUTURE_STEPS, CLEVRERDecoder


class StructuredProbeRunner:
    def __init__(self, model: CLEVRERDecoder, checkpoint: Path):
        self.model = model
        self.checkpoint = checkpoint

    @torch.no_grad()
    def __call__(self, context: torch.Tensor, future: torch.Tensor):
        outputs = self.model(context, future)
        object_valid = outputs["presence_logits"].sigmoid().ge(0.5)
        sparse_rows = object_valid.sum(dim=1).lt(2)
        if sparse_rows.any():
            row_indices = sparse_rows.nonzero(as_tuple=False).squeeze(1)
            top_objects = outputs["presence_logits"][row_indices].topk(k=2, dim=1).indices
            object_valid[row_indices] = False
            object_valid[row_indices[:, None], top_objects] = True
        first, second = self.model.probes.pair_indices[:, 0], self.model.probes.pair_indices[:, 1]
        pair_valid = object_valid[:, first] & object_valid[:, second]
        first_probabilities = outputs["first_contact_logits"].softmax(-1)
        time = torch.arange(FUTURE_STEPS + 1, device=context.device, dtype=torch.float32)
        expected_first_contact = (first_probabilities * time).sum(-1) / FUTURE_STEPS
        return {
            "object_tokens": outputs["object_tokens"],
            "pair_tokens": outputs["pair_tokens"],
            "object_valid": object_valid,
            "pair_valid": pair_valid,
            "presence_logits": outputs["presence_logits"],
            "color_logits": outputs["color_logits"],
            "material_logits": outputs["material_logits"],
            "shape_logits": outputs["shape_logits"],
            "trajectory_2d": outputs["trajectory_2d"],
            "pair_distance_2d": outputs["pair_distance_2d"],
            "contact_gt_event": outputs["contact_gt_event"],
            "first_contact_logits": outputs["first_contact_logits"],
            "time_to_contact": expected_first_contact,
        }


def load_probe_runner(config, device):
    evaluation = config.get("evaluation") or {}
    configured = evaluation.get("probe_checkpoint")
    checkpoint = Path(os.environ.get("PROBE_CHECKPOINT", configured or ""))
    if not str(checkpoint) or not checkpoint.is_file():
        raise FileNotFoundError(f"structured probe checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("protocol") != "clevrer_fullpatch_dynamics_decoder_v1":
        raise ValueError(f"unexpected decoder protocol: {payload.get('protocol')!r}")
    model = CLEVRERDecoder(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    metadata = {
        "protocol": payload["protocol"],
        "output_protocol": "clevrer_fullpatch_structured_supervised_outputs_v2",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "object_token_dim": model.decoder.hidden_dim,
        "pair_token_dim": model.decoder.hidden_dim,
        "z_dyn_shape": [8, 8, model.decoder.hidden_dim],
        "object_tokens": 6,
        "pair_tokens": 15,
        "future_steps": FUTURE_STEPS,
        "fields": [
            "object_tokens", "pair_tokens", "object_valid", "pair_valid",
            "presence_logits", "color_logits", "material_logits", "shape_logits",
            "trajectory_2d", "pair_distance_2d", "contact_gt_event",
            "first_contact_logits", "time_to_contact",
        ],
    }
    return StructuredProbeRunner(model, checkpoint), metadata
