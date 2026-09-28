"""Load the frozen structured probe for CLEVRER v2 trajectory export."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .model import CURRENT_STEPS, FUTURE_STEPS, OUTPUT_STEPS, CLEVRERDecoder


class StructuredProbeRunner:
    def __init__(self, model: CLEVRERDecoder, checkpoint: Path):
        self.model = model
        self.checkpoint = checkpoint

    @torch.no_grad()
    def __call__(self, context: torch.Tensor, future: torch.Tensor):
        tokens = torch.cat([context, future], dim=1)
        source_ids = torch.cat([
            torch.zeros(context.size(0), context.size(1), dtype=torch.long, device=context.device),
            torch.ones(future.size(0), future.size(1), dtype=torch.long, device=context.device),
        ], dim=1)
        outputs = self.model(tokens, source_ids)
        object_valid = outputs["presence_logits"].sigmoid().ge(0.5)
        sparse_rows = object_valid.sum(dim=1).lt(2)
        if sparse_rows.any():
            row_indices = sparse_rows.nonzero(as_tuple=False).squeeze(1)
            top_objects = outputs["presence_logits"][row_indices].topk(k=2, dim=1).indices
            object_valid[row_indices] = False
            object_valid[row_indices[:, None], top_objects] = True
        first, second = self.model.probes.pair_indices[:, 0], self.model.probes.pair_indices[:, 1]
        pair_valid = object_valid[:, first] & object_valid[:, second]
        # Keep the existing 16-step QA contract while also exporting the new
        # current+future predictions. Probe training/evaluation uses the full outputs.
        future_slice = slice(CURRENT_STEPS, OUTPUT_STEPS)
        future_first_contact_logits = torch.cat(
            [
                outputs["first_contact_logits"][..., CURRENT_STEPS:OUTPUT_STEPS],
                outputs["first_contact_logits"][..., OUTPUT_STEPS:],
            ],
            dim=-1,
        )
        first_probabilities = future_first_contact_logits.softmax(-1)
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
            "trajectory_2d": outputs["trajectory_2d"][..., future_slice, :],
            "pair_distance_2d": outputs["pair_distance_2d"][..., future_slice],
            "contact_gt_event": outputs["contact_gt_event"][..., future_slice],
            "first_contact_logits": future_first_contact_logits,
            "time_to_contact": expected_first_contact,
            "trajectory_2d_all": outputs["trajectory_2d"],
            "pair_distance_2d_all": outputs["pair_distance_2d"],
            "contact_gt_event_all": outputs["contact_gt_event"],
            "first_contact_logits_all": outputs["first_contact_logits"],
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
        "z_dyn_shape": [16, model.decoder.num_slots, model.decoder.hidden_dim],
        "object_tokens": 6,
        "pair_tokens": 15,
        "current_steps": 16,
        "future_steps": FUTURE_STEPS,
        "output_steps": OUTPUT_STEPS,
        "fields": [
            "object_tokens", "pair_tokens", "object_valid", "pair_valid",
            "presence_logits", "color_logits", "material_logits", "shape_logits",
            "trajectory_2d", "pair_distance_2d", "contact_gt_event",
            "first_contact_logits", "time_to_contact",
            "trajectory_2d_all", "pair_distance_2d_all", "contact_gt_event_all",
            "first_contact_logits_all",
        ],
    }
    return StructuredProbeRunner(model, checkpoint), metadata
