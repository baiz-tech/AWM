"""Frozen V-JEPA 2 encoder plus its native masked-token predictor."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from external.vjepa2.app.vjepa.utils import init_video_model
from external.vjepa2.src.masks.utils import apply_masks
from external.vjepa2.src.utils.checkpoint_loader import robust_checkpoint_loader


def _clean_state_dict(state):
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.backbone.", "backbone.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        cleaned[key] = value
    return cleaned


def make_temporal_half_masks(
    batch_size,
    *,
    total_frames=32,
    visible_frames=16,
    tubelet_size=2,
    crop_size=256,
    patch_size=16,
    device=None,
):
    """Make masks where 16 visible frames predict the following 16 frames."""
    if total_frames % tubelet_size or visible_frames % tubelet_size:
        raise ValueError("frame counts must be divisible by tubelet_size")
    if visible_frames <= 0 or visible_frames >= total_frames:
        raise ValueError("visible_frames must be strictly between 0 and total_frames")
    if crop_size % patch_size:
        raise ValueError("crop_size must be divisible by patch_size")

    temporal_tokens = total_frames // tubelet_size
    visible_temporal_tokens = visible_frames // tubelet_size
    spatial_tokens = (crop_size // patch_size) ** 2
    ids = torch.arange(temporal_tokens * spatial_tokens, device=device)
    time_ids = ids // spatial_tokens
    context = ids[time_ids < visible_temporal_tokens]
    target = ids[time_ids >= visible_temporal_tokens]
    return (
        context.view(1, -1).repeat(batch_size, 1).long(),
        target.view(1, -1).repeat(batch_size, 1).long(),
    )


class NativeVJEPA2Predictor(nn.Module):
    """Expose a clean current-16 -> future-16 latent prediction interface."""

    def __init__(self, encoder, predictor, clip_frames=16, tubelet_size=2, crop_size=256, patch_size=16):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.clip_frames = int(clip_frames)
        self.total_frames = 2 * self.clip_frames
        self.tubelet_size = int(tubelet_size)
        self.crop_size = int(crop_size)
        self.patch_size = int(patch_size)
        self.temporal_steps = self.clip_frames // self.tubelet_size
        self.spatial_tokens = (self.crop_size // self.patch_size) ** 2
        self.tokens_per_chunk = self.temporal_steps * self.spatial_tokens

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.predictor.train(mode)
        return self

    def forward(self, current, future):
        if current.shape != future.shape:
            raise ValueError(f"current/future shapes differ: {current.shape} vs {future.shape}")
        if current.ndim != 5 or current.size(2) != self.clip_frames:
            raise ValueError(f"expected [B,C,{self.clip_frames},H,W], got {tuple(current.shape)}")

        combo = torch.cat([current, future], dim=2)
        masks_x, masks_y = make_temporal_half_masks(
            combo.size(0),
            total_frames=self.total_frames,
            visible_frames=self.clip_frames,
            tubelet_size=self.tubelet_size,
            crop_size=self.crop_size,
            patch_size=self.patch_size,
            device=combo.device,
        )
        with torch.no_grad():
            context = self.encoder([combo], masks=[[masks_x]])[0][0]
            # Native V-JEPA teacher semantics: encode the complete video, then
            # select target positions. Encoding only masks_y would remove the
            # visible context from the target representation.
            full_target = self.encoder([combo])[0]
            full_target = F.layer_norm(full_target, (full_target.size(-1),))
            target = apply_masks(full_target, [masks_y])
        predicted = self.predictor([[context]], [[masks_x]], [[masks_y]])[0][0]
        return predicted, target, context

    def _masks(self, batch_size, device):
        return make_temporal_half_masks(
            batch_size,
            total_frames=self.total_frames,
            visible_frames=self.clip_frames,
            tubelet_size=self.tubelet_size,
            crop_size=self.crop_size,
            patch_size=self.patch_size,
            device=device,
        )

    def encode_current(self, current):
        """Encode Current without reading any Future pixels."""
        if current.ndim != 5 or current.size(2) != self.clip_frames:
            raise ValueError(f"expected [B,C,{self.clip_frames},H,W], got {tuple(current.shape)}")
        with torch.no_grad():
            return self.encoder([current])[0]

    def predict_from_context(self, context):
        """Apply the trained one-chunk predictor to an already encoded context."""
        masks_x, masks_y = self._masks(context.size(0), context.device)
        return self.predictor([[context]], [[masks_x]], [[masks_y]])[0][0]

    def serial_rollout(self, initial_context, chunk_mask):
        """Roll the naive predictor through every requested Future chunk.

        Unlike the multi-Future recipe, the naive checkpoint has no learned
        rollout adapter.  Its predicted target latent is therefore used
        directly as the next context, which keeps evaluation checkpoint
        compatible and prevents any Future observation from leaking in.
        """
        if chunk_mask.ndim != 2 or chunk_mask.size(0) != initial_context.size(0):
            raise ValueError("chunk_mask must be [B,K] and match the context batch")
        context = initial_context
        predictions = []
        for chunk_index in range(chunk_mask.size(1)):
            predicted = self.predict_from_context(context)
            active = chunk_mask[:, chunk_index].view(-1, 1, 1)
            predicted = predicted.masked_fill(~active, 0.0)
            predictions.append(predicted)
            context = predicted
        return torch.stack(predictions, dim=1)

    def rollout_to_length(self, current, chunk_mask):
        """Closed-loop OCP inference from real Current to the video end."""
        return self.serial_rollout(self.encode_current(current), chunk_mask)


def build_model(device, cfg_model, cfg_data, checkpoint, checkpoint_key="target_encoder"):
    clip_frames = int(cfg_data.get("clip_frames", 16))
    total_frames = 2 * clip_frames
    encoder, predictor = init_video_model(
        device=device,
        patch_size=int(cfg_model.get("patch_size", 16)),
        max_num_frames=total_frames,
        tubelet_size=int(cfg_model.get("tubelet_size", 2)),
        model_name=cfg_model.get("model_name", "vit_huge"),
        crop_size=int(cfg_data.get("crop_size", 256)),
        pred_depth=int(cfg_model.get("pred_depth", 12)),
        pred_num_heads=int(cfg_model.get("pred_num_heads", 12)),
        pred_embed_dim=int(cfg_model.get("pred_embed_dim", 384)),
        uniform_power=bool(cfg_model.get("uniform_power", True)),
        use_mask_tokens=True,
        # This recipe has one fixed future-target mask. Extra mask tokens would
        # never participate in the loss and therefore break strict DDP.
        num_mask_tokens=int(cfg_model.get("num_mask_tokens", 1)),
        zero_init_mask_tokens=bool(cfg_model.get("zero_init_mask_tokens", True)),
        use_sdpa=bool(cfg_model.get("use_sdpa", True)),
        use_rope=bool(cfg_model.get("use_rope", True)),
        use_silu=bool(cfg_model.get("use_silu", False)),
        use_pred_silu=bool(cfg_model.get("use_pred_silu", False)),
        wide_silu=bool(cfg_model.get("wide_silu", True)),
        use_activation_checkpointing=bool(cfg_model.get("use_activation_checkpointing", True)),
    )
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    source_epoch = -1
    message = None
    if rank == 0:
        payload = robust_checkpoint_loader(checkpoint, map_location=torch.device("cpu"))
        key = checkpoint_key if checkpoint_key in payload else "encoder"
        if key not in payload:
            raise KeyError(f"checkpoint has neither {checkpoint_key!r} nor 'encoder'")
        message = encoder.backbone.load_state_dict(_clean_state_dict(payload[key]), strict=False)
        critical_prefixes = ("patch_embed.", "blocks.", "norm.", "norms_block.")
        critical = [name for name in message.missing_keys if name.startswith(critical_prefixes)]
        if critical:
            raise RuntimeError(f"checkpoint misses core encoder weights: {critical[:8]}")
        source_epoch = int(payload.get("epoch", -1))
        del payload

    if distributed:
        for parameter in encoder.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in encoder.buffers():
            dist.broadcast(buffer.data, src=0)
        epoch_tensor = torch.tensor([source_epoch], dtype=torch.long, device=device)
        dist.broadcast(epoch_tensor, src=0)
        source_epoch = int(epoch_tensor.item())

    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    model = NativeVJEPA2Predictor(
        encoder,
        predictor,
        clip_frames=clip_frames,
        tubelet_size=int(cfg_model.get("tubelet_size", 2)),
        crop_size=int(cfg_data.get("crop_size", 256)),
        patch_size=int(cfg_model.get("patch_size", 16)),
    ).to(device)
    return model, source_epoch, message
