"""Small VideoMAE/VideoMAE V2 adapter and OCP readout.

The official VideoMAE V2 repositories expose slightly different constructors.
This module deliberately isolates that dependency behind ``build_encoder``;
the rest of the recipe only consumes ``[B, T', N, D]`` patch tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
import sys


class VideoMAEEncoder(nn.Module):
    """Wrap a HuggingFace-compatible VideoMAE model and return patch tokens."""

    def __init__(self, model, *, remove_cls_token=False, freeze=True):
        super().__init__()
        self.model = model
        self.remove_cls_token = bool(remove_cls_token)
        self.freeze = bool(freeze)
        if self.freeze:
            self.eval()
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, video):
        if video.ndim != 5:
            raise ValueError(f"expected [B,T,C,H,W], got {tuple(video.shape)}")
        # Official VideoMAEv2 models consume [B,C,T,H,W] and expose
        # ``forward_features``; HF models use [B,T,C,H,W].
        if hasattr(self.model, "patch_embed") and hasattr(self.model, "blocks"):
            tokens = self.model.patch_embed(video)
            if getattr(self.model, "pos_embed", None) is not None:
                pos = self.model.pos_embed
                if pos.size(1) != tokens.size(1):
                    raise ValueError(f"VideoMAEv2 positional length {pos.size(1)} != token length {tokens.size(1)}")
                tokens = tokens + pos.to(device=tokens.device, dtype=tokens.dtype)
            tokens = self.model.pos_drop(tokens) if hasattr(self.model, "pos_drop") else tokens
            for block in self.model.blocks:
                tokens = block(tokens)
            if getattr(self.model, "fc_norm", None) is not None:
                tokens = self.model.fc_norm(tokens)
            elif getattr(self.model, "norm", None) is not None:
                tokens = self.model.norm(tokens)
        elif hasattr(self.model, "forward_features"):
            tokens = self.model.forward_features(video)
        else:
            output = self.model(pixel_values=video.permute(0, 2, 1, 3, 4))
            tokens = output.last_hidden_state
        if tokens.ndim != 3:
            raise ValueError(f"VideoMAE encoder must return [B,N,D], got {tuple(tokens.shape)}")
        if self.remove_cls_token and tokens.size(1) > 0:
            tokens = tokens[:, 1:]
        return tokens


class OCPReadout(nn.Module):
    """Video-level OCP classifier over pooled Current/Future latent tokens."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * int(input_dim)),
            nn.Linear(2 * int(input_dim), int(hidden_dim)), nn.GELU(),
            nn.Dropout(float(dropout)), nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, context_tokens, future_tokens):
        context = context_tokens.mean(dim=tuple(range(1, context_tokens.ndim - 1)))
        future = future_tokens.mean(dim=tuple(range(1, future_tokens.ndim - 1)))
        return self.net(torch.cat((context, future), dim=-1)).squeeze(-1)


def build_encoder(cfg):
    """Build the official native VideoMAEv2 model and load its ``.pth``."""
    checkpoint = cfg.get("checkpoint")
    official_repo = cfg.get("official_repo")
    if not checkpoint:
        raise ValueError("model.checkpoint must point to an official VideoMAEv2 .pth file")
    if not official_repo:
        raise ValueError("model.official_repo must point to a clone of OpenGVLab/VideoMAEv2")
    repo = Path(official_repo).expanduser().resolve()
    if not (repo / "models" / "modeling_finetune.py").is_file():
        raise FileNotFoundError(f"official VideoMAEv2 repo not found: {repo}")
    sys.path.insert(0, str(repo))
    from models.modeling_finetune import vit_base_patch16_224, vit_small_patch16_224
    architecture = str(cfg.get("architecture", "vit_b")).lower()
    factories = {"vit_b": vit_base_patch16_224, "vit_base": vit_base_patch16_224,
                 "vit_s": vit_small_patch16_224, "vit_small": vit_small_patch16_224}
    if architecture not in factories:
        raise ValueError(f"unsupported VideoMAEv2 architecture: {architecture}")
    # Official implementation multiplies ``head.weight`` during init, so its
    # ``num_classes=0`` path is broken (Identity has no weight).  Construct a
    # one-class head, then discard it before loading the backbone checkpoint.
    model = factories[architecture](num_classes=1, all_frames=int(cfg.get("all_frames", 16)),
                                    tubelet_size=int(cfg.get("tubelet_size", 2)),
                                    use_mean_pooling=True)
    model.head = nn.Identity()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("module", payload.get("model", payload))
    state = {k.removeprefix("backbone."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"norm.weight", "norm.bias", "head.weight", "head.bias"}
    allowed_unexpected = {"fc_norm.weight", "fc_norm.bias", "head.weight", "head.bias"}
    if set(missing) - allowed_missing or set(unexpected) - allowed_unexpected:
        raise RuntimeError(f"VideoMAEv2 checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return VideoMAEEncoder(
        model, remove_cls_token=bool(cfg.get("remove_cls_token", False)),
        freeze=bool(cfg.get("freeze_encoder", True)),
    )
