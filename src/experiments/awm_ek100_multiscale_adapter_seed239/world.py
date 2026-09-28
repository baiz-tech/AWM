"""Local V23-style object/time/relation world extractor."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class WorldState:
    objects: torch.Tensor
    dynamics: torch.Tensor
    relations: torch.Tensor
    scene: torch.Tensor


class UnifiedWorldModel(nn.Module):
    def __init__(self, token_dim=1280, state_dim=512, slots=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, slots, state_dim) * .02)
        self.key, self.value = nn.Linear(token_dim, state_dim), nn.Linear(token_dim, state_dim)
        self.pool = nn.MultiheadAttention(state_dim, 8, batch_first=True)
        self.pool_norm = nn.LayerNorm(state_dim)
        layer = nn.TransformerEncoderLayer(state_dim, 8, 4 * state_dim, .15, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2)
        self.position = nn.Parameter(torch.randn(1, 64, state_dim) * .02)
        self.dynamics_norm = nn.LayerNorm(state_dim)
        self.relation = nn.Sequential(nn.Linear(2 * state_dim, state_dim), nn.GELU(), nn.Linear(state_dim, state_dim))
        self.future_attn, self.future_norm = nn.MultiheadAttention(state_dim, 8, batch_first=True), nn.LayerNorm(state_dim)

    def forward(self, tokens):
        b, t, _, _ = tokens.shape
        k, v = self.key(tokens), self.value(tokens)
        q = self.queries.expand(b * t, -1, -1)
        objects, _ = self.pool(q, k.flatten(0, 1), v.flatten(0, 1), need_weights=False)
        objects = self.pool_norm(objects).reshape(b, t, -1, objects.size(-1))
        temporal = self.temporal(objects.mean(2) + self.position[:, :t])
        dynamics = self.dynamics_norm(temporal.mean(1))
        final = objects[:, -1]
        relation = self.relation(torch.cat((final.unsqueeze(2).expand(-1, -1, final.size(1), -1), final.unsqueeze(1).expand(-1, final.size(1), -1, -1)), -1))
        update, _ = self.future_attn(final, final, final, need_weights=False)
        future_objects = self.future_norm(final + update).unsqueeze(1)
        future_scene = future_objects.mean((1, 2))
        state = WorldState(objects, dynamics, relation, objects.mean((1, 2)))
        future = WorldState(future_objects, future_scene - state.scene, future_objects[:, 0].unsqueeze(2) - future_objects[:, 0].unsqueeze(1), future_scene)
        return state, future
