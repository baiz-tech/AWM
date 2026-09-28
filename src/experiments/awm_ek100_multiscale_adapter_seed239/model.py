"""V23-style EK100-only residual world adapter.

The frozen readout is trained on EK100 first.  This module then optimizes only
the shared temporal/object/relation residual using EK100 labels, never Physion.
"""
from __future__ import annotations
import torch
from torch import nn
from .world import UnifiedWorldModel
from .readout import EK100MultiScaleReadout


class SharedMultiScaleWorldExtractor(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.world = UnifiedWorldModel(1280, dim, 8)
        self.domain = nn.Embedding(1, dim)
        self.norm = nn.LayerNorm(1280)
        self.delta = nn.Sequential(nn.Linear(1280 + 2 * dim, dim), nn.GELU(), nn.Linear(dim, 1280))
        nn.init.normal_(self.delta[-1].weight, std=1e-3)
        nn.init.zeros_(self.delta[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(1e-2))

    def forward(self, tokens):
        state, future = self.world(tokens.float())
        scene = future.scene + state.relations.mean((1, 2)) + self.domain.weight[0]
        b, t, n, _ = tokens.shape
        evidence = torch.cat((self.norm(tokens.float()), scene[:, None, None].expand(b, t, n, -1), state.dynamics[:, None, None].expand(b, t, n, -1)), -1)
        return tokens.float() + self.alpha.tanh() * self.delta(evidence)


class EK100OnlyWorldModel(nn.Module):
    def __init__(self, verb_classes, noun_classes, pairs):
        super().__init__()
        self.shared = SharedMultiScaleWorldExtractor()
        self.readout = EK100MultiScaleReadout(verb_classes, noun_classes, pairs)
        for parameter in self.readout.parameters():
            parameter.requires_grad_(False)

    def forward(self, grid):
        return self.readout(self.shared(grid))
