from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from src.experiments.awm_ek100_multiscale_adapter_seed239.world import UnifiedWorldModel, WorldState
from src.experiments.awm_ek100_multiscale_adapter_seed239.model import SharedMultiScaleWorldExtractor


VARIANTS = ("E0", "E1", "E2", "E3", "E4")


def _pool512(x: torch.Tensor) -> torch.Tensor:
    """Parameter-free channel pooling used by every variant interface."""
    if x.size(-1) == 512:
        return x
    return torch.nn.functional.adaptive_avg_pool1d(x.unsqueeze(1), 512).squeeze(1)


def state_feature(state: WorldState) -> torch.Tensor:
    """Convert a world state to the common [B,512] readout interface."""
    objects = state.objects.amax((1, 2))
    dynamics = state.dynamics
    relation = state.relations.mean((1, 2))
    scene = state.scene
    return objects + dynamics + relation + scene


class AttributionHead(nn.Module):
    """Identical trainable head for E0--E4 (input and hidden width are fixed)."""
    def __init__(self, verb_classes: int, noun_classes: int, action_pairs: list[tuple[int, int]], hidden_dim: int = 512):
        super().__init__()
        self.verb = nn.Linear(hidden_dim, verb_classes)
        self.noun = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, noun_classes))
        self.action = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(action_pairs)))
        pairs = torch.tensor(action_pairs, dtype=torch.long)
        self.register_buffer("pv", pairs[:, 0] if len(pairs) else torch.empty(0, dtype=torch.long))
        self.register_buffer("pn", pairs[:, 1] if len(pairs) else torch.empty(0, dtype=torch.long))

    def forward(self, feature: torch.Tensor):
        verb, noun = self.verb(feature), self.noun(feature)
        if self.pv.numel():
            action = verb[:, self.pv] + noun[:, self.pn] + 0.5 * self.action(feature)
        else:
            action = self.action(feature)
        return verb, noun, action


class AttributionExtractor(nn.Module):
    """Frozen feature paths corresponding exactly to E0--E4."""
    def __init__(self, variant: str, readout_world: UnifiedWorldModel | None = None, shared: SharedMultiScaleWorldExtractor | None = None):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}; expected {VARIANTS}")
        self.variant = variant
        self.readout_world = readout_world
        self.shared = shared
        if variant in ("E1", "E3", "E4") and readout_world is None:
            raise ValueError("readout_world is required for E1/E3/E4")
        if variant in ("E2", "E3", "E4") and shared is None:
            raise ValueError("shared extractor is required for E2/E3/E4")
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        grid = grid.float()
        if self.variant == "E0":
            return _pool512(grid.mean((1, 2)))
        if self.variant == "E1":
            return state_feature(self.readout_world(grid)[0])
        shared_state, _ = self.shared.world(grid)
        if self.variant == "E2":
            return state_feature(shared_state)
        refined = self.shared(grid)
        readout_feature = state_feature(self.readout_world(refined)[0])
        if self.variant == "E3":
            return readout_feature
        # E4 exposes the complete residual HASP interface: the readout state
        # is explicitly augmented by the shared Entity/Dynamic/Relation state.
        return readout_feature + state_feature(shared_state)


@dataclass
class LoadedComponents:
    readout_world: UnifiedWorldModel
    shared: SharedMultiScaleWorldExtractor


def load_components(readout_checkpoint: str, adapter_checkpoint: str, device: torch.device) -> LoadedComponents:
    readout_payload = torch.load(readout_checkpoint, map_location="cpu", weights_only=False)
    readout_world = UnifiedWorldModel(1280, 512, 8)
    readout_world.load_state_dict({k.removeprefix("world."): v for k, v in readout_payload["model"].items() if k.startswith("world.")}, strict=True)
    adapter_payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
    shared = SharedMultiScaleWorldExtractor()
    shared.load_state_dict(adapter_payload["shared"], strict=True)
    return LoadedComponents(readout_world.to(device).eval(), shared.to(device).eval())
