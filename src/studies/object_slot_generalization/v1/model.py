"""Object-centric slot extractor used by the isolated generalization study.

The model consumes frozen V-JEPA patch features only.  Future features are
never passed to the entity extractor; they can optionally be used as targets
for the transition/reconstruction losses in a later training stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, NamedTuple

import torch
from torch import nn
import torch.nn.functional as F


class SlotOutput(NamedTuple):
    slots: torch.Tensor                 # [B,T,S,H]
    assignment: torch.Tensor            # [B,T,N,S]
    patch_reconstruction: torch.Tensor  # [B,T,N,D]
    slot_reconstruction: torch.Tensor   # [B,T,S,N,D], per-slot contribution
    presence_logits: torch.Tensor       # [B,T,S]
    centroids: torch.Tensor             # [B,T,S,2], normalized patch coordinates


class CompetitiveSlotAttention(nn.Module):
    """Slot attention with explicit competition over slots for each patch."""

    def __init__(self, input_dim: int, hidden_dim: int, slots: int = 8,
                 iterations: int = 3, heads: int = 8) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.slots, self.iterations, self.hidden_dim = slots, iterations, hidden_dim
        self.norm_x, self.norm_s = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.key, self.value, self.query = (nn.Linear(hidden_dim, hidden_dim, bias=False)
                                            for _ in range(3))
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 4 * hidden_dim),
                                 nn.GELU(), nn.Linear(4 * hidden_dim, hidden_dim))
        self.project = nn.Linear(input_dim, hidden_dim)
        self.slot_init = nn.Parameter(torch.randn(1, slots, hidden_dim) * 0.02)
        self.presence = nn.Linear(hidden_dim, 1)
        self.patch_decoder = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                                           nn.GELU(), nn.Linear(hidden_dim, input_dim))
        self.pos = nn.Parameter(torch.randn(1, 1, 256, hidden_dim) * 0.02)

    def forward(self, tokens: torch.Tensor) -> SlotOutput:
        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [B,T,N,D]")
        b, t, n, _ = tokens.shape
        x = self.project(tokens.float())
        if n <= self.pos.size(2):
            x = x + self.pos[:, :, :n]
        else:
            # Interpolation keeps the module usable with non-16x16 patch grids.
            p = F.interpolate(self.pos.squeeze(0).permute(0, 2, 1), n, mode="linear", align_corners=False)
            x = x + p.permute(0, 2, 1).unsqueeze(0)
        x = x.reshape(b * t, n, self.hidden_dim)
        slots = self.slot_init.expand(b * t, -1, -1).contiguous()
        k, v = self.key(self.norm_x(x)), self.value(self.norm_x(x))
        assignment = None
        for _ in range(self.iterations):
            q = self.query(self.norm_s(slots))
            # Competition is across slots (dim=-1), forcing patch ownership.
            logits = torch.einsum("bsh,bnh->bns", q, k) / self.hidden_dim**0.5
            assignment = logits.softmax(-1)
            updates = torch.einsum("bns,bnh->bsh", assignment, v)
            updates = updates / assignment.sum(1).unsqueeze(-1).clamp_min(1e-6)
            slots = self.gru(updates.reshape(-1, self.hidden_dim), slots.reshape(-1, self.hidden_dim))
            slots = slots.reshape(b * t, self.slots, self.hidden_dim) + self.mlp(slots.reshape(b * t, self.slots, self.hidden_dim))
        presence = self.presence(slots).squeeze(-1)
        slot_reconstruction = self.patch_decoder(slots).unsqueeze(2).expand(-1, -1, n, -1)
        reconstruction = torch.einsum("bns,bsnd->bnd", assignment, slot_reconstruction)
        # Fixed normalized coordinates are used only to expose localization diagnostics.
        side = max(int(n ** 0.5), 1)
        coords = torch.stack(torch.meshgrid(torch.linspace(0, 1, side, device=tokens.device),
                                            torch.linspace(0, 1, side, device=tokens.device), indexing="ij"), -1).reshape(-1, 2)[:n]
        centroids = torch.einsum("bns,nd->bsd", assignment, coords) / assignment.sum(1).unsqueeze(-1).clamp_min(1e-6)
        shape = lambda z: z.reshape(b, t, *z.shape[1:])
        return SlotOutput(shape(slots), shape(assignment), shape(reconstruction),
                          slot_reconstruction.reshape(b, t, self.slots, n, -1),
                          shape(presence), shape(centroids))


class ObjectSlotModel(nn.Module):
    """Thin public wrapper; no Dynamic or Relation branch is included."""

    def __init__(self, input_dim: int = 1280, hidden_dim: int = 256, slots: int = 8,
                 iterations: int = 3) -> None:
        super().__init__()
        self.entity = CompetitiveSlotAttention(input_dim, hidden_dim, slots, iterations)

    def forward(self, context_tokens: torch.Tensor) -> SlotOutput:
        return self.entity(context_tokens)


def slot_losses(output: SlotOutput, *, target_presence: Optional[torch.Tensor] = None,
                target_centroids: Optional[torch.Tensor] = None,
                target_features: Optional[torch.Tensor] = None,
                assignment_entropy_weight: float = 1e-3,
                load_balance_weight: float = 1e-2) -> dict[str, torch.Tensor]:
    """Compute optional object-centric losses without requiring dataset labels."""
    losses: dict[str, torch.Tensor] = {}
    if target_presence is not None:
        losses["presence"] = F.binary_cross_entropy_with_logits(output.presence_logits, target_presence.float())
    if target_centroids is not None:
        losses["centroid"] = F.smooth_l1_loss(output.centroids, target_centroids.float())
    if target_features is not None:
        losses["reconstruction"] = F.smooth_l1_loss(output.patch_reconstruction, target_features.float())
    # Low entropy makes each patch choose a compact owner, while the small
    # coefficient avoids premature slot collapse during early training.
    p = output.assignment.clamp_min(1e-8)
    losses["assignment_entropy"] = -(p * p.log()).sum(-1).mean() * assignment_entropy_weight
    # Prevent a degenerate solution in which one slot owns every patch.  The
    # target is uniform usage over slots, not uniform assignment per patch.
    usage = output.assignment.mean(dim=2)
    uniform = usage.new_full(usage.shape, 1.0 / usage.size(-1))
    losses["load_balance"] = F.mse_loss(usage, uniform) * load_balance_weight
    losses["total"] = sum(losses.values())
    return losses


def temporal_slot_consistency(slots: torch.Tensor, presence_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Permutation-invariant adjacent-frame consistency for [B,T,S,H] slots.

    Matching is detached for the assignment step; gradients still flow through
    the matched cosine distances.  Presence logits optionally suppress padded
    slots, which prevents empty slots from dominating the consistency term.
    """
    if slots.ndim != 4 or slots.size(1) < 2:
        return slots.sum() * 0.0
    terms = []
    for t in range(slots.size(1) - 1):
        a, b = slots[:, t], slots[:, t + 1]
        perm = greedy_slot_matching(a.detach(), b.detach())
        valid = perm >= 0
        matched = b.gather(1, perm.clamp_min(0).unsqueeze(-1).expand(-1, -1, b.size(-1)))
        distance = 1.0 - F.cosine_similarity(a, matched, dim=-1)
        if presence_logits is not None:
            w = presence_logits[:, t].sigmoid() * presence_logits[:, t + 1].sigmoid().gather(1, perm.clamp_min(0))
            distance = distance * w
            terms.append(distance.sum() / w.sum().clamp_min(1.0))
        else:
            terms.append(distance[valid].mean() if valid.any() else distance.mean())
    return torch.stack(terms).mean()


@torch.no_grad()
def greedy_slot_matching(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return a permutation matching slots in b to slots in a by cosine score."""
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("a and b must both be [B,S,H]")
    score = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).transpose(1, 2)
    # Small dependency-free matching suitable for diagnostics and smoke tests.
    out = torch.full((a.size(0), a.size(1)), -1, dtype=torch.long, device=a.device)
    for row in range(a.size(0)):
        used = set()
        for i in score[row].amax(-1).argsort(descending=True).tolist():
            candidates = [j for j in score[row, i].argsort(descending=True).tolist() if j not in used]
            if candidates:
                out[row, i] = candidates[0]; used.add(candidates[0])
    return out
