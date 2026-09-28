"""Object-Consistent Slot Encoder.

Entity extraction is context-only.  The module explicitly separates objectness
from slot ownership, enforces competition over slots, and exposes losses for
mask grounding, local reconstruction, slot identity, and permutation-invariant
state supervision.  No Dynamic/Relation branch is present in this directory.
"""
from __future__ import annotations
from typing import NamedTuple, Optional
import torch
from torch import nn
import torch.nn.functional as F


class OCSEOutput(NamedTuple):
    slots: torch.Tensor              # [B,T,S,H]
    assignment: torch.Tensor         # [B,T,N,S]
    objectness: torch.Tensor         # [B,T,N]
    contribution: torch.Tensor       # [B,T,S,N,D]
    reconstruction: torch.Tensor     # [B,T,N,D]
    presence_logits: torch.Tensor    # [B,T,S]
    centroids: torch.Tensor          # [B,T,S,2]
    state: torch.Tensor              # [B,T,S,9], Physion object state prediction


class OCSE(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=256, slots=8, iterations=3):
        super().__init__()
        self.input_dim, self.hidden_dim, self.slots, self.iterations = input_dim, hidden_dim, slots, iterations
        self.project = nn.Linear(input_dim, hidden_dim)
        self.pos = nn.Parameter(torch.randn(1, 1, 256, hidden_dim) * .02)
        self.norm_x, self.norm_s = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.key, self.value, self.query = [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(3)]
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Linear(4 * hidden_dim, hidden_dim))
        self.slot_init = nn.Parameter(torch.randn(1, slots, hidden_dim) * .02)
        self.objectness = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.presence = nn.Linear(hidden_dim, 1)
        self.state = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 9))
        self.decoder = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, input_dim))

    def forward(self, context: torch.Tensor) -> OCSEOutput:
        if context.ndim != 4:
            raise ValueError('context must have shape [B,T,N,D]')
        b, t, n, _ = context.shape
        x = self.project(context.float())
        if n <= self.pos.size(2): x = x + self.pos[:, :, :n]
        else:
            p = F.interpolate(self.pos.squeeze(0).permute(0, 2, 1), n, mode='linear', align_corners=False)
            x = x + p.permute(0, 2, 1).unsqueeze(0)
        x = x.reshape(b * t, n, self.hidden_dim)
        xnorm = self.norm_x(x)
        objectness = self.objectness(xnorm).squeeze(-1).sigmoid()
        k, v = self.key(xnorm), self.value(xnorm)
        slots = self.slot_init.expand(b * t, -1, -1).contiguous()
        assignment = None
        for _ in range(self.iterations):
            q = self.query(self.norm_s(slots))
            logits = torch.einsum('bsh,bnh->bns', q, k) / self.hidden_dim**0.5
            # Ownership competition: each patch chooses among slots.
            assignment = logits.softmax(-1)
            weighted = assignment * objectness.unsqueeze(-1)
            updates = torch.einsum('bns,bnh->bsh', weighted, v) / weighted.sum(1).unsqueeze(-1).clamp_min(1e-6)
            flat = self.gru(updates.reshape(-1, self.hidden_dim), slots.reshape(-1, self.hidden_dim))
            slots = flat.reshape(b * t, self.slots, self.hidden_dim)
            slots = slots + self.mlp(slots)
        contribution = self.decoder(slots).unsqueeze(2).expand(-1, -1, n, -1)
        reconstruction = torch.einsum('bns,bsnd->bnd', assignment, contribution)
        side = max(int(n ** .5), 1)
        coords = torch.stack(torch.meshgrid(torch.linspace(0, 1, side, device=context.device), torch.linspace(0, 1, side, device=context.device), indexing='ij'), -1).reshape(-1, 2)[:n]
        centroids = torch.einsum('bns,nd->bsd', assignment * objectness.unsqueeze(-1), coords) / (assignment * objectness.unsqueeze(-1)).sum(1).unsqueeze(-1).clamp_min(1e-6)
        reshape = lambda z: z.reshape(b, t, *z.shape[1:])
        return OCSEOutput(reshape(slots), reshape(assignment), reshape(objectness), reshape(contribution), reshape(reconstruction), reshape(self.presence(slots).squeeze(-1)), reshape(centroids), reshape(self.state(slots)))


def match_slots(pred_state: torch.Tensor, target_state: torch.Tensor, target_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Differentiation-free greedy matching; returns [B,S] target indices."""
    if pred_state.ndim != 3 or target_state.ndim != 3: raise ValueError('states must be [B,S,H] and [B,O,H]')
    score = 1 - F.cosine_similarity(pred_state[:, :, None], target_state[:, None], dim=-1)
    if target_valid is not None: score = score.masked_fill(~target_valid[:, None].bool(), 1e6)
    out = torch.full((pred_state.size(0), pred_state.size(1)), -1, dtype=torch.long, device=pred_state.device)
    for b in range(pred_state.size(0)):
        used = set()
        for i in score[b].amin(-1).argsort().tolist():
            choices = [j for j in score[b, i].argsort().tolist() if j not in used and score[b, i, j] < 1e5]
            if choices: out[b, i] = choices[0]; used.add(choices[0])
    return out


def ocse_loss(out: OCSEOutput, target_features: Optional[torch.Tensor] = None,
              target_objectness: Optional[torch.Tensor] = None,
              target_centroids: Optional[torch.Tensor] = None,
              reconstruction_weight=.5, objectness_weight=1., balance_weight=.01,
              entropy_weight=1e-3) -> dict[str, torch.Tensor]:
    losses = {}
    if target_features is not None: losses['reconstruction'] = F.smooth_l1_loss(out.reconstruction, target_features.float()) * reconstruction_weight
    if target_objectness is not None: losses['objectness'] = F.binary_cross_entropy(out.objectness.clamp(1e-5, 1-1e-5), target_objectness.float()) * objectness_weight
    if target_centroids is not None: losses['centroid'] = F.smooth_l1_loss(out.centroids, target_centroids.float())
    p = out.assignment.clamp_min(1e-8); losses['assignment_entropy'] = -(p * p.log()).sum(-1).mean() * entropy_weight
    usage = out.assignment.mean(2); uniform = usage.new_full(usage.shape, 1 / usage.size(-1)); losses['load_balance'] = F.mse_loss(usage, uniform) * balance_weight
    losses['total'] = sum(losses.values()) if losses else out.slots.sum() * 0
    return losses


def matched_object_loss(out: OCSEOutput, target_state: torch.Tensor, target_valid: torch.Tensor,
                        state_weight: float = 1.0, presence_weight: float = 1.0) -> dict[str, torch.Tensor]:
    """Permutation-invariant Physion object supervision on the first context frame.

    Matching is detached and uses predicted 9-D state against valid target
    objects; gradients flow through the matched regression and presence terms.
    """
    pred = out.state[:, 0]
    logits = out.presence_logits[:, 0]
    total_state = pred.sum() * 0.0; total_presence = logits.sum() * 0.0; n = 0
    for b in range(pred.size(0)):
        valid = target_valid[b].bool(); ids = valid.nonzero(as_tuple=False).flatten()
        if not len(ids): continue
        p = pred[b].detach(); t = target_state[b, ids].to(p)
        cost = (p[:, None] - t[None]).abs().mean(-1)
        used = set()
        pairs = []
        for i in cost.amin(-1).argsort().tolist():
            choices = [j for j in cost[i].argsort().tolist() if j not in used]
            if choices: pairs.append((i, ids[choices[0]])); used.add(choices[0])
        if not pairs: continue
        pi = torch.tensor([q[0] for q in pairs], device=pred.device); ti = torch.tensor([q[1] for q in pairs], device=pred.device)
        tv = target_state[b, ti].to(pred); pv = pred[b, pi]
        total_state = total_state + F.smooth_l1_loss(pv, tv)
        target_presence = torch.zeros_like(logits[b]); target_presence[pi] = 1.0
        total_presence = total_presence + F.binary_cross_entropy_with_logits(logits[b], target_presence)
        n += 1
    if n == 0: return {'object_state': pred.sum() * 0.0, 'object_presence': logits.sum() * 0.0, 'object_total': pred.sum() * 0.0}
    return {'object_state': total_state / n * state_weight, 'object_presence': total_presence / n * presence_weight, 'object_total': total_state / n * state_weight + total_presence / n * presence_weight}
