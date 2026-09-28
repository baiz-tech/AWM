"""Permutation-invariant full-patch 3D object probe for Physion++."""
from __future__ import annotations

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_OBJECTS, FUTURE_STEPS, STATE_DIM = 8, 16, 9


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads, ffn_dim, dropout):
        super().__init__()
        self.q_norm, self.m_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, dim), nn.Dropout(dropout))

    def forward(self, queries, memory, return_attention=False):
        attended, weights = self.attn(self.q_norm(queries), self.m_norm(memory), self.m_norm(memory), need_weights=return_attention, average_attn_weights=True)
        queries = queries + attended
        output = queries + self.ffn(self.ffn_norm(queries))
        return (output, weights) if return_attention else output


class PhysionStructuredProbe(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=256, num_heads=8, object_layers=3, ffn_dim=1024, dropout=.1):
        super().__init__()
        self.input_dim, self.hidden_dim = int(input_dim), int(hidden_dim)
        self.project = nn.Linear(input_dim, hidden_dim)
        self.spatial = nn.Parameter(torch.empty(1, 1, 256, hidden_dim))
        self.temporal = nn.Parameter(torch.empty(1, 16, 1, hidden_dim))
        self.source = nn.Parameter(torch.empty(1, 2, 1, hidden_dim))
        self.queries = nn.Parameter(torch.randn(1, MAX_OBJECTS, hidden_dim) * .02)
        self.blocks = nn.ModuleList(CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(object_layers))
        self.object_norm = nn.LayerNorm(hidden_dim)
        self.future_time = nn.Parameter(torch.randn(1, 1, FUTURE_STEPS, hidden_dim) * .02)
        self.time_block, self.time_norm = CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout), nn.LayerNorm(hidden_dim)
        self.presence = nn.Linear(hidden_dim, 1)
        self.state = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, STATE_DIM))
        pairs = torch.tensor(list(itertools.combinations(range(MAX_OBJECTS), 2)), dtype=torch.long)
        self.register_buffer("pair_indices", pairs, persistent=True)
        self.pair_time = nn.Sequential(nn.Linear(2 * hidden_dim + 3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.pair = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        # Physion++ collision targets aggregate 16 sampled future frames into
        # 8 tubelets, whereas 3D state stays at all 16 frame positions.
        self.contact, self.first_contact = nn.Linear(hidden_dim, 1), nn.Linear(hidden_dim, 9)
        # Outcome is learned only from train/readout splits; generic collision
        # supervision remains zone-excluded and is never relabelled as OCP.
        self.ocp = nn.Sequential(nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.trunc_normal_(self.spatial, std=.02); nn.init.trunc_normal_(self.temporal, std=.02); nn.init.trunc_normal_(self.source, std=.02)

    def forward(self, context, future, return_features=False, return_attention=False):
        expected = (8, 256, self.input_dim)
        if context.ndim != 4 or tuple(context.shape[1:]) != expected or tuple(future.shape) != tuple(context.shape):
            raise ValueError(f"expected matching [B,{expected}] context/future tensors")
        values = self.project(torch.cat((context, future), 1).float())
        source = torch.cat((self.source[:, :1].expand(-1, 8, -1, -1), self.source[:, 1:].expand(-1, 8, -1, -1)), 1)
        memory = (values + self.spatial + self.temporal + source).reshape(values.size(0), 4096, self.hidden_dim)
        objects = self.queries.expand(memory.size(0), -1, -1)
        attention = None
        for block in self.blocks:
            if return_attention:
                objects, attention = block(objects, memory, return_attention=True)
            else:
                objects = block(objects, memory)
        objects = self.object_norm(objects)
        time_features = self.time_norm(self.time_block((objects[:, :, None] + self.future_time).reshape(memory.size(0), -1, self.hidden_dim), memory)).reshape(memory.size(0), MAX_OBJECTS, FUTURE_STEPS, self.hidden_dim)
        state = self.state(time_features)
        first, second = self.pair_indices.T
        relative = state[:, first, :, :3] - state[:, second, :, :3]
        pair_time = self.pair_time(torch.cat((time_features[:, first] + time_features[:, second], (time_features[:, first] - time_features[:, second]).abs(), relative), -1))
        pair = self.pair(torch.cat((objects[:, first] + objects[:, second], (objects[:, first] - objects[:, second]).abs(), pair_time.mean(2)), -1))
        output = {"object_tokens": objects, "pair_tokens": pair, "presence_logits": self.presence(objects).squeeze(-1),
                "trajectory": state, "pair_distance": torch.linalg.vector_norm(relative.float(), dim=-1),
                "contact_logits": self.contact(pair_time).squeeze(-1).reshape(memory.size(0), -1, 8, 2).mean(-1), "first_contact_logits": self.first_contact(pair),
                "ocp_logits": self.ocp(torch.cat((objects.mean(1), pair.mean(1)), -1)).squeeze(-1)}
        if return_features:
            output["time_features"] = time_features
            output["pair_time_features"] = pair_time
        if return_attention:
            output["object_memory_attention"] = attention
        return output


@torch.no_grad()
def match_objects(outputs, batch):
    assignments = []
    for row in range(outputs["presence_logits"].size(0)):
        present = batch["future_object_valid"][row].any(-1)
        count = int(present.sum())
        result = torch.full((MAX_OBJECTS,), -1, dtype=torch.long, device=outputs["presence_logits"].device)
        if count:
            targets, candidates = torch.arange(count, device=result.device), torch.tensor(list(itertools.permutations(range(MAX_OBJECTS), count)), device=result.device)
            pred = outputs["trajectory"][row, :, :, :9]
            target = batch["future_object_state"][row, present].to(pred)
            valid = batch["future_object_valid"][row, present].to(pred.dtype)
            weights = pred.new_tensor([2., 2., 2., .5, .5, .5, 1., 1., 1.])
            geometry = ((pred[:, None] - target[None]).abs() * valid[None, :, :, None] * weights).sum((2, 3)) / (valid.sum(1)[None] * weights.sum()).clamp_min(1)
            cost = geometry - .25 * outputs["presence_logits"][row].sigmoid()[:, None]
            selected = candidates[cost[candidates, targets].sum(1).argmin()]
            result[selected] = targets
        assignments.append(result)
    return torch.stack(assignments)


def loss(outputs, batch):
    assignment = match_objects(outputs, batch); matched = assignment >= 0
    losses = {"presence": F.binary_cross_entropy_with_logits(outputs["presence_logits"], matched.float())}
    rows = matched.nonzero()
    if not len(rows): raise ValueError("batch has no physical objects")
    b, p = rows.T; target = assignment[b, p]
    pred_state, true_state = outputs["trajectory"][b, p], batch["future_object_state"][b, target].to(outputs["trajectory"])
    valid = batch["future_object_valid"][b, target].bool()
    if not valid.any(): raise ValueError("batch has no valid object states")
    losses["position"] = F.smooth_l1_loss(pred_state[..., :3][valid], true_state[..., :3][valid])
    losses["velocity_extent"] = F.smooth_l1_loss(pred_state[..., 3:][valid], true_state[..., 3:][valid])
    left, right = outputs["pair_tokens"].new_tensor(list(itertools.combinations(range(MAX_OBJECTS), 2)), dtype=torch.long).T
    pairs = (matched[:, left] & matched[:, right]).nonzero()
    if not len(pairs): raise ValueError("batch has no valid pairs")
    pb, pi = pairs.T; first, second = left[pi], right[pi]; tf, ts = assignment[pb, first], assignment[pb, second]
    distance_valid = batch["future_pair_valid"][pb, tf, ts].bool()
    losses["distance"] = F.smooth_l1_loss(outputs["pair_distance"][pb, pi][distance_valid], batch["future_pair_distance"][pb, tf, ts].to(outputs["pair_distance"])[distance_valid]) if distance_valid.any() else losses["position"] * 0
    contact_valid = batch["future_contact_valid"][pb, tf, ts].bool(); contact_target = batch["future_contact"][pb, tf, ts].to(outputs["contact_logits"])[contact_valid]
    if contact_target.numel():
        pos = contact_target.sum().clamp_min(1); weight = ((contact_target.numel() - pos) / pos).clamp(max=100.)
        losses["contact"] = F.binary_cross_entropy_with_logits(outputs["contact_logits"][pb, pi][contact_valid], contact_target, pos_weight=weight)
    else: losses["contact"] = losses["position"] * 0
    first_valid = batch["future_time_to_contact_valid"][pb, tf, ts].bool()
    first_class = (batch["future_time_to_contact"][pb, tf, ts].to(outputs["trajectory"]) * 7).round().long()
    first_class = torch.where(first_valid, first_class, torch.full_like(first_class, 8))
    losses["first_contact"] = F.cross_entropy(outputs["first_contact_logits"][pb, pi], first_class)
    if "ocp_label" in batch and batch["ocp_valid"].any():
        mask = batch["ocp_valid"].bool()
        losses["ocp"] = F.binary_cross_entropy_with_logits(outputs["ocp_logits"][mask], batch["ocp_label"].to(outputs["ocp_logits"])[mask])
    else: losses["ocp"] = losses["position"] * 0
    losses["total"] = losses["presence"] + 2 * losses["position"] + losses["velocity_extent"] + losses["distance"] + 2 * losses["contact"] + losses["first_contact"] + losses["ocp"]
    if not torch.isfinite(losses["total"]): raise FloatingPointError("non-finite structured probe loss")
    return losses["total"], losses
