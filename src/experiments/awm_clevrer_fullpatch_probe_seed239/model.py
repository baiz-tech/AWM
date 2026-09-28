"""Universal dynamics decoder plus CLEVRER shallow supervision probes."""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


COLORS = ("gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow")
MATERIALS = ("rubber", "metal")
SHAPES = ("cube", "cylinder", "sphere")
MAX_OBJECTS = 6
NUM_DYNAMICS_SLOTS = 8
FUTURE_STEPS = 16
LATENT_STEPS = 8


def stable_pair_distance(relative_center: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(relative_center.float(), dim=-1)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForward(dim, ffn_dim, dropout)

    def forward(self, queries, memory):
        memory = self.memory_norm(memory)
        attended, _ = self.cross_attention(self.query_norm(queries), memory, memory, need_weights=False)
        return self.ffn(queries + attended)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForward(dim, ffn_dim, dropout)

    def forward(self, x):
        normalized = self.norm(x)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        return self.ffn(x + attended)


class UniversalDynamicsDecoder(nn.Module):
    """Produce [B,8,8,256] dynamics tokens from context and predicted future."""

    def __init__(self, input_dim=1280, hidden_dim=256, num_heads=8, ffn_dim=1024,
                 num_slots=8, slot_depth=1, transition_depth=1, interaction_depth=1,
                 dropout=0.1):
        super().__init__()
        self.input_dim, self.hidden_dim, self.num_slots = int(input_dim), int(hidden_dim), int(num_slots)
        self.context_steps = LATENT_STEPS
        self.future_steps = LATENT_STEPS
        self.num_patches = 256
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.spatial_embedding = nn.Parameter(torch.empty(1, 1, 256, hidden_dim))
        self.temporal_embedding = nn.Parameter(torch.empty(1, 16, 1, hidden_dim))
        self.source_embedding = nn.Parameter(torch.empty(1, 2, 1, 1, hidden_dim))
        self.slot_queries = nn.Parameter(torch.empty(1, 1, num_slots, hidden_dim))
        self.future_time_queries = nn.Parameter(torch.empty(1, 8, 1, hidden_dim))
        self.slot_blocks = nn.ModuleList(CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(slot_depth))
        self.temporal_blocks = nn.ModuleList(SelfAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(transition_depth))
        self.temporal_memory_blocks = nn.ModuleList(CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(transition_depth))
        self.interaction_blocks = nn.ModuleList(SelfAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(interaction_depth))
        self.interaction_memory_blocks = nn.ModuleList(CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(interaction_depth))
        self.output_norm = nn.LayerNorm(hidden_dim)
        for parameter in (self.spatial_embedding, self.temporal_embedding, self.source_embedding,
                          self.slot_queries, self.future_time_queries):
            nn.init.trunc_normal_(parameter, std=0.02)

    def _validate(self, name, value):
        expected = (8, 256, self.input_dim)
        if value.ndim != 4 or tuple(value.shape[1:]) != expected:
            raise ValueError(f"{name} must have shape [B,8,256,{self.input_dim}], got {tuple(value.shape)}")

    def build_memory(self, context, future):
        self._validate("context", context)
        self._validate("predicted_future", future)
        if context.size(0) != future.size(0):
            raise ValueError("context and predicted_future batch sizes must match")
        values = self.projection(torch.cat((context, future), dim=1))
        context_source = self.source_embedding[:, :1].expand(-1, 8, -1, -1, -1)
        future_source = self.source_embedding[:, 1:].expand(-1, 8, -1, -1, -1)
        source = torch.cat((context_source, future_source), dim=1).reshape(1, 16, 1, self.hidden_dim)
        return (values + self.spatial_embedding + self.temporal_embedding + source).reshape(values.size(0), 4096, self.hidden_dim)

    def forward(self, context, future):
        memory = self.build_memory(context, future)
        batch = context.size(0)
        queries = (self.slot_queries + self.future_time_queries).expand(batch, -1, -1, -1)
        slots = queries.reshape(batch, 8 * self.num_slots, self.hidden_dim)
        for block in self.slot_blocks:
            slots = block(slots, memory)
        slots = slots.reshape(batch, 8, self.num_slots, self.hidden_dim)
        for temporal, reread in zip(self.temporal_blocks, self.temporal_memory_blocks):
            sequence = slots.permute(0, 2, 1, 3).reshape(batch * self.num_slots, 8, self.hidden_dim)
            sequence = temporal(sequence).reshape(batch, self.num_slots, 8, self.hidden_dim).permute(0, 2, 1, 3)
            slots = reread(sequence.reshape(batch, 8 * self.num_slots, self.hidden_dim), memory).reshape(batch, 8, self.num_slots, self.hidden_dim)
        for interaction, reread in zip(self.interaction_blocks, self.interaction_memory_blocks):
            sequence = interaction(slots.reshape(batch * 8, self.num_slots, self.hidden_dim)).reshape(batch, 8, self.num_slots, self.hidden_dim)
            slots = reread(sequence.reshape(batch, 8 * self.num_slots, self.hidden_dim), memory).reshape(batch, 8, self.num_slots, self.hidden_dim)
        return self.output_norm(slots)


class CLEVRERShallowProbes(nn.Module):
    """CLEVRER heads that read only z_dyn; outputs retain the v1 public contract."""

    def __init__(self, hidden_dim=256, num_heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.object_queries = nn.Parameter(torch.randn(1, MAX_OBJECTS, hidden_dim) * 0.02)
        self.object_readout = CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout)
        self.time_queries = nn.Parameter(torch.randn(1, FUTURE_STEPS, hidden_dim) * 0.02)
        self.time_readout = CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout)
        self.object_norm = nn.LayerNorm(hidden_dim)
        self.time_norm = nn.LayerNorm(hidden_dim)
        self.presence_head = nn.Linear(hidden_dim, 1)
        self.color_head = nn.Linear(hidden_dim, len(COLORS))
        self.material_head = nn.Linear(hidden_dim, len(MATERIALS))
        self.shape_head = nn.Linear(hidden_dim, len(SHAPES))
        self.state_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 8))
        pair_indices = torch.tensor(list(itertools.combinations(range(MAX_OBJECTS), 2)), dtype=torch.long)
        self.register_buffer("pair_indices", pair_indices, persistent=True)
        self.pair_time_encoder = nn.Sequential(nn.Linear(2 * hidden_dim + 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.pair_encoder = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.contact_head = nn.Linear(hidden_dim, 1)
        self.first_contact_head = nn.Linear(hidden_dim, FUTURE_STEPS + 1)

    def forward(self, z_dyn, return_features=False):
        if z_dyn.ndim != 4 or z_dyn.shape[1:] != (8, 8, self.hidden_dim):
            raise ValueError(f"z_dyn must have shape [B,8,8,{self.hidden_dim}], got {tuple(z_dyn.shape)}")
        batch = z_dyn.size(0)
        memory = z_dyn.reshape(batch, 64, self.hidden_dim)
        objects = self.object_readout(self.object_queries.expand(batch, -1, -1), memory)
        objects = self.object_norm(objects)
        time_queries = objects[:, :, None, :] + self.time_queries[0][None, None, :, :]
        time_features = self.time_readout(time_queries.reshape(batch, MAX_OBJECTS * FUTURE_STEPS, self.hidden_dim), memory)
        time_features = self.time_norm(time_features).reshape(batch, MAX_OBJECTS, FUTURE_STEPS, self.hidden_dim)
        state = self.state_head(time_features)
        first, second = self.pair_indices[:, 0], self.pair_indices[:, 1]
        first_time, second_time = time_features[:, first], time_features[:, second]
        relative_center = state[:, first, :, :2] - state[:, second, :, :2]
        pair_time = self.pair_time_encoder(torch.cat([first_time + second_time, (first_time - second_time).abs(), relative_center], dim=-1))
        pair_tokens = self.pair_encoder(torch.cat([objects[:, first] + objects[:, second], (objects[:, first] - objects[:, second]).abs(), pair_time.mean(2)], dim=-1))
        output = {
            "z_dyn": z_dyn, "object_tokens": objects, "pair_tokens": pair_tokens,
            "presence_logits": self.presence_head(objects).squeeze(-1),
            "color_logits": self.color_head(objects), "material_logits": self.material_head(objects),
            "shape_logits": self.shape_head(objects), "trajectory_2d": state,
            "pair_distance_2d": stable_pair_distance(relative_center),
            "contact_gt_event": self.contact_head(pair_time).squeeze(-1),
            "first_contact_logits": self.first_contact_head(pair_tokens),
        }
        if return_features:
            output["time_features"] = time_features
            output["pair_time_features"] = pair_time
        return output


class CLEVRERDecoder(nn.Module):
    """Trainable universal decoder followed by CLEVRER-owned probes."""

    def __init__(self, **config):
        super().__init__()
        self.decoder = UniversalDynamicsDecoder(**config)
        self.probes = CLEVRERShallowProbes(hidden_dim=self.decoder.hidden_dim, num_heads=config.get("num_heads", 8), ffn_dim=config.get("ffn_dim", 1024), dropout=config.get("dropout", 0.1))

    def forward(self, context, future, return_features=False):
        return self.probes(self.decoder(context, future), return_features=return_features)


def _assignment(cost, object_count):
    result = torch.full((MAX_OBJECTS,), -1, dtype=torch.long, device=cost.device)
    if object_count == 0:
        return result
    candidates = torch.tensor(list(itertools.permutations(range(MAX_OBJECTS), object_count)), dtype=torch.long, device=cost.device)
    target_indices = torch.arange(object_count, device=cost.device)
    selected = candidates[cost[candidates, target_indices].sum(1).argmin()]
    result[selected] = target_indices
    return result


@torch.no_grad()
def match_objects(outputs, batch):
    assignments = []
    for sample in range(outputs["presence_logits"].size(0)):
        present = batch["object_present"][sample].bool()
        count = int(present.sum())
        if count == 0:
            assignments.append(_assignment(outputs["presence_logits"].new_zeros(MAX_OBJECTS, 0), 0))
            continue
        color, material, shape = (batch[key][sample, present].long() for key in ("color", "material", "shape"))
        cost = -0.25 * outputs["presence_logits"][sample].sigmoid()[:, None]
        cost = cost - outputs["color_logits"][sample].log_softmax(-1)[:, color]
        cost = cost - outputs["material_logits"][sample].log_softmax(-1)[:, material]
        cost = cost - outputs["shape_logits"][sample].log_softmax(-1)[:, shape]
        predicted = outputs["trajectory_2d"][sample, :, :, :5]
        target = batch["state"][sample, present, :, :5].to(predicted)
        valid = batch["state_valid"][sample, present].to(predicted.dtype)
        difference = (predicted[:, None] - target[None]).abs()
        weights = predicted.new_tensor([2.0, 2.0, 0.5, 1.0, 1.0])
        trajectory_cost = (difference * weights * valid[None, :, :, None]).sum((2, 3)) / (valid.sum(1)[None, :] * weights.sum()).clamp_min(1)
        assignments.append(_assignment(cost + trajectory_cost, count))
    return torch.stack(assignments)


def structured_probe_loss(outputs, batch):
    nonfinite = [name for name, value in outputs.items() if torch.is_tensor(value) and not torch.isfinite(value).all()]
    if nonfinite:
        raise FloatingPointError(f"decoder/probes produced non-finite outputs: {nonfinite}")
    assignment = match_objects(outputs, batch)
    matched = assignment.ge(0)
    losses = {"presence": F.binary_cross_entropy_with_logits(outputs["presence_logits"], matched.to(outputs["presence_logits"].dtype))}
    selected = torch.nonzero(matched, as_tuple=False)
    if not len(selected):
        raise ValueError("decoder batch contains no objects")
    bi, pi = selected[:, 0], selected[:, 1]
    ti = assignment[bi, pi]
    losses["attributes"] = sum(F.cross_entropy(outputs[f"{key}_logits"][bi, pi], batch[key][bi, ti].long()) for key in ("color", "material", "shape")) / 3
    predicted_state, target_state = outputs["trajectory_2d"][bi, pi], batch["state"][bi, ti].to(outputs["trajectory_2d"])
    valid = batch["state_valid"][bi, ti].bool()
    losses["center"] = outputs["trajectory_2d"].sum() * 0.0
    losses["geometry"] = outputs["trajectory_2d"].sum() * 0.0
    losses["velocity"] = outputs["trajectory_2d"].sum() * 0.0
    losses["log_area"] = outputs["trajectory_2d"].sum() * 0.0
    if valid.any():
        losses["center"] = F.smooth_l1_loss(predicted_state[..., :2][valid], target_state[..., :2][valid])
        losses["geometry"] = F.smooth_l1_loss(predicted_state[..., 2:5][valid], target_state[..., 2:5][valid])
        losses["velocity"] = F.smooth_l1_loss(predicted_state[..., 5:7][valid], target_state[..., 5:7][valid])
        losses["log_area"] = F.smooth_l1_loss(predicted_state[..., 7][valid], target_state[..., 7][valid])
    first, second = outputs["pair_tokens"].new_tensor(list(itertools.combinations(range(MAX_OBJECTS), 2)), dtype=torch.long).T
    pair_rows = torch.nonzero(matched[:, first] & matched[:, second], as_tuple=False)
    losses["distance"] = outputs["pair_distance_2d"].sum() * 0.0
    losses["contact"] = outputs["contact_gt_event"].sum() * 0.0
    losses["first_contact"] = outputs["first_contact_logits"].sum() * 0.0
    if len(pair_rows):
        pb, pi = pair_rows[:, 0], pair_rows[:, 1]
        tf, ts = assignment[pb, first[pi]], assignment[pb, second[pi]]
        distance = outputs["pair_distance_2d"][pb, pi]
        pair_valid = batch["pair_valid"][pb, tf, ts].bool()
        if pair_valid.any():
            losses["distance"] = F.smooth_l1_loss(distance[pair_valid], batch["pair_distance"][pb, tf, ts].to(distance)[pair_valid])
        contact_valid = batch["contact_valid"][pb, tf, ts].bool()
        contact_target = batch["contact"][pb, tf, ts].to(outputs["contact_gt_event"])[contact_valid]
        if contact_target.numel():
            positives = contact_target.sum().clamp_min(1)
            losses["contact"] = F.binary_cross_entropy_with_logits(outputs["contact_gt_event"][pb, pi][contact_valid], contact_target, pos_weight=((contact_target.numel() - positives) / positives).clamp(max=100.0))
        first_target = batch["first_contact_class"][pb, tf, ts].long()
        first_valid = first_target.ge(0)
        if first_valid.any():
            losses["first_contact"] = F.cross_entropy(outputs["first_contact_logits"][pb, pi][first_valid], first_target[first_valid])
    losses["total"] = losses["presence"] + losses["attributes"] + 2 * losses["center"] + losses["geometry"] + 0.5 * losses["velocity"] + 0.1 * losses["log_area"] + losses["distance"] + 2 * losses["contact"] + losses["first_contact"]
    if not torch.isfinite(losses["total"]).all():
        raise FloatingPointError("decoder/probe loss is non-finite")
    return losses["total"], losses


def model_config(model):
    decoder = model.decoder if hasattr(model, "decoder") else model
    return {"input_dim": decoder.input_dim, "hidden_dim": decoder.hidden_dim, "num_heads": decoder.slot_blocks[0].cross_attention.num_heads, "ffn_dim": decoder.slot_blocks[0].ffn.net[0].out_features, "num_slots": decoder.num_slots, "slot_depth": len(decoder.slot_blocks), "transition_depth": len(decoder.temporal_blocks), "interaction_depth": len(decoder.interaction_blocks), "dropout": decoder.slot_blocks[0].cross_attention.dropout}
