"""Unchanged v5 dynamics decoder with EK100 object and event probes."""
from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_OBJECTS, FUTURE_STEPS, STATE_DIM = 8, 16, 8
TARGET_PROTOCOL = "epic100_visor_decoder_targets_v1"
CACHE_PROTOCOL = "epic100_official_vjepa2_predicted_latents_v1"
CHECKPOINT_PROTOCOL = "epic100_v5_dynamics_decoder_v1"


class FFN(nn.Module):
    def __init__(self, dim, ffn, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, ffn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn, dim), nn.Dropout(dropout),
        )

    def forward(self, value):
        return value + self.net(self.norm(value))


class Cross(nn.Module):
    def __init__(self, dim, heads, ffn, dropout):
        super().__init__()
        self.query_norm, self.memory_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = FFN(dim, ffn, dropout)

    def forward(self, query, memory):
        value, _ = self.attention(
            self.query_norm(query), self.memory_norm(memory), self.memory_norm(memory),
            need_weights=False,
        )
        return self.ffn(query + value)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, ffn, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = FFN(dim, ffn, dropout)

    def forward(self, value):
        normalized = self.norm(value)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        return self.ffn(value + attended)


class UniversalDynamicsDecoder(nn.Module):
    """The v5 decoder trunk, kept structurally identical to Physion++ v5."""
    def __init__(self, input_dim=1280, hidden_dim=256, num_heads=8, ffn_dim=1024,
                 num_slots=8, slot_depth=1, transition_depth=1, interaction_depth=1,
                 dropout=0.1):
        super().__init__()
        self.input_dim, self.hidden_dim, self.num_slots = input_dim, hidden_dim, num_slots
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.spatial = nn.Parameter(torch.empty(1, 1, 256, hidden_dim))
        self.temporal = nn.Parameter(torch.empty(1, 16, 1, hidden_dim))
        self.source = nn.Parameter(torch.empty(1, 2, 1, 1, hidden_dim))
        self.slots = nn.Parameter(torch.empty(1, 1, num_slots, hidden_dim))
        self.times = nn.Parameter(torch.empty(1, 8, 1, hidden_dim))
        self.slot_blocks = nn.ModuleList(Cross(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(slot_depth))
        self.transition = nn.ModuleList(SelfAttention(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(transition_depth))
        self.transition_reread = nn.ModuleList(Cross(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(transition_depth))
        self.interaction = nn.ModuleList(SelfAttention(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(interaction_depth))
        self.interaction_reread = nn.ModuleList(Cross(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(interaction_depth))
        for parameter in (self.spatial, self.temporal, self.source, self.slots, self.times):
            nn.init.trunc_normal_(parameter, std=0.02)

    def forward(self, context, future):
        expected = (8, 256, self.input_dim)
        if context.ndim != 4 or tuple(context.shape[1:]) != expected or future.shape != context.shape:
            raise ValueError(f"expected context/future [B,{expected}], got {context.shape}/{future.shape}")
        batch = context.size(0)
        values = self.projection(torch.cat((context, future), 1).to(self.projection.weight.dtype))
        source = torch.cat((
            self.source[:, :1].expand(-1, 8, -1, -1, -1),
            self.source[:, 1:].expand(-1, 8, -1, -1, -1),
        ), 1).reshape(1, 16, 1, self.hidden_dim)
        memory = (values + self.spatial + self.temporal + source).reshape(batch, 4096, self.hidden_dim)
        state = (self.slots + self.times).expand(batch, -1, -1, -1).reshape(batch, 8 * self.num_slots, self.hidden_dim)
        for block in self.slot_blocks:
            state = block(state, memory)
        state = state.reshape(batch, 8, self.num_slots, self.hidden_dim)
        for block, reread in zip(self.transition, self.transition_reread):
            value = block(state.permute(0, 2, 1, 3).reshape(batch * self.num_slots, 8, self.hidden_dim))
            value = value.reshape(batch, self.num_slots, 8, self.hidden_dim).permute(0, 2, 1, 3)
            state = reread(value.reshape(batch, 8 * self.num_slots, self.hidden_dim), memory).reshape(batch, 8, self.num_slots, self.hidden_dim)
        for block, reread in zip(self.interaction, self.interaction_reread):
            value = block(state.reshape(batch * 8, self.num_slots, self.hidden_dim))
            state = reread(value.reshape(batch, 8 * self.num_slots, self.hidden_dim), memory).reshape(batch, 8, self.num_slots, self.hidden_dim)
        return state


class Epic100Probe(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, num_slots, num_categories,
                 num_verbs, num_nouns, num_actions, dropout=0.1):
        super().__init__()
        self.num_slots, self.hidden_dim = num_slots, hidden_dim
        self.object_queries = nn.Parameter(torch.randn(1, num_slots, hidden_dim) * 0.02)
        self.object_readout = Cross(hidden_dim, num_heads, ffn_dim, dropout)
        self.time_queries = nn.Parameter(torch.randn(1, FUTURE_STEPS, hidden_dim) * 0.02)
        self.time_readout = Cross(hidden_dim, num_heads, ffn_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.presence = nn.Linear(hidden_dim, 1)
        self.category = nn.Linear(hidden_dim, num_categories)
        self.state = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, STATE_DIM))
        pairs = torch.tensor(list(itertools.combinations(range(num_slots), 2)))
        self.register_buffer("pairs", pairs)
        self.pair_time = nn.Sequential(nn.Linear(2 * hidden_dim + 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.pair = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.pair_distance = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.event_query = nn.Parameter(torch.randn(1, 3, hidden_dim) * 0.02)
        self.event_readout = Cross(hidden_dim, num_heads, ffn_dim, dropout)
        self.verb = nn.Linear(hidden_dim, num_verbs)
        self.noun = nn.Linear(hidden_dim, num_nouns)
        self.action = nn.Linear(hidden_dim, num_actions)

    def forward(self, dynamics):
        batch = dynamics.size(0)
        memory = dynamics.reshape(batch, 8 * self.num_slots, self.hidden_dim)
        objects = self.norm(self.object_readout(self.object_queries.expand(batch, -1, -1), memory))
        queries = (objects[:, :, None] + self.time_queries[:, None]).reshape(batch, self.num_slots * FUTURE_STEPS, self.hidden_dim)
        object_time = self.norm(self.time_readout(queries, memory)).reshape(batch, self.num_slots, FUTURE_STEPS, self.hidden_dim)
        trajectory = self.state(object_time)
        first, second = self.pairs[:, 0], self.pairs[:, 1]
        relative = trajectory[:, first, :, :2] - trajectory[:, second, :, :2]
        pair_time = self.pair_time(torch.cat((object_time[:, first] + object_time[:, second], (object_time[:, first] - object_time[:, second]).abs(), relative), -1))
        pairs = self.pair(torch.cat((objects[:, first] + objects[:, second], (objects[:, first] - objects[:, second]).abs(), pair_time.mean(2)), -1))
        event = self.event_readout(self.event_query.expand(batch, -1, -1), memory)
        return {
            "z_dyn": dynamics, "object_tokens": objects, "pair_tokens": pairs,
            "presence_logits": self.presence(objects).squeeze(-1),
            "category_logits": self.category(objects), "trajectory_2d": trajectory,
            "pair_distance_2d": F.softplus(self.pair_distance(pair_time + pairs[:, :, None]).squeeze(-1)),
            "verb_logits": self.verb(event[:, 0]), "noun_logits": self.noun(event[:, 1]),
            "action_logits": self.action(event[:, 2]),
        }


class Epic100Decoder(nn.Module):
    def __init__(self, *, num_categories, num_verbs, num_nouns, num_actions, **decoder_config):
        super().__init__()
        self.decoder = UniversalDynamicsDecoder(**decoder_config)
        self.probe = Epic100Probe(
            self.decoder.hidden_dim, decoder_config.get("num_heads", 8),
            decoder_config.get("ffn_dim", 1024), self.decoder.num_slots,
            num_categories, num_verbs, num_nouns, num_actions,
            decoder_config.get("dropout", 0.1),
        )

    def forward(self, context, future):
        return self.probe(self.decoder(context, future))


@torch.no_grad()
def match_objects(outputs, batch):
    assignments = []
    for sample in range(outputs["presence_logits"].size(0)):
        present = batch["object_present"][sample].bool()
        count = int(present.sum())
        assignment = torch.full((MAX_OBJECTS,), -1, dtype=torch.long, device=present.device)
        if count:
            candidates = torch.tensor(list(itertools.permutations(range(MAX_OBJECTS), count)), device=present.device)
            targets = torch.arange(count, device=present.device)
            category = batch["category"][sample, present].long()
            cost = -outputs["category_logits"][sample].log_softmax(-1)[:, category]
            cost -= 0.25 * outputs["presence_logits"][sample].sigmoid()[:, None]
            predicted = outputs["trajectory_2d"][sample, :, :, :5]
            target = batch["state"][sample, present, :, :5].to(predicted)
            valid = batch["state_valid"][sample, present].to(predicted.dtype)
            weights = predicted.new_tensor([2.0, 2.0, 0.5, 1.0, 1.0])
            geometry = ((predicted[:, None] - target[None]).abs() * weights * valid[None, :, :, None]).sum((2, 3))
            geometry /= (valid.sum(1)[None] * weights.sum()).clamp_min(1)
            selected = candidates[(cost + geometry)[candidates, targets].sum(1).argmin()]
            assignment[selected] = targets
        assignments.append(assignment)
    return torch.stack(assignments)


def decoder_loss(outputs, batch, event_weight=1.0):
    assignment = match_objects(outputs, batch)
    matched = assignment.ge(0)
    losses = {"presence": F.binary_cross_entropy_with_logits(outputs["presence_logits"], matched.float())}
    rows = matched.nonzero()
    if not len(rows):
        raise ValueError("batch contains no VISOR objects")
    sample, slot = rows.T
    target_slot = assignment[sample, slot]
    losses["category"] = F.cross_entropy(outputs["category_logits"][sample, slot], batch["category"][sample, target_slot].long())
    predicted = outputs["trajectory_2d"][sample, slot]
    target = batch["state"][sample, target_slot].to(predicted)
    valid = batch["state_valid"][sample, target_slot].bool()
    if not valid.any():
        raise ValueError("batch contains no valid VISOR states")
    losses["center"] = F.smooth_l1_loss(predicted[..., :2][valid], target[..., :2][valid])
    losses["geometry"] = F.smooth_l1_loss(predicted[..., 2:5][valid], target[..., 2:5][valid])
    losses["velocity"] = F.smooth_l1_loss(predicted[..., 5:7][valid], target[..., 5:7][valid])
    losses["log_area"] = F.smooth_l1_loss(predicted[..., 7][valid], target[..., 7][valid])
    first, second = outputs["pair_tokens"].new_tensor(list(itertools.combinations(range(MAX_OBJECTS), 2)), dtype=torch.long).T
    pair_rows = (matched[:, first] & matched[:, second]).nonzero()
    zero = predicted.sum() * 0.0
    losses["distance"] = zero
    if len(pair_rows):
        pair_sample, pair_index = pair_rows.T
        target_first = assignment[pair_sample, first[pair_index]]
        target_second = assignment[pair_sample, second[pair_index]]
        pair_valid = batch["pair_valid"][pair_sample, target_first, target_second].bool()
        if pair_valid.any():
            losses["distance"] = F.smooth_l1_loss(
                outputs["pair_distance_2d"][pair_sample, pair_index][pair_valid],
                batch["pair_distance"][pair_sample, target_first, target_second].to(outputs["pair_distance_2d"])[pair_valid],
            )
    for name in ("verb", "noun", "action"):
        event_valid = batch[f"{name}_label"].ge(0)
        losses[name] = (
            F.cross_entropy(outputs[f"{name}_logits"][event_valid], batch[f"{name}_label"][event_valid].long())
            if event_valid.any() else zero
        )
    object_total = losses["presence"] + losses["category"] + 2 * losses["center"] + losses["geometry"] + 0.5 * losses["velocity"] + 0.1 * losses["log_area"] + losses["distance"]
    event_total = losses["verb"] + losses["noun"] + losses["action"]
    losses["total"] = object_total + event_weight * event_total
    return losses["total"], losses, assignment
