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
NUM_DYNAMICS_SLOTS = 6
CURRENT_STEPS = 16
FUTURE_STEPS = 16
OUTPUT_STEPS = CURRENT_STEPS + FUTURE_STEPS
LATENT_STEPS = 8
DECODER_STEPS = 2 * LATENT_STEPS


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
    """Produce [B,16,6,256] dynamics tokens from context and predicted future."""

    def __init__(self, input_dim=1280, hidden_dim=256, num_heads=8, ffn_dim=1024,
                 num_slots=NUM_DYNAMICS_SLOTS, slot_depth=1, transition_depth=1, interaction_depth=1,
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
        self.decoder_time_queries = nn.Parameter(torch.empty(1, 16, 1, hidden_dim))
        self.slot_blocks = nn.ModuleList(CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(slot_depth))
        self.temporal_blocks = nn.ModuleList(SelfAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(transition_depth))
        self.temporal_memory_blocks = nn.ModuleList(CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(transition_depth))
        self.interaction_blocks = nn.ModuleList(SelfAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(interaction_depth))
        self.interaction_memory_blocks = nn.ModuleList(CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(interaction_depth))
        self.output_norm = nn.LayerNorm(hidden_dim)
        for parameter in (self.spatial_embedding, self.temporal_embedding, self.source_embedding,
                          self.slot_queries, self.decoder_time_queries):
            nn.init.trunc_normal_(parameter, std=0.02)

    def _validate(self, name, value):
        if value.ndim != 4 or value.shape[2:] != (256, self.input_dim):
            raise ValueError(f"{name} must have shape [B,T,256,{self.input_dim}], got {tuple(value.shape)}")

    def build_memory(self, tokens, source_ids=None):
        self._validate("tokens", tokens)
        values = self.projection(tokens)
        batch_size, time_steps = values.shape[:2]
        if source_ids is None:
            source_ids = torch.zeros(batch_size, time_steps, dtype=torch.long, device=values.device)
        if source_ids.shape != (batch_size, time_steps):
            raise ValueError(f"source_ids must have shape [B,{time_steps}], got {tuple(source_ids.shape)}")
        source = self.source_embedding[:, 0].expand(values.size(0), 16, 1, self.hidden_dim)
        source = source + (self.source_embedding[:, 1] - self.source_embedding[:, 0]) * source_ids.to(values.dtype)[:, :, None, None]
        return (values + self.spatial_embedding + self.temporal_embedding[:, :time_steps] + source).reshape(batch_size, time_steps * 256, self.hidden_dim)

    def forward(self, tokens, source_ids=None):
        if tokens.size(1) != 16:
            raise ValueError("tokens must contain exactly 16 temporal latent steps")
        memory = self.build_memory(tokens, source_ids)
        batch = tokens.size(0)
        queries = (self.slot_queries + self.decoder_time_queries).expand(batch, -1, -1, -1)
        slots = queries.reshape(batch, 16 * self.num_slots, self.hidden_dim)
        for block in self.slot_blocks:
            slots = block(slots, memory)
        slots = slots.reshape(batch, 16, self.num_slots, self.hidden_dim)
        for temporal, reread in zip(self.temporal_blocks, self.temporal_memory_blocks):
            sequence = slots.permute(0, 2, 1, 3).reshape(batch * self.num_slots, 16, self.hidden_dim)
            sequence = temporal(sequence).reshape(batch, self.num_slots, 16, self.hidden_dim).permute(0, 2, 1, 3)
            slots = reread(sequence.reshape(batch, 16 * self.num_slots, self.hidden_dim), memory).reshape(batch, 16, self.num_slots, self.hidden_dim)
        for interaction, reread in zip(self.interaction_blocks, self.interaction_memory_blocks):
            sequence = interaction(slots.reshape(batch * 16, self.num_slots, self.hidden_dim)).reshape(batch, 16, self.num_slots, self.hidden_dim)
            slots = reread(sequence.reshape(batch, 16 * self.num_slots, self.hidden_dim), memory).reshape(batch, 16, self.num_slots, self.hidden_dim)
        return self.output_norm(slots)


class CLEVRERShallowProbes(nn.Module):
    """Direct linear CLEVRER heads over the decoder's six dynamics slots."""

    def __init__(self, hidden_dim=256, num_heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.presence_head = nn.Linear(hidden_dim, 1)
        self.color_head = nn.Linear(hidden_dim, len(COLORS))
        self.material_head = nn.Linear(hidden_dim, len(MATERIALS))
        self.shape_head = nn.Linear(hidden_dim, len(SHAPES))
        self.state_head = nn.Linear(hidden_dim, 8)
        pair_indices = torch.tensor(list(itertools.combinations(range(MAX_OBJECTS), 2)), dtype=torch.long)
        self.register_buffer("pair_indices", pair_indices, persistent=True)
        self.pair_contact_head = nn.Linear(2 * hidden_dim, 1)
        self.first_contact_head = nn.Linear(hidden_dim, OUTPUT_STEPS + 1)

    def forward(self, z_dyn):
        if (z_dyn.ndim != 4 or z_dyn.shape[1] != DECODER_STEPS
                or z_dyn.shape[2] != MAX_OBJECTS or z_dyn.shape[3] != self.hidden_dim):
            raise ValueError(
                f"z_dyn must have shape [B,{DECODER_STEPS},{MAX_OBJECTS},{self.hidden_dim}], "
                f"got {tuple(z_dyn.shape)}"
            )
        # Keep the direct slot readout while expanding the decoder time axis to
        # the 32-step supervision protocol used by the targets.
        batch_size = z_dyn.size(0)
        time_features = F.interpolate(
            z_dyn.permute(0, 2, 3, 1).reshape(
                batch_size * MAX_OBJECTS, self.hidden_dim, DECODER_STEPS
            ),
            size=OUTPUT_STEPS,
            mode="linear",
            align_corners=False,
        ).reshape(batch_size, MAX_OBJECTS, self.hidden_dim, OUTPUT_STEPS).permute(0, 1, 3, 2)
        objects = z_dyn.mean(dim=1)
        state = self.state_head(time_features)
        first, second = self.pair_indices[:, 0], self.pair_indices[:, 1]
        first_time, second_time = time_features[:, first], time_features[:, second]
        relative_center = state[:, first, :, :2] - state[:, second, :, :2]
        pair_time = torch.cat([first_time + second_time, (first_time - second_time).abs()], dim=-1)
        pair_tokens = objects[:, first] + objects[:, second]
        return {
            "z_dyn": z_dyn, "object_tokens": objects, "pair_tokens": pair_tokens,
            "presence_logits": self.presence_head(objects).squeeze(-1),
            "color_logits": self.color_head(objects), "material_logits": self.material_head(objects),
            "shape_logits": self.shape_head(objects), "trajectory_2d": state,
            "pair_distance_2d": stable_pair_distance(relative_center),
            "contact_gt_event": self.pair_contact_head(pair_time).squeeze(-1),
            "first_contact_logits": self.first_contact_head(pair_tokens),
        }


class CLEVRERDecoder(nn.Module):
    """Trainable universal decoder followed by CLEVRER-owned probes."""

    def __init__(self, **config):
        super().__init__()
        self.decoder = UniversalDynamicsDecoder(**config)
        self.probes = CLEVRERShallowProbes(hidden_dim=self.decoder.hidden_dim, num_heads=config.get("num_heads", 8), ffn_dim=config.get("ffn_dim", 1024), dropout=config.get("dropout", 0.1))

    def forward(self, tokens, source_ids=None):
        return self.probes(self.decoder(tokens, source_ids))


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
