"""ALOE-style Transformer QA model for V-JEPA2 CLEVRER trajectories."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AloeLikeQA(nn.Module):
    """C-JEPA/ALOE-style VQA Transformer over latent visual tokens."""

    def __init__(
        self,
        visual_dim,
        vocabulary_size,
        answer_classes,
        pad_id,
        max_question_len=20,
        max_choice_len=12,
        n_sample_frames=16,
        max_visual_tokens=16,
        learned_tokens_per_step=16,
        use_probe_tokens=False,
        probe_token_mode="legacy_four_global",
        probe_hidden_dim=256,
        answer_head_activation="relu",
        input_dim=16,
        num_layers=12,
        num_heads=8,
        ffn_dim=512,
        cls_mlp_size=128,
        dropout=0.1,
        backend="aloe_like_legacy",
        structured_time_steps=16,
    ):
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.pad_id = int(pad_id)
        self.max_question_len = int(max_question_len)
        self.max_choice_len = int(max_choice_len)
        self.text_len = self.max_question_len + self.max_choice_len
        self.n_sample_frames = int(n_sample_frames)
        self.max_visual_tokens = int(max_visual_tokens)
        self.learned_tokens_per_step = int(learned_tokens_per_step)
        self.structured_time_steps = int(structured_time_steps)
        if self.structured_time_steps <= 0:
            raise ValueError("structured_time_steps must be positive")
        self.use_probe_tokens = bool(use_probe_tokens)
        self.probe_token_mode = str(probe_token_mode)
        if self.probe_token_mode not in {
            "legacy_four_global",
            "structured_object_pair_v1",
            "structured_supervised_sequence_v2",
        }:
            raise ValueError(f"unsupported probe_token_mode={self.probe_token_mode!r}")
        self.input_dim = int(input_dim)
        if self.input_dim <= 2:
            raise ValueError("input_dim must leave room for token-type channels")
        lang_dim = self.input_dim - 2
        self.d_model = self.input_dim * int(num_heads)
        legacy_probe_count = 4 if self.use_probe_tokens and self.probe_token_mode == "legacy_four_global" else 0
        structured_probe_count = (
            6 + 6 * self.structured_time_steps + 15 + 15 * self.structured_time_steps
            if self.use_probe_tokens and self.probe_token_mode == "structured_supervised_sequence_v2"
            else 0
        )
        self.input_len = (
            1
            + 16 * self.learned_tokens_per_step
            + legacy_probe_count
            + structured_probe_count
            + self.text_len
        )

        self.q_embedding = nn.Embedding(int(vocabulary_size), lang_dim, padding_idx=self.pad_id)
        self.q_in_proj = nn.Linear(self.input_dim, self.d_model)
        self.resampler_queries = nn.Parameter(torch.randn(1, self.learned_tokens_per_step, self.d_model) * 0.02)
        self.resampler_key = nn.Linear(self.visual_dim, self.d_model)
        self.resampler_value = nn.Linear(self.visual_dim, self.d_model)
        self.resampler_attention = nn.MultiheadAttention(self.d_model, int(num_heads), dropout=float(dropout), batch_first=True)
        self.resampler_norm = nn.LayerNorm(self.d_model)
        self.patch_position_embedding = nn.Parameter(torch.zeros(1, 1, 256, self.d_model))
        self.temporal_embedding = nn.Parameter(torch.zeros(1, 16, 1, self.d_model))
        self.learned_token_embedding = nn.Parameter(torch.zeros(1, 1, self.learned_tokens_per_step, self.d_model))
        self.visual_source_embedding = nn.Parameter(torch.zeros(1, 2, 1, self.d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.register_buffer("text_type", torch.tensor([1.0, 0.0]), persistent=False)
        self.register_buffer("vision_type", torch.tensor([0.0, 1.0]), persistent=False)
        self.register_buffer("cls_question_type", torch.tensor([0.0, 1.0]), persistent=False)
        self.register_buffer("mc_question_type", torch.tensor([1.0, 0.0]), persistent=False)
        self.register_buffer("mc_choice_type", torch.tensor([0.0, 1.0]), persistent=False)
        probe_dims = {
            "trajectory_2d": 6 * 16 * 8,
            "pair_distance_2d": 6 * 6 * 16,
            "contact_gt_event": 6 * 6 * 16,
            "time_to_contact_gt_event": 6 * 6,
        }
        self.probe_names = tuple(probe_dims)
        self.probe_encoders = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.d_model), nn.GELU(), nn.LayerNorm(self.d_model))
            for name, dim in probe_dims.items()
        }) if self.use_probe_tokens and self.probe_token_mode == "legacy_four_global" else nn.ModuleDict()
        self.probe_type_embedding = nn.Parameter(torch.zeros(1, 4, self.d_model))

        self.position_embedding = nn.Parameter(torch.zeros(1, self.input_len, self.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backend = str(backend)
        if self.backend not in {"native_aloe", "aloe_like_legacy"}:
            raise ValueError(f"unsupported ALOE backend: {self.backend!r}")
        # Native ALOE's transformer builder uses norm_last=False.  The legacy
        # implementation retained a final LayerNorm; keep it available for
        # old checkpoints while making the native protocol explicit.
        final_norm = None if self.backend == "native_aloe" else nn.LayerNorm(self.d_model)
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=int(num_layers), norm=final_norm
        )
        self.answer_classes = int(answer_classes)
        answer_head_activation = str(answer_head_activation).lower()
        if answer_head_activation not in {"relu", "gelu"}:
            raise ValueError(
                f"unsupported answer_head_activation={answer_head_activation!r}"
            )
        activation = nn.ReLU if answer_head_activation == "relu" else nn.GELU
        if self.answer_classes > 0:
            self.cls_answer_mlp = nn.Sequential(
                nn.Linear(self.d_model, int(cls_mlp_size)),
                activation(),
                nn.Linear(int(cls_mlp_size), self.answer_classes),
            )
        else:
            self.cls_answer_mlp = None
        self.mc_answer_mlp = nn.Sequential(
            nn.Linear(self.d_model, int(cls_mlp_size)),
            activation(),
            nn.Linear(int(cls_mlp_size), 1),
        )
        self.structured_object_projection = None
        self.structured_pair_projection = None
        self.structured_object_prediction_projection = None
        self.structured_object_time_projection = None
        self.structured_pair_prediction_projection = None
        self.structured_pair_time_projection = None
        self.structured_sequence_norm = None
        self.structured_probe_attention = None
        self.structured_probe_norm = None
        self.register_parameter("structured_probe_type", None)
        self.register_parameter("structured_object_slot", None)
        self.register_parameter("structured_pair_slot", None)
        self.register_parameter("structured_future_time", None)
        self.register_parameter("structured_probe_gate", None)
        if self.use_probe_tokens and self.probe_token_mode == "structured_object_pair_v1":
            self.structured_object_projection = nn.Linear(int(probe_hidden_dim), self.d_model)
            self.structured_pair_projection = nn.Linear(int(probe_hidden_dim), self.d_model)
            self.structured_probe_type = nn.Parameter(torch.zeros(1, 2, 1, self.d_model))
            self.structured_probe_attention = nn.MultiheadAttention(
                self.d_model, int(num_heads), dropout=float(dropout), batch_first=True
            )
            self.structured_probe_norm = nn.LayerNorm(self.d_model)
            self.structured_probe_gate = nn.Parameter(torch.zeros(()))
        elif self.use_probe_tokens and self.probe_token_mode == "structured_supervised_sequence_v2":
            self.structured_object_projection = nn.Linear(int(probe_hidden_dim), self.d_model)
            self.structured_pair_projection = nn.Linear(int(probe_hidden_dim), self.d_model)
            self.structured_object_prediction_projection = nn.Sequential(
                nn.LayerNorm(14), nn.Linear(14, self.d_model)
            )
            self.structured_object_time_projection = nn.Sequential(
                nn.LayerNorm(8), nn.Linear(8, self.d_model)
            )
            self.structured_pair_prediction_projection = nn.Sequential(
                nn.LayerNorm(17), nn.Linear(17, self.d_model)
            )
            self.structured_pair_time_projection = nn.Sequential(
                nn.LayerNorm(2), nn.Linear(2, self.d_model)
            )
            self.structured_sequence_norm = nn.LayerNorm(self.d_model)
            self.structured_probe_type = nn.Parameter(torch.zeros(1, 4, 1, self.d_model))
            self.structured_object_slot = nn.Parameter(torch.zeros(1, 6, self.d_model))
            self.structured_pair_slot = nn.Parameter(torch.zeros(1, 15, self.d_model))
            self.structured_future_time = nn.Parameter(torch.zeros(1, self.structured_time_steps, self.d_model))

    def _append_type(self, x, type_vec):
        return torch.cat([x, type_vec.to(x.device, x.dtype).view(1, 1, 2).expand(x.size(0), x.size(1), 2)], dim=-1)

    def _resample_visual(self, video_emb):
        if video_emb.ndim != 4 or video_emb.shape[1:3] != (16, 256):
            raise ValueError(f"expected video_emb [B,16,256,D], got {tuple(video_emb.shape)}")
        batch = video_emb.size(0)
        key = self.resampler_key(video_emb.float()) + self.patch_position_embedding
        value = self.resampler_value(video_emb.float()) + self.patch_position_embedding
        key = key.reshape(batch * 16, 256, self.d_model)
        value = value.reshape(batch * 16, 256, self.d_model)
        query = self.resampler_queries.expand(batch * 16, -1, -1)
        sampled, _ = self.resampler_attention(query, key, value, need_weights=False)
        sampled = self.resampler_norm(sampled + query)
        sampled = sampled.reshape(batch, 16, self.learned_tokens_per_step, self.d_model)
        sampled = sampled + self.temporal_embedding + self.learned_token_embedding
        source = torch.cat([
            self.visual_source_embedding[:, :1].expand(-1, 8, -1, -1),
            self.visual_source_embedding[:, 1:].expand(-1, 8, -1, -1),
        ], dim=1)
        sampled = sampled + source
        return sampled.reshape(batch, 16 * self.learned_tokens_per_step, self.d_model)

    def _probe_tokens(self, probe_outputs, batch_size):
        if not self.use_probe_tokens or self.probe_token_mode != "legacy_four_global":
            return None
        if probe_outputs is None:
            raise ValueError("QA model requires probe_outputs but batch has none")
        tokens = []
        for index, name in enumerate(self.probe_names):
            value = probe_outputs[name].float().reshape(batch_size, -1)
            tokens.append((self.probe_encoders[name](value) + self.probe_type_embedding[:, index]).unsqueeze(1))
        return torch.cat(tokens, dim=1)

    def _fuse_structured_probes(self, cls_repr, probe_outputs):
        if not self.use_probe_tokens or self.probe_token_mode != "structured_object_pair_v1":
            return cls_repr
        if probe_outputs is None:
            raise ValueError("structured QA model requires probe_outputs")
        required = ("object_tokens", "pair_tokens", "object_valid", "pair_valid")
        missing = [name for name in required if name not in probe_outputs]
        if missing:
            raise ValueError(f"structured probe outputs missing {missing}")
        objects = self.structured_object_projection(probe_outputs["object_tokens"].float())
        pairs = self.structured_pair_projection(probe_outputs["pair_tokens"].float())
        objects = objects + self.structured_probe_type[:, 0]
        pairs = pairs + self.structured_probe_type[:, 1]
        tokens = torch.cat([objects, pairs], dim=1)
        valid = torch.cat(
            [probe_outputs["object_valid"].bool(), probe_outputs["pair_valid"].bool()], dim=1
        )
        if (~valid).all(1).any():
            raise ValueError("structured probe batch contains a scene with no valid tokens")
        attended, _ = self.structured_probe_attention(
            cls_repr[:, None], tokens, tokens, key_padding_mask=~valid, need_weights=False
        )
        residual = self.structured_probe_norm(attended[:, 0])
        return cls_repr + torch.tanh(self.structured_probe_gate) * residual

    def _structured_sequence_tokens(self, probe_outputs):
        if not self.use_probe_tokens or self.probe_token_mode != "structured_supervised_sequence_v2":
            return None, None
        if probe_outputs is None:
            raise ValueError("structured sequence QA model requires probe_outputs")
        required = (
            "object_tokens", "pair_tokens", "object_valid", "pair_valid",
            "presence_logits", "color_logits", "material_logits", "shape_logits",
            "trajectory_2d", "pair_distance_2d", "contact_gt_event",
            "first_contact_logits",
        )
        missing = [name for name in required if name not in probe_outputs]
        if missing:
            raise ValueError(f"structured sequence probe outputs missing {missing}")

        object_predictions = torch.cat(
            [
                probe_outputs["presence_logits"].float().unsqueeze(-1),
                probe_outputs["color_logits"].float(),
                probe_outputs["material_logits"].float(),
                probe_outputs["shape_logits"].float(),
            ],
            dim=-1,
        )
        object_base = (
            self.structured_object_projection(probe_outputs["object_tokens"].float())
            + self.structured_object_prediction_projection(object_predictions)
            + self.structured_object_slot
        )
        object_tokens = object_base + self.structured_probe_type[:, 0]
        object_time_tokens = (
            object_base[:, :, None]
            + self.structured_object_time_projection(probe_outputs["trajectory_2d"].float())
            + self.structured_future_time[:, None]
            + self.structured_probe_type[:, 1]
        )

        pair_base = (
            self.structured_pair_projection(probe_outputs["pair_tokens"].float())
            + self.structured_pair_prediction_projection(
                probe_outputs["first_contact_logits"].float()
            )
            + self.structured_pair_slot
        )
        pair_tokens = pair_base + self.structured_probe_type[:, 2]
        pair_time_values = torch.stack(
            [
                probe_outputs["pair_distance_2d"].float(),
                probe_outputs["contact_gt_event"].float(),
            ],
            dim=-1,
        )
        pair_time_tokens = (
            pair_base[:, :, None]
            + self.structured_pair_time_projection(pair_time_values)
            + self.structured_future_time[:, None]
            + self.structured_probe_type[:, 3]
        )

        object_valid = probe_outputs["object_valid"].bool()
        pair_valid = probe_outputs["pair_valid"].bool()
        if (~object_valid).all(1).any() or (~pair_valid).all(1).any():
            raise ValueError("structured probe batch contains a scene with no valid object or pair")
        tokens = torch.cat(
            [
                object_tokens,
                object_time_tokens.flatten(1, 2),
                pair_tokens,
                pair_time_tokens.flatten(1, 2),
            ],
            dim=1,
        )
        valid = torch.cat(
            [
                object_valid,
                object_valid[:, :, None].expand(-1, -1, self.structured_time_steps).flatten(1),
                pair_valid,
                pair_valid[:, :, None].expand(-1, -1, self.structured_time_steps).flatten(1),
            ],
            dim=1,
        )
        return self.structured_sequence_norm(tokens), ~valid

    def _process(self, video_emb, text_features, text_pad_mask, probe_outputs=None):
        batch = video_emb.size(0)
        vision = self._resample_visual(video_emb)
        probe = self._probe_tokens(probe_outputs, batch)
        structured_probe, structured_pad_mask = self._structured_sequence_tokens(probe_outputs)
        text = self.q_in_proj(text_features)
        cls = self.cls_token.expand(batch, 1, self.d_model)
        sequence_parts = [cls, vision]
        if probe is not None:
            sequence_parts.append(probe)
        if structured_probe is not None:
            sequence_parts.append(structured_probe)
        sequence_parts.append(text)
        sequence = torch.cat(sequence_parts, dim=1)
        sequence = sequence + self.position_embedding[:, : sequence.size(1)]
        prefix_pad = torch.zeros(
            batch, 1 + vision.size(1) + (0 if probe is None else probe.size(1)),
            dtype=torch.bool, device=sequence.device,
        )
        pad_parts = [prefix_pad]
        if structured_pad_mask is not None:
            pad_parts.append(structured_pad_mask)
        pad_parts.append(text_pad_mask)
        pad_mask = torch.cat(pad_parts, dim=1)
        cls_repr = self.transformer(sequence, src_key_padding_mask=pad_mask)[:, 0]
        return self._fuse_structured_probes(cls_repr, probe_outputs)

    def _cls_forward(self, batch):
        if batch["cls_q_tokens"].numel() == 0:
            return None
        if self.cls_answer_mlp is None:
            raise ValueError("descriptive QA samples require a non-empty answer vocabulary")
        text_ids = batch["cls_q_tokens"].long()
        text_features = self.q_embedding(text_ids)
        text_features = self._append_type(text_features, self.cls_question_type)
        cls_repr = self._process(
            batch["cls_video_emb"],
            text_features,
            text_ids.eq(self.pad_id),
            batch.get("cls_probe_outputs"),
        )
        return self.cls_answer_mlp(cls_repr)

    def _mc_forward(self, batch):
        if batch["mc_q_tokens"].numel() == 0:
            return None
        text_ids = batch["mc_q_tokens"].long()
        text_features = self.q_embedding(text_ids)
        question = text_features[:, : self.max_question_len]
        choice = text_features[:, self.max_question_len :]
        question = self._append_type(question, self.mc_question_type)
        choice = self._append_type(choice, self.mc_choice_type)
        text_features = torch.cat([question, choice], dim=1)
        video_emb = batch["mc_video_emb"][batch["mc_flag"].long()]
        probe_outputs = None
        if batch.get("mc_probe_outputs") is not None:
            probe_outputs = {name: value[batch["mc_flag"].long()] for name, value in batch["mc_probe_outputs"].items()}
        cls_repr = self._process(video_emb, text_features, text_ids.eq(self.pad_id), probe_outputs)
        return self.mc_answer_mlp(cls_repr).squeeze(-1)

    def forward(self, batch):
        return {
            "cls_answer_logits": self._cls_forward(batch),
            "mc_answer_logits": self._mc_forward(batch),
        }


def qa_loss(outputs, batch):
    losses = []
    counts = {}
    cls_logits = outputs["cls_answer_logits"]
    if cls_logits is not None:
        mask = batch["cls_label"].ge(0)
        if mask.any():
            losses.append(F.cross_entropy(cls_logits[mask], batch["cls_label"][mask].long()))
            counts["descriptive"] = int(mask.sum().item())
    mc_logits = outputs["mc_answer_logits"]
    if mc_logits is not None:
        mask = batch["mc_label"].ge(0)
        if mask.any():
            losses.append(
                F.binary_cross_entropy_with_logits(
                    mc_logits[mask], batch["mc_label"][mask].float()
                )
            )
            counts["multiple_choice"] = int(mask.sum().item())
    if not losses:
        raise ValueError("QA batch contains no labelled records")
    return torch.stack(losses).mean(), counts


def build_qa_model(cfg_qa, visual_dim, tokenizer, answer_vocabulary):
    return AloeLikeQA(
        visual_dim=visual_dim,
        vocabulary_size=len(tokenizer.tokens),
        answer_classes=len(answer_vocabulary),
        pad_id=tokenizer.pad_id,
        max_question_len=int(cfg_qa.get("max_question_len", 20)),
        max_choice_len=int(cfg_qa.get("max_choice_len", 12)),
        n_sample_frames=int(cfg_qa.get("n_sample_frames", 16)),
        max_visual_tokens=int(cfg_qa.get("max_visual_tokens", 16)),
        learned_tokens_per_step=int(cfg_qa.get("learned_tokens_per_step", 16)),
        use_probe_tokens=bool(cfg_qa.get("use_probe_tokens", False)),
        probe_token_mode=str(cfg_qa.get("probe_token_mode", "legacy_four_global")),
        probe_hidden_dim=int(cfg_qa.get("probe_hidden_dim", 256)),
        answer_head_activation=str(cfg_qa.get("answer_head_activation", "relu")),
        input_dim=int(cfg_qa.get("input_dim", 16)),
        num_layers=int(cfg_qa.get("num_layers", 12)),
        num_heads=int(cfg_qa.get("num_heads", 8)),
        ffn_dim=int(cfg_qa.get("ffn_dim", 512)),
        cls_mlp_size=int(cfg_qa.get("cls_mlp_size", 128)),
        dropout=float(cfg_qa.get("dropout", 0.1)),
        backend=str(cfg_qa.get("backend", "aloe_like_legacy")),
        structured_time_steps=int(cfg_qa.get("structured_time_steps", 16)),
    )


# Backward-compatible name for older imports.
LatentQAProbe = AloeLikeQA
