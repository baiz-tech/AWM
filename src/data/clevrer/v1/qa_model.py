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
        n_sample_frames=25,
        max_visual_tokens=7,
        input_dim=16,
        num_layers=12,
        num_heads=8,
        ffn_dim=512,
        cls_mlp_size=128,
        dropout=0.1,
        backend="aloe_like_legacy",
    ):
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.pad_id = int(pad_id)
        self.max_question_len = int(max_question_len)
        self.max_choice_len = int(max_choice_len)
        self.text_len = self.max_question_len + self.max_choice_len
        self.n_sample_frames = int(n_sample_frames)
        self.max_visual_tokens = int(max_visual_tokens)
        self.input_dim = int(input_dim)
        if self.input_dim <= 2:
            raise ValueError("input_dim must leave room for token-type channels")
        lang_dim = self.input_dim - 2
        self.d_model = self.input_dim * int(num_heads)
        self.input_len = 1 + self.n_sample_frames * self.max_visual_tokens + self.text_len

        self.q_embedding = nn.Embedding(int(vocabulary_size), lang_dim, padding_idx=self.pad_id)
        self.q_in_proj = nn.Linear(self.input_dim, self.d_model)
        self.vision_in_proj = nn.Linear(self.visual_dim + 2, self.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.register_buffer("text_type", torch.tensor([1.0, 0.0]), persistent=False)
        self.register_buffer("vision_type", torch.tensor([0.0, 1.0]), persistent=False)
        self.register_buffer("cls_question_type", torch.tensor([0.0, 1.0]), persistent=False)
        self.register_buffer("mc_question_type", torch.tensor([1.0, 0.0]), persistent=False)
        self.register_buffer("mc_choice_type", torch.tensor([0.0, 1.0]), persistent=False)

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
        if self.answer_classes > 0:
            self.cls_answer_mlp = nn.Sequential(
                nn.Linear(self.d_model, int(cls_mlp_size)),
                nn.ReLU(),
                nn.Linear(int(cls_mlp_size), self.answer_classes),
            )
        else:
            self.cls_answer_mlp = None
        self.mc_answer_mlp = nn.Sequential(
            nn.Linear(self.d_model, int(cls_mlp_size)),
            nn.ReLU(),
            nn.Linear(int(cls_mlp_size), 1),
        )

    def _append_type(self, x, type_vec):
        return torch.cat([x, type_vec.to(x.device, x.dtype).view(1, 1, 2).expand(x.size(0), x.size(1), 2)], dim=-1)

    def _process(self, video_emb, text_features, text_pad_mask):
        batch = video_emb.size(0)
        if video_emb.shape[1:3] != (self.n_sample_frames, self.max_visual_tokens):
            raise ValueError(
                f"expected video_emb [B,{self.n_sample_frames},{self.max_visual_tokens},D], "
                f"got {tuple(video_emb.shape)}"
            )
        vision = video_emb.flatten(1, 2)
        vision = self._append_type(vision, self.vision_type)
        vision = self.vision_in_proj(vision)
        text = self.q_in_proj(text_features)
        cls = self.cls_token.expand(batch, 1, self.d_model)
        sequence = torch.cat([cls, vision, text], dim=1)
        sequence = sequence + self.position_embedding[:, : sequence.size(1)]
        no_pad = torch.zeros(
            batch,
            sequence.size(1) - text_pad_mask.size(1),
            dtype=torch.bool,
            device=sequence.device,
        )
        pad_mask = torch.cat([no_pad, text_pad_mask], dim=1)
        return self.transformer(sequence, src_key_padding_mask=pad_mask)[:, 0]

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
        cls_repr = self._process(video_emb, text_features, text_ids.eq(self.pad_id))
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
        n_sample_frames=int(cfg_qa.get("n_sample_frames", 25)),
        max_visual_tokens=int(cfg_qa.get("max_visual_tokens", cfg_qa.get("max_n_objects", 7))),
        input_dim=int(cfg_qa.get("input_dim", 16)),
        num_layers=int(cfg_qa.get("num_layers", 12)),
        num_heads=int(cfg_qa.get("num_heads", 8)),
        ffn_dim=int(cfg_qa.get("ffn_dim", 512)),
        cls_mlp_size=int(cfg_qa.get("cls_mlp_size", 128)),
        dropout=float(cfg_qa.get("dropout", 0.1)),
        backend=str(cfg_qa.get("backend", "aloe_like_legacy")),
    )


# Backward-compatible name for older imports.
LatentQAProbe = AloeLikeQA
