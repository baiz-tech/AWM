"""ALOE-style CLEVRER QA dataset over exported V-JEPA2 trajectories."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch


_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
_ANNOTATION_FILES = {
    "train": "train.json",
    "validation": "validation.json",
    "val": "validation.json",
    "test": "test.json",
}
Q_SUBTYPE_TO_ID = {
    "descriptive": 0,
    "explanatory": 1,
    "predictive": 2,
    "counterfactual": 3,
}
VALID_QUESTION_TYPES = frozenset(Q_SUBTYPE_TO_ID)


def normalize_question_types(question_types):
    if question_types is None:
        return None
    if isinstance(question_types, str):
        items = [item.strip() for item in question_types.split(",")]
    else:
        items = [str(item).strip() for item in question_types]
    normalized = tuple(item for item in items if item)
    unknown = sorted(set(normalized) - VALID_QUESTION_TYPES)
    if unknown:
        raise ValueError(f"unknown CLEVRER QA question types: {unknown}")
    return normalized or None


class WordTokenizer:
    """Small word tokenizer compatible with CLEVRER QA text."""

    SPECIAL_TOKENS = ("PAD", "UNK")

    def __init__(self, vocabulary, max_question_len=20, max_choice_len=12):
        self.tokens = list(vocabulary)
        if self.tokens[: len(self.SPECIAL_TOKENS)] != list(self.SPECIAL_TOKENS):
            raise ValueError("token vocabulary must begin with PAD and UNK")
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        if len(self.token_to_id) != len(self.tokens):
            raise ValueError("token vocabulary contains duplicates")
        self.max_question_len = int(max_question_len)
        self.max_choice_len = int(max_choice_len)

    @classmethod
    def build(cls, scenes, min_frequency=1, max_question_len=20, max_choice_len=12):
        counts = Counter()
        for scene in scenes:
            for question in scene["questions"]:
                counts.update(cls.split(question["question"]))
                for choice in question.get("choices", []):
                    counts.update(cls.split(choice["choice"]))
        vocabulary = list(cls.SPECIAL_TOKENS)
        vocabulary.extend(
            token for token, count in sorted(counts.items()) if count >= int(min_frequency)
        )
        return cls(vocabulary, max_question_len=max_question_len, max_choice_len=max_choice_len)

    @classmethod
    def from_cjepa_vocab(cls, path, max_question_len=20, max_choice_len=12):
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        q_vocab = payload.get("q_vocab", payload)
        tokens = [None] * len(q_vocab)
        for token, index in q_vocab.items():
            normalized = "PAD" if token == "PAD" else "UNK" if token == "UNK" else token
            tokens[int(index)] = normalized
        if tokens[0] != "PAD":
            raise ValueError("C-JEPA vocab is expected to have PAD at index 0")
        if "UNK" not in tokens:
            tokens[1] = "UNK"
        return cls(tokens, max_question_len=max_question_len, max_choice_len=max_choice_len)

    @staticmethod
    def split(text):
        return _TOKEN.findall(str(text).lower().replace("?", ""))

    def encode_question(self, text, length=None):
        return self._encode(text, int(length or self.max_question_len))

    def encode_choice(self, text, length=None):
        return self._encode(text, int(length or self.max_choice_len))

    def encode_descriptive(self, question):
        length = self.max_question_len + self.max_choice_len
        return self._encode(question, length)

    def encode_mc(self, question, choice):
        q = self.encode_question(question, self.max_question_len)
        c = self.encode_choice(choice, self.max_choice_len)
        return torch.cat([q, c], dim=0)

    def _encode(self, text, length):
        ids = [
            self.token_to_id.get(token, self.unk_id)
            for token in self.split(text)
        ][:length]
        return torch.tensor(ids + [self.pad_id] * (length - len(ids)), dtype=torch.long)

    @property
    def pad_id(self):
        return self.token_to_id["PAD"]

    @property
    def unk_id(self):
        return self.token_to_id.get("UNK", self.pad_id)

    def state_dict(self):
        return {
            "tokens": self.tokens,
            "max_question_len": self.max_question_len,
            "max_choice_len": self.max_choice_len,
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            state["tokens"],
            max_question_len=state["max_question_len"],
            max_choice_len=state["max_choice_len"],
        )


def load_clevrer_annotations(
    annotation_root,
    split,
    scene_range=None,
    max_scenes=None,
    question_types=None,
):
    split = str(split).lower()
    if split not in _ANNOTATION_FILES:
        raise ValueError(f"unsupported QA split: {split!r}")
    path = Path(annotation_root) / _ANNOTATION_FILES[split]
    if not path.is_file():
        raise FileNotFoundError(f"CLEVRER annotation file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        scenes = json.load(handle)
    allowed_types = normalize_question_types(question_types)
    if scene_range is not None:
        lower, upper = (int(value) for value in scene_range)
        scenes = [scene for scene in scenes if lower <= int(scene["scene_index"]) <= upper]
    scenes.sort(key=lambda scene: int(scene["scene_index"]))
    if max_scenes is not None:
        scenes = scenes[: int(max_scenes)]
    if allowed_types is not None:
        allowed = set(allowed_types)
        filtered = []
        for scene in scenes:
            kept_questions = [
                question for question in scene["questions"]
                if question["question_type"] in allowed
            ]
            if kept_questions:
                item = dict(scene)
                item["questions"] = kept_questions
                filtered.append(item)
        scenes = filtered
    for scene in scenes:
        for question in scene["questions"]:
            if question["question_type"] == "descriptive":
                has_label = "answer" in question
            else:
                has_label = all("answer" in choice for choice in question.get("choices", []))
            if split == "test" and has_label:
                raise ValueError("official CLEVRER test annotations unexpectedly contain labels")
            if split != "test" and not has_label:
                raise ValueError(
                    f"missing QA label: scene={scene['scene_index']} question={question['question_id']}"
                )
    return scenes


def build_answer_vocabulary(scenes):
    answers = sorted(
        {
            str(question["answer"])
            for scene in scenes
            for question in scene["questions"]
            if question["question_type"] == "descriptive"
        }
    )
    return answers


class ClevrerAloeTrajectoryDataset(torch.utils.data.Dataset):
    """Question-level ALOE dataset backed by exported V-JEPA2 scene trajectories."""

    def __init__(
        self,
        annotation_root,
        trajectory_root,
        split,
        tokenizer,
        answer_vocabulary,
        scene_range=None,
        max_scenes=None,
        max_visual_tokens=16,
        n_sample_frames=16,
        video_len=128,
        require_labels=True,
        random_start=True,
        question_types=None,
        protocol="compatibility_stride2",
        frame_offset=None,
        predictive_window=True,
    ):
        self.split = "validation" if str(split).lower() == "val" else str(split).lower()
        self.tokenizer = tokenizer
        self.answer_to_id = {answer: index for index, answer in enumerate(answer_vocabulary)}
        self.max_visual_tokens = int(max_visual_tokens)
        self.n_sample_frames = int(n_sample_frames)
        if (self.n_sample_frames, self.max_visual_tokens) != (16, 16):
            raise ValueError("CLEVRER v2 requires n_sample_frames=16 and max_visual_tokens=16")
        self.video_len = int(video_len)
        self.protocol = str(protocol)
        self.frame_offset = int(frame_offset or (self.video_len // self.n_sample_frames))
        if self.frame_offset <= 0:
            raise ValueError("frame_offset must be positive")
        self.predictive_window = bool(predictive_window)
        self.require_labels = bool(require_labels)
        self.random_start = bool(random_start)
        if self.require_labels and self.split == "test":
            raise ValueError("official CLEVRER test has no labels; use submission mode")
        self.scenes = load_clevrer_annotations(
            annotation_root,
            self.split,
            scene_range=scene_range,
            max_scenes=max_scenes,
            question_types=question_types,
        )
        self.trajectory_root = Path(trajectory_root) / self.split
        self.samples = []
        missing = []
        for scene in self.scenes:
            scene_index = int(scene["scene_index"])
            if not self._trajectory_path(scene_index).is_file():
                missing.append(scene_index)
                continue
            for question in scene["questions"]:
                if question["question_type"] == "descriptive":
                    self.samples.append((scene, question, None))
                else:
                    self.samples.append((scene, question, list(question["choices"])))
        if missing:
            raise FileNotFoundError(
                f"missing {len(missing)} exported trajectories under {self.trajectory_root}; "
                f"first scene ids: {missing[:8]}"
            )
        if not self.samples:
            raise ValueError(f"no QA samples for split={self.split}")

    def _trajectory_path(self, scene_index):
        return self.trajectory_root / f"scene_{scene_index:05d}.pt"

    def __len__(self):
        return len(self.samples)

    def _sample_indices(self, trajectory, q_subtype):
        visual_tokens = trajectory["visual_tokens"]
        total_steps = int(visual_tokens.size(0))
        if self.protocol == "cjepa_native":
            max_start = self.video_len - (self.n_sample_frames - 1) * self.frame_offset
            if max_start <= 0:
                raise ValueError(
                    "native ALOE sampling requires video_len > (n_sample_frames-1)*frame_offset"
                )
            start = random.randrange(max_start) if self.random_start else 0
            if (
                self.predictive_window
                and q_subtype == Q_SUBTYPE_TO_ID["predictive"]
                and total_steps > 150
            ):
                start += total_steps - self.video_len
            sample_idx = [start + n * self.frame_offset for n in range(self.n_sample_frames)]
            if sample_idx[-1] >= total_steps:
                raise ValueError(
                    f"native trajectory has {total_steps} steps but requires index {sample_idx[-1]}"
                )
            return torch.tensor(sample_idx, dtype=torch.long)
        raw_times = trajectory.get("raw_frame_times")
        if raw_times is None:
            raw_times = torch.arange(total_steps, dtype=torch.long) * 4
        raw_times = raw_times.long()
        max_start = max(1, self.video_len - (self.n_sample_frames - 1) * self.frame_offset)
        start = random.randrange(max_start) if self.random_start else 0
        if q_subtype == Q_SUBTYPE_TO_ID["predictive"] and total_steps > 32:
            start += max(0, int(raw_times[-1].item()) + 1 - self.video_len)
        target_times = torch.tensor(
            [start + i * self.frame_offset for i in range(self.n_sample_frames)],
            dtype=torch.long,
        )
        distances = (raw_times[:, None] - target_times[None, :]).abs()
        return distances.argmin(dim=0)

    def _load_video_emb(self, scene_index, q_subtype):
        trajectory = torch.load(
            self._trajectory_path(scene_index), map_location="cpu", weights_only=True
        )
        if int(trajectory["scene_index"]) != int(scene_index):
            raise ValueError(f"trajectory/annotation scene mismatch for scene {scene_index}")
        visual_tokens = trajectory["visual_tokens"].float()
        if visual_tokens.ndim != 3 or visual_tokens.size(0) != 16 or visual_tokens.size(1) != 256:
            raise ValueError(
                "CLEVRER v2 expects visual_tokens [16,256,D], "
                f"got {tuple(visual_tokens.shape)}"
            )
        return visual_tokens

    def _load_probe_outputs(self, scene_index):
        trajectory = torch.load(
            self._trajectory_path(scene_index), map_location="cpu", weights_only=True
        )
        outputs = trajectory.get("probe_outputs")
        if outputs is None:
            return None
        structured = ("object_tokens", "pair_tokens", "object_valid", "pair_valid")
        legacy = ("trajectory_2d", "pair_distance_2d", "contact_gt_event", "time_to_contact_gt_event")
        if all(name in outputs for name in structured):
            return {
                name: outputs[name].bool() if name.endswith("_valid") else outputs[name].float()
                for name in outputs
            }
        missing = [name for name in legacy if name not in outputs]
        if missing:
            raise ValueError(f"scene {scene_index} probe_outputs missing {missing}")
        return {name: outputs[name].float() for name in legacy}

    def __getitem__(self, index):
        scene, question, choices = self.samples[index]
        scene_index = int(scene["scene_index"])
        q_subtype = Q_SUBTYPE_TO_ID[question["question_type"]]
        video_emb = self._load_video_emb(scene_index, q_subtype)
        common = {
            "scene_index": scene_index,
            "question_id": int(question["question_id"]),
            "q_subtype": q_subtype,
            "raw_question_type": question["question_type"],
            "video_emb": video_emb,
            "probe_outputs": self._load_probe_outputs(scene_index),
        }
        if choices is None:
            answer = question.get("answer")
            label = -1 if answer is None else self.answer_to_id.get(str(answer), -1)
            return {
                **common,
                "q_type": 0,
                "q_tokens": self.tokenizer.encode_descriptive(question["question"]),
                "a_label": int(label),
                "choice_ids": [],
            }
        labels = []
        tokens = []
        choice_ids = []
        for choice in choices:
            tokens.append(self.tokenizer.encode_mc(question["question"], choice["choice"]))
            answer = choice.get("answer")
            labels.append(-1 if answer is None else int(answer == "correct"))
            choice_ids.append(int(choice["choice_id"]))
        return {
            **common,
            "q_type": 1,
            "q_tokens": torch.stack(tokens),
            "a_label": torch.tensor(labels, dtype=torch.long),
            "choice_ids": torch.tensor(choice_ids, dtype=torch.long),
        }


def _empty_tokens(length):
    return torch.empty(0, length, dtype=torch.long)


def clevrer_aloe_collate_fn(samples):
    cls_samples = [sample for sample in samples if sample["q_type"] == 0]
    mc_samples = [sample for sample in samples if sample["q_type"] == 1]
    token_len = samples[0]["q_tokens"].numel() if samples[0]["q_type"] == 0 else samples[0]["q_tokens"].size(1)
    visual_shape = samples[0]["video_emb"].shape
    empty_video = torch.empty(0, *visual_shape, dtype=samples[0]["video_emb"].dtype)
    batch = {
        "scene_index": torch.tensor([sample["scene_index"] for sample in samples], dtype=torch.long),
        "question_id": torch.tensor([sample["question_id"] for sample in samples], dtype=torch.long),
        "q_type": torch.tensor([sample["q_type"] for sample in samples], dtype=torch.long),
        "q_subtype": torch.tensor([sample["q_subtype"] for sample in samples], dtype=torch.long),
    }
    def stack_probe(group):
        values = [sample["probe_outputs"] for sample in group]
        if not values:
            return None
        if all(value is None for value in values):
            # Visual-only runs deliberately export no probe outputs.
            return None
        if any(value is None for value in values):
            raise ValueError("a QA batch mixes trajectories with and without probe_outputs")
        names = tuple(values[0])
        if any(tuple(value) != names for value in values[1:]):
            raise ValueError("a QA batch mixes incompatible probe output protocols")
        return {name: torch.stack([value[name] for value in values]) for name in names}
    if cls_samples:
        batch["cls_video_emb"] = torch.stack([sample["video_emb"] for sample in cls_samples])
        batch["cls_q_tokens"] = torch.stack([sample["q_tokens"] for sample in cls_samples])
        batch["cls_label"] = torch.tensor([sample["a_label"] for sample in cls_samples], dtype=torch.long)
        batch["cls_meta"] = [
            {
                "scene_index": sample["scene_index"],
                "question_id": sample["question_id"],
                "question_type": sample["raw_question_type"],
                "choice_id": -1,
            }
            for sample in cls_samples
        ]
        batch["cls_probe_outputs"] = stack_probe(cls_samples)
    else:
        batch["cls_video_emb"] = empty_video
        batch["cls_q_tokens"] = _empty_tokens(token_len)
        batch["cls_label"] = torch.empty(0, dtype=torch.long)
        batch["cls_meta"] = []
        batch["cls_probe_outputs"] = None
    if mc_samples:
        batch["mc_video_emb"] = torch.stack([sample["video_emb"] for sample in mc_samples])
        batch["mc_subtype"] = torch.tensor([sample["q_subtype"] for sample in mc_samples], dtype=torch.long)
        batch["mc_q_tokens"] = torch.cat([sample["q_tokens"] for sample in mc_samples], dim=0)
        batch["mc_label"] = torch.cat([sample["a_label"] for sample in mc_samples], dim=0)
        mc_flag = []
        for flag, sample in enumerate(mc_samples):
            mc_flag.extend([flag] * int(sample["q_tokens"].size(0)))
        batch["mc_flag"] = torch.tensor(mc_flag, dtype=torch.long)
        batch["mc_choice_id"] = torch.cat([sample["choice_ids"] for sample in mc_samples], dim=0)
        batch["mc_meta"] = [
            {
                "scene_index": sample["scene_index"],
                "question_id": sample["question_id"],
                "question_type": sample["raw_question_type"],
                "choice_ids": sample["choice_ids"].tolist(),
            }
            for sample in mc_samples
        ]
        batch["mc_probe_outputs"] = stack_probe(mc_samples)
    else:
        batch["mc_video_emb"] = empty_video
        batch["mc_subtype"] = torch.empty(0, dtype=torch.long)
        batch["mc_q_tokens"] = _empty_tokens(token_len)
        batch["mc_label"] = torch.empty(0, dtype=torch.long)
        batch["mc_flag"] = torch.empty(0, dtype=torch.long)
        batch["mc_choice_id"] = torch.empty(0, dtype=torch.long)
        batch["mc_meta"] = []
        batch["mc_probe_outputs"] = None
    return batch


# Backward-compatible alias used by older tests/scripts while this recipe now
# follows an ALOE-style question dataset.
QASceneDataset = ClevrerAloeTrajectoryDataset
collate_qa_scenes = clevrer_aloe_collate_fn
