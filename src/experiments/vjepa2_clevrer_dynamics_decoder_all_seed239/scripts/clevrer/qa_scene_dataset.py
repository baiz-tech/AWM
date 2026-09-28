"""Question-level dataset backed by exported decoder scene outputs."""
from __future__ import annotations

from pathlib import Path
import torch

from src.data.clevrer.qa_dataset import (
    Q_SUBTYPE_TO_ID, load_clevrer_annotations, clevrer_aloe_collate_fn,
)


class DecoderSceneQADataset(torch.utils.data.Dataset):
    def __init__(self, annotation_root, scene_output_root, split, tokenizer, answers,
                 max_scenes=None, random_start=False, question_types=None, require_labels=True):
        self.split = "validation" if split == "val" else split
        self.tokenizer = tokenizer
        self.answer_to_id = {str(x): i for i, x in enumerate(answers)}
        self.root = Path(scene_output_root)
        self.random_start = random_start
        self.scenes = load_clevrer_annotations(annotation_root, self.split, max_scenes=max_scenes, question_types=question_types)
        self.samples = []
        for scene in self.scenes:
            sid = int(scene["scene_index"])
            for question in scene["questions"]:
                mode = "predictive_32_159" if question["question_type"] == "predictive" else "nonpredictive"
                path = self.root / mode / self.split / f"scene_{sid:05d}_window_{32 if mode.startswith('predictive') else 0:03d}.pt"
                if not path.is_file():
                    raise FileNotFoundError(path)
                choices = None if question["question_type"] == "descriptive" else list(question["choices"])
                self.samples.append((scene, question, choices, path))
        if not self.samples:
            raise ValueError(f"no QA samples for split={self.split}")

    def __len__(self): return len(self.samples)

    def _load(self, path):
        record = torch.load(path, map_location="cpu", weights_only=True)
        video = record["tokens"].float()
        if video.shape != (16, 256, 1280):
            raise ValueError(f"invalid scene tokens {path}: {tuple(video.shape)}")
        outputs = {k: record[k].float() for k in (
            "object_tokens", "pair_tokens", "presence_logits", "color_logits",
            "material_logits", "shape_logits", "trajectory_2d", "pair_distance_2d",
            "contact_gt_event", "first_contact_logits")}
        valid = outputs["presence_logits"].sigmoid().ge(0.5)
        if valid.sum() < 2:
            valid[:] = False
            valid[outputs["presence_logits"].topk(2).indices] = True
        first, second = torch.triu_indices(6, 6, offset=1)
        outputs["object_valid"] = valid
        outputs["pair_valid"] = valid[first] & valid[second]
        outputs["time_to_contact"] = outputs["first_contact_logits"].softmax(-1).mul(torch.arange(33).float()).sum(-1) / 32.0
        return video, outputs

    def __getitem__(self, index):
        scene, question, choices, path = self.samples[index]
        video, probe = self._load(path)
        common = {"scene_index": int(scene["scene_index"]), "question_id": int(question["question_id"]),
                  "q_subtype": Q_SUBTYPE_TO_ID[question["question_type"]], "raw_question_type": question["question_type"],
                  "video_emb": video, "probe_outputs": probe}
        if choices is None:
            return {**common, "q_type": 0, "q_tokens": self.tokenizer.encode_descriptive(question["question"]),
                    "a_label": self.answer_to_id.get(str(question.get("answer")), -1), "choice_ids": []}
        tokens, labels, ids = [], [], []
        for choice in choices:
            tokens.append(self.tokenizer.encode_mc(question["question"], choice["choice"]))
            labels.append(int(choice.get("answer") == "correct")); ids.append(int(choice["choice_id"]))
        return {**common, "q_type": 1, "q_tokens": torch.stack(tokens), "a_label": torch.tensor(labels), "choice_ids": torch.tensor(ids)}


__all__ = ["DecoderSceneQADataset", "clevrer_aloe_collate_fn"]
