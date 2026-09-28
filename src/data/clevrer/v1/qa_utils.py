"""ALOE-style CLEVRER QA losses, records, submissions, and metrics."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json

import torch
import torch.distributed as dist

from src.data.clevrer.v1.qa_dataset import normalize_question_types


def gather_records(records):
    if not (dist.is_available() and dist.is_initialized()):
        return records
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, records)
    return [record for group in gathered for record in group]


def records_from_outputs(outputs, batch, threshold=0.5):
    records = []
    cls_logits = outputs["cls_answer_logits"]
    if cls_logits is not None:
        predictions = cls_logits.argmax(dim=-1).detach().cpu().tolist()
        labels = batch["cls_label"].detach().cpu().tolist()
        for meta, label, prediction in zip(batch["cls_meta"], labels, predictions):
            records.append(
                {
                    **meta,
                    "task": 0,
                    "label": int(label),
                    "prediction": int(prediction),
                    "probability": None,
                }
            )
    mc_logits = outputs["mc_answer_logits"]
    if mc_logits is not None:
        probabilities = mc_logits.sigmoid().detach().cpu().tolist()
        labels = batch["mc_label"].detach().cpu().tolist()
        choice_ids = batch["mc_choice_id"].detach().cpu().tolist()
        flags = batch["mc_flag"].detach().cpu().tolist()
        for index, (label, probability, choice_id, flag) in enumerate(
            zip(labels, probabilities, choice_ids, flags)
        ):
            meta = batch["mc_meta"][int(flag)]
            records.append(
                {
                    "scene_index": int(meta["scene_index"]),
                    "question_id": int(meta["question_id"]),
                    "question_type": meta["question_type"],
                    "choice_id": int(choice_id),
                    "task": 1,
                    "label": int(label),
                    "prediction": int(float(probability) >= float(threshold)),
                    "probability": float(probability),
                }
            )
    return records


def _accuracy(correct, total):
    return None if total == 0 else float(correct) / float(total)


def compute_metrics(records):
    labelled = [record for record in records if int(record["label"]) >= 0]
    metrics = {
        "records": len(labelled),
        "descriptive": {},
        "multiple_choice": {},
    }
    desc = [record for record in labelled if int(record["task"]) == 0]
    desc_correct = sum(record["prediction"] == record["label"] for record in desc)
    metrics["descriptive"] = {
        "records": len(desc),
        "correct": desc_correct,
        "accuracy": _accuracy(desc_correct, len(desc)),
    }
    mc = [record for record in labelled if int(record["task"]) == 1]
    groups = defaultdict(list)
    for record in mc:
        groups[(record["scene_index"], record["question_id"], record["question_type"])].append(record)
    option_correct = sum(record["prediction"] == record["label"] for record in mc)
    question_correct = sum(
        all(record["prediction"] == record["label"] for record in group)
        for group in groups.values()
    )
    by_type = {}
    for question_type in ("explanatory", "predictive", "counterfactual"):
        typed_groups = [group for key, group in groups.items() if key[2] == question_type]
        typed_options = [record for group in typed_groups for record in group]
        typed_option_correct = sum(
            record["prediction"] == record["label"] for record in typed_options
        )
        typed_question_correct = sum(
            all(record["prediction"] == record["label"] for record in group)
            for group in typed_groups
        )
        by_type[question_type] = {
            "questions": len(typed_groups),
            "options": len(typed_options),
            "option_accuracy": _accuracy(typed_option_correct, len(typed_options)),
            "question_accuracy": _accuracy(typed_question_correct, len(typed_groups)),
        }
    metrics["multiple_choice"] = {
        "options": len(mc),
        "correct_options": option_correct,
        "option_accuracy": _accuracy(option_correct, len(mc)),
        "questions": len(groups),
        "correct_questions": question_correct,
        "question_accuracy": _accuracy(question_correct, len(groups)),
        "by_type": by_type,
    }
    total_questions = len(desc) + len(groups)
    total_correct = desc_correct + question_correct
    metrics["overall_question_accuracy"] = _accuracy(total_correct, total_questions)
    return metrics


def choose_threshold(records):
    candidates = [index / 20.0 for index in range(1, 20)]
    best_threshold, best_score = 0.5, -1.0
    for threshold in candidates:
        converted = []
        for record in records:
            item = dict(record)
            if item["task"] == 1 and item.get("probability") is not None:
                item["prediction"] = int(item["probability"] >= threshold)
            converted.append(item)
        score = compute_metrics(converted)["multiple_choice"]["question_accuracy"]
        score = -1.0 if score is None else score
        if score > best_score:
            best_threshold, best_score = threshold, score
    return best_threshold


def make_submission(annotation_path, records, answer_vocabulary, scene_indices=None, question_types=None):
    with Path(annotation_path).open(encoding="utf-8") as handle:
        submission = json.load(handle)
    allowed_types = normalize_question_types(question_types)
    if scene_indices is not None:
        selected = {int(scene_index) for scene_index in scene_indices}
        submission = [
            scene for scene in submission
            if int(scene["scene_index"]) in selected
        ]
    if allowed_types is not None:
        allowed = set(allowed_types)
        filtered = []
        for scene in submission:
            kept_questions = [
                question for question in scene["questions"]
                if question["question_type"] in allowed
            ]
            if kept_questions:
                item = dict(scene)
                item["questions"] = kept_questions
                filtered.append(item)
        submission = filtered
    lookup = {
        (record["scene_index"], record["question_id"], record["choice_id"]): record
        for record in records
    }
    missing = []
    for scene in submission:
        scene_index = int(scene["scene_index"])
        for question in scene["questions"]:
            question_id = int(question["question_id"])
            if question["question_type"] == "descriptive":
                key = (scene_index, question_id, -1)
                if key not in lookup:
                    missing.append(key)
            else:
                for choice in question["choices"]:
                    key = (scene_index, question_id, int(choice["choice_id"]))
                    if key not in lookup:
                        missing.append(key)
    if missing:
        first = [
            f"scene={scene_index},question={question_id},choice={choice_id}"
            for scene_index, question_id, choice_id in missing[:8]
        ]
        raise ValueError(
            f"missing {len(missing)} QA predictions for submission; first missing: {first}"
        )
    for scene in submission:
        scene_index = int(scene["scene_index"])
        for question in scene["questions"]:
            question_id = int(question["question_id"])
            if question["question_type"] == "descriptive":
                record = lookup[(scene_index, question_id, -1)]
                prediction = int(record["prediction"])
                if not 0 <= prediction < len(answer_vocabulary):
                    raise ValueError(
                        f"descriptive prediction index out of range: scene={scene_index} "
                        f"question={question_id} prediction={prediction} "
                        f"vocabulary_size={len(answer_vocabulary)}"
                    )
                question["answer"] = answer_vocabulary[prediction]
            else:
                for choice in question["choices"]:
                    record = lookup[(scene_index, question_id, int(choice["choice_id"]))]
                    choice["answer"] = "correct" if int(record["prediction"]) else "wrong"
    return submission
