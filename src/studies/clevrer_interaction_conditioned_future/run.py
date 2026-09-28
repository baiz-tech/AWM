#!/usr/bin/env python3
"""Evaluate frozen CLEVRER pair predictions by interaction condition."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from src.experiments.awm_clevrer_fullpatch_probe_seed239.model import (
    CLEVRERDecoder,
    match_objects,
)

MAX_OBJECTS = 6
FUTURE_STEPS = 16
PAIR_INDICES = list(itertools.combinations(range(MAX_OBJECTS), 2))
GROUPS = ("approaching", "contact", "separating", "no-contact")


def _load_model(checkpoint: Path, device: torch.device) -> CLEVRERDecoder:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = CLEVRERDecoder(**payload.get("model_config", {})).to(device)
    state = payload.get("model")
    if state is None and "decoder" in payload and "clevrer_probes" in payload:
        state = {f"decoder.{k}": v for k, v in payload["decoder"].items()}
        state.update({f"probes.{k}": v for k, v in payload["clevrer_probes"].items()})
    model.load_state_dict(state if state is not None else payload, strict=True)
    model.eval()
    return model


def _refs(root: Path, targets: dict, limit: int | None) -> list[tuple[Path, int]]:
    lookup = {(int(s), int(w)): i for i, (s, w) in enumerate(zip(targets["scene_id"], targets["window_start"]))}
    paths = sorted(root.glob("*.pt"))
    if limit is not None:
        paths = paths[:limit]
    refs = []
    for path in paths:
        row = torch.load(path, map_location="cpu", weights_only=True)
        key = (int(row["scene_id"]), int(row["window_start"]))
        if key in lookup:
            refs.append((path, lookup[key]))
    return refs


def _batch(targets: dict, index: int, device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("object_present", "color", "material", "shape", "state", "state_valid")
    return {key: targets[key][index:index + 1].to(device) for key in keys}


def _group(distance: torch.Tensor, valid: torch.Tensor, first_class: int) -> str:
    if first_class < FUTURE_STEPS:
        return "contact"
    values = distance[valid]
    if values.numel() < 2:
        return "no-contact"
    half = max(1, values.numel() // 2)
    delta = float(values[-half:].mean() - values[:half].mean())
    tolerance = max(0.005, float(values.abs().mean()) * 0.02)
    if delta < -tolerance:
        return "approaching"
    if delta > tolerance:
        return "separating"
    return "no-contact"


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


@torch.inference_mode()
def evaluate(root: Path, targets: dict, checkpoint: Path, device: torch.device, limit: int | None) -> dict:
    model = _load_model(checkpoint, device)
    metrics = {name: {"pairs": 0, "distance_mae": [], "speed_change_mae": [], "contact_scores": [], "contact_labels": [], "ttc_abs_error": [], "relation_norm": []} for name in GROUPS}
    for path, target_index in _refs(root, targets, limit):
        record = torch.load(path, map_location="cpu", weights_only=True)
        context = record["context_tokens"].unsqueeze(0).to(device).float()
        future = record["future_tokens"].unsqueeze(0).to(device).float()
        output = model(context, future, return_features=True)
        assignment = match_objects(output, _batch(targets, target_index, device))[0].cpu()
        target_to_pred = {int(t): int(p) for p, t in enumerate(assignment.tolist()) if int(t) >= 0}
        count = int(targets["object_present"][target_index].sum())
        state = targets["state"][target_index].float()
        state_valid = targets["state_valid"][target_index].bool()
        pair_distance = targets["pair_distance"][target_index].float()
        pair_valid = targets["pair_valid"][target_index].bool()
        first = targets["first_contact_class"][target_index].long()
        pred_state = output["trajectory_2d"][0].float().cpu()
        pred_distance = output["pair_distance_2d"][0].float().cpu()
        pred_contact = output["contact_gt_event"][0].float().cpu()
        pred_first = output["first_contact_logits"][0].float().cpu()
        relation_features = output["pair_time_features"][0].float().mean(1).cpu()
        for left, right in itertools.combinations(range(count), 2):
            if left not in target_to_pred or right not in target_to_pred:
                continue
            pl, pr = target_to_pred[left], target_to_pred[right]
            pair_slot = PAIR_INDICES.index(tuple(sorted((pl, pr))))
            valid = pair_valid[left, right]
            if not valid.any() or int(first[left, right]) < 0:
                continue
            group = _group(pair_distance[left, right], valid, int(first[left, right]))
            item = metrics[group]
            item["pairs"] += 1
            target_d = pair_distance[left, right][valid]
            item["distance_mae"].extend((pred_distance[pair_slot][valid] - target_d).abs().tolist())
            object_valid = state_valid[left] & state_valid[right]
            if object_valid.any():
                target_speed = (state[left, :, 5:7] - state[right, :, 5:7]).norm(dim=-1)
                predicted_speed = (pred_state[pl, :, 5:7] - pred_state[pr, :, 5:7]).norm(dim=-1)
                target_change = target_speed - target_speed[object_valid][0]
                predicted_change = predicted_speed - predicted_speed[object_valid][0]
                item["speed_change_mae"].append(float((predicted_change[object_valid] - target_change[object_valid]).abs().mean()))
            contact_valid = targets["contact_valid"][target_index, left, right].bool()
            if contact_valid.any():
                item["contact_scores"].extend(pred_contact[pair_slot][contact_valid].tolist())
                item["contact_labels"].extend(targets["contact"][target_index, left, right][contact_valid].float().tolist())
            target_first = int(first[left, right])
            probabilities = pred_first[pair_slot].softmax(-1)
            predicted_ttc = float((probabilities[:FUTURE_STEPS] * torch.arange(FUTURE_STEPS)).sum() / (FUTURE_STEPS - 1))
            if target_first < FUTURE_STEPS:
                item["ttc_abs_error"].append(abs(predicted_ttc - target_first / (FUTURE_STEPS - 1)))
            item["relation_norm"].append(float(relation_features[pair_slot].norm()))
        del record, output
    summary = {}
    for group, item in metrics.items():
        labels = torch.tensor(item.pop("contact_labels"), dtype=torch.bool)
        scores = torch.tensor(item.pop("contact_scores"), dtype=torch.float32)
        auroc = None
        if labels.numel() and labels.any() and (~labels).any():
            order = scores.argsort().argsort().float() + 1
            positive, negative = int(labels.sum()), int((~labels).sum())
            auroc = float((order[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))
        summary[group] = {
            "pairs": item["pairs"],
            "distance_mae": _safe_mean(item["distance_mae"]),
            "speed_change_mae": _safe_mean(item["speed_change_mae"]),
            "contact_auroc": auroc,
            "contact_samples": int(labels.numel()),
            "ttc_mae": _safe_mean(item["ttc_abs_error"]),
            "relation_norm_mean": _safe_mean(item["relation_norm"]),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-latents", type=Path, required=True)
    parser.add_argument("--validation-targets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    targets = torch.load(args.validation_targets, map_location="cpu", weights_only=False)
    summary = evaluate(args.validation_latents, targets, args.checkpoint, device, args.max_samples)
    payload = {"protocol": "clevrer_interaction_conditioned_future_prediction_v1", "checkpoint": str(args.checkpoint.resolve()), "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
