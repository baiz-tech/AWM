#!/usr/bin/env python3
"""Evaluate a trained EK100 v5 decoder on a frozen validation cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .model import CHECKPOINT_PROTOCOL, Epic100Decoder, decoder_loss
from .train_decoder import CachedDataset, LOSS_KEYS
from src.core.run_context import apply_cli_defaults, task_context


def topk_correct(logits, labels, k):
    return logits.topk(min(k, logits.size(1)), dim=1).indices.eq(labels[:, None]).any(1)


def mean_class_recall(correct, labels, classes):
    recalls = []
    for label in range(classes):
        selected = labels.eq(label)
        if selected.any():
            recalls.append(correct[selected].float().mean())
    return float(torch.stack(recalls).mean()) if recalls else 0.0


def main():
    ctx = task_context()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    apply_cli_defaults(parser, ctx)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CachedDataset(args.cache, args.targets)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("protocol") != CHECKPOINT_PROTOCOL:
        raise ValueError(f"unexpected checkpoint protocol: {payload.get('protocol')!r}")
    model = Epic100Decoder(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    sums = {key: 0.0 for key in LOSS_KEYS}
    predictions = {key: [] for key in ("verb", "noun", "action")}
    labels = {key: [] for key in predictions}
    category_correct = category_count = samples = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
                outputs = model(batch["context"], batch["future"])
                _, losses, assignment = decoder_loss(outputs, batch)
            size = batch["context"].size(0); samples += size
            for key in LOSS_KEYS:
                sums[key] += float(losses[key]) * size
            matched = assignment.ge(0)
            selected = matched.nonzero()
            if len(selected):
                rows, slots = selected.T
                targets = assignment[rows, slots]
                category_correct += int(outputs["category_logits"][rows, slots].argmax(1).eq(batch["category"][rows, targets]).sum())
                category_count += len(selected)
            for key in predictions:
                predictions[key].append(outputs[f"{key}_logits"].float().cpu())
                labels[key].append(batch[f"{key}_label"].cpu())
    result = {
        "protocol": "epic100_v5_dynamics_decoder_evaluation_v1",
        "checkpoint": str(args.checkpoint.resolve()), "samples": samples,
        "loss": {key: value / max(samples, 1) for key, value in sums.items()},
        "object": {"matched_category_accuracy": category_correct / max(category_count, 1), "matched_objects": category_count},
        "event": {},
    }
    for key in predictions:
        logits, target = torch.cat(predictions[key]), torch.cat(labels[key])
        valid = target.ge(0); logits, target = logits[valid], target[valid]
        top1, top5 = topk_correct(logits, target, 1), topk_correct(logits, target, 5)
        result["event"][key] = {
            "samples": int(valid.sum()), "unknown": int((~valid).sum()),
            "top1": float(top1.float().mean()), "top5": float(top5.float().mean()),
            "top1_mean_class_recall": mean_class_recall(top1, target, logits.size(1)),
            "top5_mean_class_recall": mean_class_recall(top5, target, logits.size(1)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
