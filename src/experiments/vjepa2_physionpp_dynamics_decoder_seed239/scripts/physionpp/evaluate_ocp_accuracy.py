#!/usr/bin/env python3
"""Evaluate OCP classification metrics for Physion++ probe checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .model import PhysionDecoder
from .model_target_only import TargetOnlyDecoder


def cache_name(path: str | Path) -> str:
    return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:16] + ".pt"


class ReadoutDataset(Dataset):
    def __init__(self, cache_root: Path, targets_path: Path) -> None:
        targets = torch.load(targets_path, map_location="cpu", weights_only=False)
        self.labels = targets["ocp_label"].bool()
        self.valid = targets["ocp_valid"].bool()
        self.files = []
        for path in targets["path"]:
            file = cache_root / cache_name(path)
            if not file.is_file():
                raise FileNotFoundError(f"missing latent cache for {path}: {file}")
            self.files.append(file)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        record = torch.load(self.files[index], map_location="cpu", weights_only=True)
        return (
            record["context_tokens"],
            record["future_tokens"],
            self.labels[index],
            self.valid[index],
        )


def binary_metrics(labels: torch.Tensor, scores: torch.Tensor, threshold: float) -> dict:
    labels = labels.bool()
    predictions = scores >= threshold
    tp = int((predictions & labels).sum())
    tn = int((~predictions & ~labels).sum())
    fp = int((predictions & ~labels).sum())
    fn = int((~predictions & labels).sum())
    n = labels.numel()
    positive = int(labels.sum())
    negative = n - positive
    accuracy = (tp + tn) / n if n else float("nan")
    tpr = tp / positive if positive else float("nan")
    tnr = tn / negative if negative else float("nan")

    # Rank-statistic AUROC; ties receive the standard half-credit.
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ranks = torch.arange(1, n + 1, dtype=torch.float64)
    start = 0
    rank_sum = 0.0
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Average rank for tied scores (ranks are one-based).
        rank_sum += float(((start + 1 + end) / 2) * int(sorted_labels[start:end].sum()))
        start = end
    auroc = (rank_sum - positive * (positive + 1) / 2) / (positive * negative) if positive and negative else float("nan")
    return {
        "n": n,
        "positive": positive,
        "negative": negative,
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": (tpr + tnr) / 2 if positive and negative else float("nan"),
        "positive_recall": tpr,
        "negative_recall": tnr,
        "auroc": auroc,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def evaluate(name: str, model_cls, checkpoint: Path, loader: DataLoader, threshold: float, device: torch.device) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = model_cls(**payload.get("model_config", {})).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for context, future, target, valid in loader:
            valid = valid.bool()
            output = model(context.to(device), future.to(device))["ocp_logits"].detach().cpu()
            scores.append(output[valid])
            labels.append(target[valid])
    if not labels:
        raise RuntimeError(f"no valid OCP labels for {name}")
    result = binary_metrics(torch.cat(labels), torch.cat(scores).sigmoid(), threshold)
    result.update({"name": name, "checkpoint": str(checkpoint.resolve())})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--target-only-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset = ReadoutDataset(args.cache_root, args.targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {
        "protocol": "physionpp_ocp_accuracy_v1",
        "targets": str(args.targets.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "results": [
            evaluate("full_object", PhysionDecoder, args.full_checkpoint, loader, args.threshold, device),
            evaluate("target_only", TargetOnlyDecoder, args.target_only_checkpoint, loader, args.threshold, device),
        ],
    }
    text = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
