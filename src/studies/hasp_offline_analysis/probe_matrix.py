#!/usr/bin/env python3
"""Train equal-capacity offline probes and emit a layer x target matrix."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from src.core.run_context import apply_cli_defaults, task_context


def load_rows(root: str, max_samples: int | None):
    paths = sorted(Path(root).glob("sample_*.pt"))[:max_samples]
    if not paths: raise FileNotFoundError(root)
    return [torch.load(p, map_location="cpu", weights_only=False) for p in paths]


def _binary_auroc(scores: torch.Tensor, target: torch.Tensor) -> float | None:
    positives, negatives = int(target.sum()), int((1 - target).sum())
    if not positives or not negatives:
        return None
    order = scores.argsort().argsort().float() + 1
    return float((order[target.bool()].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def classify(x: np.ndarray, y: np.ndarray, test_x: np.ndarray | None = None, test_y: np.ndarray | None = None) -> dict[str, float | int | None]:
    """Equal-capacity linear probe without an external sklearn dependency."""
    keep = np.isfinite(y)
    x, y = x[keep], y[keep].astype(np.int64)
    if test_x is not None and test_y is not None:
        test_keep = np.isfinite(test_y)
        test_x, test_y = test_x[test_keep], test_y[test_keep].astype(np.int64)
    labels = np.unique(y)
    if len(labels) < 2:
        return {"n": int(len(y)), "accuracy": None}
    # Re-indexing prevents sparse EPIC class IDs from inflating the head.
    y = np.searchsorted(labels, y)
    if test_y is not None:
        keep_test = np.isin(test_y, labels)
        test_x, test_y = test_x[keep_test], np.searchsorted(labels, test_y[keep_test])
        split = len(y)
    else:
        split = max(1, int(.8 * len(y)))
        test_x, test_y = x[split:], y[split:]
    if split >= len(y) and test_y is None:
        return {"n": 0, "accuracy": None}
    train_x, test_x = torch.from_numpy(x[:split]).float(), torch.from_numpy(test_x).float()
    train_y, test_y = torch.from_numpy(y[:split]), torch.from_numpy(test_y)
    mean, std = train_x.mean(0, keepdim=True), train_x.std(0, keepdim=True).clamp_min(1e-6)
    train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
    torch.manual_seed(239)
    head = torch.nn.Linear(train_x.shape[1], len(labels))
    counts = torch.bincount(train_y, minlength=len(labels)).float().clamp_min(1)
    weight = (counts.sum() / counts) / len(labels)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(head(train_x), train_y, weight=weight)
        loss.backward(); optimizer.step()
    logits = head(test_x).detach(); pred = logits.argmax(1)
    accuracy = float((pred == test_y).float().mean())
    recall = []
    for label in range(len(labels)):
        mask = test_y == label
        if mask.any(): recall.append(float((pred[mask] == label).float().mean()))
    result: dict[str, float | int | None] = {"n": int(len(test_y)), "accuracy": accuracy, "balanced_accuracy": float(np.mean(recall))}
    if len(labels) == 2:
        result["auroc"] = _binary_auroc(logits[:, 1], test_y)
    return result


def prepare(rows, task):
    target_key = {"ocp": "ocp_label", "verb": "verb", "noun": "noun"}[task]
    y = np.asarray([float(r[target_key]) for r in rows])
    if task == "ocp":
        valid = np.asarray([bool(r["ocp_valid"]) for r in rows])
        rows, y = [row for row, ok in zip(rows, valid) if ok], y[valid]
    return rows, y


def main():
    ctx=task_context();p = argparse.ArgumentParser(); p.add_argument("--features", required=True); p.add_argument("--test-features"); p.add_argument("--output", required=True); p.add_argument("--max-samples", type=int); p.add_argument("--task", choices=("ocp", "verb", "noun"), required=True); p.add_argument("--combinations", action="store_true"); apply_cli_defaults(p,ctx);a = p.parse_args()
    rows = load_rows(a.features, a.max_samples); layers = ["entity", "dynamic", "relation", "full"]
    rows, y = prepare(rows, a.task)
    if not rows: raise ValueError("the selected features contain no valid labels")
    test_rows = test_y = None
    if a.test_features:
        test_rows, test_y = prepare(load_rows(a.test_features, a.max_samples), a.task)
        if not test_rows: raise ValueError("the selected test features contain no valid labels")
    result = {}
    groups = [(layer,) for layer in layers]
    if a.combinations:
        groups += [group for size in (2, 3) for group in itertools.combinations(layers[:3], size)]
    for group in groups:
        x = np.concatenate([np.stack([r[layer].float().numpy() for r in rows]) for layer in group], axis=1)
        tx = None if test_rows is None else np.concatenate([np.stack([r[layer].float().numpy() for r in test_rows]) for layer in group], axis=1)
        result["+".join(group)] = classify(x, y, tx, test_y)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps({"task": a.task, "count": len(rows), "results": result}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
