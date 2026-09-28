#!/usr/bin/env python3
"""Calibrate an OCP threshold on validation cache and evaluate once on test."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .structured_probe import PhysionStructuredProbe
from .train_probe import Cache
from src.core.run_context import apply_cli_defaults, task_context


@torch.no_grad()
def predictions(model, cache, device):
    probabilities, labels = [], []
    for raw in DataLoader(Cache(cache), batch_size=8, shuffle=False, num_workers=2):
        batch = {key: value.to(device, non_blocking=True) for key, value in raw.items() if torch.is_tensor(value)}
        valid = batch["ocp_valid"].bool()
        probabilities.append(model(batch["context"], batch["future"])["ocp_logits"].sigmoid()[valid].cpu())
        labels.append(batch["ocp_label"][valid].bool().cpu())
    return torch.cat(probabilities), torch.cat(labels)


def metrics(probability, label, threshold):
    predicted = probability.ge(threshold); label = label.bool()
    tp, tn = int((predicted & label).sum()), int((~predicted & ~label).sum())
    fp, fn = int((predicted & ~label).sum()), int((~predicted & label).sum())
    recall_pos, recall_neg = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {"accuracy": (tp + tn) / max(len(label), 1), "balanced_accuracy": .5 * (recall_pos + recall_neg), "f1": 2 * precision * recall_pos / max(precision + recall_pos, 1e-12), "precision": precision, "positive_recall": recall_pos, "negative_recall": recall_neg, "positive_rate": float(label.float().mean()), "predicted_positive_rate": float(predicted.float().mean())}


def auroc(probability, label):
    positive, negative = probability[label], probability[~label]
    if not len(positive) or not len(negative): return None
    return float(((positive[:, None] > negative[None]).float() + .5 * (positive[:, None] == negative[None]).float()).mean())


def main():
    ctx = task_context()
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--validation-cache", required=True); parser.add_argument("--test-cache", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--device", default="cuda:0")
    apply_cli_defaults(parser, ctx)
    args = parser.parse_args(); output = Path(args.output_dir)
    if output.exists(): raise FileExistsError(f"output exists: {output}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu"); payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PhysionStructuredProbe(**payload["model_config"]).to(device); model.load_state_dict(payload["model"], strict=True); model.eval()
    validation_prob, validation_label = predictions(model, args.validation_cache, device)
    candidates = torch.unique(torch.cat((torch.tensor([0., .5, 1.]), validation_prob))).sort().values
    scores = torch.tensor([metrics(validation_prob, validation_label, float(value))["balanced_accuracy"] for value in candidates])
    threshold = float(candidates[scores.argmax()]); test_prob, test_label = predictions(model, args.test_cache, device)
    result = {"protocol": "physionpp_structured_probe_ocp_calibrated_v1", "checkpoint": str(Path(args.checkpoint).resolve()), "threshold": threshold, "threshold_source": "readout_data_v1/max_validation_balanced_accuracy", "validation": {**metrics(validation_prob, validation_label, threshold), "auroc": auroc(validation_prob, validation_label), "samples": len(validation_label)}, "test": {**metrics(test_prob, test_label, threshold), "auroc": auroc(test_prob, test_label), "samples": len(test_label)} }
    output.mkdir(parents=True); (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n"); torch.save({"probabilities": test_prob, "labels": test_label, "threshold": threshold}, output / "test_predictions.pt"); print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
