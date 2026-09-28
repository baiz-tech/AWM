#!/usr/bin/env python3
"""V12-style frozen physical-feature OCP readout for the structured probe."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .structured_probe import PhysionStructuredProbe
from .train_probe import Cache
from .eval_ocp_calibrated import auroc, metrics
from src.core.run_context import apply_cli_defaults, task_context


def split(labels, fraction, seed):
    generator = torch.Generator().manual_seed(seed); train, validation = [], []
    for value in torch.unique(labels).tolist():
        indices = torch.where(labels.eq(value))[0][torch.randperm(int(labels.eq(value).sum()), generator=generator)]
        count = min(max(1, round(len(indices) * fraction)), len(indices) - 1)
        validation.append(indices[:count]); train.append(indices[count:])
    return torch.cat(train).sort().values, torch.cat(validation).sort().values


@torch.no_grad()
def extract(model, cache, device):
    feature, labels = [], []
    for raw in DataLoader(Cache(cache), batch_size=4, shuffle=False, num_workers=2):
        batch = {key: value.to(device, non_blocking=True) for key, value in raw.items() if torch.is_tensor(value)}; valid = batch["ocp_valid"].bool()
        output = model(batch["context"], batch["future"])
        # No global pooling: retain every canonical object, time, and pair feature.
        physical = torch.cat((output["object_tokens"].float().flatten(1), output["presence_logits"].float(), output["trajectory"].float().flatten(1), output["pair_tokens"].float().flatten(1), output["pair_distance"].float().flatten(1), output["contact_logits"].sigmoid().float().flatten(1), output["first_contact_logits"].softmax(-1).float().flatten(1)), 1)
        feature.append(physical[valid].cpu()); labels.append(batch["ocp_label"][valid].float().cpu())
    return torch.cat(feature), torch.cat(labels)


def fit(train_x, train_y, validation_x, validation_y, test_x, test_y, seed, epochs, lr, weight_decay, device):
    torch.manual_seed(seed); mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-6); train_x, validation_x, test_x = (train_x - mean) / std, (validation_x - mean) / std, (test_x - mean) / std
    classifier = nn.Linear(train_x.size(1), 1).to(device); optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    train_x, train_y = train_x.to(device), train_y.to(device)
    for _ in range(epochs): optimizer.zero_grad(set_to_none=True); F.binary_cross_entropy_with_logits(classifier(train_x).squeeze(1), train_y).backward(); optimizer.step()
    with torch.no_grad(): validation_p, test_p = classifier(validation_x.to(device)).sigmoid().squeeze(1).cpu(), classifier(test_x.to(device)).sigmoid().squeeze(1).cpu()
    candidates = torch.unique(torch.cat((torch.tensor([0., .5, 1.]), validation_p))).sort().values
    score = torch.tensor([metrics(validation_p, validation_y, float(value))["balanced_accuracy"] for value in candidates]); threshold = float(candidates[score.argmax()])
    return threshold, {**metrics(test_p, test_y, threshold), "auroc": auroc(test_p, test_y.bool())}, test_p


def main():
    ctx = task_context()
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--readout-cache", required=True); parser.add_argument("--test-cache", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--seeds", type=int, nargs="+", default=(239, 240, 241)); parser.add_argument("--epochs", type=int, default=300); parser.add_argument("--learning-rate", type=float, default=1e-2); parser.add_argument("--weight-decay", type=float, default=1e-3); parser.add_argument("--device", default="cuda:0")
    apply_cli_defaults(parser, ctx)
    args = parser.parse_args(); output = Path(args.output_dir)
    if output.exists(): raise FileExistsError(f"output exists: {output}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu"); payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False); model = PhysionStructuredProbe(**payload["model_config"]).to(device); model.load_state_dict(payload["model"], strict=True); model.eval()
    readout_x, readout_y = extract(model, args.readout_cache, device)
    test_x, test_y = extract(model, args.test_cache, device); runs, predictions = {}, {}
    for seed in args.seeds:
        train, validation = split(readout_y, .2, seed); threshold, result, prediction = fit(readout_x[train], readout_y[train], readout_x[validation], readout_y[validation], test_x, test_y, seed, args.epochs, args.learning_rate, args.weight_decay, device); runs[str(seed)] = {"threshold": threshold, **result}; predictions[f"seed{seed}"] = prediction
    names = tuple(next(iter(runs.values())).keys() - {"threshold"}); aggregate = {name: {"mean": float(np.mean([runs[str(seed)][name] for seed in args.seeds])), "std": float(np.std([runs[str(seed)][name] for seed in args.seeds])), "values": [runs[str(seed)][name] for seed in args.seeds]} for name in names}
    result = {"protocol": "physionpp_structured_probe_v12_frozen_readout_v1", "checkpoint": str(Path(args.checkpoint).resolve()), "feature_definition": "all object/pair tokens and explicit physical outputs; no global pooling", "feature_dim": int(readout_x.size(1)), "num_readout": len(readout_y), "num_test": len(test_y), "seeds": args.seeds, "per_seed": runs, "aggregate": aggregate}
    output.mkdir(parents=True); (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n"); torch.save({"labels": test_y, **predictions}, output / "test_predictions.pt"); print(json.dumps(result, indent=2))
if __name__ == "__main__": main()
