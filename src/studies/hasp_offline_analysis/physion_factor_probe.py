#!/usr/bin/env python3
"""Decode native Physion++ factors from each frozen HASP level.

This is the primary layer-functionality analysis.  It does not use OCP as a
proxy for object/dynamics/relations; every target is derived directly from the
cached physical annotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from src.core.run_context import apply_cli_defaults, task_context


LAYERS = ("entity", "dynamic", "relation", "full")


def rows(root: str, limit: int | None):
    paths = sorted(Path(root).glob("sample_*.pt"))
    if limit: paths = paths[:limit]
    return [torch.load(p, map_location="cpu", weights_only=False) for p in paths]


def targets(row):
    state = row["future_object_state"].float()
    valid = row["future_object_valid"].bool()
    # Object-level targets.  All reductions are masked by the physical target
    # validity, so padding slots cannot become artificial supervision.
    present = valid.any(-1)
    count = present.float().sum()
    pos = state[..., :3][valid]
    vel = state[..., 3:6][valid]
    extent = state[..., 6:9][valid]
    entity_position = pos.mean(0) if len(pos) else torch.zeros(3)
    entity_extent = extent.mean(0) if len(extent) else torch.zeros(3)
    speed = vel.norm(dim=-1).mean() if len(vel) else torch.tensor(0.)
    displacement = (state[:, -1, :3] - state[:, 0, :3])[present].norm(dim=-1).mean() if present.any() else torch.tensor(0.)
    # Relation targets are taken from pair annotations, not the probe output.
    distance = row["future_pair_distance"].float()
    distance_valid = row["future_pair_valid"].bool()
    contact = row["future_contact"].float()
    contact_valid = row["future_contact_valid"].bool()
    d = distance[distance_valid]
    c = contact[contact_valid]
    relation_distance = d.mean() if len(d) else torch.tensor(0.)
    relation_contact = (c > 0.5).any().float() if len(c) else torch.tensor(0.)
    relation_contact_rate = (c > 0.5).float().mean() if len(c) else torch.tensor(0.)
    ttc = row["future_time_to_contact"].float()
    ttc_valid = row["future_time_to_contact_valid"].bool()
    ttc_values = ttc[ttc_valid]
    ttc_min = ttc_values.min() if len(ttc_values) else torch.tensor(1.)
    return {
        "entity_count": count, "entity_position": entity_position, "entity_extent": entity_extent,
        "dynamic_speed": speed, "dynamic_displacement": displacement, "dynamic_ttc": ttc_min,
        "relation_distance": relation_distance, "relation_contact": relation_contact,
        "relation_contact_rate": relation_contact_rate,
    }


def fit_regression(x, y, tx, ty):
    x, y, tx, ty = map(lambda z: torch.as_tensor(z).float(), (x, y, tx, ty))
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-5)
    x, tx = (x - mean) / std, (tx - mean) / std
    head = torch.nn.Linear(x.shape[1], y.shape[1] if y.ndim > 1 else 1)
    if y.ndim == 1: y, ty = y[:, None], ty[:, None]
    opt = torch.optim.AdamW(head.parameters(), lr=.02, weight_decay=1e-4)
    for _ in range(150):
        opt.zero_grad(set_to_none=True); loss = torch.nn.functional.smooth_l1_loss(head(x), y); loss.backward(); opt.step()
    with torch.no_grad():
        pred = head(tx)
        ss_res = ((pred - ty) ** 2).sum()
        ss_tot = ((ty - ty.mean(0)) ** 2).sum().clamp_min(1e-8)
        return {"mae": float((pred - ty).abs().mean()), "r2": float(1 - ss_res / ss_tot)}


def fit_binary(x, y, tx, ty):
    x, y, tx, ty = map(lambda z: torch.as_tensor(z).float(), (x, y, tx, ty))
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-5)
    x, tx = (x - mean) / std, (tx - mean) / std
    head = torch.nn.Linear(x.shape[1], 1); opt = torch.optim.AdamW(head.parameters(), lr=.02, weight_decay=1e-4)
    pos = y.sum().clamp_min(1); weight = ((len(y) - pos) / pos).clamp(max=50)
    for _ in range(150):
        opt.zero_grad(set_to_none=True); loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x).squeeze(-1), y, pos_weight=weight); loss.backward(); opt.step()
    with torch.no_grad():
        score = head(tx).squeeze(-1); pred = score.sigmoid() > 0.5; target = ty > 0.5
        order = score.argsort().argsort().float() + 1; p, n = int(target.sum()), int((~target).sum())
        auroc = None if not p or not n else float((order[target].sum() - p * (p + 1) / 2) / (p * n))
        return {"accuracy": float((pred == target).float().mean()), "auroc": auroc, "positive_rate": float(target.float().mean())}


def main():
    ctx=task_context();p = argparse.ArgumentParser(); p.add_argument("--train", required=True); p.add_argument("--test", required=True); p.add_argument("--output", required=True); p.add_argument("--max-train", type=int); p.add_argument("--max-test", type=int); apply_cli_defaults(p,ctx);a = p.parse_args()
    train_rows, test_rows = rows(a.train, a.max_train), rows(a.test, a.max_test)
    train_targets, test_targets = [targets(x) for x in train_rows], [targets(x) for x in test_rows]
    specs = {
        "entity_count": ("regression",), "entity_position": ("regression",), "entity_extent": ("regression",),
        "dynamic_speed": ("regression",), "dynamic_displacement": ("regression",), "dynamic_ttc": ("regression",),
        "relation_distance": ("regression",), "relation_contact": ("binary",), "relation_contact_rate": ("regression",),
    }
    result = {}
    for name, (kind,) in specs.items():
        result[name] = {}
        y = torch.stack([x[name] if x[name].ndim else x[name][None] for x in train_targets]).numpy()
        ty = torch.stack([x[name] if x[name].ndim else x[name][None] for x in test_targets]).numpy()
        for layer in LAYERS:
            x = torch.stack([r[layer] for r in train_rows]).numpy(); tx = torch.stack([r[layer] for r in test_rows]).numpy()
            result[name][layer] = fit_binary(x, y.squeeze(-1), tx, ty.squeeze(-1)) if kind == "binary" else fit_regression(x, y, tx, ty)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps({"protocol": "physion_native_factor_layer_probe_v1", "train": len(train_rows), "test": len(test_rows), "results": result}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

