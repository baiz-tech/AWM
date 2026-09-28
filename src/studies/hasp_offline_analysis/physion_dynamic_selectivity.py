"""Physion++ Dynamic selectivity analysis.

This module is deliberately offline: it trains tiny readout heads on exported
Physion++ features and, when a latent cache/checkpoint is supplied, reruns the
frozen structured probe under temporal interventions.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from src.experiments.awm_physionpp_fullpatch_probe_seed239.structured_probe import PhysionStructuredProbe
from src.core.run_context import apply_cli_defaults, task_context


def load_rows(root: str, limit: int | None = None) -> list[dict]:
    paths = sorted(Path(root).glob("sample_*.pt"))
    if limit is not None:
        paths = paths[:limit]
    return [torch.load(p, map_location="cpu", weights_only=False) for p in paths]


def dynamic_targets(row: dict) -> dict[str, torch.Tensor]:
    """Return masked, scene-level physical targets for a feature row."""
    state = row["future_object_state"].float()
    valid = row["future_object_valid"].bool()
    speed = state[..., 3:6].norm(dim=-1)
    speed_values = speed[valid]
    # Acceleration is defined only between two consecutive valid samples.
    pair_valid = valid[:, 1:] & valid[:, :-1]
    acceleration = (state[:, 1:, 3:6] - state[:, :-1, 3:6]).norm(dim=-1)
    acceleration_values = acceleration[pair_valid]
    ttc = row["future_time_to_contact"].float()
    ttc_valid = row["future_time_to_contact_valid"].bool()
    return {
        "speed": speed_values.mean() if speed_values.numel() else torch.tensor(float("nan")),
        "acceleration": acceleration_values.mean() if acceleration_values.numel() else torch.tensor(float("nan")),
        "ttc": ttc[ttc_valid].min() if ttc_valid.any() else torch.tensor(float("nan")),
        "contact_event": ttc_valid.any().float(),
    }


def _standardize(train_x: torch.Tensor, test_x: torch.Tensor):
    mean = train_x.mean(0, keepdim=True)
    std = train_x.std(0, keepdim=True).clamp_min(1e-5)
    return (train_x - mean) / std, (test_x - mean) / std


def fit_regression(train_x, train_y, test_x, test_y, epochs: int = 120) -> dict:
    train_x, train_y, test_x, test_y = [torch.as_tensor(x).float() for x in (train_x, train_y, test_x, test_y)]
    train_x, test_x = _standardize(train_x, test_x)
    head = torch.nn.Linear(train_x.size(1), 1)
    opt = torch.optim.AdamW(head.parameters(), lr=0.02, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.smooth_l1_loss(head(train_x).squeeze(-1), train_y)
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = head(test_x).squeeze(-1)
    ss_tot = ((test_y - test_y.mean()) ** 2).sum().clamp_min(1e-8)
    return {"mae": float((pred - test_y).abs().mean()), "rmse": float((pred - test_y).square().mean().sqrt()), "r2": float(1 - (pred - test_y).square().sum() / ss_tot), "samples": int(len(test_y))}


def fit_binary(train_x, train_y, test_x, test_y, epochs: int = 120) -> dict:
    train_x, train_y, test_x, test_y = [torch.as_tensor(x).float() for x in (train_x, train_y, test_x, test_y)]
    train_x, test_x = _standardize(train_x, test_x)
    head = torch.nn.Linear(train_x.size(1), 1)
    opt = torch.optim.AdamW(head.parameters(), lr=0.02, weight_decay=1e-4)
    pos = train_y.sum().clamp_min(1); weight = ((len(train_y) - pos) / pos).clamp(max=50)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(train_x).squeeze(-1), train_y, pos_weight=weight)
        loss.backward(); opt.step()
    with torch.no_grad(): score = head(test_x).squeeze(-1)
    target = test_y > .5; p, n = int(target.sum()), int((~target).sum())
    if not p or not n: auc = None
    else:
        order = score.argsort().argsort().float() + 1
        auc = float((order[target].sum() - p * (p + 1) / 2) / (p * n))
    return {"auroc": auc, "accuracy": float(((score.sigmoid() > .5) == target).float().mean()), "samples": int(len(test_y))}


def decode_matrix(train_rows: list[dict], test_rows: list[dict]) -> dict:
    layers = {"entity": "entity", "dynamic": "dynamic", "relation": "relation", "full": "full"}
    train_targets, test_targets = [dynamic_targets(r) for r in train_rows], [dynamic_targets(r) for r in test_rows]
    result = {}
    for target_name in ("speed", "acceleration", "ttc"):
        result[target_name] = {}
        keep_train = torch.tensor([torch.isfinite(x[target_name]) for x in train_targets])
        keep_test = torch.tensor([torch.isfinite(x[target_name]) for x in test_targets])
        for name, key in layers.items():
            x = torch.stack([r[key] for r, keep in zip(train_rows, keep_train) if keep])
            tx = torch.stack([r[key] for r, keep in zip(test_rows, keep_test) if keep])
            y = torch.stack([x[target_name] for x, keep in zip(train_targets, keep_train) if keep])
            ty = torch.stack([x[target_name] for x, keep in zip(test_targets, keep_test) if keep])
            result[target_name][name] = fit_regression(x, y, tx, ty) if len(y) >= 2 and len(ty) >= 2 else {"mae": None, "r2": None, "samples": int(len(ty))}
    result["contact_event"] = {}
    y = torch.stack([x["contact_event"] for x in train_targets]); ty = torch.stack([x["contact_event"] for x in test_targets])
    for name, key in layers.items():
        result["contact_event"][name] = fit_binary(torch.stack([r[key] for r in train_rows]), y, torch.stack([r[key] for r in test_rows]), ty)
    return result


def temporal_intervention(context: torch.Tensor, future: torch.Tensor, mode: str, seed: int = 239):
    """Apply a deterministic temporal intervention without changing space."""
    if mode == "baseline": return context, future
    if mode == "future_reverse": return context, future.flip(1)
    if mode == "future_static":
        mean = future.mean(1, keepdim=True); return context, mean.expand_as(future).clone()
    if mode == "future_repeat": return context, future[:, :1].expand_as(future).clone()
    if mode == "future_shuffle":
        g = torch.Generator(device=future.device).manual_seed(seed)
        return context, future[:, torch.randperm(future.size(1), generator=g, device=future.device)]
    raise ValueError(f"unknown temporal intervention: {mode}")


def _assignment(output: dict, row: dict) -> torch.Tensor:
    valid_obj = row["future_object_valid"].bool().any(-1)
    ids = torch.where(valid_obj)[0].tolist(); result = torch.full((8,), -1, dtype=torch.long)
    if not ids: return result
    pred, target, mask = output["trajectory"][0].float().cpu(), row["future_object_state"].float()[ids], row["future_object_valid"].bool()[ids]
    costs = []
    for slot in range(8):
        costs.append([float((pred[slot].sub(target[j]).abs()[mask[j]]).mean()) for j in range(len(ids))])
    slots, cols = linear_sum_assignment(np.asarray(costs))
    for slot, col in zip(slots, cols): result[int(slot)] = ids[int(col)]
    return result


def intervention_metrics(output: dict, row: dict, assignment: torch.Tensor) -> dict:
    state, valid = row["future_object_state"].float(), row["future_object_valid"].bool()
    speed_err, accel_err = [], []
    for slot, target_id in enumerate(assignment.tolist()):
        if target_id < 0: continue
        mask = valid[target_id]; pred = output["trajectory"][0, slot].float().cpu()
        if mask.any():
            speed_err.append(float((pred[:, 3:6].norm(dim=-1) - state[target_id, :, 3:6].norm(dim=-1))[mask].abs().mean()))
        pair = mask[1:] & mask[:-1]
        if pair.any():
            pa = (pred[1:, 3:6] - pred[:-1, 3:6]).norm(dim=-1)
            ta = (state[target_id, 1:, 3:6] - state[target_id, :-1, 3:6]).norm(dim=-1)
            accel_err.append(float((pa - ta)[pair].abs().mean()))
    return {"speed_mae": float(np.mean(speed_err)) if speed_err else None, "acceleration_mae": float(np.mean(accel_err)) if accel_err else None}


def run_interventions(cache: str, checkpoint: str, device: str, limit: int | None) -> dict:
    files = sorted(Path(cache).glob("sample_*.pt")); files = files[:limit] if limit else files
    model = PhysionStructuredProbe().to(device); payload = torch.load(checkpoint, map_location="cpu", weights_only=False); model.load_state_dict(payload.get("model", payload), strict=True); model.eval()
    modes = ("baseline", "future_reverse", "future_shuffle", "future_static", "future_repeat"); collected = {m: [] for m in modes}
    with torch.inference_mode():
        for path in files:
            row = torch.load(path, map_location="cpu", weights_only=False); context = row["context"].unsqueeze(0).to(device); future = row["future"].unsqueeze(0).to(device)
            base = model(context, future, return_features=True); assignment = _assignment(base, row)
            for mode in modes:
                x, y = temporal_intervention(context, future, mode); out = model(x, y, return_features=True); collected[mode].append(intervention_metrics(out, row, assignment))
    summary = {m: {k: float(np.mean([x[k] for x in values if x[k] is not None])) if any(x[k] is not None for x in values) else None for k in ("speed_mae", "acceleration_mae")} for m, values in collected.items()}
    base = summary["baseline"]
    delta = {m: {k: (None if summary[m][k] is None or base[k] is None else summary[m][k] - base[k]) for k in base} for m in modes if m != "baseline"}
    return {"samples": len(files), "summary": summary, "delta_vs_baseline": delta}


def main() -> None:
    ctx = task_context()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-features", required=True); p.add_argument("--test-features", required=True); p.add_argument("--output", required=True)
    p.add_argument("--cache"); p.add_argument("--checkpoint"); p.add_argument("--device", default="cpu"); p.add_argument("--max-samples", type=int)
    apply_cli_defaults(p, ctx)
    a = p.parse_args(); torch.manual_seed(239); np.random.seed(239)
    result = {"protocol": "physion_dynamic_selectivity_v1", "decode": decode_matrix(load_rows(a.train_features, a.max_samples), load_rows(a.test_features, a.max_samples))}
    if bool(a.cache) != bool(a.checkpoint): p.error("--cache and --checkpoint must be provided together")
    if a.cache: result["interventions"] = run_interventions(a.cache, a.checkpoint, a.device, a.max_samples)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
