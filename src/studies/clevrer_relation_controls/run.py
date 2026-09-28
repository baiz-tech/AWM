#!/usr/bin/env python3
"""Pair-level Relation-only/no-Relation controls for CLEVRER.

The decoder and structured probe remain frozen. Small linear readouts are
trained here only to compare the information available in representation
subsets. Targets are aligned with the frozen probe's object assignment.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.experiments.awm_clevrer_fullpatch_probe_seed239.model import (
    CLEVRERDecoder,
    match_objects,
)
from src.core.run_context import apply_cli_defaults, task_context

PAIR_COUNT = 6
FEATURES = ("entity", "dynamic", "relation")
VARIANTS = {
    "Entity": ("entity",),
    "Dynamic": ("dynamic",),
    "Relation-only": ("relation",),
    "Entity+Dynamic": ("entity", "dynamic"),
    "Entity+Relation": ("entity", "relation"),
    "Dynamic+Relation": ("dynamic", "relation"),
    "Full": FEATURES,
}
LOGGER = logging.getLogger("clevrer_relation_controls")


def _load_model(checkpoint: Path, device: torch.device) -> CLEVRERDecoder:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("model_config", {})
    model = CLEVRERDecoder(**config).to(device)
    state = payload.get("model")
    if state is None and "decoder" in payload and "clevrer_probes" in payload:
        state = {f"decoder.{k}": v for k, v in payload["decoder"].items()}
        state.update({f"probes.{k}": v for k, v in payload["clevrer_probes"].items()})
    if state is None:
        state = payload
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _refs(root: Path, targets: dict, limit: int | None, rank: int = 0, world_size: int = 1) -> list[tuple[Path, int]]:
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
    return refs[rank::world_size]


def _assignment_batch(target: dict, index: int, device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("object_present", "color", "material", "shape", "state", "state_valid")
    return {key: target[key][index:index + 1].to(device) for key in keys}


@torch.inference_mode()
def extract(root: Path, targets: dict, checkpoint: Path, device: torch.device, limit: int | None, rank: int = 0, world_size: int = 1) -> list[dict]:
    model = _load_model(checkpoint, device)
    rows: list[dict] = []
    refs = _refs(root, targets, limit, rank, world_size)
    LOGGER.info("rank=%d/%d extracting %d latent files from %s", rank, world_size, len(refs), root)
    started = time.time()
    for file_index, (path, target_index) in enumerate(refs, 1):
        record = torch.load(path, map_location="cpu", weights_only=True)
        context = record["context_tokens"].unsqueeze(0).to(device).float()
        future = record["future_tokens"].unsqueeze(0).to(device).float()
        output = model(context, future, return_features=True)
        assignment = match_objects(output, _assignment_batch(targets, target_index, device))[0].cpu()
        target_to_pred = {
            int(target_slot): int(pred_slot)
            for pred_slot, target_slot in enumerate(assignment.tolist())
            if int(target_slot) >= 0
        }
        objects = output["object_tokens"][0].float()
        dynamics = output["time_features"][0].float()
        relations = output["pair_time_features"][0].float()
        first = targets["first_contact_class"][target_index].long()
        count = int(targets["object_present"][target_index].sum())
        pair_indices = list(itertools.combinations(range(PAIR_COUNT), 2))
        for left, right in itertools.combinations(range(count), 2):
            if left not in target_to_pred or right not in target_to_pred:
                continue
            pred_left, pred_right = target_to_pred[left], target_to_pred[right]
            pair_slot = pair_indices.index(tuple(sorted((pred_left, pred_right))))
            first_class = int(first[left, right])
            if first_class < 0:
                continue
            rows.append({
                "entity": (objects[pred_left] + objects[pred_right]).mul(0.5).cpu(),
                "dynamic": (dynamics[pred_left] + dynamics[pred_right]).mean(0).mul(0.5).cpu(),
                "relation": relations[pair_slot].mean(0).cpu(),
                "contact": float(first_class < 16),
                "first_contact": first_class,
                "ttc": float(first_class / 15.0) if first_class < 16 else float("nan"),
            })
        del record, output
        if file_index == 1 or file_index % 100 == 0 or file_index == len(refs):
            LOGGER.info("rank=%d progress=%d/%d valid_pairs=%d elapsed=%.1fs", rank, file_index, len(refs), len(rows), time.time() - started)
    LOGGER.info("rank=%d extraction complete files=%d valid_pairs=%d elapsed=%.1fs", rank, len(refs), len(rows), time.time() - started)
    return rows


def _matrix(rows: list[dict], names: tuple[str, ...], indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.cat([torch.cat([rows[i][name] for name in names]).view(1, -1) for i in indices])
    contact = torch.tensor([rows[i]["contact"] for i in indices], dtype=torch.float32)
    first = torch.tensor([rows[i]["first_contact"] for i in indices], dtype=torch.long)
    return x, contact, first


def _standardize(train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train.mean(0, keepdim=True)
    std = train.std(0, keepdim=True).clamp_min(1e-5)
    return (train - mean) / std, (test - mean) / std


def _auroc(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    labels = labels.bool()
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    order = scores.argsort().argsort().float() + 1
    return float((order[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _binary(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor) -> dict:
    train_x, test_x = _standardize(train_x, test_x)
    model = torch.nn.Linear(train_x.size(1), 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    weight = ((len(train_y) - train_y.sum()) / train_y.sum().clamp_min(1)).clamp(max=50)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(train_x).squeeze(-1), train_y, pos_weight=weight)
        loss.backward(); optimizer.step()
    with torch.no_grad():
        scores = model(test_x).squeeze(-1)
    return {"auroc": _auroc(test_y, scores), "positive_rate": float(test_y.mean()), "samples": len(test_y)}


def _multiclass(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor) -> dict:
    train_x, test_x = _standardize(train_x, test_x)
    classes = int(max(train_y.max(), test_y.max()).item() + 1)
    model = torch.nn.Linear(train_x.size(1), classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_x), train_y)
        loss.backward(); optimizer.step()
    with torch.no_grad():
        predicted = model(test_x).argmax(-1)
    return {"accuracy": float((predicted == test_y).float().mean()), "samples": len(test_y), "classes": classes}


def _regression(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor) -> dict:
    if not len(train_y) or not len(test_y):
        return {"mae": None, "samples": len(test_y)}
    train_x, test_x = _standardize(train_x, test_x)
    model = torch.nn.Linear(train_x.size(1), 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = F.smooth_l1_loss(model(train_x).squeeze(-1), train_y)
        loss.backward(); optimizer.step()
    with torch.no_grad():
        prediction = model(test_x).squeeze(-1)
    return {"mae": float((prediction - test_y).abs().mean()), "samples": len(test_y)}


def evaluate(train_rows: list[dict], validation_rows: list[dict]) -> dict:
    result = {name: {} for name in VARIANTS}
    train_indices = list(range(len(train_rows)))
    validation_indices = list(range(len(validation_rows)))
    for variant, names in VARIANTS.items():
        train_x, train_contact, train_first = _matrix(train_rows, names, train_indices)
        validation_x, validation_contact, validation_first = _matrix(validation_rows, names, validation_indices)
        result[variant]["contact"] = _binary(train_x, train_contact, validation_x, validation_contact)
        result[variant]["first_contact"] = _multiclass(train_x, train_first, validation_x, validation_first)
        train_ttc_mask, validation_ttc_mask = torch.isfinite(torch.tensor([r["ttc"] for r in train_rows])), torch.isfinite(torch.tensor([r["ttc"] for r in validation_rows]))
        result[variant]["ttc"] = _regression(train_x[train_ttc_mask], torch.tensor([r["ttc"] for r in train_rows])[train_ttc_mask], validation_x[validation_ttc_mask], torch.tensor([r["ttc"] for r in validation_rows])[validation_ttc_mask])
        result[variant]["feature_dim"] = int(train_x.size(1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-latents", type=Path, required=True)
    parser.add_argument("--validation-latents", type=Path, required=True)
    parser.add_argument("--train-targets", type=Path, required=True)
    parser.add_argument("--validation-targets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--shard-output", type=Path)
    parser.add_argument("--merge-shards", action="store_true")
    apply_cli_defaults(parser, task_context())
    args = parser.parse_args()
    args.rank = int(os.environ.get("RANK", args.rank))
    args.world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    if args.world_size > 1 and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", args.rank))
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [pid=%(process)d] %(message)s")
    LOGGER.info("start rank=%d world_size=%d merge=%s max_samples=%s", args.rank, args.world_size, args.merge_shards, args.max_samples or "all")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_targets = torch.load(args.train_targets, map_location="cpu", weights_only=False)
    validation_targets = torch.load(args.validation_targets, map_location="cpu", weights_only=False)
    if args.merge_shards:
        if args.shard_output is None:
            raise ValueError("--shard-output is required with --merge-shards")
        train_rows = []
        validation_rows = []
        for rank in range(args.world_size):
            train_path = args.shard_output / f"train_rank{rank:02d}.pt"
            validation_path = args.shard_output / f"validation_rank{rank:02d}.pt"
            LOGGER.info("loading rank=%d train=%s validation=%s", rank, train_path, validation_path)
            train_rows.extend(torch.load(train_path, map_location="cpu", weights_only=False))
            validation_rows.extend(torch.load(validation_path, map_location="cpu", weights_only=False))
        LOGGER.info("merged train_pairs=%d validation_pairs=%d", len(train_rows), len(validation_rows))
    else:
        train_rows = extract(args.train_latents, train_targets, args.checkpoint, device, args.max_samples, args.rank, args.world_size)
        validation_rows = extract(args.validation_latents, validation_targets, args.checkpoint, device, args.max_samples, args.rank, args.world_size)
        if args.world_size > 1:
            if args.shard_output is None:
                raise ValueError("--shard-output is required when --world-size > 1")
            args.shard_output.mkdir(parents=True, exist_ok=True)
            torch.save(train_rows, args.shard_output / f"train_rank{args.rank:02d}.pt")
            torch.save(validation_rows, args.shard_output / f"validation_rank{args.rank:02d}.pt")
            LOGGER.info("wrote rank shards to %s", args.shard_output)
            return
    if not train_rows or not validation_rows:
        raise RuntimeError("no aligned valid pair rows were found")
    payload = {"protocol": "clevrer_relation_controls_v1", "train_pairs": len(train_rows), "validation_pairs": len(validation_rows), "checkpoint": str(args.checkpoint.resolve()), "metrics": evaluate(train_rows, validation_rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("metrics written to %s", args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
