#!/usr/bin/env python3
"""Train the v5 decoder and EK100 object/event probes from frozen caches."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.training.distributed import ExactDistributedSampler

from .model import CACHE_PROTOCOL, CHECKPOINT_PROTOCOL, TARGET_PROTOCOL, Epic100Decoder, decoder_loss
from src.core.run_context import apply_cli_defaults, task_context

TOKEN_SHAPE = (8, 256, 1280)
LOSS_KEYS = ("total", "presence", "category", "center", "geometry", "velocity", "log_area", "distance", "verb", "noun", "action")


class CachedDataset(Dataset):
    def __init__(self, cache_dir: Path, target_path: Path):
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        if manifest.get("protocol") != CACHE_PROTOCOL or not manifest.get("complete"):
            raise ValueError(f"invalid/incomplete latent cache: {cache_dir}")
        target = torch.load(target_path, map_location="cpu", weights_only=False)
        if target.get("protocol") != TARGET_PROTOCOL:
            raise ValueError(f"unexpected target protocol: {target.get('protocol')!r}")
        if int(manifest["samples"]) != int(target["samples"]):
            raise ValueError("cache and target sample counts differ")
        self.targets = target
        self.cache_dir = cache_dir
        self.count = int(target["samples"])
        self.context = np.memmap(cache_dir / "context.f16", mode="r", dtype=np.float16, shape=(self.count, *TOKEN_SHAPE))
        self.future = np.memmap(cache_dir / "future.f16", mode="r", dtype=np.float16, shape=(self.count, *TOKEN_SHAPE))

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {
            "context": torch.from_numpy(np.array(self.context[index], copy=True)),
            "future": torch.from_numpy(np.array(self.future[index], copy=True)),
            "object_present": self.targets["object_present"][index],
            "category": self.targets["category"][index],
            "state": self.targets["state"][index],
            "state_valid": self.targets["state_valid"][index],
            "pair_distance": self.targets["pair_distance"][index],
            "pair_valid": self.targets["pair_valid"][index],
            "verb_label": self.targets["verb_label"][index],
            "noun_label": self.targets["noun_label"][index],
            "action_label": self.targets["action_label"][index],
        }


def setup(seed):
    if "RANK" not in os.environ:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        return 0, 1, 0
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    return rank, world, local


def reduce_stats(values, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor.cpu().tolist()


def main():
    ctx = task_context()
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, required=True)
    # Two-phase parser: this module needs --config before it can build the
    # rest of the parser, so only that option is defaulted here.
    apply_cli_defaults(preliminary, ctx, options={"--config"})
    known, _ = preliminary.parse_known_args()
    config = yaml.safe_load(known.config.read_text())
    decoder_defaults = config.get("decoder", {})
    optimization_defaults = config.get("optimization", {})
    parser = argparse.ArgumentParser(parents=[preliminary])
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--train-targets", type=Path, required=True)
    parser.add_argument("--validation-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=int(optimization_defaults.get("epochs", 30)))
    parser.add_argument("--batch-size", type=int, default=int(optimization_defaults.get("batch_size_per_gpu", 1)))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=int(decoder_defaults.get("hidden_dim", 256)))
    parser.add_argument("--num-heads", type=int, default=int(decoder_defaults.get("num_heads", 8)))
    parser.add_argument("--ffn-dim", type=int, default=int(decoder_defaults.get("ffn_dim", 1024)))
    parser.add_argument("--num-slots", type=int, default=int(decoder_defaults.get("num_slots", 8)))
    parser.add_argument("--slot-depth", type=int, default=int(decoder_defaults.get("slot_depth", 1)))
    parser.add_argument("--transition-depth", type=int, default=int(decoder_defaults.get("transition_depth", 1)))
    parser.add_argument("--interaction-depth", type=int, default=int(decoder_defaults.get("interaction_depth", 1)))
    parser.add_argument("--dropout", type=float, default=float(decoder_defaults.get("dropout", 0.1)))
    parser.add_argument("--learning-rate", type=float, default=float(optimization_defaults.get("learning_rate", 2e-4)))
    parser.add_argument("--weight-decay", type=float, default=float(optimization_defaults.get("weight_decay", 0.04)))
    parser.add_argument("--event-weight", type=float, default=float(optimization_defaults.get("event_weight", 1.0)))
    parser.add_argument("--seed", type=int, default=int(optimization_defaults.get("seed", 239)))
    parser.add_argument("--resume", action="store_true")
    apply_cli_defaults(parser, ctx)
    args = parser.parse_args()
    rank, world, local = setup(args.seed)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    train = CachedDataset(args.train_cache, args.train_targets)
    validation = CachedDataset(args.validation_cache, args.validation_targets)
    train_vocab = train.targets["vocabulary"]
    val_vocab = validation.targets["vocabulary"]
    if train_vocab != val_vocab:
        raise ValueError("train/validation label vocabularies differ")
    train_sampler = DistributedSampler(train, world, rank, shuffle=True, seed=args.seed)
    val_sampler = ExactDistributedSampler(validation, rank, world)
    options = {"batch_size": args.batch_size, "num_workers": args.num_workers,
               "pin_memory": device.type == "cuda", "persistent_workers": args.num_workers > 0}
    train_loader, val_loader = DataLoader(train, sampler=train_sampler, **options), DataLoader(validation, sampler=val_sampler, **options)
    decoder_config = {
        "input_dim": 1280, "hidden_dim": args.hidden_dim, "num_heads": args.num_heads,
        "ffn_dim": args.ffn_dim, "num_slots": args.num_slots,
        "slot_depth": args.slot_depth, "transition_depth": args.transition_depth,
        "interaction_depth": args.interaction_depth, "dropout": args.dropout,
    }
    model = Epic100Decoder(
        num_categories=int(train.targets["num_categories"]),
        num_verbs=len(train_vocab["verb_ids"]), num_nouns=len(train_vocab["noun_ids"]),
        num_actions=len(train_vocab["action_pairs"]), **decoder_config,
    ).to(device)
    if world > 1:
        # VISOR samples can contain fewer than two simultaneously valid tracks;
        # pair-distance branches are therefore legitimately unused on some
        # ranks/batches. DDP must account for those sparse supervision paths.
        model = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest, best_path = args.output_dir / "latest.pt", args.output_dir / "best.pt"
    start_epoch, best, history = 0, float("inf"), []
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError(f"resume checkpoint missing: {latest}")
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload.get("protocol") != CHECKPOINT_PROTOCOL:
            raise ValueError("resume checkpoint protocol mismatch")
        module = model.module if isinstance(model, DDP) else model
        module.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"])
        start_epoch, best, history = int(payload["epoch"]), float(payload["best_validation_loss"]), list(payload.get("history", []))
    elif latest.exists() or best_path.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")

    def run_epoch(loader, training, epoch):
        model.train(training)
        sums = {key: 0.0 for key in LOSS_KEYS}; count = 0
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
                outputs = model(batch["context"], batch["future"])
                value, losses, _ = decoder_loss(outputs, batch, args.event_weight)
            if training:
                optimizer.zero_grad(set_to_none=True); value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            size = batch["context"].size(0); count += size
            for key in LOSS_KEYS: sums[key] += float(losses[key].detach()) * size
        stats = reduce_stats([sums[key] for key in LOSS_KEYS] + [count], device)
        denominator = max(stats[-1], 1.0)
        return {key: stats[index] / denominator for index, key in enumerate(LOSS_KEYS)}

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_stats = run_epoch(train_loader, True, epoch)
        with torch.no_grad():
            validation_stats = run_epoch(val_loader, False, epoch)
        scheduler.step()
        row = {"epoch": epoch, "learning_rate": scheduler.get_last_lr()[0], "train": train_stats, "validation": validation_stats}
        history.append(row)
        if rank == 0:
            module = model.module if isinstance(model, DDP) else model
            payload = {"protocol": CHECKPOINT_PROTOCOL, "model": module.state_dict(), "model_config": {
                **decoder_config, "num_categories": int(train.targets["num_categories"]),
                "num_verbs": len(train_vocab["verb_ids"]), "num_nouns": len(train_vocab["noun_ids"]),
                "num_actions": len(train_vocab["action_pairs"]),
            }, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "epoch": epoch, "best_validation_loss": best, "history": history,
                "train_cache": str(args.train_cache.resolve()), "validation_cache": str(args.validation_cache.resolve()),
                "train_targets": str(args.train_targets.resolve()), "validation_targets": str(args.validation_targets.resolve()),
                "config": str(args.config.resolve()), "resolved_config": config,
                "official_encoder_predictor_frozen": True, "seed": args.seed,
            }
            if validation_stats["total"] < best:
                best = validation_stats["total"]; payload["best_validation_loss"] = best; torch.save(payload, best_path)
            payload["best_validation_loss"] = best
            torch.save(payload, latest)
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
