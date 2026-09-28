#!/usr/bin/env python3
"""Train the current-only structured CLEVRER predictability probe with DDP."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from .model import CLEVRERDecoder, model_config, structured_probe_loss
from src.training.distributed import ExactDistributedSampler
from src.core.run_context import apply_cli_defaults, task_context


LOGGER = logging.getLogger("reproduce_v1_clevrer_only_analysePredictability.clevrer.train")
TARGET_KEYS = (
    "object_present", "color", "material", "shape", "state", "state_valid",
    "pair_distance", "pair_valid", "contact", "contact_valid", "first_contact_class",
)


class FullPatchProbeDataset(Dataset):
    def __init__(self, cache_root: Path, target_path: Path, split: str, max_scenes=None):
        self.cache_dir = Path(cache_root) / split
        target = torch.load(target_path, map_location="cpu", weights_only=False)
        scene_ids = target["scene_id"].long()
        window_starts = target["window_start"].long()
        if max_scenes is not None:
            selected_scenes = torch.unique_consecutive(scene_ids)[: int(max_scenes)]
            keep = torch.isin(scene_ids, selected_scenes)
            scene_ids = scene_ids[keep]
            window_starts = window_starts[keep]
        else:
            keep = torch.ones_like(scene_ids, dtype=torch.bool)
        self.scene_ids = scene_ids
        self.window_starts = window_starts
        self.targets = {key: target[key][keep] for key in TARGET_KEYS}
        missing = [
            (int(scene_id), int(window_start))
            for scene_id, window_start in zip(scene_ids, window_starts)
            if not (self.cache_dir / f"scene_{int(scene_id):05d}_window_{int(window_start):03d}.pt").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"missing {len(missing)} full-patch cache scenes under {self.cache_dir}; first={missing[:8]}"
            )

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, index):
        scene_id = int(self.scene_ids[index])
        window_start = int(self.window_starts[index])
        record = torch.load(
            self.cache_dir / f"scene_{scene_id:05d}_window_{window_start:03d}.pt",
            map_location="cpu", weights_only=True
        )
        if int(record["scene_id"]) != scene_id or int(record["window_start"]) != window_start:
            raise ValueError(f"cache scene mismatch at index={index}")
        return {
            "scene_id": torch.tensor(scene_id),
            "window_start": torch.tensor(window_start),
            "context": record["context_tokens"],
            **{key: value[index] for key, value in self.targets.items()},
        }


def init_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return rank, world_size, local_rank


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def batch_scene_ids(batch):
    return [int(value) for value in batch["scene_id"].detach().cpu().tolist()]


@torch.no_grad()
def evaluate(model, loader, device, use_amp, amp_dtype):
    model.eval()
    totals: dict[str, torch.Tensor] = {}
    samples = torch.zeros(1, dtype=torch.float64, device=device)
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp, dtype=amp_dtype if use_amp else None
        ):
            outputs = model(batch["context"])
            try:
                _, losses = structured_probe_loss(outputs, batch)
            except (FloatingPointError, ValueError) as error:
                raise RuntimeError(
                    f"validation failed for scene_ids={batch_scene_ids(batch)}: {error}"
                ) from error
        count = batch["context"].size(0)
        samples += count
        for name, value in losses.items():
            totals.setdefault(name, torch.zeros(1, dtype=torch.float64, device=device))
            totals[name] += value.detach().double() * count
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(samples)
        for value in totals.values():
            dist.all_reduce(value)
    return {name: float((value / samples.clamp_min(1)).item()) for name, value in totals.items()}


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, args):
    module = model.module if isinstance(model, DistributedDataParallel) else model
    payload = {
        "protocol": "clevrer_current_only_predictability_v1",
        "epoch": int(epoch),
        "model": module.state_dict(),
        "decoder": module.decoder.state_dict(),
        "clevrer_probes": module.probes.state_dict(),
        "model_config": model_config(module),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "validation_metrics": metrics,
        "seed": args.seed,
        "cache_root": str(args.cache_root.resolve()),
        "train_targets": str(args.train_targets.resolve()),
        "validation_targets": str(args.validation_targets.resolve()),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    ctx = task_context()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--train-targets", type=Path, required=True)
    parser.add_argument("--validation-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--slot-depth", type=int, default=1)
    parser.add_argument("--transition-depth", type=int, default=1)
    parser.add_argument("--interaction-depth", type=int, default=1)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--max-train-scenes", type=int)
    parser.add_argument("--max-validation-scenes", type=int)
    apply_cli_defaults(parser, ctx)
    args = parser.parse_args()
    rank, world_size, local_rank = init_distributed()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format="[%(levelname)-8s][%(asctime)s][%(name)s] %(message)s",
    )
    seed_everything(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    train = FullPatchProbeDataset(
        args.cache_root, args.train_targets, "train", args.max_train_scenes
    )
    validation = FullPatchProbeDataset(
        args.cache_root, args.validation_targets, "validation", args.max_validation_scenes
    )
    train_sampler = DistributedSampler(
        train, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    )
    validation_sampler = ExactDistributedSampler(validation, rank=rank, world_size=world_size)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train, sampler=train_sampler, **loader_options)
    validation_loader = DataLoader(validation, sampler=validation_sampler, **loader_options)
    model = CLEVRERDecoder(
        input_dim=1280, hidden_dim=args.hidden_dim, num_heads=args.num_heads,
        ffn_dim=args.ffn_dim, num_slots=args.num_slots,
        slot_depth=args.slot_depth, transition_depth=args.transition_depth,
        interaction_depth=args.interaction_depth, dropout=args.dropout,
    ).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        train_total = torch.zeros(2, dtype=torch.float64, device=device)
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=use_amp, dtype=amp_dtype if use_amp else None
            ):
                outputs = model(batch["context"])
                try:
                    loss, _ = structured_probe_loss(outputs, batch)
                except (FloatingPointError, ValueError) as error:
                    raise RuntimeError(
                        f"training failed for scene_ids={batch_scene_ids(batch)}: {error}"
                    ) from error
            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip, error_if_nonfinite=True
                )
            except RuntimeError as error:
                raise RuntimeError(
                    "non-finite probe gradient for "
                    f"scene_ids={batch_scene_ids(batch)} loss={float(loss.detach())}"
                ) from error
            optimizer.step()
            train_total[0] += loss.detach().double() * batch["context"].size(0)
            train_total[1] += batch["context"].size(0)
        scheduler.step()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(train_total)
        train_loss = float((train_total[0] / train_total[1].clamp_min(1)).item())
        validation_metrics = evaluate(model, validation_loader, device, use_amp, amp_dtype)
        record = {"epoch": epoch, "train_loss": train_loss, "validation": validation_metrics}
        history.append(record)
        if rank == 0:
            LOGGER.info("epoch=%d train_loss=%.6f validation=%s", epoch, train_loss, validation_metrics)
            save_checkpoint(
                args.output_dir / "latest.pt", model, optimizer, scheduler,
                epoch, validation_metrics, args,
            )
            if validation_metrics["total"] < best:
                best = validation_metrics["total"]
                save_checkpoint(
                    args.output_dir / "best.pt", model, optimizer, scheduler,
                    epoch, validation_metrics, args,
                )
                (args.output_dir / "best_metrics.json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    if rank == 0:
        manifest = {
            "protocol": "clevrer_current_only_predictability_v1",
            "best_validation_loss": best,
            "epochs": args.epochs,
            "world_size": world_size,
            "seed": args.seed,
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
