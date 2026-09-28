#!/usr/bin/env python3
"""Train the full-patch structured CLEVRER probe with DDP."""

from __future__ import annotations

import argparse
import datetime
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


LOGGER = logging.getLogger("vjepa2_naive_probe_v5_decoder.train")
TARGET_KEYS = (
    "object_present", "color", "material", "shape", "state", "state_valid",
    "pair_distance", "pair_valid", "contact", "contact_valid", "first_contact_class",
)
MODE_IDS = {"nonpredictive": 0, "predictive": 1}


class FullPatchProbeDataset(Dataset):
    def __init__(self, cache_roots: dict[str, Path], target_path: Path, split: str, max_scenes=None):
        if not cache_roots:
            raise ValueError("at least one latent cache is required")
        self.modes = tuple(cache_roots)
        self.cache_dirs = {mode: Path(root) / split for mode, root in cache_roots.items()}
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
        missing = []
        for mode, cache_dir in self.cache_dirs.items():
            missing.extend(
                (mode, int(scene_id), int(window_start))
                for scene_id, window_start in zip(scene_ids, window_starts)
                if not (cache_dir / f"scene_{int(scene_id):05d}_window_{int(window_start):03d}.pt").is_file()
            )
        if missing:
            raise FileNotFoundError(
                f"missing {len(missing)} full-patch cache samples; first={missing[:8]}"
            )

    def __len__(self):
        return len(self.scene_ids) * len(self.modes)

    def __getitem__(self, index):
        target_index = index % len(self.scene_ids)
        mode_index = index // len(self.scene_ids)
        mode = self.modes[mode_index]
        scene_id = int(self.scene_ids[target_index])
        window_start = int(self.window_starts[target_index])
        record = torch.load(
            self.cache_dirs[mode] / f"scene_{scene_id:05d}_window_{window_start:03d}.pt",
            map_location="cpu", weights_only=True
        )
        if int(record["scene_id"]) != scene_id or int(record["window_start"]) != window_start:
            raise ValueError(f"cache scene mismatch at index={index}")
        return {
            "scene_id": torch.tensor(scene_id),
            "window_start": torch.tensor(window_start),
            "latent_mode": torch.tensor(MODE_IDS[mode]),
            "tokens": record.get("tokens", torch.cat([record["context_tokens"], record["future_tokens"]], dim=0)),
            "source_ids": record.get("source_ids", torch.zeros(16, dtype=torch.long)),
            **{key: value[target_index] for key, value in self.targets.items()},
        }


def init_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
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
    evaluation_model = model.module if isinstance(model, DistributedDataParallel) else model
    totals: dict[str, torch.Tensor] = {}
    samples = torch.zeros(1, dtype=torch.float64, device=device)
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp, dtype=amp_dtype if use_amp else None
        ):
            tokens = batch["tokens"] if use_amp else batch["tokens"].float()
            outputs = evaluation_model(tokens, batch["source_ids"])
            try:
                _, losses = structured_probe_loss(outputs, batch)
            except (FloatingPointError, ValueError) as error:
                raise RuntimeError(
                    f"validation failed for scene_ids={batch_scene_ids(batch)}: {error}"
                ) from error
        count = batch["tokens"].size(0)
        samples += count
        for name, value in losses.items():
            totals.setdefault(name, torch.zeros(1, dtype=torch.float64, device=device))
            totals[name] += value.detach().double() * count
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(samples)
        for value in totals.values():
            dist.all_reduce(value)
    return (
        {name: float((value / samples.clamp_min(1)).item()) for name, value in totals.items()},
        int(samples.item()),
    )


def combine_metrics(results):
    total_samples = sum(samples for _, samples in results)
    names = results[0][0]
    return {
        name: sum(metrics[name] * samples for metrics, samples in results) / max(1, total_samples)
        for name in names
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, args):
    module = model.module if isinstance(model, DistributedDataParallel) else model
    payload = {
        "protocol": "clevrer_fullpatch_dynamics_decoder_v1",
        "epoch": int(epoch),
        "model": module.state_dict(),
        "decoder": module.decoder.state_dict(),
        "clevrer_probes": module.probes.state_dict(),
        "model_config": model_config(module),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "validation_metrics": metrics,
        "best_validation_loss": float(args.best_validation_loss),
        "history": args.history,
        "seed": args.seed,
        "world_size": dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1,
        "latent_modes": ["nonpredictive", "predictive"],
        "nonpredictive_cache_root": str(args.nonpredictive_cache_root.resolve()),
        "predictive_cache_root": str(args.predictive_cache_root.resolve()),
        "train_targets": str(args.train_targets.resolve()),
        "validation_targets": str(args.validation_targets.resolve()),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    ctx = task_context()
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonpredictive-cache-root", type=Path, required=True)
    parser.add_argument("--predictive-cache-root", type=Path, required=True)
    parser.add_argument("--train-targets", type=Path, required=True)
    parser.add_argument("--validation-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-slots", type=int, default=6)
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
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--log-every", type=int, default=25)
    apply_cli_defaults(parser, ctx)
    args = parser.parse_args()
    rank, world_size, local_rank = init_distributed()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format="[%(levelname)-8s][%(asctime)s][%(name)s] %(message)s",
    )
    seed_everything(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    cache_roots = {
        "nonpredictive": args.nonpredictive_cache_root,
        "predictive": args.predictive_cache_root,
    }
    train = FullPatchProbeDataset(cache_roots, args.train_targets, "train", args.max_train_scenes)
    validation_nonpredictive = FullPatchProbeDataset(
        {"nonpredictive": args.nonpredictive_cache_root},
        args.validation_targets, "validation", args.max_validation_scenes,
    )
    validation_predictive = FullPatchProbeDataset(
        {"predictive": args.predictive_cache_root},
        args.validation_targets, "validation", args.max_validation_scenes,
    )
    train_sampler = DistributedSampler(
        train, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    )
    validation_samplers = {
        "nonpredictive": ExactDistributedSampler(validation_nonpredictive, rank=rank, world_size=world_size),
        "predictive": ExactDistributedSampler(validation_predictive, rank=rank, world_size=world_size),
    }
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train, sampler=train_sampler, **loader_options)
    validation_loaders = {
        "nonpredictive": DataLoader(
            validation_nonpredictive, sampler=validation_samplers["nonpredictive"], **loader_options
        ),
        "predictive": DataLoader(
            validation_predictive, sampler=validation_samplers["predictive"], **loader_options
        ),
    }
    model = CLEVRERDecoder(
        input_dim=1280, hidden_dim=args.hidden_dim, num_heads=args.num_heads,
        ffn_dim=args.ffn_dim, num_slots=args.num_slots,
        slot_depth=args.slot_depth, transition_depth=args.transition_depth,
        interaction_depth=args.interaction_depth, dropout=args.dropout,
    ).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best, history, start_epoch = float("inf"), [], 1
    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        module = model.module if isinstance(model, DistributedDataParallel) else model
        if payload.get("protocol") != "clevrer_fullpatch_dynamics_decoder_v1":
            raise ValueError(f"unexpected resume protocol: {payload.get('protocol')!r}")
        if payload.get("model_config") != model_config(module):
            raise ValueError("resume checkpoint model config does not match requested model")
        if int(payload.get("world_size", world_size)) != world_size:
            raise ValueError(
                f"resume world_size={payload.get('world_size')} does not match current {world_size}"
            )
        module.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        best = float(payload.get("best_validation_loss", float("inf")))
        history = list(payload.get("history", []))
        if rank == 0:
            LOGGER.info("resumed checkpoint=%s start_epoch=%d best=%.6f", args.resume_checkpoint, start_epoch, best)
    args.history = history
    args.best_validation_loss = best
    if rank == 0:
        LOGGER.info(
            "train_samples=%d nonpredictive_validation=%d predictive_validation=%d "
            "world_size=%d batches_per_rank=%d",
            len(train), len(validation_nonpredictive), len(validation_predictive),
            world_size, len(train_loader),
        )
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        train_total = torch.zeros(2, dtype=torch.float64, device=device)
        for step, batch in enumerate(train_loader, 1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=use_amp, dtype=amp_dtype if use_amp else None
            ):
                tokens = batch["tokens"] if use_amp else batch["tokens"].float()
                outputs = model(tokens, batch["source_ids"])
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
            train_total[0] += loss.detach().double() * batch["tokens"].size(0)
            train_total[1] += batch["tokens"].size(0)
            if rank == 0 and (step == 1 or step % args.log_every == 0 or step == len(train_loader)):
                LOGGER.info(
                    "epoch=%d step=%d/%d loss=%.6f",
                    epoch, step, len(train_loader), float(loss.detach()),
                )
        scheduler.step()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(train_total)
        train_loss = float((train_total[0] / train_total[1].clamp_min(1)).item())
        nonpredictive_result = evaluate(
            model, validation_loaders["nonpredictive"], device, use_amp, amp_dtype
        )
        predictive_result = evaluate(
            model, validation_loaders["predictive"], device, use_amp, amp_dtype
        )
        validation_metrics = {
            "nonpredictive": nonpredictive_result[0],
            "predictive": predictive_result[0],
            "combined": combine_metrics([nonpredictive_result, predictive_result]),
        }
        record = {"epoch": epoch, "train_loss": train_loss, "validation": validation_metrics}
        history.append(record)
        if rank == 0:
            LOGGER.info("epoch=%d train_loss=%.6f validation=%s", epoch, train_loss, validation_metrics)
            args.history = history
            args.best_validation_loss = min(best, validation_metrics["combined"]["total"])
            save_checkpoint(
                args.output_dir / "latest.pt", model, optimizer, scheduler,
                epoch, validation_metrics, args,
            )
            if validation_metrics["combined"]["total"] < best:
                best = validation_metrics["combined"]["total"]
                args.best_validation_loss = best
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
            "protocol": "clevrer_fullpatch_dynamics_decoder_v1",
            "best_validation_loss": best,
            "epochs": args.epochs,
            "world_size": world_size,
            "nodes": int(os.environ.get("NNODES", 1)),
            "latent_modes": ["nonpredictive", "predictive"],
            "nonpredictive_cache_root": str(args.nonpredictive_cache_root.resolve()),
            "predictive_cache_root": str(args.predictive_cache_root.resolve()),
            "train_targets": str(args.train_targets.resolve()),
            "validation_targets": str(args.validation_targets.resolve()),
            "seed": args.seed,
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
