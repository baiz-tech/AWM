#!/usr/bin/env python3
"""Train the native V-JEPA 2 predictor on current-16 -> future-16 Physion clips."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from external.vjepa2.app.vjepa.transforms import make_transforms
from src.data.physionpp.dataset import init_data
from src.core.backbone import build_model
from src.core.run_context import apply_cli_defaults, task_context


LOGGER = logging.getLogger("vjepa2_naive")


def init_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def cosine_lr(step, total_steps, start_lr, ref_lr, final_lr, warmup_fraction):
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if step < warmup_steps:
        return start_lr + (ref_lr - start_lr) * step / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return final_lr + 0.5 * (ref_lr - final_lr) * (1.0 + math.cos(math.pi * progress))


def append_metrics(path, row):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def prediction_loss(predicted, target, loss_type):
    if loss_type == "l1":
        return F.l1_loss(predicted.float(), target.float())
    if loss_type == "mse":
        return F.mse_loss(predicted.float(), target.float())
    raise ValueError(f"Unsupported loss.type={loss_type!r}; expected 'l1' or 'mse'")


def state_metrics(predicted, target):
    mse = F.mse_loss(predicted.float(), target.float())
    cosine = F.cosine_similarity(predicted.flatten(1).float(), target.flatten(1).float(), dim=1).mean()
    return mse, cosine


@torch.no_grad()
def evaluate(loader, model, device, dtype, use_amp, loss_type, max_batches=None):
    model.eval()
    loss_sum = mse_sum = cosine_sum = 0.0
    count = 0
    effective_max_batches = max_batches
    if max_batches is not None and dist.is_available() and dist.is_initialized():
        effective_max_batches = math.ceil(int(max_batches) / dist.get_world_size())
    for iteration, batch in enumerate(loader):
        if effective_max_batches is not None and iteration >= effective_max_batches:
            break
        current = batch["current"].to(device, non_blocking=True)
        future = batch["future"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=dtype if use_amp else None):
            predicted, target, _ = model(current, future)
        loss = prediction_loss(predicted, target, loss_type)
        mse, cosine = state_metrics(predicted, target)
        batch_size = current.size(0)
        loss_sum += float(loss.item()) * batch_size
        mse_sum += float(mse.item()) * batch_size
        cosine_sum += float(cosine.item()) * batch_size
        count += batch_size
    totals = torch.tensor([loss_sum, mse_sum, cosine_sum, count], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    total_count = max(float(totals[3].item()), 1.0)
    return {
        "loss": float(totals[0].item()) / total_count,
        "mse": float(totals[1].item()) / total_count,
        "cosine": float(totals[2].item()) / total_count,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    step,
    config,
    source_epoch,
    best_validation_loss=float("inf"),
    best_epoch=-1,
):
    module = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "predictor": module.predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "source_checkpoint_epoch": source_epoch,
            "best_validation_loss": float(best_validation_loss),
            "best_epoch": int(best_epoch),
            "protocol": "current_16_to_future_16",
            "config": config,
        },
        path.with_suffix(path.suffix + ".tmp"),
    )
    os.replace(path.with_suffix(path.suffix + ".tmp"), path)


def main():
    ctx = task_context()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-train-videos", type=int, default=None)
    parser.add_argument("--max-eval-videos", type=int, default=None)
    parser.add_argument("--sampling-mode", choices=("random", "prediction_start"), default=None)
    parser.add_argument("--current-frame-step", type=int, default=None)
    parser.add_argument("--future-frame-step", type=int, default=None)
    parser.add_argument("--clip-gap", type=int, default=None)
    apply_cli_defaults(parser, ctx)
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    meta = config.get("meta", {})
    cfg_model = config.get("model", {})
    cfg_data = config.get("data", {})
    cfg_aug = config.get("data_aug", {})
    cfg_opt = config.get("optimization", {})
    cfg_log = config.get("logging", {})
    cfg_loss = config.get("loss", {})
    loss_type = str(cfg_loss.get("type", "l1")).lower()
    cli_data_overrides = {
        "sampling_mode": args.sampling_mode,
        "current_frame_step": args.current_frame_step,
        "future_frame_step": args.future_frame_step,
        "clip_gap": args.clip_gap,
    }
    for key, value in cli_data_overrides.items():
        if value is not None:
            cfg_data[key] = value
    if loss_type not in ("l1", "mse"):
        raise ValueError(f"loss.type must be 'l1' or 'mse', got {loss_type!r}")

    rank, world_size, local_rank = init_distributed()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format="[%(levelname)-8s][%(asctime)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    seed = int(meta.get("seed", 239)) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    output_dir = Path(config.get("folder", "outputs/runs/vjepa2_naive/physion_vith_16to16"))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    train_transform = make_transforms(
        random_horizontal_flip=cfg_aug.get("horizontal_flip", True),
        random_resize_aspect_ratio=tuple(cfg_aug.get("random_resize_aspect_ratio", [0.9, 1.1])),
        random_resize_scale=tuple(cfg_aug.get("random_resize_scale", [0.8, 1.0])),
        reprob=float(cfg_aug.get("reprob", 0.0)),
        auto_augment=cfg_aug.get("auto_augment", False),
        motion_shift=cfg_aug.get("motion_shift", False),
        crop_size=int(cfg_data.get("crop_size", 256)),
    )
    eval_transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=int(cfg_data.get("crop_size", 256)),
    )

    common = dict(
        root=cfg_data.get("root"),
        video_glob=cfg_data.get("video_glob", "**/*_img.mp4"),
        clip_frames=int(cfg_data.get("clip_frames", 16)),
        sampling_mode=cfg_data.get("sampling_mode", "prediction_start"),
        current_frame_step=int(cfg_data.get("current_frame_step", 2)),
        future_frame_step=int(cfg_data.get("future_frame_step", 4)),
        clip_gap=int(cfg_data.get("clip_gap", 32)),
        num_workers=int(cfg_data.get("num_workers", 4)),
        pin_mem=cfg_data.get("pin_mem", True),
        persistent_workers=cfg_data.get("persistent_workers", True),
    )
    train_dataset, train_loader, train_sampler = init_data(
        **common,
        split=cfg_data.get("train_split", "data_v1"),
        data_paths=cfg_data.get("datasets"),
        batch_size=int(cfg_data.get("batch_size", 2)),
        transform=train_transform,
        rank=rank,
        world_size=world_size,
        deterministic=False,
        max_videos=args.max_train_videos or cfg_data.get("max_train_videos"),
    )
    _, eval_loader, _ = init_data(
        **common,
        split=cfg_data.get("eval_split", "readout_data_v1"),
        data_paths=cfg_data.get("eval_datasets"),
        batch_size=int(cfg_data.get("eval_batch_size", cfg_data.get("batch_size", 2))),
        transform=eval_transform,
        rank=rank,
        world_size=world_size,
        deterministic=True,
        drop_last=False,
        max_videos=args.max_eval_videos or cfg_data.get("max_eval_videos"),
    )

    model, source_epoch, load_message = build_model(
        device,
        cfg_model,
        cfg_data,
        checkpoint=meta["pretrain_checkpoint"],
        checkpoint_key=meta.get("encoder_checkpoint_key", "target_encoder"),
    )
    if rank == 0:
        LOGGER.info("protocol=current_16_to_future_16 train_videos=%d source_epoch=%d", len(train_dataset), source_epoch)
        LOGGER.info(
            "sampling_mode=%s current_frame_step=%d future_frame_step=%d clip_gap=%d",
            cfg_data.get("sampling_mode", "prediction_start"),
            int(cfg_data.get("current_frame_step", 2)),
            int(cfg_data.get("future_frame_step", 4)),
            int(cfg_data.get("clip_gap", 32)),
        )
        LOGGER.info(
            "encoder checkpoint missing=%d unexpected=%d loss_type=%s",
            len(load_message.missing_keys),
            len(load_message.unexpected_keys),
            loss_type,
        )
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    module = model.module if hasattr(model, "module") else model
    optimizer = torch.optim.AdamW(
        module.predictor.parameters(),
        lr=float(cfg_opt.get("lr", 3.0e-4)),
        betas=tuple(cfg_opt.get("betas", [0.9, 0.999])),
        eps=float(cfg_opt.get("eps", 1.0e-8)),
        weight_decay=float(cfg_opt.get("weight_decay", 0.04)),
    )
    epochs = int(cfg_opt.get("epochs", 20))
    iterations_per_epoch = int(cfg_opt.get("ipe") or len(train_loader))
    total_steps = max(1, epochs * iterations_per_epoch)
    dtype_name = str(meta.get("dtype", "bfloat16")).lower()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16 if dtype_name == "float16" else torch.float32
    use_amp = dtype_name in ("bfloat16", "float16") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if dtype_name == "float16" and torch.cuda.is_available() else None

    start_epoch = step = 0
    best_validation_loss = float("inf")
    best_epoch = -1
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu")
        module.predictor.load_state_dict(payload["predictor"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload.get("epoch", 0))
        step = int(payload.get("step", start_epoch * iterations_per_epoch))
        best_validation_loss = float(payload.get("best_validation_loss", float("inf")))
        best_epoch = int(payload.get("best_epoch", -1))
        LOGGER.info(
            "resume state: start_epoch=%d step=%d best_validation_loss=%.6f best_epoch=%d",
            start_epoch,
            step,
            best_validation_loss,
            best_epoch,
        )

    metrics_path = output_dir / "metrics.csv"
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        iterator = iter(train_loader)
        for iteration in range(iterations_per_epoch):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            lr = cosine_lr(
                step,
                total_steps,
                float(cfg_opt.get("start_lr", 5.0e-5)),
                float(cfg_opt.get("lr", 3.0e-4)),
                float(cfg_opt.get("final_lr", 1.0e-5)),
                float(cfg_opt.get("warmup", 0.05)),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            current = batch["current"].to(device, non_blocking=True)
            future = batch["future"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=dtype if use_amp else None):
                predicted, target, _ = model(current, future)
                loss = prediction_loss(predicted, target, loss_type)
                mse, cosine = state_metrics(predicted, target)
            optimizer.zero_grad(set_to_none=True)
            if scaler is None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(module.predictor.parameters(), float(cfg_opt.get("grad_clip", 1.0)))
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(module.predictor.parameters(), float(cfg_opt.get("grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()

            if rank == 0 and iteration % int(cfg_log.get("log_freq", 10)) == 0:
                row = {"split": "train", "epoch": epoch, "iteration": iteration, "step": step, "loss": float(loss.item()), "mse": float(mse.item()), "cosine": float(cosine.item()), "lr": lr}
                append_metrics(metrics_path, row)
                LOGGER.info("epoch=%d itr=%d loss=%.6f cosine=%.6f lr=%.3e", epoch, iteration, row["loss"], row["cosine"], lr)
            step += 1

        eval_metrics = evaluate(
            eval_loader,
            model,
            device,
            dtype,
            use_amp,
            loss_type,
            cfg_log.get("max_eval_batches"),
        )
        if rank == 0:
            append_metrics(metrics_path, {"split": "validation", "epoch": epoch, "iteration": -1, "step": step, **eval_metrics, "lr": optimizer.param_groups[0]["lr"]})
            LOGGER.info("epoch=%d validation=%s", epoch, eval_metrics)
            validation_loss = float(eval_metrics["loss"])
            improved = math.isfinite(validation_loss) and validation_loss < best_validation_loss
            if improved:
                best_validation_loss = validation_loss
                best_epoch = epoch + 1
                save_checkpoint(
                    output_dir / "best.pt", module, optimizer, scaler, epoch + 1, step,
                    config, source_epoch, best_validation_loss, best_epoch,
                )
                LOGGER.info(
                    "new best checkpoint: epoch=%d validation_loss=%.6f path=%s",
                    best_epoch,
                    best_validation_loss,
                    output_dir / "best.pt",
                )
            save_checkpoint(
                output_dir / "latest.pt", module, optimizer, scaler, epoch + 1, step,
                config, source_epoch, best_validation_loss, best_epoch,
            )
            if (epoch + 1) % int(meta.get("save_every_freq", 5)) == 0:
                save_checkpoint(
                    output_dir / f"epoch-{epoch + 1}.pt", module, optimizer, scaler,
                    epoch + 1, step, config, source_epoch,
                    best_validation_loss, best_epoch,
                )
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
