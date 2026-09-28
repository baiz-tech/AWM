#!/usr/bin/env python3
"""Train an ALOE-style CLEVRER QA model on exported V-JEPA2 trajectories."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

from src.training.distributed import ExactDistributedSampler
from src.core.config_utils import merged_task_experiment
from src.data.clevrer.v1.qa_dataset import (
    ClevrerAloeTrajectoryDataset,
    WordTokenizer,
    build_answer_vocabulary,
    clevrer_aloe_collate_fn,
    load_clevrer_annotations,
    normalize_question_types,
)
from src.data.clevrer.v1.qa_model import build_qa_model, qa_loss
from src.data.clevrer.v1.qa_utils import (
    choose_threshold,
    compute_metrics,
    gather_records,
    records_from_outputs,
)
from src.data.clevrer.v1.protocol import resolve_qa_protocol


LOGGER = logging.getLogger("shared.evaluate.clevrer.train_qa")


def experiment_config(raw_config):
    return merged_task_experiment(raw_config, "qa_eval")


def init_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    tensor_keys = (
        "scene_index",
        "question_id",
        "q_type",
        "q_subtype",
        "cls_video_emb",
        "cls_q_tokens",
        "cls_label",
        "mc_video_emb",
        "mc_subtype",
        "mc_q_tokens",
        "mc_label",
        "mc_flag",
        "mc_choice_id",
    )
    for key in tensor_keys:
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


@torch.no_grad()
def evaluate(model, loader, device, use_amp, dtype, threshold=0.5):
    model.eval()
    records = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp, dtype=dtype if use_amp else None
        ):
            outputs = model(batch)
        records.extend(records_from_outputs(outputs, batch, threshold=threshold))
    return gather_records(records)


def save_checkpoint(path, model, optimizer, scheduler, epoch, metric, threshold, tokenizer, answers, cfg_qa, visual_dim):
    backend = str(cfg_qa.get("backend", "aloe_like_legacy"))
    payload = {
        "epoch": int(epoch),
        "metric": float(metric),
        "mc_threshold": float(threshold),
        "model": model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "tokenizer": tokenizer.state_dict(),
        "answer_vocabulary": list(answers),
        "qa_config": dict(cfg_qa),
        "visual_dim": int(visual_dim),
        "protocol": f"clevrer_{backend}_qa_v1",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trajectory-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-scenes", type=int, default=None)
    parser.add_argument("--max-val-scenes", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = experiment_config(yaml.safe_load(handle))
    cfg_qa = config.get("qa", {})
    resolve_qa_protocol(config)
    cfg_outputs = config.get("outputs", {})
    rank, world_size, local_rank = init_distributed()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format="[%(levelname)-8s][%(asctime)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    seed = int(config.get("meta", {}).get("seed", 239))
    seed_everything(seed + rank)

    run_dir = Path(config["folder"])
    output_dir = Path(args.output_dir or cfg_outputs.get("qa_checkpoint_root", run_dir / "evaluations" / "qa_aloe"))
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_root = Path(
        args.trajectory_root
        or cfg_outputs.get("qa_trajectory_root")
        or cfg_qa.get("trajectory_root", run_dir / "qa" / "trajectories")
    )
    annotation_root = Path(cfg_qa.get("annotation_root", "/data/shared/datasets/CLEVRER/official_code/executor/data"))
    question_types = normalize_question_types(cfg_qa.get("question_types"))

    train_scenes = load_clevrer_annotations(
        annotation_root,
        "train",
        max_scenes=args.max_train_scenes,
        question_types=question_types,
    )
    tokenizer = WordTokenizer.build(
        train_scenes,
        min_frequency=int(cfg_qa.get("min_token_frequency", 1)),
        max_question_len=int(cfg_qa.get("max_question_len", 20)),
        max_choice_len=int(cfg_qa.get("max_choice_len", 12)),
    )
    answers = build_answer_vocabulary(train_scenes)
    dataset_kwargs = {
        "max_visual_tokens": int(cfg_qa.get("max_visual_tokens", cfg_qa.get("max_n_objects", 7))),
        "n_sample_frames": int(cfg_qa.get("n_sample_frames", 25)),
        "video_len": int(cfg_qa.get("video_len", 128)),
        "protocol": str(cfg_qa.get("protocol", "compatibility_stride2")),
        "frame_offset": int(
            (cfg_qa.get("temporal") or {}).get(
                "frame_offset", cfg_qa.get("video_len", 128) // cfg_qa.get("n_sample_frames", 25)
            )
        ),
        "predictive_window": bool(
            (cfg_qa.get("temporal") or {}).get("predictive_window", True)
        ),
    }
    train_dataset = ClevrerAloeTrajectoryDataset(
        annotation_root,
        trajectory_root,
        "train",
        tokenizer,
        answers,
        max_scenes=args.max_train_scenes,
        random_start=True,
        question_types=question_types,
        **dataset_kwargs,
    )
    val_dataset = ClevrerAloeTrajectoryDataset(
        annotation_root,
        trajectory_root,
        "validation",
        tokenizer,
        answers,
        max_scenes=args.max_val_scenes,
        random_start=False,
        question_types=question_types,
        **dataset_kwargs,
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = ExactDistributedSampler(val_dataset, rank, world_size)
    loader_kwargs = {
        "batch_size": int(cfg_qa.get("train_batch_size", cfg_qa.get("scene_batch_size", 96))),
        "num_workers": int(cfg_qa.get("num_workers", 4)),
        "pin_memory": True,
        "persistent_workers": int(cfg_qa.get("num_workers", 4)) > 0,
        "collate_fn": clevrer_aloe_collate_fn,
    }
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, drop_last=False, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        drop_last=False,
        batch_size=int(cfg_qa.get("val_batch_size", loader_kwargs["batch_size"] * 2)),
        num_workers=loader_kwargs["num_workers"],
        pin_memory=True,
        persistent_workers=loader_kwargs["persistent_workers"],
        collate_fn=clevrer_aloe_collate_fn,
    )
    first_trajectory = train_dataset[0]["video_emb"]
    visual_dim = int(first_trajectory.size(-1))
    model = build_qa_model(cfg_qa, visual_dim, tokenizer, answers).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_qa.get("learning_rate", 1e-3)),
        weight_decay=float(cfg_qa.get("weight_decay", 0.0)),
    )
    epochs = int(args.epochs or cfg_qa.get("epochs", 400))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    dtype_name = str(config.get("meta", {}).get("dtype", "bfloat16")).lower()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    use_amp = dtype_name in ("bfloat16", "float16") and device.type == "cuda"
    best_metric = -1.0
    best_threshold = 0.5
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        loss_sum = torch.zeros(2, dtype=torch.float64, device=device)
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=use_amp, dtype=dtype if use_amp else None
            ):
                outputs = model(batch)
                loss, _ = qa_loss(outputs, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg_qa.get("grad_clip", 1.0)))
            optimizer.step()
            loss_sum[0] += loss.detach().double()
            loss_sum[1] += 1
        scheduler.step()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        validation_records_raw = evaluate(model, val_loader, device, use_amp, dtype, threshold=0.5)
        threshold = choose_threshold(validation_records_raw)
        validation_records = [
            {**record, "prediction": record["prediction"] if record["task"] == 0 else int(record["probability"] >= threshold)}
            for record in validation_records_raw
        ]
        validation_metrics = compute_metrics(validation_records)
        metric = validation_metrics["overall_question_accuracy"] or 0.0
        if rank == 0:
            LOGGER.info(
                "epoch=%d train_loss=%.6f val_metric=%.6f threshold=%.2f metrics=%s",
                epoch + 1,
                (loss_sum[0] / loss_sum[1].clamp_min(1)).item(),
                metric,
                threshold,
                validation_metrics,
            )
            save_checkpoint(
                output_dir / "latest.pt", model, optimizer, scheduler, epoch + 1,
                metric, threshold, tokenizer, answers, cfg_qa, visual_dim,
            )
            if metric > best_metric:
                best_metric, best_threshold = metric, threshold
                save_checkpoint(
                    output_dir / "best.pt", model, optimizer, scheduler, epoch + 1,
                    metric, threshold, tokenizer, answers, cfg_qa, visual_dim,
                )
                (output_dir / "best_metrics.json").write_text(
                    json.dumps(validation_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
        if dist.is_available() and dist.is_initialized():
            payload = [best_metric, best_threshold]
            dist.broadcast_object_list(payload, src=0)
            best_metric, best_threshold = payload
    if rank == 0:
        manifest = {
            "protocol": f"clevrer_{cfg_qa.get('backend', 'aloe_like_legacy')}_qa_v1",
            "trajectory_root": str(trajectory_root.resolve()),
            "train_split": "train",
            "validation_split": "validation",
            "question_types": question_types,
            "answer_vocabulary": answers,
            "tokenizer": tokenizer.state_dict(),
            "best_metric": best_metric,
            "best_threshold": best_threshold,
            "world_size": world_size,
            "config": str(Path(args.config).resolve()),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
