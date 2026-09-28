#!/usr/bin/env python3
"""Retrieve the true future window among same-video temporal candidates."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset, Subset

from external.vjepa2.app.vjepa.transforms import make_transforms
from src.core.config_utils import training_experiment
from src.data.physionpp.evaluate.world_model_adapter import load_world_model


def experiment_config(raw_config):
    return training_experiment(raw_config)


def _candidate_offsets(offset_min, offset_max, offset_step, correct_offset):
    if offset_step <= 0 or offset_min > offset_max:
        raise ValueError("offset_step must be positive and offset_min <= offset_max")
    offsets = list(range(int(offset_min), int(offset_max) + 1, int(offset_step)))
    if len(offsets) < 6:
        raise ValueError("at least six temporal candidate offsets are required")
    if int(correct_offset) not in offsets:
        raise ValueError("correct_offset must appear in the candidate offset grid")
    return offsets


def _temporal_pool(tokens, num_time_steps):
    if tokens.ndim != 3 or tokens.size(1) % int(num_time_steps):
        raise ValueError("tokens must be [B,N,D] with N divisible by num_time_steps")
    return tokens.float().reshape(
        tokens.size(0), int(num_time_steps), -1, tokens.size(-1)
    ).mean(dim=2)


def _rank(scores, correct_index):
    order = torch.argsort(scores, descending=True, stable=True)
    rank = int((order == int(correct_index)).nonzero(as_tuple=False)[0, 0]) + 1
    return rank, int(order[0])


def _gather(local, rank, world_size):
    if world_size == 1:
        gathered = [local]
    else:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local, gathered, dst=0)
        if rank != 0:
            return None
    merged = {key: [] for key in local}
    for shard in gathered:
        for key, values in shard.items():
            merged[key].extend(values)
    order = np.argsort(np.asarray(merged["paths"], dtype=object))
    for key, values in merged.items():
        dtype = object if key == "paths" else None
        merged[key] = np.asarray(values, dtype=dtype)[order]
    return merged


def _retrieval_metrics(ranks, top_offsets, correct_offset, candidate_count):
    ranks = np.asarray(ranks, dtype=np.int64)
    top_offsets = np.asarray(top_offsets, dtype=np.int64)
    if ranks.size == 0 or ranks.shape != top_offsets.shape:
        raise ValueError("retrieval ranks and offsets must be non-empty with matching shapes")
    errors = np.abs(top_offsets - int(correct_offset))
    return {
        "num_queries": int(ranks.size),
        "candidate_count": int(candidate_count),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "top1_offset_mae_frames": float(np.mean(errors)),
        "top1_offset_median_ae_frames": float(np.median(errors)),
        "top1_offset_mean_frames": float(np.mean(top_offsets)),
        "top1_offset_median_frames": float(np.median(top_offsets)),
        "top1_before_correct_rate": float(np.mean(top_offsets < int(correct_offset))),
        "top1_after_correct_rate": float(np.mean(top_offsets > int(correct_offset))),
        "top1_within_8_frames": float(np.mean(errors <= 8)),
        "top1_within_16_frames": float(np.mean(errors <= 16)),
        "chance_recall_at_1": 1.0 / candidate_count,
        "chance_recall_at_5": min(5.0 / candidate_count, 1.0),
        "chance_mrr": sum(1.0 / rank for rank in range(1, candidate_count + 1)) / candidate_count,
    }


def _bootstrap(model_ranks, current_ranks, model_offsets, current_offsets, args):
    rng = np.random.default_rng(239)
    model_ranks = np.asarray(model_ranks, dtype=np.int64)
    current_ranks = np.asarray(current_ranks, dtype=np.int64)
    model_errors = np.abs(np.asarray(model_offsets, dtype=np.int64) - args.correct_offset)
    current_errors = np.abs(np.asarray(current_offsets, dtype=np.int64) - args.correct_offset)
    values = {name: [] for name in ("recall_at_1", "recall_at_5", "mrr", "offset_mae")}
    for _ in range(int(args.bootstrap_samples)):
        indices = rng.integers(0, model_ranks.size, model_ranks.size)
        model_sample, current_sample = model_ranks[indices], current_ranks[indices]
        values["recall_at_1"].append(np.mean(model_sample <= 1) - np.mean(current_sample <= 1))
        values["recall_at_5"].append(np.mean(model_sample <= 5) - np.mean(current_sample <= 5))
        values["mrr"].append(np.mean(1.0 / model_sample) - np.mean(1.0 / current_sample))
        values["offset_mae"].append(
            np.mean(model_errors[indices]) - np.mean(current_errors[indices])
        )
    return {
        key: {
            "delta": float(np.mean(samples)),
            "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        }
        for key, samples in values.items()
    }


def _bootstrap_random(model_ranks, model_offsets, offsets, args):
    rng = np.random.default_rng(239)
    model_ranks = np.asarray(model_ranks, dtype=np.int64)
    model_errors = np.abs(np.asarray(model_offsets, dtype=np.int64) - args.correct_offset)
    random_errors = np.abs(np.asarray(offsets, dtype=np.int64) - args.correct_offset)
    chance = {
        "recall_at_1": 1.0 / len(offsets),
        "recall_at_5": min(5.0 / len(offsets), 1.0),
        "mrr": sum(1.0 / rank for rank in range(1, len(offsets) + 1)) / len(offsets),
        "within_8": float(np.mean(random_errors <= 8)),
        "offset_mae": float(np.mean(random_errors)),
    }
    values = {key: [] for key in chance}
    for _ in range(int(args.bootstrap_samples)):
        indices = rng.integers(0, model_ranks.size, model_ranks.size)
        ranks, errors = model_ranks[indices], model_errors[indices]
        values["recall_at_1"].append(np.mean(ranks <= 1) - chance["recall_at_1"])
        values["recall_at_5"].append(np.mean(ranks <= 5) - chance["recall_at_5"])
        values["mrr"].append(np.mean(1.0 / ranks) - chance["mrr"])
        values["within_8"].append(np.mean(errors <= 8) - chance["within_8"])
        values["offset_mae"].append(np.mean(errors) - chance["offset_mae"])
    return {
        key: {
            "delta": float(np.mean(samples)),
            "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        }
        for key, samples in values.items()
    }


def _evaluate(store, representation, args):
    model_ranks = store[f"model_{representation}_rank"].astype(np.int64)
    current_ranks = store[f"current_{representation}_rank"].astype(np.int64)
    model_offsets = store[f"model_{representation}_top_offset"].astype(np.int64)
    current_offsets = store[f"current_{representation}_top_offset"].astype(np.int64)
    offsets = _candidate_offsets(
        args.offset_min, args.offset_max, args.offset_step, args.correct_offset
    )
    random_errors = np.abs(np.asarray(offsets, dtype=np.int64) - args.correct_offset)
    return {
        "model": _retrieval_metrics(
            model_ranks,
            model_offsets,
            args.correct_offset,
            len(offsets),
        ),
        "current_copy": _retrieval_metrics(
            current_ranks,
            current_offsets,
            args.correct_offset,
            len(offsets),
        ),
        "random_chance": {
            "recall_at_1": 1.0 / len(offsets),
            "recall_at_5": min(5.0 / len(offsets), 1.0),
            "mrr": sum(1.0 / rank for rank in range(1, len(offsets) + 1)) / len(offsets),
            "top1_offset_mae_frames": float(np.mean(random_errors)),
            "top1_offset_median_ae_frames": float(np.median(random_errors)),
            "top1_within_8_frames": float(np.mean(random_errors <= 8)),
            "top1_within_16_frames": float(np.mean(random_errors <= 16)),
        },
        "model_minus_current_bootstrap": _bootstrap(
            model_ranks, current_ranks, model_offsets, current_offsets, args
        ),
        "model_minus_random_bootstrap": _bootstrap_random(
            model_ranks, model_offsets, offsets, args
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--world-model-adapter",
        default=None,
        help="Recipe-local adapter override for legacy saved configs",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--offset-min", type=int, default=-60)
    parser.add_argument("--offset-max", type=int, default=64)
    parser.add_argument("--offset-step", type=int, default=4)
    parser.add_argument("--correct-offset", type=int, default=None,
                        help="True future offset from prediction_start; defaults to data.clip_gap")
    parser.add_argument("--num-time-steps", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--max-test-videos", type=int, default=None)
    parser.add_argument("--distributed", action="store_true")
    return parser.parse_args()


class SameVideoCandidates(Dataset):
    def __init__(self, cfg, transform, offsets, max_videos=None):
        data = cfg["data"]
        samples = []
        manifests = data.get("test_datasets") or []
        if isinstance(manifests, str):
            manifests = [manifests]
        for manifest in manifests:
            with open(manifest, encoding="utf-8") as handle:
                samples.extend(line.split()[0] for line in handle if line.split())
        root = data.get("root")
        if root:
            split_root = Path(root) / "testdata_v1"
            if not split_root.is_dir():
                raise FileNotFoundError(f"Physion split directory does not exist: {split_root}")
            samples.extend(
                glob.glob(str(split_root / data.get("video_glob", "**/*_img.mp4")), recursive=True)
            )
        samples = sorted(dict.fromkeys(samples))
        if not samples:
            raise FileNotFoundError("No Physion++ test videos found")
        self.num_index_records = len(samples)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.clip_frames = int(data.get("clip_frames", 16))
        self.current_step = int(data.get("current_frame_step", 2))
        self.future_step = int(data.get("future_frame_step", 4))
        self.transform = transform
        records = []
        for path in samples:
            anchor = self._read_prediction_start(path)
            if anchor is None:
                continue
            try:
                reader = VideoReader(path, num_threads=1, ctx=cpu(0))
            except Exception as exc:
                warnings.warn(f"discarding video that failed to open: {path}: {exc}")
                continue
            current_start = anchor - self.clip_frames * self.current_step
            last_candidate = anchor + int(self.offsets.max()) + (self.clip_frames - 1) * self.future_step
            if current_start >= 0 and anchor + int(self.offsets.min()) >= 0 and last_candidate < len(reader):
                records.append((path, anchor))
        self.records = records[: int(max_videos)] if max_videos is not None else records
        if not self.records:
            raise FileNotFoundError("No test videos support the requested candidate offset range")

    @staticmethod
    def _read_prediction_start(video_path):
        suffix = "_img.mp4"
        if not str(video_path).endswith(suffix):
            warnings.warn(f"Physion++ video does not end in {suffix!r}: {video_path}")
            return None
        metadata_path = str(video_path)[:-len(suffix)] + ".pkl"
        try:
            with open(metadata_path, "rb") as handle:
                metadata = pickle.load(handle)
            return int(metadata["static"]["start_frame_for_prediction"])
        except (OSError, EOFError, KeyError, TypeError, ValueError, pickle.UnpicklingError) as exc:
            warnings.warn(f"failed to read prediction start from {metadata_path}: {exc}")
            return None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, anchor = self.records[index]
        current = anchor - self.clip_frames * self.current_step + np.arange(self.clip_frames) * self.current_step
        candidates = np.stack([
            anchor + offset + np.arange(self.clip_frames) * self.future_step for offset in self.offsets
        ]).astype(np.int64)
        unique = np.unique(np.concatenate([current, candidates.reshape(-1)]))
        frames = VideoReader(path, num_threads=-1, ctx=cpu(0)).get_batch(unique).asnumpy()
        frames = self.transform(frames)
        lookup = {int(value): position for position, value in enumerate(unique)}
        return {
            "frames": frames,
            "current_positions": torch.tensor([lookup[int(x)] for x in current]),
            "candidate_positions": torch.tensor([[lookup[int(x)] for x in row] for row in candidates]),
            "path": path,
        }


@torch.no_grad()
def extract(loader, model, device, use_amp, dtype, args, offsets):
    correct_index = offsets.index(args.correct_offset)
    store = {name: [] for name in (
        "model_temporal_rank", "current_temporal_rank", "model_global_rank", "current_global_rank",
        "model_temporal_top_offset", "current_temporal_top_offset",
        "model_global_top_offset", "current_global_top_offset", "paths",
    )}
    for batch in loader:
        frames = batch["frames"][0].to(device, non_blocking=True)
        current = frames.index_select(1, batch["current_positions"][0].to(device)).unsqueeze(0)
        candidate_positions = batch["candidate_positions"][0].to(device)
        candidate_temporal, candidate_global = [], []
        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp, dtype=dtype if use_amp else None
        ):
            context = model.encode_current(current)
            predicted = model.predict_next(context)
        for start in range(0, len(offsets), args.candidate_batch_size):
            positions = candidate_positions[start:start + args.candidate_batch_size]
            futures = torch.stack([frames.index_select(1, item) for item in positions])
            currents = current.expand(futures.size(0), -1, -1, -1, -1)
            with torch.amp.autocast(
                device_type=device.type, enabled=use_amp, dtype=dtype if use_amp else None
            ):
                targets = model.encode_target(currents, futures)
            candidate_temporal.append(_temporal_pool(targets, args.num_time_steps).cpu())
            candidate_global.append(targets.float().mean(1).cpu())
        candidate_temporal = F.normalize(torch.cat(candidate_temporal).flatten(1), dim=1)
        candidate_global = F.normalize(torch.cat(candidate_global), dim=1)
        queries = {
            "temporal": (
                F.normalize(_temporal_pool(predicted, args.num_time_steps).cpu().flatten(1), dim=1)[0],
                F.normalize(_temporal_pool(context, args.num_time_steps).cpu().flatten(1), dim=1)[0],
                candidate_temporal,
            ),
            "global": (
                F.normalize(predicted.float().mean(1).cpu(), dim=1)[0],
                F.normalize(context.float().mean(1).cpu(), dim=1)[0], candidate_global,
            ),
        }
        for representation, (model_query, current_query, candidates) in queries.items():
            model_rank, model_top = _rank(candidates @ model_query, correct_index)
            current_rank, current_top = _rank(candidates @ current_query, correct_index)
            store[f"model_{representation}_rank"].append(model_rank)
            store[f"current_{representation}_rank"].append(current_rank)
            store[f"model_{representation}_top_offset"].append(offsets[model_top])
            store[f"current_{representation}_top_offset"].append(offsets[current_top])
        store["paths"].append(batch["path"][0])
    return store


def write_summary(path, report):
    result = report["representations"]["temporal"]
    lines = [
        "# Physion++ Same-Video Temporal Retrieval", "",
        f"Candidates: {len(report['candidate_offsets'])}; offsets: {report['candidate_offsets'][0]}..{report['candidate_offsets'][-1]}; correct offset: {report['correct_offset']}.", "",
        "| Query | Recall@1 | Recall@5 | MRR | Median rank | Offset MAE |", "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("model", "Predicted future"), ("current_copy", "Current-copy baseline")):
        item = result[key]
        lines.append(f"| {label} | {item['recall_at_1']:.4f} | {item['recall_at_5']:.4f} | {item['mrr']:.4f} | {item['median_rank']:.1f} | {item['top1_offset_mae_frames']:.2f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-8s][%(asctime)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    with open(args.config, encoding="utf-8") as handle:
        cfg = experiment_config(yaml.safe_load(handle))
    if args.correct_offset is None:
        args.correct_offset = int(cfg["data"].get("clip_gap", 32))
    offsets = _candidate_offsets(args.offset_min, args.offset_max, args.offset_step, args.correct_offset)
    rank, world_size, local_rank = 0, 1, 0
    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed evaluation requires CUDA/NCCL")
        local_rank = int(os.environ.get("LOCAL_RANK", "0")); torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl"); rank, world_size = dist.get_rank(), dist.get_world_size()
    seed = int(cfg.get("meta", {}).get("seed", 239))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(f"cuda:{local_rank}" if args.distributed else args.device if torch.cuda.is_available() else "cpu")
    dtype_name = str(cfg.get("meta", {}).get("dtype", "bfloat16")).lower()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    use_amp = device.type == "cuda" and dtype_name in ("bfloat16", "float16")
    model, checkpoint_metadata = load_world_model(
        cfg, args.checkpoint, device, adapter_name=args.world_model_adapter
    )
    transform = make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0), random_resize_scale=(1.0, 1.0), reprob=0.0, auto_augment=False, motion_shift=False, crop_size=int(cfg["data"].get("crop_size", 256)))
    dataset = SameVideoCandidates(cfg, transform, offsets, args.max_test_videos)
    shard = Subset(dataset, range(rank, len(dataset), world_size)) if world_size > 1 else dataset
    loader = DataLoader(shard, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    merged = _gather(extract(loader, model, device, use_amp, dtype, args, offsets), rank, world_size)
    if rank == 0:
        report = {"model_label": checkpoint_metadata.get("model_label", "World-model predicted future"), "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_epoch": int(checkpoint_metadata.get("checkpoint_epoch", -1)), "num_index_records": dataset.num_index_records, "num_test": len(merged["paths"]), "candidate_offsets": offsets, "correct_offset": args.correct_offset, "representations": {name: _evaluate(merged, name, args) for name in ("temporal", "global")}}
        output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(output / "ranks_and_offsets.npz", **merged)
        write_summary(output / "summary.md", report)
        print(json.dumps(report["representations"]["temporal"], indent=2))
    if args.distributed:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
