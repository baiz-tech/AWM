#!/usr/bin/env python3
"""Evaluate one-step Physion++ latent Future prediction on ``testdata_v1``."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pickle
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from decord import VideoReader, cpu

from external.vjepa2.app.vjepa.transforms import make_transforms
from src.training.distributed import ExactDistributedSampler
from src.data.physionpp.evaluate.world_model_adapter import load_world_model
from src.core.config_utils import training_experiment


LOGGER = logging.getLogger("shared.evaluate.physionpp.future_prediction")
METRIC_NAMES = (
    "mae",
    "mse",
    "rmse",
    "cosine_similarity",
    "cosine_distance",
    "relative_l2",
)


def experiment_config(raw_config):
    return training_experiment(raw_config)


def read_future_prediction_spec(
    video_path,
    *,
    clip_frames,
    current_frame_step,
    future_frame_step,
    clip_gap,
    video_length,
):
    """Return deterministic Current/Future indices anchored at prediction start."""
    path = str(video_path)
    if not path.endswith("_img.mp4"):
        return None
    try:
        with open(path[: -len("_img.mp4")] + ".pkl", "rb") as handle:
            metadata = pickle.load(handle)
        prediction_start = int(metadata["static"]["start_frame_for_prediction"])
    except (OSError, EOFError, KeyError, TypeError, ValueError, pickle.UnpicklingError):
        return None
    current_start = prediction_start - int(clip_frames) * int(current_frame_step)
    current_indices = current_start + np.arange(int(clip_frames)) * int(current_frame_step)
    future_indices = (
        prediction_start
        + int(clip_gap)
        + np.arange(int(clip_frames)) * int(future_frame_step)
    )
    if current_start < 0 or int(future_indices[-1]) >= int(video_length):
        return None
    return {
        "prediction_start": prediction_start,
        "current_indices": current_indices.astype(np.int64),
        "future_indices": future_indices.astype(np.int64),
    }


class PhysionFuturePredictionDataset(torch.utils.data.Dataset):
    """Decode aligned Current and true Future clips from one Physion++ split."""

    def __init__(
        self,
        *,
        root,
        split="testdata_v1",
        video_glob="**/*_img.mp4",
        clip_frames=16,
        current_frame_step=2,
        future_frame_step=4,
        clip_gap=0,
        transform=None,
        max_videos=None,
    ):
        self.clip_frames = int(clip_frames)
        self.current_frame_step = int(current_frame_step)
        self.future_frame_step = int(future_frame_step)
        self.clip_gap = int(clip_gap)
        self.transform = transform
        if min(self.clip_frames, self.current_frame_step, self.future_frame_step) <= 0:
            raise ValueError("clip_frames and frame steps must be positive")
        if self.clip_gap < 0:
            raise ValueError("clip_gap must be non-negative")
        split_root = Path(root) / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Physion split directory does not exist: {split_root}")
        paths = sorted(glob.glob(str(split_root / video_glob), recursive=True))
        if max_videos is not None:
            paths = paths[: int(max_videos)]
        self.specs = []
        for path in paths:
            try:
                reader = VideoReader(path, num_threads=1, ctx=cpu(0))
                video_length = len(reader)
            except Exception as exc:
                warnings.warn(f"discarding video that failed to open: {path}: {exc}")
                continue
            spec = read_future_prediction_spec(
                path,
                clip_frames=self.clip_frames,
                current_frame_step=self.current_frame_step,
                future_frame_step=self.future_frame_step,
                clip_gap=self.clip_gap,
                video_length=video_length,
            )
            if spec is not None:
                self.specs.append({"path": path, **spec})
        if not self.specs:
            raise RuntimeError(f"No valid future-prediction videos remain in split={split}")

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, index):
        spec = self.specs[index]
        decode_indices = np.concatenate(
            (spec["current_indices"], spec["future_indices"])
        ).astype(np.int64)
        try:
            reader = VideoReader(spec["path"], num_threads=-1, ctx=cpu(0))
            frames = reader.get_batch(decode_indices).asnumpy()
        except Exception as exc:
            raise RuntimeError(
                f"failed to decode deterministic future-prediction sample={spec['path']}: {exc}"
            ) from exc
        if self.transform is None:
            clips = torch.as_tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
        else:
            # One transform call guarantees identical spatial processing for both clips.
            clips = self.transform(frames)
        expected = 2 * self.clip_frames
        if clips.ndim != 4 or clips.size(1) != expected:
            raise ValueError(
                f"transform must return [C,{expected},H,W], got {tuple(clips.shape)}"
            )
        return {
            "current": clips[:, : self.clip_frames],
            "future": clips[:, self.clip_frames :],
            "current_indices": torch.as_tensor(spec["current_indices"], dtype=torch.long),
            "future_indices": torch.as_tensor(spec["future_indices"], dtype=torch.long),
            "path": spec["path"],
        }


def make_loader(config, transform, batch_size, workers, max_test_videos):
    data = config["data"]
    dataset = PhysionFuturePredictionDataset(
        root=data.get("root"),
        split="testdata_v1",
        video_glob=data.get("video_glob", "**/*_img.mp4"),
        clip_frames=int(data.get("clip_frames", 16)),
        current_frame_step=int(data.get("current_frame_step", 2)),
        future_frame_step=int(data.get("future_frame_step", 4)),
        clip_gap=int(data.get("clip_gap", 0)),
        transform=transform,
        max_videos=max_test_videos,
    )
    sampler = None
    if dist.is_initialized():
        sampler = ExactDistributedSampler(
            dataset, rank=dist.get_rank(), world_size=dist.get_world_size()
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=data.get("pin_mem", True),
        persistent_workers=workers > 0,
        drop_last=False,
    )


def prediction_triplet(batch, model, device, dtype, use_amp):
    """Return Current context, predicted Future, and true teacher Future."""
    current = batch["current"].to(device, non_blocking=True)
    future = batch["future"].to(device, non_blocking=True)
    with torch.amp.autocast(
        device_type=device.type,
        enabled=use_amp,
        dtype=dtype if use_amp else None,
    ):
        context = model.encode_current(current)
        predicted = model.predict_next(context)
        target = model.encode_target(current, future)
    return context, predicted, target


def metric_rows(source, target):
    """Compute one metric value per video over all latent tokens and channels."""
    if source.shape != target.shape or source.ndim < 2:
        raise ValueError(
            f"source/target latent shapes must match and include a batch: {source.shape} vs "
            f"{target.shape}"
        )
    source = source.float().flatten(1)
    target = target.float().flatten(1)
    difference = source - target
    mse = difference.square().mean(dim=1)
    cosine = F.cosine_similarity(source, target, dim=1)
    return {
        "mae": difference.abs().mean(dim=1).cpu(),
        "mse": mse.cpu(),
        "rmse": mse.sqrt().cpu(),
        "cosine_similarity": cosine.cpu(),
        "cosine_distance": (1.0 - cosine).cpu(),
        "relative_l2": (
            torch.linalg.vector_norm(difference, dim=1)
            / torch.linalg.vector_norm(target, dim=1).clamp_min(torch.finfo(target.dtype).eps)
        ).cpu(),
    }


@torch.inference_mode()
def extract_local_metrics(loader, model, device, dtype, use_amp, log_every_batches):
    values = {
        source: {metric: [] for metric in METRIC_NAMES}
        for source in ("prediction", "current_copy")
    }
    paths = []
    model.eval()
    rank = dist.get_rank() if dist.is_initialized() else 0
    started = time.monotonic()
    total_batches = len(loader)
    for iteration, batch in enumerate(loader, start=1):
        context, predicted, target = prediction_triplet(
            batch, model, device, dtype, use_amp
        )
        if predicted.shape != target.shape:
            raise ValueError(
                f"predicted/target latent shapes differ: {predicted.shape} vs {target.shape}"
            )
        if context.shape != target.shape:
            raise ValueError(
                "current-copy baseline requires matching context/target shapes: "
                f"{context.shape} vs {target.shape}"
            )
        for source_name, source in (("prediction", predicted), ("current_copy", context)):
            rows = metric_rows(source, target)
            for metric in METRIC_NAMES:
                values[source_name][metric].append(rows[metric])
        paths.extend(batch["path"])
        if iteration == 1 or iteration % log_every_batches == 0 or iteration == total_batches:
            LOGGER.info(
                "future prediction progress: rank=%d batch=%d/%d samples=%d elapsed=%.1fs",
                rank,
                iteration,
                total_batches,
                len(paths),
                time.monotonic() - started,
            )
    if not paths:
        raise RuntimeError(f"rank {rank} has no future-prediction samples")
    return {
        "paths": paths,
        **{
            source: {
                metric: torch.cat(values[source][metric]).numpy()
                for metric in METRIC_NAMES
            }
            for source in values
        },
    }


def merge_metric_payloads(payloads):
    paths = [path for payload in payloads for path in payload["paths"]]
    if len(set(paths)) != len(paths):
        raise RuntimeError("distributed future-prediction sharding produced duplicate paths")
    order = np.argsort(np.asarray(paths, dtype=object))
    merged = {"paths": np.asarray(paths, dtype=object)[order]}
    for source in ("prediction", "current_copy"):
        merged[source] = {}
        for metric in METRIC_NAMES:
            values = np.concatenate([payload[source][metric] for payload in payloads])
            if values.shape != (len(paths),):
                raise RuntimeError("distributed future-prediction metric sizes do not match")
            merged[source][metric] = values[order]
    return merged


def gather_metrics(local_result):
    if not dist.is_initialized():
        return merge_metric_payloads([local_result])
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_result)
    return merge_metric_payloads(gathered) if dist.get_rank() == 0 else None


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("metric values must be a finite non-empty vector")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def write_results(output_dir, store, args, config, checkpoint_metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        source: {metric: summarize(store[source][metric]) for metric in METRIC_NAMES}
        for source in ("prediction", "current_copy")
    }
    data = config["data"]
    result = {
        "protocol": "physionpp_test_single_future_latent_prediction",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_epoch": int(checkpoint_metadata.get("checkpoint_epoch", -1)),
        "split": "testdata_v1",
        "num_test": int(len(store["paths"])),
        "distributed_world_size": int(dist.get_world_size() if dist.is_initialized() else 1),
        "seed": int(args.seed),
        "clip_frames": int(data.get("clip_frames", 16)),
        "current_frame_step": int(data.get("current_frame_step", 2)),
        "future_frame_step": int(data.get("future_frame_step", 4)),
        "clip_gap": int(data.get("clip_gap", 0)),
        "future_rgb_role": "teacher target only; never consumed by the prediction branch",
        "aggregation": "macro over videos; each video metric first reduces all latent tokens/channels",
        "metrics": summaries,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_video_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for index, path in enumerate(store["paths"]):
            row = {"path": str(path)}
            for source in ("prediction", "current_copy"):
                row[source] = {
                    metric: float(store[source][metric][index]) for metric in METRIC_NAMES
                }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--world-model-adapter", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-test-videos", type=int, default=None)
    parser.add_argument("--progress-every-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=239)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.progress_every_batches <= 0:
        parser.error("batch-size/progress interval must be positive and num-workers non-negative")
    return args


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 8:
        raise RuntimeError(f"future_prediction requires exactly 8 GPUs, got WORLD_SIZE={world_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("future_prediction requires CUDA/NCCL")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank


def _validate_checkpoint(checkpoint):
    error = [None]
    if dist.get_rank() == 0:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            error[0] = f"future_prediction checkpoint does not exist: {path}"
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise FileNotFoundError(error[0])


def main():
    local_rank = _init_distributed()
    try:
        args = parse_args()
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)-8s][%(asctime)s][%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _validate_checkpoint(args.checkpoint)
        rank = dist.get_rank()
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        with open(args.config, encoding="utf-8") as handle:
            config = experiment_config(yaml.safe_load(handle))
        device = torch.device(f"cuda:{local_rank}")
        dtype_name = str(config.get("meta", {}).get("dtype", "bfloat16")).lower()
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype_name)
        if dtype is None:
            raise ValueError(f"unsupported evaluation dtype={dtype_name!r}")
        model, checkpoint_metadata = load_world_model(
            config,
            args.checkpoint,
            device,
            adapter_name=args.world_model_adapter,
        )
        transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
            crop_size=int(config["data"].get("crop_size", 256)),
        )
        loader = make_loader(
            config, transform, args.batch_size, args.num_workers, args.max_test_videos
        )
        store = gather_metrics(
            extract_local_metrics(
                loader,
                model,
                device,
                dtype,
                dtype_name in ("bfloat16", "float16"),
                args.progress_every_batches,
            )
        )
        if rank == 0:
            result = write_results(
                args.output_dir, store, args, config, checkpoint_metadata
            )
            print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
