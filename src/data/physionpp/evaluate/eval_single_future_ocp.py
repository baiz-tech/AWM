#!/usr/bin/env python3
"""Evaluate contact during exactly one predicted Physion++ Future clip."""

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
import yaml
from decord import VideoReader, cpu

from external.vjepa2.app.vjepa.transforms import make_transforms
from src.training.distributed import ExactDistributedSampler
from src.data.physionpp.evaluate.diagnostics import (
    select_single_future_probe_features,
    stratified_probe_split,
)
from src.data.physionpp.evaluate.eval_ocp import (
    auroc,
    best_threshold,
    binary_metrics,
    train_readout,
)
from src.data.physionpp.evaluate.world_model_adapter import load_world_model
from src.core.config_utils import (
    task_experiment,
    task_launch,
    training_experiment,
)


LOGGER = logging.getLogger("shared.evaluate.physionpp.single_future_ocp")
PROBE_VARIANTS = ("full", "current", "predicted_future", "delta")
LABEL_SCOPES = ("single_clip", "all_future")


def experiment_config(raw_config):
    return training_experiment(raw_config)


def _frame_at(frames, index):
    for key in (index, str(index), f"{index:04d}"):
        if key in frames:
            return frames[key]
    return None


def read_single_future_spec(
    video_path,
    *,
    clip_frames,
    current_frame_step,
    future_frame_step,
    clip_gap,
    video_length,
    label_scope="single_clip",
):
    """Return one-clip indices and contact over the configured label interval."""
    path = str(video_path)
    if not path.endswith("_img.mp4"):
        return None
    if label_scope not in LABEL_SCOPES:
        raise ValueError(f"unknown single-Future label scope: {label_scope!r}")
    try:
        with open(path[: -len("_img.mp4")] + ".pkl", "rb") as handle:
            metadata = pickle.load(handle)
        prediction_start = int(metadata["static"]["start_frame_for_prediction"])
        current_start = prediction_start - int(clip_frames) * int(current_frame_step)
        current_indices = current_start + np.arange(int(clip_frames)) * int(
            current_frame_step
        )
        future_start = prediction_start + int(clip_gap)
        future_indices = future_start + np.arange(int(clip_frames)) * int(
            future_frame_step
        )
        if current_start < 0 or int(future_indices[-1]) >= int(video_length):
            return None
        frames = metadata.get("frames", {})
        contact = False
        label_end = (
            int(future_indices[-1])
            if label_scope == "single_clip"
            else int(video_length) - 1
        )
        for frame_index in range(int(future_indices[0]), label_end + 1):
            frame = _frame_at(frames, frame_index)
            if frame is None:
                return None
            contact = contact or bool(
                frame.get("labels", {}).get("target_contacting_zone", False)
            )
        return {
            "prediction_start": prediction_start,
            "current_indices": current_indices.astype(np.int64),
            "future_indices": future_indices.astype(np.int64),
            "label_start": int(future_indices[0]),
            "label_end": label_end,
            "label_scope": label_scope,
            "contact_label": int(contact),
        }
    except (OSError, EOFError, KeyError, TypeError, ValueError, pickle.UnpicklingError):
        return None


class PhysionSingleFutureOCPDataset(torch.utils.data.Dataset):
    """Decode Current only and label its one predicted Future time interval."""

    def __init__(
        self,
        *,
        root,
        split,
        video_glob="**/*_img.mp4",
        clip_frames=16,
        current_frame_step=2,
        future_frame_step=4,
        clip_gap=0,
        label_scope="single_clip",
        transform=None,
        max_videos=None,
    ):
        self.clip_frames = int(clip_frames)
        self.current_frame_step = int(current_frame_step)
        self.future_frame_step = int(future_frame_step)
        self.clip_gap = int(clip_gap)
        self.label_scope = str(label_scope)
        self.transform = transform
        if min(self.clip_frames, self.current_frame_step, self.future_frame_step) <= 0:
            raise ValueError("clip_frames and frame steps must be positive")
        if self.clip_gap < 0:
            raise ValueError("clip_gap must be non-negative")
        if self.label_scope not in LABEL_SCOPES:
            raise ValueError(f"unknown single-Future label scope: {self.label_scope!r}")
        split_root = os.path.join(str(root), split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"Physion split directory does not exist: {split_root}")
        paths = sorted(glob.glob(os.path.join(split_root, video_glob), recursive=True))
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
            spec = read_single_future_spec(
                path,
                clip_frames=self.clip_frames,
                current_frame_step=self.current_frame_step,
                future_frame_step=self.future_frame_step,
                clip_gap=self.clip_gap,
                video_length=video_length,
                label_scope=self.label_scope,
            )
            if spec is not None:
                self.specs.append({"path": path, **spec})
        if not self.specs:
            raise RuntimeError(f"No valid single-Future OCP videos remain in split={split}")

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, index):
        spec = self.specs[index]
        path = spec["path"]
        try:
            reader = VideoReader(path, num_threads=-1, ctx=cpu(0))
            frames = reader.get_batch(spec["current_indices"]).asnumpy()
        except Exception as exc:
            raise RuntimeError(f"failed to decode deterministic OCP sample={path}: {exc}") from exc
        if self.transform is None:
            current = (
                torch.as_tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2)
                / 255.0
            )
        else:
            current = self.transform(frames)
        if current.ndim != 4 or current.size(1) != self.clip_frames:
            raise ValueError(
                f"transform must return [C,{self.clip_frames},H,W], got {tuple(current.shape)}"
            )
        return {
            "current": current,
            "current_indices": torch.as_tensor(spec["current_indices"], dtype=torch.long),
            "future_indices": torch.as_tensor(spec["future_indices"], dtype=torch.long),
            "label_start": torch.tensor(spec["label_start"], dtype=torch.long),
            "label_end": torch.tensor(spec["label_end"], dtype=torch.long),
            "contact_label": torch.tensor(spec["contact_label"], dtype=torch.long),
            "path": path,
        }


def build_data_loader(dataset, data, batch_size, workers):
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


def make_loader(config, split, transform, batch_size, workers, max_videos):
    data = config["data"]
    dataset = PhysionSingleFutureOCPDataset(
        root=data.get("root"),
        split=split,
        video_glob=data.get("video_glob", "**/*_img.mp4"),
        clip_frames=int(data["_ocp_clip_frames"]),
        current_frame_step=int(data["_ocp_current_frame_step"]),
        future_frame_step=int(data["_ocp_future_frame_step"]),
        clip_gap=int(data.get("clip_gap", 0)),
        label_scope=data.get("_ocp_label_scope", "single_clip"),
        transform=transform,
        max_videos=max_videos,
    )
    return build_data_loader(dataset, data, batch_size, workers)


def resolve_single_future_protocol(config, model):
    """Resolve the one-step clip protocol from the loaded model when available."""
    data = config.setdefault("data", {})
    clip_frames = getattr(model, "clip_frames", None)
    if clip_frames is None:
        clip_frames = data.get("clip_frames", data.get("future_frames", data.get("context_frames", 16)))
    current_step = getattr(model, "current_frame_step", None)
    if current_step is None:
        current_step = data.get("current_frame_step", data.get("frame_step", 2))
    future_step = getattr(model, "future_frame_step", None)
    if future_step is None:
        future_step = data.get("future_frame_step", data.get("frame_step", 2))
    values = {
        "_ocp_clip_frames": int(clip_frames),
        "_ocp_current_frame_step": int(current_step),
        "_ocp_future_frame_step": int(future_step),
    }
    if min(values.values()) <= 0:
        raise ValueError(f"single-Future OCP protocol must be positive: {values}")
    data.update(values)
    return values


def feature_result(current_context, predicted_future, labels, paths):
    if current_context.shape != predicted_future.shape:
        raise ValueError(
            f"current/predicted token shapes differ: {current_context.shape} vs "
            f"{predicted_future.shape}"
        )
    current = current_context.float().mean(dim=1).cpu()
    predicted = predicted_future.float().mean(dim=1).cpu()
    return {
        "features": torch.cat((current, predicted, predicted - current), dim=1),
        "labels": labels.float().cpu(),
        "paths": list(paths),
    }


@torch.no_grad()
def extract_local_features(
    loader, model, device, dtype, use_amp, split_name, log_every_batches
):
    feature_batches, labels, paths = [], [], []
    count = 0
    rank = dist.get_rank() if dist.is_initialized() else 0
    started = time.monotonic()
    total_batches = len(loader)
    model.eval()
    LOGGER.info(
        "feature extraction started: split=%s rank=%d batches=%d",
        split_name,
        rank,
        total_batches,
    )
    for iteration, batch in enumerate(loader, start=1):
        current = batch["current"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=dtype if use_amp else None,
        ):
            context = model.encode_current(current)
            predicted = model.predict_next(context)
        result = feature_result(context, predicted, batch["contact_label"], batch["path"])
        feature_batches.append(result["features"])
        labels.append(result["labels"])
        paths.extend(result["paths"])
        count += len(result["paths"])
        if iteration == 1 or iteration % log_every_batches == 0 or iteration == total_batches:
            LOGGER.info(
                "feature extraction progress: split=%s rank=%d batch=%d/%d "
                "samples=%d elapsed=%.1fs",
                split_name,
                rank,
                iteration,
                total_batches,
                count,
                time.monotonic() - started,
            )
    if not labels:
        raise RuntimeError(f"rank {rank} has no single-Future OCP samples")
    return {
        "features": torch.cat(feature_batches),
        "labels": torch.cat(labels),
        "paths": paths,
    }


def merge_feature_payloads(payloads):
    features = torch.cat([payload["features"] for payload in payloads], dim=0)
    labels = torch.cat([payload["labels"] for payload in payloads], dim=0)
    paths = [path for payload in payloads for path in payload["paths"]]
    if features.size(0) != labels.numel() or features.size(0) != len(paths):
        raise RuntimeError("distributed single-Future payload sizes do not match")
    if len(set(paths)) != len(paths):
        raise RuntimeError("distributed single-Future sharding produced duplicate paths")
    order = sorted(range(len(paths)), key=paths.__getitem__)
    index = torch.as_tensor(order, dtype=torch.long)
    return {
        "features": features[index],
        "labels": labels[index],
        "paths": [paths[item] for item in order],
    }


def gather_features(local_result):
    if not dist.is_initialized():
        return local_result
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_result)
    return merge_feature_payloads(gathered) if dist.get_rank() == 0 else None


def run_probe_suite(readout, test, args, readout_device, output_dir):
    fit_indices, threshold_indices = stratified_probe_split(
        readout["paths"],
        readout["labels"],
        validation_fraction=args.probe_validation_fraction,
        seed=args.seed,
    )
    (output_dir / "probe_split.json").write_text(
        json.dumps({
            "seed": int(args.seed),
            "validation_fraction": float(args.probe_validation_fraction),
            "fit_paths": [readout["paths"][index] for index in fit_indices.tolist()],
            "threshold_validation_paths": [
                readout["paths"][index] for index in threshold_indices.tolist()
            ],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results = {}
    for variant in args.probe_variants:
        readout_x = select_single_future_probe_features(readout["features"], variant)
        test_x = select_single_future_probe_features(test["features"], variant)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        classifier, mean, std = train_readout(
            readout_x[fit_indices],
            readout["labels"][fit_indices],
            args.readout_epochs,
            args.readout_lr,
            args.readout_weight_decay,
            args.readout_log_every_epochs,
            readout_device,
        )
        with torch.no_grad():
            validation_x = readout_x[threshold_indices].to(readout_device)
            validation_probs = torch.sigmoid(
                classifier((validation_x - mean) / std).squeeze(1)
            ).cpu()
            device_test_x = test_x.to(readout_device)
            test_probs = torch.sigmoid(
                classifier((device_test_x - mean) / std).squeeze(1)
            ).cpu()
        threshold = best_threshold(
            validation_probs, readout["labels"][threshold_indices]
        )
        metrics = binary_metrics(test_probs, test["labels"], threshold)
        results[variant] = {
            "feature_dim": int(test_x.size(1)),
            "num_fit": int(fit_indices.numel()),
            "num_threshold_validation": int(threshold_indices.numel()),
            "num_test": int(test["labels"].numel()),
            "threshold": float(threshold),
            "threshold_source": "readout_data_v1/held_out_by_path_hash",
            "test_auroc": auroc(test_probs.numpy(), test["labels"].numpy()),
            **{f"test_{key}": float(value) for key, value in metrics.items()},
        }
        torch.save({
            "classifier": {
                key: value.detach().cpu()
                for key, value in classifier.state_dict().items()
            },
            "feature_mean": mean.detach().cpu(),
            "feature_std": std.detach().cpu(),
            "threshold": float(threshold),
            "fit_indices": fit_indices,
            "threshold_validation_indices": threshold_indices,
        }, output_dir / f"readout_{variant}.pt")
        np.savez_compressed(
            output_dir / f"test_predictions_{variant}.npz",
            probabilities=test_probs.numpy(),
            labels=test["labels"].numpy(),
            paths=np.asarray(test["paths"], dtype=object),
            threshold=np.asarray(threshold),
        )
    return results


def parse_args(task_name="single_future_ocp"):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--eval-config", default=None)
    known, _ = config_parser.parse_known_args()
    eval_config = {}
    if known.eval_config:
        with open(known.eval_config) as handle:
            eval_config = yaml.safe_load(handle) or {}
    task_launch_config = task_launch(eval_config, task_name)
    task = task_experiment(eval_config, task_name)
    evaluation = task.get("evaluation", {})
    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--config", default=task_launch_config.get("training_config"))
    parser.add_argument("--checkpoint", default=task_launch_config.get("checkpoint"))
    parser.add_argument("--output-dir", default=task_launch_config.get("output_dir"))
    parser.add_argument("--device", default=evaluation.get("device", "cuda:0"))
    parser.add_argument("--batch-size", type=int, default=int(evaluation.get("batch_size", 1)))
    parser.add_argument("--num-workers", type=int, default=int(evaluation.get("num_workers", 4)))
    parser.add_argument("--max-readout-videos", type=int, default=evaluation.get("max_readout_videos"))
    parser.add_argument("--max-test-videos", type=int, default=evaluation.get("max_test_videos"))
    parser.add_argument("--readout-epochs", type=int, default=int(evaluation.get("readout_epochs", 300)))
    parser.add_argument("--readout-lr", type=float, default=float(evaluation.get("readout_lr", 3.0e-4)))
    parser.add_argument(
        "--readout-weight-decay",
        type=float,
        default=float(evaluation.get("readout_weight_decay", 1.0e-3)),
    )
    parser.add_argument("--readout-device", default=evaluation.get("readout_device", "same"))
    parser.add_argument("--progress-every-batches", type=int, default=int(evaluation.get("progress_every_batches", 10)))
    parser.add_argument(
        "--readout-log-every-epochs",
        type=int,
        default=int(evaluation.get("readout_log_every_epochs", 50)),
    )
    parser.add_argument("--seed", type=int, default=int(evaluation.get("seed", 239)))
    parser.add_argument(
        "--probe-validation-fraction",
        type=float,
        default=float(evaluation.get("probe_validation_fraction", 0.25)),
    )
    parser.add_argument(
        "--probe-variants",
        nargs="+",
        choices=PROBE_VARIANTS,
        default=evaluation.get("probe_variants", PROBE_VARIANTS),
    )
    args = parser.parse_args()
    if not args.config or not args.checkpoint:
        parser.error("provide --eval-config, or both --config and --checkpoint")
    if args.progress_every_batches <= 0 or args.readout_log_every_epochs <= 0:
        parser.error("progress logging intervals must be positive")
    return args, evaluation


def _init_distributed(task_name="single_future_ocp"):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 8:
        raise RuntimeError(f"{task_name} requires exactly 8 GPUs, got WORLD_SIZE={world_size}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{task_name} requires CUDA/NCCL")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank


def _validate_checkpoint(checkpoint):
    error = [None]
    if dist.get_rank() == 0:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            error[0] = f"single_future_ocp checkpoint does not exist: {path}"
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise FileNotFoundError(error[0])


def main(task_name="single_future_ocp", default_label_scope="single_clip"):
    local_rank = _init_distributed(task_name)
    try:
        args, evaluation = parse_args(task_name)
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)-8s][%(asctime)s][%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _validate_checkpoint(args.checkpoint)
        random.seed(args.seed + dist.get_rank())
        np.random.seed(args.seed + dist.get_rank())
        torch.manual_seed(args.seed + dist.get_rank())
        with open(args.config) as handle:
            config = experiment_config(yaml.safe_load(handle))
        label_scope = str(evaluation.get("label_scope", default_label_scope))
        if label_scope not in LABEL_SCOPES:
            raise ValueError(
                f"evaluation.label_scope must be one of {LABEL_SCOPES}, got {label_scope!r}"
            )
        config.setdefault("data", {})["_ocp_label_scope"] = label_scope
        device = torch.device(f"cuda:{local_rank}")
        dtype_name = str(config.get("meta", {}).get("dtype", "bfloat16")).lower()
        dtype = (
            torch.bfloat16 if dtype_name == "bfloat16"
            else torch.float16 if dtype_name == "float16"
            else torch.float32
        )
        adapter_name = evaluation.get("world_model_adapter")
        model, checkpoint_metadata = load_world_model(
            config, args.checkpoint, device, adapter_name=adapter_name
        )
        protocol = resolve_single_future_protocol(config, model)
        LOGGER.info("single-Future protocol resolved from model/config: %s", protocol)
        crop_size = int(config["data"].get("crop_size", 256))
        transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
            crop_size=crop_size,
        )
        readout_loader = make_loader(
            config, "readout_data_v1", transform, args.batch_size,
            args.num_workers, args.max_readout_videos,
        )
        readout = gather_features(extract_local_features(
            readout_loader, model, device, dtype,
            dtype_name in ("bfloat16", "float16"),
            "readout_data_v1", args.progress_every_batches,
        ))
        test_loader = make_loader(
            config, "testdata_v1", transform, args.batch_size,
            args.num_workers, args.max_test_videos,
        )
        test = gather_features(extract_local_features(
            test_loader, model, device, dtype,
            dtype_name in ("bfloat16", "float16"),
            "testdata_v1", args.progress_every_batches,
        ))
        if dist.get_rank() != 0:
            return
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        readout_device = device if args.readout_device == "same" else torch.device(args.readout_device)
        probes = run_probe_suite(readout, test, args, readout_device, output_dir)
        data = config["data"]
        result = {
            "protocol": (
                "single_future_clip_ocp"
                if label_scope == "single_clip"
                else "single_future_clip_all_future_label_ocp"
            ),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_epoch": int(checkpoint_metadata.get("checkpoint_epoch", -1)),
            "future_rgb_consumed": False,
            "label_scope": label_scope,
            "label_definition": (
                "any target_contacting_zone on every raw frame from the first "
                "through the last sampled Future frame, inclusive"
                if label_scope == "single_clip"
                else "any target_contacting_zone on every raw frame in "
                "[P+clip_gap, video_end)"
            ),
            "future_indices_definition": "P+clip_gap+arange(clip_frames)*future_frame_step",
            "label_checks_intermediate_frames": True,
            "feature_definition": "concat(current, predicted_future, predicted_future-current)",
            "primary_probe_variant": "predicted_future",
            "split_definition": (
                "path-hash class-stratified readout fit/threshold split; test fits nothing"
            ),
            "distributed_world_size": dist.get_world_size(),
            "seed": int(args.seed),
            "clip_frames": int(protocol["_ocp_clip_frames"]),
            "current_frame_step": int(protocol["_ocp_current_frame_step"]),
            "future_frame_step": int(protocol["_ocp_future_frame_step"]),
            "clip_gap": int(data.get("clip_gap", 0)),
            "num_readout": int(readout["labels"].numel()),
            "num_test": int(test["labels"].numel()),
            "readout_label_positive_rate": float(readout["labels"].mean()),
            "test_label_positive_rate": float(test["labels"].mean()),
            "probes": probes,
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
