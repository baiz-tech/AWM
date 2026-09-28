#!/usr/bin/env python3
"""Export observed CLEVRER latents plus one closed-loop imagined chunk for QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from decord import VideoReader, cpu

from external.vjepa2.app.vjepa.transforms import make_transforms
from src.training.distributed import ExactDistributedSampler
from src.core.world_model_loader import load_configured_model
from src.core.config_utils import merged_task_experiment
from src.data.clevrer.v1.protocol import resolve_qa_protocol


LOGGER = logging.getLogger("shared.evaluate.clevrer.export_qa_trajectories")
_SPLIT_DIRECTORIES = {"train": "train", "validation": "val", "val": "val", "test": "test"}
_VIDEO_NAME = re.compile(r"^video_(\d{5})\.mp4$")


def experiment_config(raw_config):
    return merged_task_experiment(raw_config, "qa_eval")


def load_world_model(config, checkpoint, device):
    return load_configured_model(config, checkpoint, device)


def init_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FullFrameResize:
    """VideoSAUR release preprocessing: resize the complete frame, then normalize."""

    def __init__(self, size):
        self.size = int(size)

    def __call__(self, buffer):
        frames = torch.as_tensor(buffer, dtype=torch.float32).permute(0, 3, 1, 2)
        frames = F.interpolate(frames, size=(self.size, self.size), mode="bicubic", align_corners=False)
        mean = frames.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = frames.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return ((frames / 255.0 - mean) / std).permute(1, 0, 2, 3)


class CLEVRERFullVideoDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform, scene_range=None, max_videos=None):
        split = str(split).lower()
        if split not in _SPLIT_DIRECTORIES:
            raise ValueError(f"unsupported split: {split!r}")
        self.split = "validation" if split == "val" else split
        self.transform = transform
        split_root = Path(root) / "videos" / _SPLIT_DIRECTORIES[split]
        if not split_root.is_dir():
            raise FileNotFoundError(f"CLEVRER video directory does not exist: {split_root}")
        samples = []
        for path in sorted(split_root.glob("*.mp4")):
            match = _VIDEO_NAME.fullmatch(path.name)
            if match is None:
                raise ValueError(f"unexpected CLEVRER video name: {path}")
            scene_index = int(match.group(1))
            if scene_range is None or int(scene_range[0]) <= scene_index <= int(scene_range[1]):
                samples.append((path, scene_index))
        self.samples = samples[: int(max_videos)] if max_videos is not None else samples
        if not self.samples:
            raise FileNotFoundError(f"no CLEVRER videos found under {split_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, scene_index = self.samples[index]
        reader = VideoReader(str(path), num_threads=-1, ctx=cpu(0))
        if len(reader) != 128:
            raise ValueError(f"QA protocol requires 128 frames, got {len(reader)} for {path}")
        indices = np.arange(0, 128, 2, dtype=np.int64)
        frames = reader.get_batch(indices).asnumpy()
        video = self.transform(frames)
        if video.ndim != 4 or video.size(1) != 64:
            raise ValueError(f"transform must return [C,64,H,W], got {tuple(video.shape)}")
        return {
            "scene_index": torch.tensor(scene_index, dtype=torch.long),
            "video": video,
        }


def spatial_pool(tokens, spatial_size, pool_grid):
    """Average patch tokens into a deterministic spatial grid."""
    if tokens.ndim != 4 or tokens.size(2) != spatial_size * spatial_size:
        raise ValueError(
            f"expected [B,T,{spatial_size * spatial_size},D], got {tuple(tokens.shape)}"
        )
    batch, steps, _, dim = tokens.shape
    feature_map = tokens.reshape(batch * steps, spatial_size, spatial_size, dim)
    feature_map = feature_map.permute(0, 3, 1, 2).float()
    pooled = F.adaptive_avg_pool2d(feature_map, (pool_grid, pool_grid))
    return pooled.permute(0, 2, 3, 1).reshape(batch, steps, pool_grid * pool_grid, dim)


@torch.no_grad()
def export_batch(model, video, pool_grid, use_amp, dtype, variant="naive"):
    if variant not in ("naive", "observed_only", "copy_future"):
        raise ValueError(f"unsupported QA trajectory variant: {variant!r}")
    clip_frames = model.clip_frames
    if video.size(2) != 4 * clip_frames:
        raise ValueError(f"expected {4 * clip_frames} sampled observed frames")
    with torch.amp.autocast(
        device_type=video.device.type, enabled=use_amp, dtype=dtype if use_amp else None
    ):
        if getattr(model, "token_layout", "patches") == "objects" and hasattr(
            model, "encode_current_sequence"
        ):
            observed = model.encode_current_sequence(video)
            last_context = observed[:, -model.tokens_per_chunk :]
        else:
            encoded_chunks = []
            for chunk_index in range(4):
                clip = video[:, :, chunk_index * clip_frames : (chunk_index + 1) * clip_frames]
                encoded_chunks.append(model.encode_current(clip))
            last_context = encoded_chunks[-1]
            observed = torch.cat(encoded_chunks, dim=1)
    observed = observed.reshape(
        video.size(0), 4 * model.temporal_steps, model.spatial_tokens, observed.size(-1)
    )
    if variant == "observed_only":
        trajectory = observed
    else:
        if variant == "naive":
            with torch.amp.autocast(
                device_type=video.device.type, enabled=use_amp, dtype=dtype if use_amp else None
            ):
                imagined = model.predict_next(last_context)
        else:
            imagined = last_context
        imagined = imagined.reshape(
            video.size(0), model.temporal_steps, model.spatial_tokens, imagined.size(-1)
        )
        trajectory = torch.cat([observed, imagined], dim=1)
    if getattr(model, "token_layout", "patches") == "objects":
        return trajectory.float()
    spatial_size = model.crop_size // model.patch_size
    return spatial_pool(
        trajectory.float(), spatial_size=spatial_size, pool_grid=pool_grid
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=("train", "validation", "val", "test"), required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scene-range", type=int, nargs=2, default=None, metavar=("START", "END"))
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--variant", choices=("naive", "observed_only", "copy_future"), default="naive"
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    config = experiment_config(raw_config)
    meta = config.get("meta", {})
    cfg_data = config.get("data", {})
    cfg_qa = config.get("qa", {})
    protocol_info = resolve_qa_protocol(config)
    if protocol_info["token_source"] != "online":
        raise ValueError(
            "export_qa_trajectories is the online exporter; use precomputed trajectories "
            "directly with train_qa/eval_qa"
        )
    expected_generation = {"observed_only": "none", "naive": "rollout", "copy_future": "ground_truth"}[args.variant]
    if protocol_info["future_generation"] != expected_generation:
        raise ValueError(
            f"qa.data.future_generation={protocol_info['future_generation']!r} does not match "
            f"export variant={args.variant!r} ({expected_generation!r})"
        )
    configured_representation = str(
        (config.get("qa", {}).get("representation") or {}).get("type", "object_slots")
    )
    cfg_outputs = config.get("outputs", {})
    if (
        int(cfg_data.get("clip_frames", 16)),
        int(cfg_data.get("current_frame_step", 2)),
    ) != (16, 2):
        raise ValueError("QA trajectory export requires clip_frames=16 and current_frame_step=2")
    pool_grid = int(cfg_qa.get("spatial_pool_grid", 2))
    if pool_grid <= 0:
        raise ValueError("qa.spatial_pool_grid must be positive")

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
    split = "validation" if args.split == "val" else args.split
    run_dir = Path(config["folder"])
    checkpoint = Path(args.checkpoint or run_dir / "best.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"world-model checkpoint does not exist: {checkpoint}")
    trajectory_root = Path(
        args.output_dir
        or cfg_outputs.get("qa_trajectory_root")
        or cfg_qa.get("trajectory_root", run_dir / "qa" / "trajectories")
    )
    output_dir = trajectory_root / args.variant / split
    output_dir.mkdir(parents=True, exist_ok=True)

    token_layout_hint = str(config.get("model", {}).get("object_encoder", "")).lower()
    if token_layout_hint == "videosaur" and bool(cfg_data.get("full_frame_resize", True)):
        transform = FullFrameResize(int(cfg_data.get("crop_size", 256)))
    else:
        transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
            crop_size=int(cfg_data.get("crop_size", 256)),
        )
    dataset = CLEVRERFullVideoDataset(
        cfg_data["root"], split, transform=transform, scene_range=args.scene_range,
        max_videos=args.max_videos,
    )
    sampler = ExactDistributedSampler(dataset, rank=rank, world_size=world_size)
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else cfg_qa.get("export_num_workers", cfg_data.get("num_workers", 4))
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=int(cfg_qa.get("export_batch_size", 1)),
        num_workers=num_workers,
        pin_memory=bool(cfg_data.get("pin_mem", True)),
        persistent_workers=bool(cfg_data.get("persistent_workers", True) and num_workers > 0),
        drop_last=False,
    )
    model, checkpoint_metadata = load_world_model(config, checkpoint, device)
    model.eval()
    token_layout = str(getattr(model, "token_layout", "patches"))
    if token_layout not in ("patches", "objects"):
        raise ValueError(f"unsupported world-model token_layout={token_layout!r}")
    actual_representation = "object_slots" if token_layout == "objects" else "global_tokens"
    if configured_representation != actual_representation:
        raise ValueError(
            f"qa.representation.type={configured_representation!r} does not match "
            f"model output {actual_representation!r}"
        )
    visual_tokens_per_step = (
        int(model.spatial_tokens) if token_layout == "objects" else pool_grid * pool_grid
    )
    encoder_checkpoint = checkpoint_metadata.get("slot_checkpoint") or meta.get(
        "pretrain_checkpoint"
    )
    if not encoder_checkpoint:
        raise ValueError(
            "world-model adapter metadata must provide slot_checkpoint or config "
            "meta.pretrain_checkpoint"
        )
    encoder_source_epoch = checkpoint_metadata.get(
        "slot_checkpoint_epoch", checkpoint_metadata.get("source_epoch", -1)
    )
    dtype_name = str(meta.get("dtype", "bfloat16")).lower()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        dtype_name
    ]
    use_amp = dtype_name in ("bfloat16", "float16") and device.type == "cuda"
    raw_frame_step = int(model.tubelet_size) * int(
        cfg_data.get("current_frame_step", 2)
    )
    checkpoint_sha = sha256(checkpoint)
    encoder_path = Path(encoder_checkpoint).resolve()
    encoder_sha = sha256(encoder_path) if encoder_path.is_file() else None
    feature_dim = getattr(model.predictor, "slot_dim", None)
    if feature_dim is None:
        feature_dim = getattr(getattr(model, "encoder", None), "embed_dim", None)
    if feature_dim is None:
        raise AttributeError("world model must expose predictor.slot_dim or encoder.embed_dim")
    trajectory_steps = (
        4 * model.temporal_steps
        if args.variant == "observed_only"
        else 5 * model.temporal_steps
    )
    trajectory_shape = (trajectory_steps, visual_tokens_per_step, int(feature_dim))
    fingerprint_payload = {
        "protocol": protocol_info,
        "variant": args.variant,
        "token_layout": token_layout,
        "crop_size": int(cfg_data.get("crop_size", 256)),
        "raw_frame_step": raw_frame_step,
        "trajectory_shape": trajectory_shape,
        "feature_normalization": "none",
    }
    trajectory_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    def validate_or_stale(path):
        if not path.is_file():
            return False
        try:
            record = torch.load(path, map_location="cpu", weights_only=False)
            visual_tokens = record.get("visual_tokens")
            observed_mask = record.get("observed_mask")
            raw_times = record.get("raw_frame_times")
            valid = (
                record.get("world_model_checkpoint_sha256") == checkpoint_sha
                and record.get("encoder_checkpoint_sha256") == encoder_sha
                and record.get("trajectory_fingerprint") == trajectory_fingerprint
                and record.get("token_layout") == token_layout
                and record.get("protocol_version") == f"clevrer_qa_{args.variant}_v3"
                and torch.is_tensor(visual_tokens)
                and tuple(visual_tokens.shape) == trajectory_shape
                and torch.is_tensor(observed_mask)
                and tuple(observed_mask.shape) == (trajectory_steps,)
                and torch.is_tensor(raw_times)
                and torch.equal(
                    raw_times,
                    torch.arange(trajectory_steps, dtype=torch.long) * raw_frame_step,
                )
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            valid = False
        if valid:
            return True
        stale = path.with_name(f"{path.stem}.stale.{time.time_ns()}{path.suffix}")
        path.replace(stale)
        LOGGER.warning("stale trajectory moved aside: %s -> %s", path, stale)
        return False

    exported = 0
    skipped = 0
    for batch in loader:
        scene_indices = batch["scene_index"].tolist()
        paths = [output_dir / f"scene_{scene_index:05d}.pt" for scene_index in scene_indices]
        exists = [validate_or_stale(path) for path in paths]
        if any(exists) and not args.resume:
            existing = [str(path) for path, present in zip(paths, exists) if present]
            raise FileExistsError(
                f"trajectory files already exist; use --resume to keep them: {existing[:4]}"
            )
        if all(exists):
            skipped += len(paths)
            continue
        video = batch["video"].to(device, non_blocking=True)
        tokens = export_batch(
            model, video, pool_grid, use_amp, dtype, variant=args.variant
        ).cpu().to(torch.float16)
        observed_steps = 4 * model.temporal_steps
        raw_times = torch.arange(tokens.size(1), dtype=torch.long) * raw_frame_step
        observed_mask = torch.ones(tokens.size(1), dtype=torch.bool)
        if args.variant != "observed_only":
            observed_mask[observed_steps:] = False
        for row, (scene_index, path, present) in enumerate(zip(scene_indices, paths, exists)):
            if present:
                skipped += 1
                continue
            record = {
                "scene_index": int(scene_index),
                "visual_tokens": tokens[row].contiguous(),
                "observed_mask": observed_mask,
                "raw_frame_times": raw_times,
                "world_model_variant": args.variant,
                "world_model_checkpoint": str(checkpoint.resolve()),
                "encoder_checkpoint": str(encoder_path),
                "world_model_checkpoint_sha256": checkpoint_sha,
                "encoder_checkpoint_sha256": encoder_sha,
                "trajectory_fingerprint": trajectory_fingerprint,
                "token_layout": token_layout,
                "protocol_version": f"clevrer_qa_{args.variant}_v3",
                "qa_protocol": protocol_info["protocol"],
                "representation": configured_representation,
                "feature_normalization": "none",
            }
            temporary = path.with_suffix(f".rank{rank}.tmp")
            torch.save(record, temporary)
            temporary.replace(path)
            exported += 1
    counts = torch.tensor([exported, skipped], dtype=torch.long, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        dist.barrier()
    if rank == 0:
        manifest = {
            "protocol": f"clevrer_qa_{args.variant}_v3",
            "variant": args.variant,
            "qa_protocol": protocol_info["protocol"],
            "qa_backend": protocol_info["backend"],
            "representation": configured_representation,
            "split": split,
            "scene_range": args.scene_range,
            "scenes": len(dataset),
            "exported": int(counts[0].item()),
            "skipped": int(counts[1].item()),
            "trajectory_shape": list(trajectory_shape),
            "trajectory_fingerprint": trajectory_fingerprint,
            "token_layout": token_layout,
            "feature_normalization": "none",
            "observed_raw_frame_times": list(range(0, 128, raw_frame_step)),
            "imagined_raw_frame_times": list(
                range(128, 128 + model.temporal_steps * raw_frame_step, raw_frame_step)
            ),
            "imagined_reads_future_rgb": False,
            "world_model_checkpoint": str(checkpoint.resolve()),
            "world_model_checkpoint_sha256": checkpoint_sha,
            "world_model_checkpoint_epoch": int(checkpoint_metadata.get("checkpoint_epoch", -1)),
            "encoder_checkpoint": str(Path(encoder_checkpoint).resolve()),
            "encoder_checkpoint_sha256": encoder_sha,
            "encoder_source_epoch": int(encoder_source_epoch),
            "config": str(Path(args.config).resolve()),
            "world_size": world_size,
        }
        manifest_name = (
            f"manifest_{args.scene_range[0]:05d}_{args.scene_range[1]:05d}.json"
            if args.scene_range is not None else "manifest.json"
        )
        (output_dir / manifest_name).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        LOGGER.info("trajectory export complete: %s", manifest)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
