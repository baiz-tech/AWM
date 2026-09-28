#!/usr/bin/env python3
"""Export per-scene full-patch context and naive-predicted future latents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from decord import VideoReader, cpu
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.vjepa.transforms import make_transforms
from recipe.shared.distributed import ExactDistributedSampler
from recipe.vjepa2_naive.dataset import CLEVRERCurrentFutureDataset
from recipe.vjepa2_naive.model import build_model


class MultiWindow(CLEVRERCurrentFutureDataset):
    def __init__(self, *args, window_starts=(0, 32, 64), **kwargs):
        super().__init__(*args, **kwargs)
        self.window_starts = tuple(int(value) for value in window_starts)
        if not self.window_starts or any(value < 0 for value in self.window_starts):
            raise ValueError("window starts must contain non-negative frame indices")
        self.samples = [
            (path, scene_id, window_start)
            for path, scene_id in self.samples
            for window_start in self.window_starts
        ]

    def _indices_for_window(self, video_length, window_start):
        current = window_start + np.arange(self.clip_frames) * self.current_frame_step
        anchor = window_start + self.clip_frames * self.current_frame_step
        future = anchor + self.clip_gap + np.arange(
            self.clip_frames * self.num_future_chunks
        ) * self.future_frame_step
        if int(future[-1]) >= int(video_length):
            return None
        return (
            anchor,
            current.astype(np.int64),
            future.reshape(self.num_future_chunks, self.clip_frames).astype(np.int64),
        )

    def __getitem__(self, index):
        path, scene_id, window_start = self.samples[index]
        reader = VideoReader(path, num_threads=-1, ctx=cpu(0))
        selection = self._indices_for_window(len(reader), window_start)
        if selection is None:
            raise RuntimeError(f"invalid CLEVRER window: scene={scene_id} start={window_start}")
        anchor, current_indices, future_indices = selection
        indices = np.concatenate((current_indices, future_indices.reshape(-1)))
        frames = reader.get_batch(indices).asnumpy()
        merged = self.transform(frames) if self.transform is not None else torch.as_tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
        current = merged[:, : self.clip_frames]
        future_flat = merged[:, self.clip_frames :]
        future_chunks = torch.stack([
            future_flat[:, chunk * self.clip_frames : (chunk + 1) * self.clip_frames]
            for chunk in range(self.num_future_chunks)
        ])
        return {
            "scene_index": torch.tensor(scene_id), "window_start": torch.tensor(window_start),
            "anchor_frame": torch.tensor(anchor), "current": current,
            "future": future_chunks[0], "future_chunks": future_chunks,
            "current_indices": torch.as_tensor(current_indices),
            "future_indices": torch.as_tensor(future_indices),
        }


def init_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return rank, world_size, local_rank


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--window-starts", default="0,32,64")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    experiment = config["tasks"]["train"]["experiment"]
    meta, model_cfg, data_cfg = experiment["meta"], experiment["model"], experiment["data"]
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model, _, _ = build_model(
        device, model_cfg, data_cfg, meta["pretrain_checkpoint"],
        meta.get("encoder_checkpoint_key", "target_encoder"),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "predictor" not in checkpoint:
        raise KeyError("naive checkpoint must contain predictor")
    model.predictor.load_state_dict(checkpoint["predictor"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    transform = make_transforms(
        False, (1.0, 1.0), (1.0, 1.0), 0.0, False, False, int(data_cfg["crop_size"])
    )
    window_starts = tuple(int(value) for value in args.window_starts.split(",") if value.strip())
    dataset = MultiWindow(
        root=data_cfg["root"], split=args.split, video_glob=data_cfg.get("video_glob", "*.mp4"),
        clip_frames=int(data_cfg["clip_frames"]), sampling_mode="random",
        current_frame_step=int(data_cfg["current_frame_step"]),
        future_frame_step=int(data_cfg["future_frame_step"]),
        clip_gap=int(data_cfg.get("clip_gap", 0)), num_future_chunks=1,
        transform=transform, deterministic=True, max_videos=args.max_scenes,
        window_starts=window_starts,
    )
    sampler = ExactDistributedSampler(dataset, rank=rank, world_size=world_size)
    loader = DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    split_dir = args.output_dir / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    dtype_name = str(meta.get("dtype", "bfloat16"))
    use_amp = device.type == "cuda" and dtype_name in ("bfloat16", "float16")
    amp_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    written = 0
    with torch.no_grad():
        for batch in loader:
            pending = []
            for row, (scene_id, window_start) in enumerate(zip(batch["scene_index"].tolist(), batch["window_start"].tolist())):
                path = split_dir / f"scene_{scene_id:05d}_window_{window_start:03d}.pt"
                if args.resume and path.is_file():
                    continue
                pending.append((row, scene_id, window_start, path))
            if not pending:
                continue
            rows = torch.tensor([item[0] for item in pending], dtype=torch.long)
            current = batch["current"][rows].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=use_amp, dtype=amp_dtype if use_amp else None
            ):
                context = model.encode_current(current)
                future = model.predict_next(context)
            context = context.reshape(len(pending), 8, 256, 1280).half().cpu()
            future = future.reshape(len(pending), 8, 256, 1280).half().cpu()
            for row, (_, scene_id, window_start, path) in enumerate(pending):
                torch.save(
                    {
                        "protocol": "clevrer_fullpatch_multiwindow_naive_v1",
                        "scene_id": scene_id,
                        "window_start": window_start,
                        "current_indices": batch["current_indices"][pending[row][0]],
                        "future_indices": batch["future_indices"][pending[row][0]],
                        "context_tokens": context[row],
                        "future_tokens": future[row],
                    },
                    path,
                )
                written += 1
            if rank == 0 and written % 100 == 0:
                print(f"rank0 wrote {written} scenes", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        scene_files = sorted(split_dir.glob("scene_*_window_*.pt"))
        manifest = {
            "protocol": "clevrer_fullpatch_multiwindow_naive_v1",
            "split": args.split,
            "samples": len(scene_files),
            "window_starts": list(window_starts),
            "context_shape": [8, 256, 1280],
            "future_shape": [8, 256, 1280],
            "dtype": "float16",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "world_size": world_size,
        }
        (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
