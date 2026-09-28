#!/usr/bin/env python3
"""Cache current and official V-JEPA2 predicted-future full-patch latents."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

from external.vjepa2.app.vjepa.transforms import make_transforms
from src.experiments.vjepa2_clevrer_naive_world_model_seed239.model import _clean_state_dict, build_model
from src.training.distributed import ExactDistributedSampler
from external.vjepa2.src.utils.checkpoint_loader import robust_checkpoint_loader

from .event_data import TargetClipDataset
from .model import CACHE_PROTOCOL, TARGET_PROTOCOL
from src.core.run_context import apply_cli_defaults, task_context

TOKEN_SHAPE = (8, 256, 1280)


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    return rank, world, local


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_model(config_path, device):
    config = yaml.safe_load(Path(config_path).read_text())
    experiment = config["experiment"]
    meta, model_config, data_config = experiment["meta"], experiment["model"], experiment["data"]
    checkpoint = Path(meta["pretrain_checkpoint"])
    model, _, encoder_message = build_model(
        device, model_config, data_config, checkpoint,
        meta.get("encoder_checkpoint_key", "target_encoder"),
    )
    expected = model.predictor.state_dict()
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        payload = robust_checkpoint_loader(checkpoint, map_location="cpu")
        if "predictor" not in payload:
            raise KeyError(f"official checkpoint does not contain predictor: {checkpoint}")
        raw_predictor = payload["predictor"]
        candidates = [
            raw_predictor,
            {key.removeprefix("module."): value for key, value in raw_predictor.items()},
            _clean_state_dict(raw_predictor),
        ]
        predictor_state = next((candidate for candidate in candidates if set(candidate) == set(expected)), None)
        if predictor_state is None:
            details = []
            for candidate in candidates:
                missing = sorted(set(expected).difference(candidate))
                unexpected = sorted(set(candidate).difference(expected))
                mismatched = sorted(key for key in set(expected).intersection(candidate) if expected[key].shape != candidate[key].shape)
                details.append(f"missing={missing[:4]} unexpected={unexpected[:4]} mismatched={mismatched[:4]}")
            raise RuntimeError("official predictor is incompatible: " + " | ".join(details))
        mismatched = sorted(key for key in expected if expected[key].shape != predictor_state[key].shape)
        if mismatched:
            raise RuntimeError(f"official predictor tensor shapes differ: {mismatched[:8]}")
        model.predictor.load_state_dict(predictor_state, strict=True)
        del payload
    if dist.is_initialized():
        for parameter in model.predictor.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in model.predictor.buffers():
            dist.broadcast(buffer.data, src=0)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint, encoder_message, data_config


def main():
    ctx = task_context()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    apply_cli_defaults(parser, ctx)
    args = parser.parse_args()
    target = torch.load(args.targets, map_location="cpu", weights_only=False)
    if target.get("protocol") != TARGET_PROTOCOL:
        raise ValueError(f"unexpected target protocol: {target.get('protocol')!r}")
    rank, world, local = setup_distributed()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    model, checkpoint, encoder_message, data_config = load_official_model(args.config, device)
    transform = make_transforms(False, (1.0, 1.0), (1.0, 1.0), 0.0, False, False, int(data_config.get("crop_size", 256)))
    dataset = TargetClipDataset(target["records"], transform)
    loader = DataLoader(dataset, sampler=ExactDistributedSampler(dataset, rank, world),
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    manifest_path = args.output_dir / "manifest.json"
    completed_path = args.output_dir / "completed.u8"
    context_path, future_path = args.output_dir / "context.f16", args.output_dir / "future.f16"
    count = len(dataset)
    target_hash = sha256(args.targets) if rank == 0 else None
    checkpoint_hash = sha256(checkpoint) if rank == 0 else None
    if dist.is_initialized():
        hashes = [target_hash, checkpoint_hash]
        dist.broadcast_object_list(hashes, src=0, device=device)
        target_hash, checkpoint_hash = hashes
    cache_id = hashlib.sha256((CACHE_PROTOCOL + target_hash + checkpoint_hash).encode()).hexdigest()
    if rank == 0:
        if manifest_path.exists() and not args.resume:
            raise FileExistsError(f"cache exists; use --resume: {args.output_dir}")
        if not manifest_path.exists():
            for path in (completed_path, context_path, future_path):
                if path.exists():
                    raise RuntimeError(f"cache data exists without manifest: {path}")
            for path in (context_path, future_path):
                with path.open("wb") as handle:
                    handle.truncate(count * int(np.prod(TOKEN_SHAPE)) * np.dtype(np.float16).itemsize)
            completed = np.memmap(completed_path, mode="w+", dtype=np.uint8, shape=(count,))
            completed[:] = 0
            completed.flush()
            manifest_path.write_text(json.dumps({
                "protocol": CACHE_PROTOCOL, "cache_id": cache_id, "complete": False,
                "samples": count, "shape": list(TOKEN_SHAPE), "dtype": "float16",
                "targets": str(args.targets.resolve()), "targets_sha256": target_hash,
                "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
                "encoder_source": "official_checkpoint", "predictor_source": "official_checkpoint",
                "encoder_frozen": True, "predictor_frozen": True,
                "future_source": "official_predictor_rollout_from_current_only", "world_size": world,
            }, indent=2) + "\n")
    if dist.is_initialized():
        dist.barrier()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("cache_id") != cache_id or int(manifest.get("samples", -1)) != count:
        raise ValueError("existing cache is incompatible with targets/checkpoint")
    completed = np.memmap(completed_path, mode="r+", dtype=np.uint8, shape=(count,))
    context_output = np.memmap(context_path, mode="r+", dtype=np.float16, shape=(count, *TOKEN_SHAPE))
    future_output = np.memmap(future_path, mode="r+", dtype=np.float16, shape=(count, *TOKEN_SHAPE))
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, 1):
            indices = batch["index"].tolist()
            pending = [row for row, index in enumerate(indices) if not completed[index]]
            if pending:
                rows = torch.tensor(pending)
                current = batch["current"][rows].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
                    context = model.encode_current(current)
                    future = model.predict_next(context)
                context = context.reshape(len(pending), *TOKEN_SHAPE).half().cpu().numpy()
                future = future.reshape(len(pending), *TOKEN_SHAPE).half().cpu().numpy()
                for output_row, batch_row in enumerate(pending):
                    index = indices[batch_row]
                    context_output[index], future_output[index], completed[index] = context[output_row], future[output_row], 1
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(loader):
                context_output.flush(); future_output.flush(); completed.flush()
                print(f"rank={rank} batch={batch_index}/{len(loader)} local_progress", flush=True)
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        done = int(np.memmap(completed_path, mode="r", dtype=np.uint8, shape=(count,)).sum())
        manifest.update({"complete": done == count, "completed_samples": done, "encoder_load_message": str(encoder_message)})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if done != count:
            raise RuntimeError(f"cache incomplete: {done}/{count}")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
