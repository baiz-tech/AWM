#!/usr/bin/env python3
"""Export frozen decoder/probe outputs for every scene latent cache record."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .model import CLEVRERDecoder


def distributed_info():
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--mode", choices=("nonpredictive", "predictive", "predictive_32_159"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rank, world_size = distributed_info()
    device = torch.device(args.device or (f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("protocol") != "clevrer_fullpatch_dynamics_decoder_v1":
        raise ValueError(f"unexpected checkpoint protocol: {payload.get('protocol')!r}")
    model = CLEVRERDecoder(**payload["model_config"]).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    source = args.cache_root / args.split
    output = args.output_root / args.mode / args.split
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("scene_*_window_*.pt"))
    files = files[rank::world_size]
    fields = None
    with torch.inference_mode():
        for offset in range(0, len(files), args.batch_size):
            batch_files = files[offset : offset + args.batch_size]
            records = [torch.load(path, map_location="cpu", weights_only=True) for path in batch_files]
            tokens = torch.stack([record.get("tokens", torch.cat([record["context_tokens"], record["future_tokens"]], dim=0)) for record in records]).to(device)
            source_ids = torch.stack([record.get("source_ids", torch.zeros(16, dtype=torch.long)) for record in records]).to(device)
            input_tokens = tokens.float()
            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
                dtype=torch.bfloat16 if device.type == "cuda" else None,
            ):
                outputs = model(input_tokens, source_ids)
            if fields is None:
                fields = tuple(name for name, value in outputs.items() if torch.is_tensor(value))
            for index, path in enumerate(batch_files):
                target = output / path.name
                if args.resume and target.is_file():
                    continue
                record = {
                    "protocol": "clevrer_scene_decoder_outputs_v1",
                    "scene_id": int(records[index]["scene_id"]),
                    "window_start": int(records[index]["window_start"]),
                    "latent_mode": args.mode,
                    "tokens": tokens[index].detach().float().cpu(),
                    "source_ids": source_ids[index].detach().cpu(),
                }
                record.update({name: outputs[name][index].detach().float().cpu() for name in fields})
                torch.save(record, target)
    if world_size > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("gloo")
        dist.barrier()
    if rank == 0:
        manifest = {
            "protocol": "clevrer_scene_decoder_outputs_v1",
            "mode": args.mode,
            "split": args.split,
            "samples": len(list(output.glob("scene_*_window_*.pt"))),
            "checkpoint": str(args.checkpoint.resolve()),
            "fields": list(fields or ()),
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
