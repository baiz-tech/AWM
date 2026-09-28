#!/usr/bin/env python3
"""Evaluate the structured probe on CLEVRER validation scenes."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from .structured_probe_evaluation import evaluate_validation, load_probe, load_targets
from src.core.run_context import apply_cli_defaults, task_context


def main():
    ctx = task_context()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--distributed", action="store_true")
    apply_cli_defaults(parser, ctx)
    args = parser.parse_args()
    if args.distributed:
        rank = int(__import__("os").environ["RANK"])
        world_size = int(__import__("os").environ["WORLD_SIZE"])
        local_rank = int(__import__("os").environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device("cuda", local_rank)
    else:
        rank, world_size = 0, 1
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[rank {rank}/{world_size}] loading checkpoint={args.checkpoint}", flush=True)
    print(f"[rank {rank}/{world_size}] cache={args.cache_root} targets={args.targets}", flush=True)
    summary = evaluate_validation(
        load_probe(args.checkpoint, device), args.cache_root, load_targets(args.targets),
        device, args.output_dir / (f"rank{rank:05d}" if args.distributed else ""),
        args.batch_size, args.max_scenes, rank, world_size,
    )
    print(json.dumps(summary, indent=2), flush=True)
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
