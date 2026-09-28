#!/usr/bin/env python3
"""Evaluate the structured probe on CLEVRER validation scenes."""

import argparse
import json
from pathlib import Path

import torch

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
    apply_cli_defaults(parser, ctx)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    summary = evaluate_validation(
        load_probe(args.checkpoint, device), args.cache_root, load_targets(args.targets),
        device, args.output_dir, args.batch_size, args.max_scenes,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
