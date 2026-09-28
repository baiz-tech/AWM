#!/usr/bin/env python3
"""Validate a node-local CLEVRER decoder training copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="check every target/cache mapping")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.data_root.resolve()
    modes = ("nonpredictive", "predictive")
    expected_sources = {
        "nonpredictive": [0] * 16,
        "predictive": [0] * 12 + [1] * 4,
    }
    summary: dict[str, object] = {"data_root": str(root), "splits": {}}
    for split in ("train", "validation"):
        target_path = root / "targets" / f"targets_{split}.pt"
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        targets = torch.load(target_path, map_location="cpu", weights_only=False)
        target_keys = [
            (int(scene_id), int(window_start))
            for scene_id, window_start in zip(targets["scene_id"], targets["window_start"])
        ]
        split_summary = {"targets": len(target_keys), "modes": {}}
        for mode in modes:
            cache_dir = root / f"latents_{mode}" / split
            manifest_path = cache_dir / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("mode") != mode or manifest.get("split") != split:
                raise ValueError(
                    f"{manifest_path}: mode/split mismatch: "
                    f"{manifest.get('mode')}/{manifest.get('split')}"
                )
            files = sorted(cache_dir.glob("scene_*_window_*.pt"))
            if int(manifest.get("samples", -1)) != len(files):
                raise ValueError(f"{mode}/{split}: manifest samples do not match file count")
            check_keys = target_keys if args.full else target_keys[: min(64, len(target_keys))]
            missing = [
                key for key in check_keys
                if not (cache_dir / f"scene_{key[0]:05d}_window_{key[1]:03d}.pt").is_file()
            ]
            if missing:
                raise FileNotFoundError(f"{mode}/{split}: missing target caches: {missing[:8]}")
            if not files:
                raise ValueError(f"{mode}/{split}: cache is empty")
            record = torch.load(files[0], map_location="cpu", weights_only=True)
            tokens = record.get(
                "tokens", torch.cat([record["context_tokens"], record["future_tokens"]], dim=0)
            )
            source_ids = record.get("source_ids", torch.zeros(16, dtype=torch.long))
            if tuple(tokens.shape) != (16, 256, 1280):
                raise ValueError(f"{files[0]} tokens shape={tuple(tokens.shape)}")
            if source_ids.tolist() != expected_sources[mode]:
                raise ValueError(f"{files[0]} source_ids={source_ids.tolist()}")
            split_summary["modes"][mode] = {
                "files": len(files),
                "manifest_mode": manifest.get("mode"),
                "sample": files[0].name,
            }
        summary["splits"][split] = split_summary
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
