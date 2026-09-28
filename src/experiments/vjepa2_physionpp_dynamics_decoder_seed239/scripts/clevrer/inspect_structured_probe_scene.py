#!/usr/bin/env python3
"""Create a matched structured-probe report and optional video overlay."""

import argparse
import json
import random
from pathlib import Path

import torch

from .structured_probe_evaluation import (
    build_scene_report,
    load_probe,
    load_targets,
    predict_scene,
    render_scene,
    write_scene_report,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/shared/datasets/CLEVRER"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-id", type=int)
    parser.add_argument("--window-start", type=int)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-video", action="store_true")
    args = parser.parse_args()
    target = load_targets(args.targets)
    scene_id = args.scene_id
    if scene_id is None:
        scene_id = random.Random(args.seed).choice(target["scene_id"].tolist())
    window_start = args.window_start
    if window_start is None:
        candidates = [int(value) for value in target["window_start"][target["scene_id"].eq(scene_id)].tolist()]
        if not candidates:
            raise ValueError(f"scene {scene_id} has no target windows")
        window_start = candidates[0]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result = predict_scene(
        load_probe(args.checkpoint, device), args.cache_root, target, scene_id, window_start, device
    )
    _, batch, outputs, _, target_to_pred, pair_lookup = result
    proposal_path = args.dataset_root / "processed_proposals" / f"sim_{scene_id:05d}.json"
    video_path = args.dataset_root / "videos" / "val" / f"video_{scene_id:05d}.mp4"
    with proposal_path.open(encoding="utf-8") as handle:
        annotation = json.load(handle)
    report = build_scene_report(
        scene_id, window_start, annotation, batch, outputs, target_to_pred, pair_lookup,
        video_path, proposal_path,
    )
    write_scene_report(report, args.output_dir)
    if args.render_video:
        render_scene(
            video_path, proposal_path, batch, outputs, target_to_pred, pair_lookup,
            args.output_dir,
        )
    print(json.dumps({"scene_id": scene_id, "window_start": window_start, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
