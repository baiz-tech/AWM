#!/usr/bin/env python3
"""Render current frames followed by future frames with training GT overlays."""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from decord import VideoReader, cpu

from .model import TARGET_PROTOCOL

COLORS = (
    (40, 210, 255), (80, 220, 80), (255, 120, 60), (220, 80, 220),
    (80, 180, 255), (255, 180, 40), (180, 255, 80), (255, 80, 130),
)


def category_names(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["id"]): row["key"] for row in csv.DictReader(handle)}


def choose_sample(payload, sample_index, narration_id, seed):
    if narration_id is not None:
        matches = [index for index, record in enumerate(payload["records"])
                   if record["narration_id"] == narration_id]
        if not matches:
            raise ValueError(f"narration ID not found in targets: {narration_id}")
        return matches[0]
    if sample_index is None:
        return random.Random(seed).randrange(int(payload["samples"]))
    if not 0 <= sample_index < int(payload["samples"]):
        raise IndexError(f"sample index {sample_index} outside [0,{payload['samples']})")
    return sample_index


def outlined_text(frame, text, position, scale=0.48, color=(255, 255, 255)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_future_gt(frame, payload, sample, step, names):
    height, width = frame.shape[:2]
    for slot in range(payload["object_present"].shape[1]):
        if not bool(payload["object_present"][sample, slot]) or not bool(payload["state_valid"][sample, slot, step]):
            continue
        state = payload["state"][sample, slot, step].float().numpy()
        center_x, center_y, _, box_width, box_height = state[:5]
        x0 = int(round((center_x - box_width / 2) * width))
        y0 = int(round((center_y - box_height / 2) * height))
        x1 = int(round((center_x + box_width / 2) * width))
        y1 = int(round((center_y + box_height / 2) * height))
        x0, x1 = np.clip((x0, x1), 0, width - 1)
        y0, y1 = np.clip((y0, y1), 0, height - 1)
        color = COLORS[slot % len(COLORS)]
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        cv2.circle(frame, (int(center_x * width), int(center_y * height)), 3, color, -1, cv2.LINE_AA)
        category = int(payload["category"][sample, slot])
        object_name = payload["records"][sample].get("object_names", [])[slot]
        label = f"s{slot} {object_name} [{names.get(category, str(category))}]"
        label_y = max(34, y0 - 5)
        outlined_text(frame, label, (x0, label_y), 0.42, color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--noun-classes", type=Path, default=Path(
        "/data/shared/datasets/EPIC-KITCHENS-VISOR-processed_v1/EPIC_100_noun_classes_v2.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--narration-id")
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--display-fps", type=float, default=4.0)
    args = parser.parse_args()
    payload = torch.load(args.targets, map_location="cpu", weights_only=False)
    if payload.get("protocol") != TARGET_PROTOCOL:
        raise ValueError(f"unexpected target protocol: {payload.get('protocol')!r}")
    sample = choose_sample(payload, args.sample_index, args.narration_id, args.seed)
    record = payload["records"][sample]
    reader = VideoReader(record["video_path"], num_threads=1, ctx=cpu(0))
    current_indices = [int(value) for value in record["current_indices"]]
    future_indices = [int(value) for value in record["future_indices"]]
    frames = reader.get_batch(current_indices + future_indices).asnumpy()
    names = category_names(args.noun_classes)
    height, width = frames.shape[1:3]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".mp4v.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), args.display_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {temporary}")
    verb = record.get("verb", str(record["verb_class"]))
    noun = record.get("noun", str(record["noun_class"]))
    action_label = int(payload["action_label"][sample])
    for output_step, source in enumerate(frames):
        frame = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        future = output_step >= len(current_indices)
        phase_step = output_step - len(current_indices) if future else output_step
        phase = "FUTURE + TRAINING GT" if future else "CURRENT INPUT"
        cv2.rectangle(frame, (0, 0), (width, 58), (0, 0, 0), -1)
        outlined_text(frame, f"{phase}  {phase_step + 1:02d}/16  source_frame={current_indices[phase_step] if not future else future_indices[phase_step]}", (8, 20), 0.48)
        outlined_text(frame, f"objective: verb={verb} ({record['verb_class']})  noun={noun} ({record['noun_class']})  action_label={action_label}", (8, 43), 0.43)
        if future:
            draw_future_gt(frame, payload, sample, phase_step, names)
        writer.write(frame)
    writer.release()
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(temporary),
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ], check=True)
    temporary.unlink()
    manifest = {
        "protocol": "epic100_current_future_gt_visualization_v1",
        "targets": str(args.targets.resolve()), "sample_index": sample,
        "seed": args.seed,
        "narration_id": record["narration_id"], "video": record["video_path"],
        "current_indices": current_indices, "future_indices": future_indices,
        "verb_class": record["verb_class"], "noun_class": record["noun_class"],
        "action_label": action_label, "output": str(args.output.resolve()),
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
