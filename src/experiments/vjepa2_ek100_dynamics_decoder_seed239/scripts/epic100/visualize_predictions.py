#!/usr/bin/env python3
"""Render trained decoder predictions and VISOR GT on a validation clip."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from decord import VideoReader, cpu

from .model import CHECKPOINT_PROTOCOL, TARGET_PROTOCOL, decoder_loss
from .train_decoder import TOKEN_SHAPE
from .visualize_targets import COLORS, category_names, outlined_text


def draw_box(frame, state, color, label, dashed=False):
    height, width = frame.shape[:2]
    center_x, center_y, _, box_width, box_height = map(float, state[:5])
    x0 = int(round((center_x - box_width / 2) * width)); y0 = int(round((center_y - box_height / 2) * height))
    x1 = int(round((center_x + box_width / 2) * width)); y1 = int(round((center_y + box_height / 2) * height))
    x0, x1 = np.clip((x0, x1), 0, width - 1); y0, y1 = np.clip((y0, y1), 0, height - 1)
    if dashed:
        for start in range(x0, x1, 8):
            cv2.line(frame, (start, y0), (min(start + 4, x1), y0), color, 2, cv2.LINE_AA)
            cv2.line(frame, (start, y1), (min(start + 4, x1), y1), color, 2, cv2.LINE_AA)
        for start in range(y0, y1, 8):
            cv2.line(frame, (x0, start), (x0, min(start + 4, y1)), color, 2, cv2.LINE_AA)
            cv2.line(frame, (x1, start), (x1, min(start + 4, y1)), color, 2, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    cv2.circle(frame, (int(center_x * width), int(center_y * height)), 3, color, -1, cv2.LINE_AA)
    outlined_text(frame, label, (x0, max(74, y0 - 5)), 0.38, color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--display-fps", type=float, default=4.0)
    parser.add_argument("--hand-only", action="store_true", help="只绘制 VISOR 的 left/right hand 对象")
    parser.add_argument("--tap-hand-only", action="store_true", help="只绘制 tap、left hand、right hand 对象")
    args = parser.parse_args()
    target_path = args.run_dir / "targets" / f"{args.split}.pt"
    cache_dir = args.run_dir / "cache" / args.split
    checkpoint_path = args.run_dir / "checkpoints" / "best.pt"
    payload = torch.load(target_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != TARGET_PROTOCOL or checkpoint.get("protocol") != CHECKPOINT_PROTOCOL:
        raise ValueError("target/checkpoint protocol mismatch")
    sample = int(np.random.default_rng(args.seed).integers(int(payload["samples"])))
    record = payload["records"][sample]
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError(f"latent cache is incomplete: {cache_dir}")
    count = int(manifest["samples"])
    context_mem = np.memmap(cache_dir / "context.f16", mode="r", dtype=np.float16, shape=(count, *TOKEN_SHAPE))
    future_mem = np.memmap(cache_dir / "future.f16", mode="r", dtype=np.float16, shape=(count, *TOKEN_SHAPE))
    from .model import Epic100Decoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Epic100Decoder(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    batch = {"context": torch.from_numpy(np.array(context_mem[sample], copy=True)).unsqueeze(0).to(device), "future": torch.from_numpy(np.array(future_mem[sample], copy=True)).unsqueeze(0).to(device)}
    for key in ("object_present", "category", "state", "state_valid", "pair_distance", "pair_valid", "verb_label", "noun_label", "action_label"):
        batch[key] = payload[key][sample:sample + 1].to(device)
    with torch.no_grad():
        outputs = model(batch["context"], batch["future"])
        _, losses, assignment = decoder_loss(outputs, batch)
    assignment = assignment[0].cpu().tolist()
    predicted_state = outputs["trajectory_2d"][0].float().cpu().numpy()
    predicted_category = outputs["category_logits"][0].argmax(-1).cpu().tolist()
    names = category_names(Path("/data/shared/datasets/EPIC-KITCHENS-VISOR-processed_v1/EPIC_100_noun_classes_v2.csv"))
    reader = VideoReader(record["video_path"], num_threads=1, ctx=cpu(0))
    indices = [int(v) for v in record["current_indices"] + record["future_indices"]]
    frames = reader.get_batch(indices).asnumpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gt_output = args.output.parent / "current_future_gt.mp4"
    pred_output = args.output.parent / "current_future_pred.mp4"
    combo_output = args.output.parent / "current_future_gt_pred.mp4"
    gt_temporary = gt_output.with_suffix(".mp4v.mp4")
    pred_temporary = pred_output.with_suffix(".mp4v.mp4")
    combo_temporary = combo_output.with_suffix(".mp4v.mp4")
    height, width = frames.shape[1:3]
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    gt_writer = cv2.VideoWriter(str(gt_temporary), codec, args.display_fps, (width, height))
    pred_writer = cv2.VideoWriter(str(pred_temporary), codec, args.display_fps, (width, height))
    combo_writer = cv2.VideoWriter(str(combo_temporary), codec, args.display_fps, (width, height))
    if not gt_writer.isOpened() or not pred_writer.isOpened() or not combo_writer.isOpened():
        raise RuntimeError("failed to open GT/prediction/combined video writers")
    for frame_index, source in enumerate(frames):
        base = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        gt_frame, pred_frame, combo_frame = base.copy(), base.copy(), base.copy()
        is_future = frame_index >= 16; step = frame_index - 16 if is_future else frame_index
        for frame, label in (
            (gt_frame, "VISOR GT: green dashed"),
            (pred_frame, "DECODER PRED: magenta solid"),
            (combo_frame, "GT dashed + PRED solid"),
        ):
            phase = f"FUTURE: {label}" if is_future else "CURRENT INPUT"
            cv2.rectangle(frame, (0, 0), (width, 67), (0, 0, 0), -1)
            outlined_text(frame, f"{phase}  {step + 1:02d}/16  source_frame={indices[frame_index]}", (8, 20), 0.44)
            outlined_text(frame, f"verb={record.get('verb', record['verb_class'])}  noun={record.get('noun', record['noun_class'])}  sample={sample} seed={args.seed}", (8, 42), 0.40)
        if is_future:
            for slot in range(8):
                target_slot = assignment[slot]
                target_name = ""
                if target_slot >= 0:
                    object_names = record.get("object_names", [])
                    if target_slot < len(object_names):
                        target_name = str(object_names[target_slot]).strip().lower()
                if args.hand_only and target_name not in {"left hand", "right hand"}:
                    continue
                if args.tap_hand_only and target_name not in {"tap", "left hand", "right hand"}:
                    continue
                if target_slot >= 0 and bool(payload["state_valid"][sample, target_slot, step]):
                    label = f"GT s{slot} {record.get('object_names', [''])[target_slot]}"
                    draw_box(gt_frame, payload["state"][sample, target_slot, step].numpy(), (60, 220, 60), label, dashed=True)
                    draw_box(combo_frame, payload["state"][sample, target_slot, step].numpy(), (60, 220, 60), label, dashed=True)
                if bool(torch.sigmoid(outputs["presence_logits"][0, slot]) > 0.5):
                    label = f"PRED s{slot} {names.get(predicted_category[slot], str(predicted_category[slot]))}"
                    draw_box(pred_frame, predicted_state[slot, step], (220, 60, 220), label)
                    draw_box(combo_frame, predicted_state[slot, step], (220, 60, 220), label)
        gt_writer.write(gt_frame); pred_writer.write(pred_frame); combo_writer.write(combo_frame)
    gt_writer.release(); pred_writer.release(); combo_writer.release()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for temporary, output in ((gt_temporary, gt_output), (pred_temporary, pred_output), (combo_temporary, combo_output)):
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(temporary), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True)
        temporary.unlink()
    metadata = {"protocol": "epic100_decoder_prediction_visualization_v3", "run_dir": str(args.run_dir.resolve()), "checkpoint": str(checkpoint_path.resolve()), "split": args.split, "sample_index": sample, "seed": args.seed, "narration_id": record["narration_id"], "hand_only": args.hand_only, "tap_hand_only": args.tap_hand_only, "gt_output": str(gt_output.resolve()), "prediction_output": str(pred_output.resolve()), "combined_output": str(combo_output.resolve()), "losses": {key: float(value) for key, value in losses.items()}}
    (args.output.parent / "visualization.json").write_text(json.dumps(metadata, indent=2) + "\n"); print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
