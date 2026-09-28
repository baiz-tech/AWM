"""Matched metrics, reports, and rendering for the structured CLEVRER probe."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch

try:
    from .model import COLORS, MATERIALS, SHAPES, CLEVRERDecoder, match_objects
except ImportError:
    from model import COLORS, MATERIALS, SHAPES, CLEVRERDecoder, match_objects


TARGET_KEYS = (
    "object_present", "color", "material", "shape", "state", "state_valid",
    "pair_distance", "pair_valid", "contact", "contact_valid", "first_contact_class",
)


def load_probe(checkpoint: Path, device: torch.device) -> CLEVRERDecoder:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("protocol") != "clevrer_fullpatch_dynamics_decoder_v1":
        raise ValueError(f"unexpected decoder protocol: {payload.get('protocol')!r}")
    model = CLEVRERDecoder(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


def load_targets(path: Path) -> dict[str, torch.Tensor]:
    target = torch.load(path, map_location="cpu", weights_only=False)
    if target.get("protocol") != "clevrer_multiwindow_structured_object_set_targets_v1":
        raise ValueError(f"unexpected target protocol: {target.get('protocol')!r}")
    return target


def load_scene_batch(
    cache_root: Path,
    target: dict[str, torch.Tensor],
    indices: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    contexts, futures = [], []
    for index in indices:
        scene_id = int(target["scene_id"][index])
        window_start = int(target["window_start"][index])
        record = torch.load(
            cache_root / "validation" / f"scene_{scene_id:05d}_window_{window_start:03d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if int(record["scene_id"]) != scene_id or int(record["window_start"]) != window_start:
            raise ValueError(f"cache/target scene mismatch for scene {scene_id}")
        contexts.append(record["context_tokens"])
        futures.append(record["future_tokens"])
    batch = {
        "scene_id": target["scene_id"][indices],
        "window_start": target["window_start"][indices],
        "context": torch.stack(contexts),
        "future": torch.stack(futures),
        **{key: target[key][indices] for key in TARGET_KEYS},
    }
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def predict_batch(model, batch, use_amp=True):
    enabled = bool(use_amp and batch["context"].device.type == "cuda")
    with torch.amp.autocast(
        device_type=batch["context"].device.type,
        enabled=enabled,
        dtype=torch.bfloat16 if enabled else None,
    ):
        outputs = model(batch["context"], batch["future"])
    assignment = match_objects(outputs, batch)
    return (
        {key: value.detach().float().cpu() for key, value in outputs.items()},
        assignment.cpu(),
    )


def aligned_scene(outputs, assignment, batch, row):
    object_count = int(batch["object_present"][row].sum())
    target_to_pred = torch.full((object_count,), -1, dtype=torch.long)
    for predicted_slot, target_slot in enumerate(assignment[row].tolist()):
        if 0 <= target_slot < object_count:
            target_to_pred[target_slot] = predicted_slot
    if target_to_pred.lt(0).any():
        raise ValueError(f"incomplete object assignment: {target_to_pred.tolist()}")

    pair_lookup = {}
    pair_indices = list(model_pair_indices())
    for target_first in range(object_count):
        for target_second in range(target_first + 1, object_count):
            predicted = sorted(
                (int(target_to_pred[target_first]), int(target_to_pred[target_second]))
            )
            pair_lookup[(target_first, target_second)] = pair_indices.index(tuple(predicted))
    return target_to_pred, pair_lookup


def model_pair_indices():
    for first in range(6):
        for second in range(first + 1, 6):
            yield first, second


def _binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    logits, labels = logits.flatten().float(), labels.flatten().bool()
    predicted = logits.sigmoid().ge(0.5)
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    fn = int((~predicted & labels).sum())
    tn = int((~predicted & ~labels).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {
        "samples": int(labels.numel()),
        "positives": int(labels.sum()),
        "accuracy_at_0_5": (tp + tn) / max(labels.numel(), 1),
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "f1_at_0_5": 2 * precision * recall / max(precision + recall, 1e-12),
        "auroc": None,
        "average_precision": None,
    }
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives and negatives:
        order = torch.argsort(logits, descending=True)
        sorted_labels = labels[order].float()
        _, group_counts = torch.unique_consecutive(logits[order], return_counts=True)
        group_positives = torch.stack(
            [values.sum() for values in torch.split(sorted_labels, group_counts.tolist())]
        )
        group_total = group_counts.to(torch.float32)
        true_positive = group_positives.cumsum(0)
        false_positive = (group_total - group_positives).cumsum(0)
        tpr = torch.cat([torch.zeros(1), true_positive / positives])
        fpr = torch.cat([torch.zeros(1), false_positive / negatives])
        result["auroc"] = float(torch.trapezoid(tpr, fpr))
        precision_curve = true_positive / group_total.cumsum(0)
        result["average_precision"] = float(
            (precision_curve * group_positives).sum() / positives
        )
    return result


def _safe_ratio(total, count):
    return None if not count else float(total / count)


def evaluate_validation(
    model,
    cache_root,
    target,
    device,
    output_dir,
    batch_size=4,
    max_scenes=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_count = len(target["scene_id"])
    if max_scenes is not None:
        scene_count = min(scene_count, int(max_scenes))
    totals = {
        "center_abs": 0.0, "center_count": 0,
        "geometry_abs": 0.0, "geometry_count": 0,
        "velocity_abs": 0.0, "velocity_count": 0,
        "log_area_abs": 0.0, "log_area_count": 0,
        "distance_abs": 0.0, "distance_count": 0,
        "objects": 0, "object_count_correct": 0,
        "presence_tp": 0, "presence_fp": 0, "presence_fn": 0,
        "color_correct": 0, "material_correct": 0, "shape_correct": 0,
        "first_correct": 0, "first_count": 0,
        "first_collision_correct": 0, "first_collision_count": 0,
        "first_no_contact_correct": 0, "first_no_contact_count": 0,
    }
    contact_logits, contact_labels, per_scene = [], [], []
    pair_indices = list(model_pair_indices())
    for start in range(0, scene_count, batch_size):
        indices = list(range(start, min(start + batch_size, scene_count)))
        batch = load_scene_batch(cache_root, target, indices, device)
        outputs, assignment = predict_batch(model, batch)
        cpu_batch = {key: value.cpu() for key, value in batch.items()}
        for row, index in enumerate(indices):
            scene_id = int(cpu_batch["scene_id"][row])
            target_to_pred, pair_lookup = aligned_scene(outputs, assignment, cpu_batch, row)
            count = len(target_to_pred)
            predicted_presence = outputs["presence_logits"][row].sigmoid().ge(0.5)
            matched_presence = assignment[row].ge(0)
            totals["presence_tp"] += int((predicted_presence & matched_presence).sum())
            totals["presence_fp"] += int((predicted_presence & ~matched_presence).sum())
            totals["presence_fn"] += int((~predicted_presence & matched_presence).sum())
            totals["object_count_correct"] += int(int(predicted_presence.sum()) == count)
            totals["objects"] += count

            pred_color = outputs["color_logits"][row, target_to_pred].argmax(-1)
            pred_material = outputs["material_logits"][row, target_to_pred].argmax(-1)
            pred_shape = outputs["shape_logits"][row, target_to_pred].argmax(-1)
            totals["color_correct"] += int(
                pred_color.eq(cpu_batch["color"][row, :count]).sum()
            )
            totals["material_correct"] += int(
                pred_material.eq(cpu_batch["material"][row, :count]).sum()
            )
            totals["shape_correct"] += int(
                pred_shape.eq(cpu_batch["shape"][row, :count]).sum()
            )

            predicted_state = outputs["trajectory_2d"][row, target_to_pred]
            target_state = cpu_batch["state"][row, :count]
            state_valid = cpu_batch["state_valid"][row, :count].bool()
            state_error = (predicted_state - target_state).abs()
            for name, channels in (
                ("center", slice(0, 2)),
                ("geometry", slice(2, 5)),
                ("velocity", slice(5, 7)),
                ("log_area", slice(7, 8)),
            ):
                values = state_error[..., channels]
                mask = state_valid.unsqueeze(-1).expand_as(values)
                totals[f"{name}_abs"] += float(values[mask].sum())
                totals[f"{name}_count"] += int(mask.sum())

            scene_distance_sum, scene_distance_count = 0.0, 0
            scene_contact_logits, scene_contact_labels = [], []
            first_correct, first_count = 0, 0
            for (target_first, target_second), predicted_pair in pair_lookup.items():
                valid = cpu_batch["pair_valid"][row, target_first, target_second].bool()
                distance_error = (
                    outputs["pair_distance_2d"][row, predicted_pair]
                    - cpu_batch["pair_distance"][row, target_first, target_second]
                ).abs()
                totals["distance_abs"] += float(distance_error[valid].sum())
                totals["distance_count"] += int(valid.sum())
                scene_distance_sum += float(distance_error[valid].sum())
                scene_distance_count += int(valid.sum())

                contact_valid = cpu_batch["contact_valid"][row, target_first, target_second].bool()
                logits = outputs["contact_gt_event"][row, predicted_pair][contact_valid]
                labels = cpu_batch["contact"][row, target_first, target_second][contact_valid]
                contact_logits.append(logits)
                contact_labels.append(labels)
                scene_contact_logits.append(logits)
                scene_contact_labels.append(labels)

                target_class = int(
                    cpu_batch["first_contact_class"][row, target_first, target_second]
                )
                predicted_class = int(
                    outputs["first_contact_logits"][row, predicted_pair].argmax()
                )
                correct = int(target_class == predicted_class)
                totals["first_correct"] += correct
                totals["first_count"] += 1
                first_correct += correct
                first_count += 1
                if target_class == 16:
                    totals["first_no_contact_correct"] += correct
                    totals["first_no_contact_count"] += 1
                else:
                    totals["first_collision_correct"] += correct
                    totals["first_collision_count"] += 1

            valid_state_mask = state_valid.unsqueeze(-1).expand_as(state_error[..., :2])
            per_scene.append({
                "scene_id": scene_id,
                "window_start": int(cpu_batch["window_start"][row]),
                "objects": count,
                "predicted_objects_at_0_5": int(predicted_presence.sum()),
                "center_mae": float(state_error[..., :2][valid_state_mask].mean()),
                "pair_distance_mae": _safe_ratio(scene_distance_sum, scene_distance_count),
                "first_contact_accuracy": _safe_ratio(first_correct, first_count),
                "contact_positives": int(torch.cat(scene_contact_labels).sum()),
            })

    precision = totals["presence_tp"] / max(
        totals["presence_tp"] + totals["presence_fp"], 1
    )
    recall = totals["presence_tp"] / max(
        totals["presence_tp"] + totals["presence_fn"], 1
    )
    summary = {
        "checkpoint_selection": "best validation total loss",
        "scenes": scene_count,
        "object_presence": {
            "count_accuracy": totals["object_count_correct"] / scene_count,
            "precision_at_0_5": precision,
            "recall_at_0_5": recall,
            "f1_at_0_5": 2 * precision * recall / max(precision + recall, 1e-12),
        },
        "attributes": {
            "objects": totals["objects"],
            "color_accuracy": totals["color_correct"] / totals["objects"],
            "material_accuracy": totals["material_correct"] / totals["objects"],
            "shape_accuracy": totals["shape_correct"] / totals["objects"],
        },
        "trajectory": {
            name + "_mae": _safe_ratio(totals[f"{name}_abs"], totals[f"{name}_count"])
            for name in ("center", "geometry", "velocity", "log_area")
        },
        "pair_distance": {
            "mae": _safe_ratio(totals["distance_abs"], totals["distance_count"])
        },
        "contact": _binary_metrics(torch.cat(contact_logits), torch.cat(contact_labels)),
        "first_contact": {
            "accuracy": _safe_ratio(totals["first_correct"], totals["first_count"]),
            "collision_pair_accuracy": _safe_ratio(
                totals["first_collision_correct"], totals["first_collision_count"]
            ),
            "no_contact_pair_accuracy": _safe_ratio(
                totals["first_no_contact_correct"], totals["first_no_contact_count"]
            ),
            "pairs": totals["first_count"],
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_scene.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_scene[0].keys())
        writer.writeheader()
        writer.writerows(per_scene)
    return summary


def predict_scene(model, cache_root, target, scene_id, window_start, device):
    positions = torch.nonzero(
        target["scene_id"].eq(int(scene_id)) & target["window_start"].eq(int(window_start)),
        as_tuple=False,
    ).flatten()
    if len(positions) != 1:
        raise ValueError(f"validation sample scene={scene_id} window_start={window_start} is unavailable")
    index = int(positions.item())
    batch = load_scene_batch(cache_root, target, [index], device)
    outputs, assignment = predict_batch(model, batch)
    cpu_batch = {key: value.cpu() for key, value in batch.items()}
    target_to_pred, pair_lookup = aligned_scene(outputs, assignment, cpu_batch, 0)
    return index, cpu_batch, outputs, assignment, target_to_pred, pair_lookup


def build_scene_report(
    scene_id,
    window_start,
    annotation,
    batch,
    outputs,
    target_to_pred,
    pair_lookup,
    video_path,
    proposal_path,
):
    objects = sorted(annotation["ground_truth"]["objects"], key=lambda item: int(item["id"]))
    report_objects = []
    for target_slot, obj in enumerate(objects):
        predicted_slot = int(target_to_pred[target_slot])
        valid = batch["state_valid"][0, target_slot].bool()
        error = (
            outputs["trajectory_2d"][0, predicted_slot]
            - batch["state"][0, target_slot]
        ).abs()
        report_objects.append({
            "target_slot": target_slot,
            "object_id": int(obj["id"]),
            "ground_truth": {
                "color": obj["color"], "material": obj["material"], "shape": obj["shape"]
            },
            "predicted_slot": predicted_slot,
            "presence_probability": float(outputs["presence_logits"][0, predicted_slot].sigmoid()),
            "predicted": {
                "color": COLORS[int(outputs["color_logits"][0, predicted_slot].argmax())],
                "material": MATERIALS[int(outputs["material_logits"][0, predicted_slot].argmax())],
                "shape": SHAPES[int(outputs["shape_logits"][0, predicted_slot].argmax())],
            },
            "center_mae": float(error[..., :2][valid].mean()),
            "geometry_mae": float(error[..., 2:5][valid].mean()),
            "velocity_mae": float(error[..., 5:7][valid].mean()),
        })
    pairs = []
    for (first, second), predicted_pair in pair_lookup.items():
        target_class = int(batch["first_contact_class"][0, first, second])
        predicted_class = int(outputs["first_contact_logits"][0, predicted_pair].argmax())
        pairs.append({
            "object_ids": [int(objects[first]["id"]), int(objects[second]["id"])],
            "predicted_pair_slot": predicted_pair,
            "ground_truth_first_contact": "no_contact" if target_class == 16 else target_class,
            "predicted_first_contact": "no_contact" if predicted_class == 16 else predicted_class,
            "prediction_correct": target_class == predicted_class,
        })
    return {
        "scene_id": int(scene_id),
        "window_start": int(window_start),
        "video": str(video_path),
        "proposal_annotation": str(proposal_path),
        "objects": report_objects,
        "pairs": pairs,
        "current_raw_frames": list(range(window_start, window_start + 32, 2)),
        "future_raw_frames": list(range(window_start + 32, window_start + 64, 2)),
    }


def write_scene_report(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scene_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Dynamics decoder sample: scene {report['scene_id']} window {report['window_start']}",
        "",
        "## Matched objects",
        "",
        "| GT object | Ground truth | Predicted slot | Prediction | Presence | Center MAE |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for item in report["objects"]:
        gt = item["ground_truth"]
        pred = item["predicted"]
        lines.append(
            f"| {item['object_id']} | {gt['color']} {gt['material']} {gt['shape']} "
            f"| {item['predicted_slot']} | {pred['color']} {pred['material']} {pred['shape']} "
            f"| {item['presence_probability']:.3f} | {item['center_mae']:.5f} |"
        )
    lines.extend([
        "", "## First contact", "",
        "| Object IDs | GT | Prediction | Correct |",
        "| --- | --- | --- | --- |",
    ])
    for pair in report["pairs"]:
        lines.append(
            f"| {pair['object_ids']} | {pair['ground_truth_first_contact']} "
            f"| {pair['predicted_first_contact']} | {pair['prediction_correct']} |"
        )
    (output_dir / "scene_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def decode_rle(rle):
    height, width = (int(value) for value in rle["size"])
    encoded, counts, position = rle["counts"], [], 0
    while position < len(encoded):
        value, shift = 0, 0
        while True:
            code = ord(encoded[position]) - 48
            position += 1
            value |= (code & 31) << (5 * shift)
            if not code & 32:
                if code & 16:
                    value |= -1 << (5 * (shift + 1))
                break
            shift += 1
        counts.append(value + counts[-2] if len(counts) > 2 else value)
    flat, offset = np.zeros(height * width, dtype=np.uint8), 0
    for index, run in enumerate(counts):
        if run < 0 or offset + run > flat.size:
            raise ValueError("invalid CLEVRER proposal RLE")
        if index % 2:
            flat[offset : offset + run] = 1
        offset += run
    if offset != flat.size:
        raise ValueError("CLEVRER proposal RLE does not cover its declared size")
    return flat.reshape((height, width), order="F").astype(bool)


def draw_box(frame, state, color, thickness, label=None):
    height, width = frame.shape[:2]
    center_x, center_y, _, box_width, box_height = state[:5]
    x0 = int((float(center_x) - float(box_width) / 2) * width)
    x1 = int((float(center_x) + float(box_width) / 2) * width)
    y0 = int((float(center_y) - float(box_height) / 2) * height)
    y1 = int((float(center_y) + float(box_height) / 2) * height)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)
    if label:
        cv2.putText(
            frame, label, (x0, max(12, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
            0.35, color, 1, cv2.LINE_AA,
        )


def encode_h264(intermediate, output):
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i",
        str(intermediate), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ], check=True)
    intermediate.unlink()


def render_scene(video_path, proposal_path, batch, outputs, target_to_pred, pair_lookup, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with proposal_path.open(encoding="utf-8") as handle:
        annotation = json.load(handle)
    reader = cv2.VideoCapture(str(video_path))
    fps = reader.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_temp = output_dir / "video_raw_mp4v.mp4"
    overlay_temp = output_dir / "video_overlay_mp4v.mp4"
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    raw_writer = cv2.VideoWriter(str(raw_temp), codec, fps, (width, height))
    overlay_writer = cv2.VideoWriter(str(overlay_temp), codec, fps, (width, height))
    if not raw_writer.isOpened() or not overlay_writer.isOpened():
        raise RuntimeError("failed to open MP4V video writers")
    objects = sorted(annotation["ground_truth"]["objects"], key=lambda item: int(item["id"]))
    window_start = int(batch["window_start"][0])
    video_start = window_start
    video_end = window_start + 64
    reader.set(cv2.CAP_PROP_POS_FRAMES, video_start)
    for raw_frame in range(video_start, video_end):
        ok, frame = reader.read()
        if not ok:
            raise RuntimeError(f"video ended before frame {raw_frame}: {video_path}")
        raw_writer.write(frame)
        overlay = frame.copy()
        for proposal_index, proposal in enumerate(annotation["frames"][raw_frame]["objects"]):
            if float(proposal["score"]) < 0.5:
                continue
            mask = decode_rle(proposal["mask"])
            color = np.asarray((45, 140, 230))
            overlay[mask] = (0.72 * overlay[mask] + 0.28 * color).astype(np.uint8)
        step = (raw_frame - (window_start + 32)) // 2 if raw_frame >= window_start + 32 and raw_frame % 2 == window_start % 2 else None
        if step is not None and step < 16:
            for target_slot, predicted_slot in enumerate(target_to_pred.tolist()):
                object_id = int(objects[target_slot]["id"])
                if batch["state_valid"][0, target_slot, step]:
                    draw_box(
                        overlay, batch["state"][0, target_slot, step], (0, 220, 0), 2,
                        f"GT {object_id}",
                    )
                draw_box(
                    overlay, outputs["trajectory_2d"][0, predicted_slot, step],
                    (220, 0, 220), 1, f"P{predicted_slot}",
                )
            gt_contacts, predicted_contacts = [], []
            for (first, second), predicted_pair in pair_lookup.items():
                ids = (int(objects[first]["id"]), int(objects[second]["id"]))
                if batch["contact"][0, first, second, step] > 0:
                    gt_contacts.append(ids)
                if outputs["contact_gt_event"][0, predicted_pair, step].sigmoid() >= 0.5:
                    predicted_contacts.append(ids)
            cv2.putText(
                overlay, f"frame={raw_frame} GT contact={gt_contacts} pred={predicted_contacts}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
            )
        cv2.putText(
            overlay, "orange=proposal mask  green=GT box  magenta=matched probe box",
            (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        overlay_writer.write(overlay)
    reader.release()
    raw_writer.release()
    overlay_writer.release()
    encode_h264(raw_temp, output_dir / "video_raw.mp4")
    encode_h264(overlay_temp, output_dir / "video_overlay.mp4")
