#!/usr/bin/env python3
"""Create object-set, trajectory, and event targets for the structured probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import COLORS, FUTURE_STEPS, MATERIALS, MAX_OBJECTS, SHAPES
from src.core.run_context import apply_cli_defaults, task_context


SPLITS = {"train": range(0, 10_000), "validation": range(10_000, 15_000)}


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
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
            raise ValueError("invalid COCO RLE run length")
        if index % 2:
            flat[offset : offset + run] = 1
        offset += run
    if offset != flat.size:
        raise ValueError("COCO RLE does not cover declared mask size")
    return flat.reshape((height, width), order="F").astype(bool)


def object_key(obj: dict[str, Any]) -> tuple[str, str, str]:
    return str(obj["color"]), str(obj["material"]), str(obj["shape"])


def state_from_mask(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty proposal mask")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    area = mask.mean()
    return np.asarray(
        [
            xs.mean() / width,
            ys.mean() / height,
            area,
            (x1 - x0 + 1) / width,
            (y1 - y0 + 1) / height,
            0.0,
            0.0,
            np.log(max(area, 1.0 / (height * width))),
        ],
        dtype=np.float32,
    )


def scene_targets(annotation: dict[str, Any], future_indices: list[int], score_threshold: float):
    objects = sorted(annotation["ground_truth"]["objects"], key=lambda obj: int(obj["id"]))
    if len(objects) > MAX_OBJECTS:
        raise ValueError(f"scene has {len(objects)} objects, max={MAX_OBJECTS}")
    lookup = {object_key(obj): index for index, obj in enumerate(objects)}
    if len(lookup) != len(objects):
        raise ValueError("object attribute tuples must be unique for safe proposal matching")
    count = len(objects)
    object_present = np.zeros(MAX_OBJECTS, dtype=bool)
    object_present[:count] = True
    color = np.full(MAX_OBJECTS, -1, dtype=np.int64)
    material = np.full(MAX_OBJECTS, -1, dtype=np.int64)
    shape = np.full(MAX_OBJECTS, -1, dtype=np.int64)
    for slot, obj in enumerate(objects):
        color[slot] = COLORS.index(str(obj["color"]))
        material[slot] = MATERIALS.index(str(obj["material"]))
        shape[slot] = SHAPES.index(str(obj["shape"]))

    state = np.zeros((MAX_OBJECTS, FUTURE_STEPS, 8), dtype=np.float32)
    state_valid = np.zeros((MAX_OBJECTS, FUTURE_STEPS), dtype=bool)
    previous_state = np.zeros((MAX_OBJECTS, 8), dtype=np.float32)
    previous_valid = np.zeros(MAX_OBJECTS, dtype=bool)
    raw_step = future_indices[1] - future_indices[0]
    for sampled_index, frame_index in enumerate([future_indices[0] - raw_step, *future_indices]):
        matched: dict[int, tuple[float, dict[str, Any]]] = {}
        for proposal in annotation["frames"][frame_index]["objects"]:
            slot = lookup.get(object_key(proposal))
            score = float(proposal["score"])
            if slot is None or score < score_threshold:
                continue
            if slot not in matched or score > matched[slot][0]:
                matched[slot] = (score, proposal)
        for slot, (_, proposal) in matched.items():
            value = state_from_mask(decode_rle(proposal["mask"]))
            if sampled_index == 0:
                previous_state[slot], previous_valid[slot] = value, True
            else:
                step = sampled_index - 1
                state[slot, step], state_valid[slot, step] = value, True
    for slot in range(count):
        for step in np.flatnonzero(state_valid[slot]):
            if step > 0 and state_valid[slot, step - 1]:
                state[slot, step, 5:7] = state[slot, step, :2] - state[slot, step - 1, :2]
            elif step == 0 and previous_valid[slot]:
                state[slot, step, 5:7] = state[slot, step, :2] - previous_state[slot, :2]

    pair_distance = np.zeros((MAX_OBJECTS, MAX_OBJECTS, FUTURE_STEPS), dtype=np.float32)
    pair_valid = np.zeros((MAX_OBJECTS, MAX_OBJECTS, FUTURE_STEPS), dtype=bool)
    contact = np.zeros_like(pair_distance)
    contact_valid = np.zeros_like(pair_valid)
    first_contact_class = np.full((MAX_OBJECTS, MAX_OBJECTS), -1, dtype=np.int64)
    id_to_slot = {int(obj["id"]): slot for slot, obj in enumerate(objects)}
    collisions = annotation["ground_truth"]["collisions"]
    for first in range(count):
        for second in range(first + 1, count):
            valid = state_valid[first] & state_valid[second]
            pair_valid[first, second] = pair_valid[second, first] = valid
            distances = np.linalg.norm(state[first, valid, :2] - state[second, valid, :2], axis=-1)
            pair_distance[first, second, valid] = distances
            pair_distance[second, first, valid] = distances
            contact_valid[first, second] = contact_valid[second, first] = valid
            event_frames = sorted(
                int(event["frame"])
                for event in collisions
                if {id_to_slot.get(int(event["object"][0])), id_to_slot.get(int(event["object"][1]))}
                == {first, second}
            )
            collision_steps = [
                step
                for step, frame in enumerate(future_indices)
                if any(frame <= event < frame + raw_step for event in event_frames)
            ]
            for step in collision_steps:
                contact[first, second, step] = contact[second, first, step] = 1.0
            first_class = collision_steps[0] if collision_steps else FUTURE_STEPS
            first_contact_class[first, second] = first_contact_class[second, first] = first_class
    return {
        "object_present": object_present,
        "color": color,
        "material": material,
        "shape": shape,
        "state": state,
        "state_valid": state_valid,
        "pair_distance": pair_distance,
        "pair_valid": pair_valid,
        "contact": contact,
        "contact_valid": contact_valid,
        "first_contact_class": first_contact_class,
    }


def main() -> None:
    ctx = task_context()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/shared/datasets/CLEVRER"))
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-starts", default="0,32,64")
    parser.add_argument("--future-step", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--max-scenes", type=int)
    apply_cli_defaults(parser, ctx)
    args = parser.parse_args()
    scene_ids = list(SPLITS[args.split])[: args.max_scenes]
    values: dict[str, list[torch.Tensor]] = {}
    window_starts = [int(value) for value in args.window_starts.split(",") if value.strip()]
    sample_scene_ids, sample_window_starts = [], []
    for position, scene_id in enumerate(scene_ids):
        path = args.dataset_root / "processed_proposals" / f"sim_{scene_id:05d}.json"
        with path.open(encoding="utf-8") as handle:
            annotation = json.load(handle)
        for window_start in window_starts:
            future_indices = [window_start + 32 + args.future_step * step for step in range(FUTURE_STEPS)]
            if future_indices[-1] >= len(annotation["frames"]):
                continue
            target = scene_targets(annotation, future_indices, args.score_threshold)
            sample_scene_ids.append(scene_id)
            sample_window_starts.append(window_start)
            for name, value in target.items():
                values.setdefault(name, []).append(torch.from_numpy(value))
        if (position + 1) % 500 == 0:
            print(f"processed {position + 1}/{len(scene_ids)} scenes", flush=True)
    payload = {
        "protocol": "clevrer_multiwindow_structured_object_set_targets_v1",
        "split": args.split,
        "scene_id": torch.tensor(sample_scene_ids, dtype=torch.long),
        "window_start": torch.tensor(sample_window_starts, dtype=torch.long),
        "attribute_vocabulary": {
            "color": list(COLORS), "material": list(MATERIALS), "shape": list(SHAPES)
        },
        "score_threshold": args.score_threshold,
        "window_starts": window_starts,
        **{name: torch.stack(items) for name, items in values.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"output": str(args.output), "scenes": len(scene_ids), "samples": len(sample_scene_ids)}, indent=2))


if __name__ == "__main__":
    main()
