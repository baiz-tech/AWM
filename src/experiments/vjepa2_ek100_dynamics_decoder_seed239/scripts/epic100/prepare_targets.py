#!/usr/bin/env python3
"""Build event-aligned EK100 and VISOR targets for decoder training."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .event_data import build_event_samples, read_events
from .model import FUTURE_STEPS, MAX_OBJECTS, STATE_DIM, TARGET_PROTOCOL
from src.core.run_context import apply_cli_defaults, task_context


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        ids = sorted(int(row["id"]) for row in csv.DictReader(handle))
    if not ids or ids != list(range(ids[-1] + 1)):
        raise ValueError(f"VISOR category IDs must be contiguous from zero: {path}")
    return ids[-1] + 1


def label_vocabulary(train_annotations: Path):
    events = read_events(train_annotations)
    verbs = sorted({event.verb_class for event in events})
    nouns = sorted({event.noun_class for event in events})
    actions = sorted({(event.verb_class, event.noun_class) for event in events})
    return {
        "verb_ids": verbs, "noun_ids": nouns,
        "action_pairs": [list(pair) for pair in actions],
    }


def read_needed_frames(path: Path, needed: set[int]):
    if not path.is_file() or not needed:
        return {}
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            frame = int(record["processed_frame_index"])
            if frame in needed:
                result[frame] = record["annotations"]
            if result.keys() >= needed:
                break
    return result


def annotation_state(annotation):
    x0, y0, x1, y1 = map(float, annotation["bbox_xyxy"])
    center_x, center_y = map(float, annotation["center_xy"])
    area = float(annotation["area_fraction"])
    return np.asarray([
        center_x / 256.0, center_y / 256.0, area,
        (x1 - x0) / 256.0, (y1 - y0) / 256.0,
        0.0, 0.0, np.log(max(area, 1.0 / 65536.0)),
    ], dtype=np.float32)


def track_annotations(frames, max_distance):
    tracks = []
    for step, annotations in enumerate(frames):
        used = set()
        for annotation in sorted(annotations, key=lambda value: float(value["area_fraction"]), reverse=True):
            category = int(annotation["class_id"])
            state = annotation_state(annotation)
            candidates = []
            for index, track in enumerate(tracks):
                if index in used or track["category"] != category:
                    continue
                gap = step - track["last_step"]
                distance = float(np.linalg.norm(state[:2] - track["last_state"][:2]))
                if distance <= max_distance * max(gap, 1):
                    candidates.append((distance, index))
            if candidates:
                _, index = min(candidates)
                track = tracks[index]
                track["states"][step] = state
                track["last_step"], track["last_state"] = step, state
                used.add(index)
            else:
                tracks.append({
                    "category": category, "name": str(annotation["name"]),
                    "states": {step: state}, "last_step": step, "last_state": state,
                })
                used.add(len(tracks) - 1)
    return tracks


def event_target(frames, event_noun, max_distance):
    tracks = track_annotations(frames, max_distance)
    if not tracks:
        return None

    def priority(track):
        event_object = track["category"] == event_noun
        hand = track["category"] in (300, 301, 303, 304)
        area = sum(float(state[2]) for state in track["states"].values())
        return event_object, hand, len(track["states"]), area

    tracks = sorted(tracks, key=priority, reverse=True)[:MAX_OBJECTS]
    present = np.zeros(MAX_OBJECTS, dtype=bool)
    category = np.full(MAX_OBJECTS, -1, dtype=np.int64)
    state = np.zeros((MAX_OBJECTS, FUTURE_STEPS, STATE_DIM), dtype=np.float32)
    valid = np.zeros((MAX_OBJECTS, FUTURE_STEPS), dtype=bool)
    names = []
    for slot, track in enumerate(tracks):
        present[slot], category[slot] = True, track["category"]
        names.append(track["name"])
        for step, value in track["states"].items():
            state[slot, step], valid[slot, step] = value, True
        for step in np.flatnonzero(valid[slot]):
            if step > 0 and valid[slot, step - 1]:
                state[slot, step, 5:7] = state[slot, step, :2] - state[slot, step - 1, :2]
    distance = np.zeros((MAX_OBJECTS, MAX_OBJECTS, FUTURE_STEPS), dtype=np.float32)
    pair_valid = np.zeros_like(distance, dtype=bool)
    for first in range(len(tracks)):
        for second in range(first + 1, len(tracks)):
            both = valid[first] & valid[second]
            values = np.linalg.norm(state[first, both, :2] - state[second, both, :2], axis=-1)
            pair_valid[first, second] = pair_valid[second, first] = both
            distance[first, second, both] = distance[second, first, both] = values
    return {
        "object_present": present, "category": category, "state": state,
        "state_valid": valid, "pair_distance": distance, "pair_valid": pair_valid,
        "object_names": names,
    }


def main():
    ctx = task_context()
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--visor-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--max-track-distance", type=float, default=0.25)
    apply_cli_defaults(parser, ctx)
    args = parser.parse_args()
    manifest_path = args.visor_root / "manifest.json"
    noun_classes = args.visor_root / "EPIC_100_noun_classes_v2.csv"
    for path in (args.processed_root, args.visor_root, args.annotations, args.train_annotations, manifest_path, noun_classes):
        if not path.exists():
            parser.error(f"required input does not exist: {path}")
    if not json.loads(manifest_path.read_text()).get("complete"):
        raise RuntimeError(f"VISOR processed dataset is incomplete: {manifest_path}")

    samples, sampling_skipped = build_event_samples(args.processed_root, args.annotations, args.max_events)
    by_video = {}
    for sample in samples:
        by_video.setdefault(sample[0].video_id, []).append(sample)
    vocabulary = label_vocabulary(args.train_annotations)
    verb_map = {value: index for index, value in enumerate(vocabulary["verb_ids"])}
    noun_map = {value: index for index, value in enumerate(vocabulary["noun_ids"])}
    action_map = {tuple(value): index for index, value in enumerate(vocabulary["action_pairs"])}
    values = {key: [] for key in ("object_present", "category", "state", "state_valid", "pair_distance", "pair_valid")}
    labels = {key: [] for key in ("verb_label", "noun_label", "action_label")}
    records, skipped = [], {**sampling_skipped, "no_annotation_frames": 0, "no_objects": 0}
    for video_index, (video_id, video_samples) in enumerate(sorted(by_video.items()), 1):
        needed = {int(index) for sample in video_samples for index in sample[-1]}
        dense = args.visor_root / "annotations" / "dense" / args.split / f"{video_id}.jsonl.gz"
        sparse = args.visor_root / "annotations" / "sparse" / args.split / f"{video_id}.jsonl.gz"
        annotations = read_needed_frames(dense, needed)
        annotations.update(read_needed_frames(sparse, needed.difference(annotations)))
        for sample in video_samples:
            event, path, fps, current_window, future_window, current, future = sample
            frames = [annotations.get(int(index), []) for index in future]
            if not any(frames):
                skipped["no_annotation_frames"] += 1
                continue
            target = event_target(frames, event.noun_class, args.max_track_distance)
            if target is None:
                skipped["no_objects"] += 1
                continue
            record = {
                "narration_id": event.narration_id, "participant_id": event.participant_id,
                "video_id": event.video_id, "video_path": str(path.resolve()),
                "verb_class": event.verb_class, "noun_class": event.noun_class,
                "verb": event.verb, "noun": event.noun,
                "current_indices": current.tolist(), "future_indices": future.tolist(),
                "current_window": list(current_window), "future_window": list(future_window),
                "source_fps": fps, "object_names": target.pop("object_names"),
            }
            records.append(record)
            for key in values:
                values[key].append(torch.from_numpy(target[key]))
            labels["verb_label"].append(verb_map.get(event.verb_class, -1))
            labels["noun_label"].append(noun_map.get(event.noun_class, -1))
            labels["action_label"].append(action_map.get((event.verb_class, event.noun_class), -1))
        print(f"videos={video_index}/{len(by_video)} selected={len(records)}", flush=True)
    if not records:
        raise RuntimeError("no event has usable future VISOR supervision")
    payload = {
        "protocol": TARGET_PROTOCOL, "split": args.split, "samples": len(records),
        "records": records, "num_categories": category_count(noun_classes),
        "vocabulary": vocabulary, "skipped": skipped,
        "annotations": str(args.annotations.resolve()), "annotations_sha256": sha256(args.annotations),
        "train_annotations": str(args.train_annotations.resolve()), "train_annotations_sha256": sha256(args.train_annotations),
        "visor_manifest_sha256": sha256(manifest_path), "max_track_distance": args.max_track_distance,
        **{key: torch.stack(items) for key, items in values.items()},
        **{key: torch.tensor(items, dtype=torch.long) for key, items in labels.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "samples": len(records), "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
