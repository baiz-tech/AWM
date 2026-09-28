"""Event-centred EPIC-KITCHENS sampling shared by target and latent stages."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu


def timestamp_seconds(value: str) -> float:
    hour, minute, second = value.split(":")
    return int(hour) * 3600.0 + int(minute) * 60.0 + float(second)


@dataclass(frozen=True)
class Event:
    narration_id: str
    participant_id: str
    video_id: str
    start: float
    stop: float
    verb_class: int
    noun_class: int
    verb: str
    noun: str


def read_events(path: Path) -> list[Event]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        required = {
            "narration_id", "participant_id", "video_id", "start_timestamp",
            "stop_timestamp", "verb_class", "noun_class", "verb", "noun",
        }
        missing = required.difference(rows.fieldnames or ())
        if missing:
            raise ValueError(f"annotation CSV missing columns: {sorted(missing)}")
        events = []
        for row in rows:
            start = timestamp_seconds(row["start_timestamp"])
            stop = timestamp_seconds(row["stop_timestamp"])
            if stop > start:
                events.append(Event(
                    narration_id=row["narration_id"], participant_id=row["participant_id"],
                    video_id=row["video_id"], start=start, stop=stop,
                    verb_class=int(row["verb_class"]), noun_class=int(row["noun_class"]),
                    verb=row["verb"], noun=row["noun"],
                ))
    return events


def event_windows(start: float, stop: float, frames: int = 16, min_fps: float = 4.0):
    midpoint = start + 0.5 * (stop - start)
    half_duration = midpoint - start
    if half_duration <= 0:
        return None
    current_start = max(start, midpoint - frames / min_fps)
    return (current_start, midpoint), (midpoint, stop)


def uniform_indices(begin, end, fps, count, frames=16, require_unique=True):
    if end <= begin or fps <= 0 or count <= 0:
        return None
    times = begin + (np.arange(frames, dtype=np.float64) + 0.5) * (end - begin) / frames
    indices = np.floor(times * fps).astype(np.int64)
    if indices[-1] >= count or (require_unique and len(np.unique(indices)) != frames):
        return None
    return indices


def build_event_samples(root: Path, annotations: Path, max_events: int | None = None):
    events = read_events(annotations)
    if max_events is not None:
        events = events[:max_events]
    samples, metadata = [], {}
    skipped = {"missing_video": 0, "invalid_window": 0}
    for event in events:
        path = root / event.participant_id / "videos" / f"{event.video_id}.MP4"
        if not path.is_file():
            skipped["missing_video"] += 1
            continue
        if path not in metadata:
            reader = VideoReader(str(path), num_threads=1, ctx=cpu(0))
            metadata[path] = (float(reader.get_avg_fps()), len(reader))
        fps, count = metadata[path]
        windows = event_windows(event.start, event.stop)
        if windows is None:
            skipped["invalid_window"] += 1
            continue
        current_window, future_window = windows
        current = uniform_indices(*current_window, fps, count, require_unique=True)
        future = uniform_indices(*future_window, fps, count, require_unique=False)
        if current is None or future is None:
            skipped["invalid_window"] += 1
            continue
        samples.append((event, path, fps, current_window, future_window, current, future))
    return samples, skipped


class TargetClipDataset(torch.utils.data.Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        reader = VideoReader(record["video_path"], num_threads=1, ctx=cpu(0))
        frames = reader.get_batch(record["current_indices"]).asnumpy()
        return {
            "index": torch.tensor(index),
            "current": self.transform(frames),
        }
