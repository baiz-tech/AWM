from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset


class EK100Dataset(Dataset):
    def __init__(self, annotation_csv, video_root, indices=None, crop_size=256):
        self.annotation_csv = Path(annotation_csv)
        self.video_root = Path(video_root)
        with self.annotation_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = rows if indices is None else [rows[int(i)] for i in indices]
        self.crop_size = int(crop_size)
        if not self.rows:
            raise ValueError(f"no EK100 rows in {self.annotation_csv}")

    def __len__(self):
        return len(self.rows)

    def _video_path(self, row):
        path = self.video_root / row["participant_id"] / "videos" / f"{row['video_id']}.MP4"
        if not path.exists():
            lower = path.with_suffix(".mp4")
            if lower.exists(): return lower
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _resize_crop(frames, size):
        value = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        value = torch.nn.functional.interpolate(value, (size, size), mode="bilinear", align_corners=False)
        mean = value.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = value.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return ((value - mean) / std).permute(1, 0, 2, 3)

    def __getitem__(self, index):
        row = self.rows[index]
        reader = VideoReader(str(self._video_path(row)), num_threads=1, ctx=cpu(0))
        # Official EPIC-KITCHENS annotation frame numbers are one-based;
        # decord indices are zero-based.
        start, stop = int(row["start_frame"]) - 1, int(row["stop_frame"]) - 1
        start = max(0, min(start, len(reader) - 1)); stop = max(start, min(stop, len(reader) - 1))
        positions = np.linspace(start, stop, 16).round().astype(np.int64)
        frames = reader.get_batch(positions).asnumpy()
        video = self._resize_crop(frames, self.crop_size)
        return {
            "video": video,
            "verb_class": torch.tensor(int(row["verb_class"]), dtype=torch.long),
            "noun_class": torch.tensor(int(row["noun_class"]), dtype=torch.long),
            "narration_id": row["narration_id"], "participant_id": row["participant_id"],
            "video_id": row["video_id"], "narration": row["narration"],
            "verb": row["verb"], "noun": row["noun"],
        }


def read_rows(path):
    with Path(path).open(newline="") as handle: return list(csv.DictReader(handle))


def split_by_video(annotation_csv, validation_fraction=0.1, seed=239):
    rows = read_rows(annotation_csv)
    videos = sorted({row["video_id"] for row in rows})
    generator = np.random.default_rng(seed); generator.shuffle(videos)
    cut = max(1, int(round(len(videos) * validation_fraction)))
    validation_videos = set(videos[:cut])
    train = [i for i, row in enumerate(rows) if row["video_id"] not in validation_videos]
    validation = [i for i, row in enumerate(rows) if row["video_id"] in validation_videos]
    if not train or not validation: raise ValueError("video split produced an empty partition")
    return train, validation
