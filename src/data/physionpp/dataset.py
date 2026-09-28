"""Physion current-16 to future-16 dataset used by the native V-JEPA 2 recipe."""

from __future__ import annotations

import glob
import os
import pickle
import random
import warnings
import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu


def _read_paths(manifest):
    table = pd.read_csv(manifest, header=None, delimiter=" ")
    return [str(row[0]) for row in table.values if len(row) and str(row[0]).strip()]


class PhysionCurrentFutureDataset(torch.utils.data.Dataset):
    """Return non-overlapping current and future clips with identical lengths."""

    def __init__(
        self,
        root=None,
        split="data_v1",
        data_paths=None,
        video_glob="**/*_img.mp4",
        clip_frames=16,
        sampling_mode="prediction_start",
        current_frame_step=2,
        future_frame_step=4,
        clip_gap=32,
        transform=None,
        deterministic=False,
        max_videos=None,
    ):
        self.clip_frames = int(clip_frames)
        self.sampling_mode = str(sampling_mode).lower()
        self.current_frame_step = int(current_frame_step)
        self.future_frame_step = int(future_frame_step)
        self.clip_gap = int(clip_gap)
        self.transform = transform
        self.deterministic = bool(deterministic)
        self._prediction_start_cache = {}
        self._metadata_cache = {}

        if self.sampling_mode not in ("random", "prediction_start"):
            raise ValueError("sampling_mode must be 'random' or 'prediction_start'")
        if min(self.clip_frames, self.current_frame_step, self.future_frame_step) <= 0:
            raise ValueError("clip_frames, current_frame_step, and future_frame_step must be positive")
        if self.clip_gap < 0:
            raise ValueError("clip_gap must be non-negative")

        if isinstance(data_paths, str):
            data_paths = [data_paths]
        samples = []
        for manifest in data_paths or []:
            samples.extend(_read_paths(manifest))

        if root:
            split_root = os.path.join(root, split)
            if not os.path.isdir(split_root):
                raise FileNotFoundError(f"Physion split directory does not exist: {split_root}")
            samples.extend(sorted(glob.glob(os.path.join(split_root, video_glob), recursive=True)))

        self.samples = sorted(dict.fromkeys(samples))
        if max_videos is not None:
            self.samples = self.samples[: int(max_videos)]
        if not self.samples:
            raise FileNotFoundError(f"No videos found from root={root} split={split} manifests={data_paths}")

        valid_samples = []
        for path in self.samples:
            try:
                reader = VideoReader(path, num_threads=1, ctx=cpu(0))
            except Exception as exc:
                warnings.warn(f"discarding video that failed to open: video={path}: {exc}")
                continue
            if self._select_indices(path, len(reader)) is None:
                continue
            valid_samples.append(path)
        discarded = len(self.samples) - len(valid_samples)
        self.samples = valid_samples
        if discarded:
            warnings.warn(f"discarded {discarded} invalid or too-short videos during dataset initialization")
        if not self.samples:
            raise RuntimeError("No valid videos remain after checking prediction starts and clip lengths")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.deterministic:
            sample = self._load(index)
            if sample is None:
                raise RuntimeError(f"Deterministic sample is invalid or too short: {self.samples[index]}")
            return sample
        for _ in range(20):
            sample = self._load(index)
            if sample is not None:
                return sample
            index = random.randrange(len(self.samples))
        raise RuntimeError("Failed to decode a sufficiently long Physion video after 20 attempts")

    @staticmethod
    def _pkl_path(video_path):
        suffix = "_img.mp4"
        if not video_path.endswith(suffix):
            raise ValueError(f"prediction_start mode requires a video ending in {suffix!r}: {video_path}")
        return video_path[: -len(suffix)] + ".pkl"

    def _read_metadata(self, video_path):
        if video_path in self._metadata_cache:
            return self._metadata_cache[video_path]

        pkl_path = self._pkl_path(video_path)
        try:
            # Physion++ PKLs are trusted local dataset files. Never unpickle
            # metadata from an untrusted source.
            with open(pkl_path, "rb") as handle:
                metadata = pickle.load(handle)
            if not isinstance(metadata, dict):
                raise TypeError(f"expected a metadata dictionary, got {type(metadata).__name__}")
        except (OSError, EOFError, TypeError, ValueError, pickle.UnpicklingError) as exc:
            warnings.warn(f"failed to read metadata from pkl={pkl_path}: {exc}")
            metadata = None
        self._metadata_cache[video_path] = metadata
        return metadata

    def _read_prediction_start(self, video_path):
        if video_path in self._prediction_start_cache:
            return self._prediction_start_cache[video_path]

        try:
            metadata = self._read_metadata(video_path)
            prediction_start = int(metadata["static"]["start_frame_for_prediction"])
        except (KeyError, TypeError, ValueError) as exc:
            warnings.warn(f"failed to read prediction start for video={video_path}: {exc}")
            return None

        self._prediction_start_cache[video_path] = prediction_start
        return prediction_start

    def _supervision_labels(self, video_path, current_indices):
        """Return outcome/contact labels plus masks for optional auxiliary losses."""
        metadata = self._read_metadata(video_path)
        outcome = contact = 0.0
        outcome_valid = contact_valid = False
        if metadata is None:
            return outcome, outcome_valid, contact, contact_valid

        static = metadata.get("static", {})
        outcome_value = static.get("does_target_contact_zone")
        if outcome_value is not None:
            outcome = float(bool(outcome_value))
            outcome_valid = True

        target_id = static.get("target_id")
        frames = metadata.get("frames")
        try:
            target_id = int(target_id)
        except (TypeError, ValueError):
            target_id = None
        if target_id is not None and isinstance(frames, dict):
            contact_valid = True
            for frame_index in current_indices:
                frame = frames.get(f"{int(frame_index):04d}", {})
                object_ids = np.asarray(
                    frame.get("collisions", {}).get("object_ids", []), dtype=np.int64
                )
                if object_ids.ndim == 2 and object_ids.shape[1] == 2 and np.any(object_ids == target_id):
                    contact = 1.0
                    break
        return outcome, outcome_valid, contact, contact_valid

    def _select_indices(self, video_path, video_length):
        # The anchor is the exclusive right boundary of current. Future starts
        # clip_gap raw frames after it:
        # current = P-F*Sc, ..., P-Sc; future = P+G, ..., P+G+(F-1)*Sf.
        min_anchor = self.clip_frames * self.current_frame_step
        max_anchor = (
            video_length
            - 1
            - self.clip_gap
            - (self.clip_frames - 1) * self.future_frame_step
        )
        if min_anchor > max_anchor:
            return None

        if self.sampling_mode == "prediction_start":
            anchor = self._read_prediction_start(video_path)
            if anchor is None or not min_anchor <= anchor <= max_anchor:
                if anchor is not None:
                    warnings.warn(
                        f"prediction start is outside valid range for video={video_path}: "
                        f"P={anchor}, valid=[{min_anchor}, {max_anchor}]"
                    )
                return None
        else:
            anchor = (min_anchor + max_anchor) // 2 if self.deterministic else random.randint(min_anchor, max_anchor)

        current_start = anchor - self.clip_frames * self.current_frame_step
        current_indices = current_start + np.arange(self.clip_frames) * self.current_frame_step
        future_indices = anchor + self.clip_gap + np.arange(self.clip_frames) * self.future_frame_step
        return anchor, current_indices, future_indices

    def _load(self, index):
        path = self.samples[index]
        try:
            reader = VideoReader(path, num_threads=-1, ctx=cpu(0))
        except Exception as exc:
            warnings.warn(f"failed to open video={path}: {exc}")
            return None

        selection = self._select_indices(path, len(reader))
        if selection is None:
            return None
        anchor, current_indices, future_indices = selection

        indices = np.concatenate([current_indices, future_indices]).astype(np.int64)
        try:
            frames = reader.get_batch(indices).asnumpy()
        except Exception as exc:
            warnings.warn(f"failed to decode video={path}: {exc}")
            return None

        if self.transform is None:
            merged = torch.as_tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
        else:
            # One transform call guarantees identical spatial augmentation for
            # the current and future halves.
            merged = self.transform(frames)

        f = self.clip_frames
        outcome, outcome_valid, contact, contact_valid = self._supervision_labels(
            path, current_indices
        )
        return {
            "current": merged[:, :f],
            "future": merged[:, f : 2 * f],
            "current_indices": torch.as_tensor(current_indices, dtype=torch.long),
            "future_indices": torch.as_tensor(future_indices, dtype=torch.long),
            "anchor_frame": torch.tensor(anchor, dtype=torch.long),
            "ocp_label": torch.tensor(outcome, dtype=torch.float32),
            "ocp_valid": torch.tensor(outcome_valid, dtype=torch.bool),
            "contact_label": torch.tensor(contact, dtype=torch.float32),
            "contact_valid": torch.tensor(contact_valid, dtype=torch.bool),
            "path": path,
        }


def init_data(
    *,
    root=None,
    split="data_v1",
    data_paths=None,
    video_glob="**/*_img.mp4",
    batch_size=2,
    clip_frames=16,
    sampling_mode="prediction_start",
    current_frame_step=2,
    future_frame_step=4,
    clip_gap=32,
    transform=None,
    rank=0,
    world_size=1,
    num_workers=4,
    pin_mem=True,
    persistent_workers=True,
    drop_last=True,
    deterministic=False,
    max_videos=None,
):
    dataset = PhysionCurrentFutureDataset(
        root=root,
        split=split,
        data_paths=data_paths,
        video_glob=video_glob,
        clip_frames=clip_frames,
        sampling_mode=sampling_mode,
        current_frame_step=current_frame_step,
        future_frame_step=future_frame_step,
        clip_gap=clip_gap,
        transform=transform,
        deterministic=deterministic,
        max_videos=max_videos,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not deterministic,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=bool(persistent_workers and num_workers > 0),
        drop_last=drop_last,
    )
    return dataset, loader, sampler
