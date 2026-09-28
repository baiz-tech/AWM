#!/usr/bin/env python3
"""Train an OCP readout on readout_data_v1 and evaluate testdata_v1."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import pickle
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from decord import VideoReader, cpu

from external.vjepa2.app.vjepa.transforms import make_transforms
from src.data.physionpp.evaluate.diagnostics import select_probe_features, stratified_probe_split
from src.core.config_utils import task_experiment, task_launch, training_experiment
from src.data.physionpp.evaluate.world_model_adapter import load_world_model


LOGGER = logging.getLogger("shared.evaluate.physionpp.ocp")


def experiment_config(raw_config):
    return training_experiment(raw_config)


def rollout_from_context(model, initial_context, chunk_mask):
    """Apply the shared one-step predictor recurrently for this OCP batch."""
    if chunk_mask.ndim != 2 or chunk_mask.size(0) != initial_context.size(0):
        raise ValueError("chunk_mask must be [B,K] and match initial_context")
    context = initial_context
    predictions = []
    for chunk_index in range(chunk_mask.size(1)):
        predicted = model.predict_next(context)
        active = chunk_mask[:, chunk_index].view(-1, 1, 1)
        predicted = predicted.masked_fill(~active, 0.0)
        predictions.append(predicted)
        context = predicted
    return torch.stack(predictions, dim=1)


def read_ocp_label(video_path, clip_gap=0):
    """Return whether contact occurs anywhere from P+gap through video end."""
    path = str(video_path)
    if not path.endswith("_img.mp4"):
        return -1
    try:
        with open(path[: -len("_img.mp4")] + ".pkl", "rb") as handle:
            metadata = pickle.load(handle)
        frames = metadata.get("frames", {})
        if not frames:
            return -1
        prediction_start = int(metadata["static"]["start_frame_for_prediction"])
        future_start = prediction_start + int(clip_gap)
        for key in sorted(frames, key=lambda value: int(value)):
            if int(key) < future_start:
                continue
            if bool(frames[key].get("labels", {}).get("target_contacting_zone", False)):
                return 1
        return 0
    except (OSError, EOFError, KeyError, TypeError, ValueError, pickle.UnpicklingError):
        return -1


def _frame_at(frames, index):
    for key in (index, str(index), f"{index:04d}"):
        if key in frames:
            return frames[key]
    return None


def _read_ocp_spec(video_path, clip_gap, video_length):
    """Read P and the strict all-Future label used by the OCP dataset."""
    path = str(video_path)
    if not path.endswith("_img.mp4"):
        return None
    try:
        with open(path[:-len("_img.mp4")] + ".pkl", "rb") as handle:
            metadata = pickle.load(handle)
        prediction_start = int(metadata["static"]["start_frame_for_prediction"])
        future_start = prediction_start + int(clip_gap)
        frames = metadata.get("frames", {})
        contact = False
        for index in range(future_start, video_length):
            frame = _frame_at(frames, index)
            if frame is None:
                return None
            contact = contact or bool(
                frame.get("labels", {}).get("target_contacting_zone", False)
            )
        return prediction_start, int(contact)
    except (OSError, EOFError, KeyError, TypeError, ValueError, pickle.UnpicklingError):
        return None


class PhysionFullFutureOCPDataset(torch.utils.data.Dataset):
    """Decode Current only and describe all step-sampled Future chunks."""

    def __init__(self, *, root, split, video_glob="**/*_img.mp4", clip_frames=16,
                 current_frame_step=2, future_frame_step=2, clip_gap=0,
                 transform=None, max_videos=None):
        self.clip_frames = int(clip_frames)
        self.current_frame_step = int(current_frame_step)
        self.future_frame_step = int(future_frame_step)
        self.clip_gap = int(clip_gap)
        self.transform = transform
        if min(self.clip_frames, self.current_frame_step, self.future_frame_step) <= 0:
            raise ValueError("clip_frames and frame steps must be positive")
        if self.clip_gap < 0:
            raise ValueError("clip_gap must be non-negative")
        split_root = os.path.join(str(root), split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"Physion split directory does not exist: {split_root}")
        paths = sorted(glob.glob(os.path.join(split_root, video_glob), recursive=True))
        if max_videos is not None:
            paths = paths[:int(max_videos)]
        self.specs = []
        self.max_future_chunks = 0
        for path in paths:
            try:
                reader = VideoReader(path, num_threads=1, ctx=cpu(0))
                video_length = len(reader)
            except Exception as exc:
                warnings.warn(f"discarding video that failed to open: {path}: {exc}")
                continue
            metadata = _read_ocp_spec(path, self.clip_gap, video_length)
            if metadata is None:
                continue
            prediction_start, contact_label = metadata
            current_start = prediction_start - self.clip_frames * self.current_frame_step
            future_start = prediction_start + self.clip_gap
            if current_start < 0 or future_start >= video_length:
                continue
            sampled_count = 1 + (video_length - 1 - future_start) // self.future_frame_step
            num_chunks = (sampled_count + self.clip_frames - 1) // self.clip_frames
            self.specs.append({
                "path": path,
                "prediction_start": prediction_start,
                "video_length": video_length,
                "sampled_count": sampled_count,
                "num_chunks": num_chunks,
                "contact_label": contact_label,
            })
            self.max_future_chunks = max(self.max_future_chunks, num_chunks)
        if not self.specs:
            raise RuntimeError(f"No valid OCP videos remain in split={split}")

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, index):
        spec = self.specs[index]
        path = spec["path"]
        current_start = spec["prediction_start"] - self.clip_frames * self.current_frame_step
        current_indices = current_start + np.arange(self.clip_frames) * self.current_frame_step
        try:
            reader = VideoReader(path, num_threads=-1, ctx=cpu(0))
            frames = reader.get_batch(current_indices.astype(np.int64)).asnumpy()
        except Exception as exc:
            raise RuntimeError(f"failed to decode deterministic OCP sample={path}: {exc}") from exc
        if self.transform is None:
            current = torch.as_tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
        else:
            current = self.transform(frames)
        padded_frames = self.max_future_chunks * self.clip_frames
        frame_mask = torch.arange(padded_frames) < spec["sampled_count"]
        return {
            "current": current,
            "chunk_frame_mask": frame_mask.reshape(self.max_future_chunks, self.clip_frames),
            "chunk_mask": torch.arange(self.max_future_chunks) < spec["num_chunks"],
            "num_future_chunks": torch.tensor(spec["num_chunks"], dtype=torch.long),
            "contact_label": torch.tensor(spec["contact_label"], dtype=torch.long),
            "path": path,
        }


def binary_metrics(probabilities, labels, threshold):
    predictions = probabilities >= threshold
    truth = labels.bool()
    tp = int((predictions & truth).sum())
    tn = int((~predictions & ~truth).sum())
    fp = int((predictions & ~truth).sum())
    fn = int((~predictions & truth).sum())
    positive_recall = tp / max(tp + fn, 1)
    negative_recall = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "accuracy": (tp + tn) / max(labels.numel(), 1),
        "balanced_accuracy": 0.5 * (positive_recall + negative_recall),
        "f1": 2.0 * precision * positive_recall / max(precision + positive_recall, 1.0e-12),
        "precision": precision,
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "predicted_positive_rate": float(predictions.float().mean()),
    }


def best_threshold(probabilities, labels):
    candidates = torch.unique(torch.cat((torch.tensor([0.0, 0.5, 1.0]), probabilities))).sort().values
    best = (float("-inf"), -1.0, 0.5)
    for threshold in candidates:
        value = float(threshold)
        score = binary_metrics(probabilities, labels, value)["balanced_accuracy"]
        # Prefer thresholds nearer 0.5 when balanced accuracy is tied.
        candidate = (score, -abs(value - 0.5), value)
        if candidate > best:
            best = candidate
    return best[2]


def auroc(probabilities, labels):
    scores = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    positives = int((truth == 1).sum())
    negatives = int((truth == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = ranks[truth == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def stimulus_id(video_path):
    """Convert an extracted Physion++ video path to the human-study stimulus id."""
    path = Path(str(video_path))
    if len(path.parents) < 2:
        raise ValueError(f"Cannot form a stimulus id from path={video_path!r}")
    return f"{path.parent.parent.name}_{path.parent.name}_{path.stem}"


def _parse_bool(value):
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return 1
    if text in ("false", "0", "no"):
        return 0
    raise ValueError(f"Unsupported boolean value {value!r}")


def load_human_accuracy(humans_dir):
    """Load and response-count-weight human accuracy across all study runs."""
    humans_dir = Path(humans_dir)
    files = sorted(humans_dir.glob("human_accuracy-*.csv"))
    if not files:
        raise FileNotFoundError(f"No human_accuracy-*.csv files found in {humans_dir}")
    totals = {}
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"stim_ID", "correct", "c", "target_hit_zone_label"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(f"Human accuracy CSV has missing columns: {path}")
            for row in reader:
                stim_id = row["stim_ID"].strip()
                count = int(float(row["c"]))
                correct = float(row["correct"])
                label = _parse_bool(row["target_hit_zone_label"])
                if not stim_id or count <= 0 or not 0.0 <= correct <= 1.0:
                    raise ValueError(f"Invalid human accuracy row in {path}: {row}")
                item = totals.setdefault(stim_id, {"correct": 0.0, "count": 0, "label": label})
                if item["label"] != label:
                    raise ValueError(f"Conflicting human-study labels for stimulus {stim_id}")
                item["correct"] += correct * count
                item["count"] += count
    return {
        stim_id: {"accuracy": item["correct"] / item["count"], "count": item["count"], "label": item["label"]}
        for stim_id, item in totals.items()
    }


def human_subset_metrics(probabilities, labels, paths, threshold, human_accuracy, hard_threshold=0.5):
    """Evaluate model accuracy on all human stimuli and the at/below-chance subset."""
    if len(paths) != labels.numel() or probabilities.numel() != labels.numel():
        raise ValueError("Probabilities, labels, and paths must have equal length")
    predictions = probabilities.ge(float(threshold)).long()
    matched, hard = [], []
    per_test_human_accuracy = np.full(labels.numel(), np.nan, dtype=np.float32)
    per_test_human_label = np.full(labels.numel(), -1, dtype=np.int8)
    label_disagreements = 0
    seen = set()
    for index, path in enumerate(paths):
        stim_id = stimulus_id(path)
        item = human_accuracy.get(stim_id)
        if item is None:
            continue
        if stim_id in seen:
            raise ValueError(f"Duplicate test prediction for human stimulus {stim_id}")
        seen.add(stim_id)
        if int(labels[index].item()) != int(item["label"]):
            label_disagreements += 1
        matched.append(index)
        per_test_human_accuracy[index] = float(item["accuracy"])
        per_test_human_label[index] = int(item["label"])
        if float(item["accuracy"]) <= float(hard_threshold):
            hard.append(index)
    missing = sorted(set(human_accuracy) - seen)
    if missing:
        LOGGER.warning("human-study stimuli missing from test predictions: count=%d examples=%s", len(missing), missing[:5])
    if not matched:
        raise RuntimeError("No test predictions matched the human-study stimuli")

    def accuracy(indices):
        if not indices:
            return float("nan")
        index = torch.as_tensor(indices, dtype=torch.long)
        targets = torch.as_tensor([human_accuracy[stimulus_id(paths[i])]["label"] for i in indices], dtype=torch.long)
        return float(predictions[index].eq(targets).float().mean())

    return {
        "human_all_accuracy": accuracy(matched),
        "human_hard_accuracy": accuracy(hard),
        "human_all_count": len(matched),
        "human_hard_count": len(hard),
        "human_stimuli_total": len(human_accuracy),
        "human_stimuli_missing": len(missing),
        "human_label_disagreements_with_final_frame": label_disagreements,
        "human_hard_threshold": float(hard_threshold),
        "human_hard_definition": "weighted_human_accuracy<=threshold",
    }, per_test_human_accuracy, per_test_human_label


@torch.no_grad()
def extract_features(loader, model, device, dtype, use_amp, split_name, log_every_batches):
    current_features, predicted_features, last_features, delta_features = [], [], [], []
    labels, paths = [], []
    count = 0
    model.eval()
    started = time.monotonic()
    total_batches = len(loader)
    LOGGER.info("feature extraction started: split=%s batches=%d", split_name, total_batches)
    for iteration, batch in enumerate(loader, start=1):
        batch_labels = batch["contact_label"].long()
        valid = batch_labels >= 0
        if not valid.any():
            continue
        current = batch["current"].to(device, non_blocking=True)
        chunk_mask = batch["chunk_mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=dtype if use_amp else None):
            initial_context = model.encode_current(current)
            predicted = rollout_from_context(model, initial_context, chunk_mask)
        temporal_mask = batch["chunk_frame_mask"].to(device, non_blocking=True).reshape(
            current.size(0), chunk_mask.size(1), model.temporal_steps, model.tubelet_size
        ).any(dim=-1)
        batch_size, chunks, tokens, dim = predicted.shape
        spatial_tokens = tokens // model.temporal_steps
        predicted_5d = predicted.reshape(
            batch_size, chunks, model.temporal_steps, spatial_tokens, dim
        )
        mask = temporal_mask[:, :, :, None, None]
        denominator = mask.sum(dim=(1, 2, 3, 4)).clamp_min(1).float() * spatial_tokens
        pooled_predicted = (
            (predicted_5d.float() * mask).sum(dim=(1, 2, 3)) / denominator.unsqueeze(1)
        ).cpu()
        pooled_current = initial_context.float().mean(dim=1).cpu()
        pooled_last = []
        for sample_index in range(batch_size):
            last_chunk = int(batch["num_future_chunks"][sample_index]) - 1
            valid_steps = temporal_mask[sample_index, last_chunk]
            last_tokens = predicted_5d[sample_index, last_chunk, valid_steps]
            pooled_last.append(last_tokens.float().mean(dim=(0, 1)).cpu())
        pooled_last = torch.stack(pooled_last)
        keep = valid.nonzero(as_tuple=False).flatten()
        current_features.append(pooled_current[keep])
        predicted_features.append(pooled_predicted[keep])
        last_features.append(pooled_last[keep])
        delta_features.append((pooled_predicted - pooled_current)[keep])
        labels.append(batch_labels[keep].float())
        paths.extend(path for path, is_valid in zip(batch["path"], valid.tolist()) if is_valid)
        count += int(keep.numel())
        if iteration == 1 or iteration % log_every_batches == 0 or iteration == total_batches:
            elapsed = time.monotonic() - started
            LOGGER.info(
                "feature extraction progress: split=%s batch=%d/%d valid_samples=%d elapsed=%.1fs",
                split_name, iteration, total_batches, count, elapsed,
            )
    if not labels:
        raise RuntimeError("No samples with valid OCP labels were found")
    result = {
        "features": torch.cat((
            torch.cat(current_features), torch.cat(predicted_features),
            torch.cat(last_features), torch.cat(delta_features),
        ), dim=1),
        "labels": torch.cat(labels),
        "paths": paths,
    }
    LOGGER.info(
        "feature extraction finished: split=%s valid_samples=%d elapsed=%.1fs",
        split_name, count, time.monotonic() - started,
    )
    return result


def train_readout(train_x, train_y, epochs, lr, weight_decay, log_every_epochs, device):
    # Feature extraction is gradient-free, but the linear probe needs ordinary
    # tensors (not inference-mode tensors) because autograd saves its inputs
    # while computing the classifier weight gradient.
    train_x = train_x.detach().clone().to(device, non_blocking=True)
    train_y = train_y.detach().clone().to(device, non_blocking=True)
    mean = train_x.mean(0, keepdim=True)
    std = train_x.std(0, keepdim=True).clamp_min(1.0e-6)
    normalized = (train_x - mean) / std
    classifier = nn.Linear(normalized.size(1), 1, device=device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    positives = train_y.sum().clamp_min(1.0)
    pos_weight = (train_y.numel() - train_y.sum()).clamp_min(1.0) / positives
    started = time.monotonic()
    LOGGER.info(
        "readout training started: device=%s samples=%d feature_dim=%d epochs=%d positive_rate=%.4f",
        device, train_y.numel(), train_x.size(1), epochs, float(train_y.mean()),
    )
    for epoch in range(1, epochs + 1):
        logits = classifier(normalized).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % log_every_epochs == 0 or epoch == epochs:
            LOGGER.info(
                "readout training progress: epoch=%d/%d loss=%.6f elapsed=%.1fs",
                epoch, epochs, float(loss.item()), time.monotonic() - started,
            )
    LOGGER.info("readout training finished: elapsed=%.1fs", time.monotonic() - started)
    return classifier.eval(), mean, std


def run_unified_probe_suite(readout, test, args, readout_device, output_dir):
    fit_indices, threshold_indices = stratified_probe_split(
        readout["paths"], readout["labels"],
        validation_fraction=args.probe_validation_fraction, seed=args.seed,
    )
    split_payload = {
        "seed": int(args.seed),
        "validation_fraction": float(args.probe_validation_fraction),
        "fit_paths": [readout["paths"][index] for index in fit_indices.tolist()],
        "threshold_validation_paths": [
            readout["paths"][index] for index in threshold_indices.tolist()
        ],
    }
    (output_dir / "probe_split.json").write_text(
        json.dumps(split_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    results = {}
    for variant in args.probe_variants:
        all_readout_x = select_probe_features(readout["features"], variant)
        test_x = select_probe_features(test["features"], variant)
        fit_x = all_readout_x[fit_indices]
        fit_y = readout["labels"][fit_indices]
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        classifier, mean, std = train_readout(
            fit_x, fit_y,
            args.readout_epochs, args.readout_lr, args.readout_weight_decay,
            args.readout_log_every_epochs, readout_device,
        )
        with torch.no_grad():
            validation_x = all_readout_x[threshold_indices].to(
                readout_device, non_blocking=True
            )
            validation_probs = torch.sigmoid(
                classifier((validation_x - mean) / std).squeeze(1)
            ).cpu()
            device_test_x = test_x.to(readout_device, non_blocking=True)
            test_probs = torch.sigmoid(
                classifier((device_test_x - mean) / std).squeeze(1)
            ).cpu()
        threshold = best_threshold(
            validation_probs, readout["labels"][threshold_indices]
        )
        metrics = binary_metrics(test_probs, test["labels"], threshold)
        results[variant] = {
            "feature_dim": int(test_x.size(1)),
            "num_fit": int(fit_indices.numel()),
            "num_threshold_validation": int(threshold_indices.numel()),
            "num_test": int(test["labels"].numel()),
            "threshold": float(threshold),
            "threshold_source": "readout_data_v1/held_out_by_path_hash",
            "test_auroc": auroc(test_probs.numpy(), test["labels"].numpy()),
            **{f"test_{key}": float(value) for key, value in metrics.items()},
        }
        torch.save({
            "classifier": {key: value.detach().cpu() for key, value in classifier.state_dict().items()},
            "feature_mean": mean.detach().cpu(),
            "feature_std": std.detach().cpu(),
            "threshold": float(threshold),
            "fit_indices": fit_indices,
            "threshold_validation_indices": threshold_indices,
        }, output_dir / f"readout_{variant}.pt")
        np.savez_compressed(
            output_dir / f"test_predictions_{variant}.npz",
            probabilities=test_probs.numpy(), labels=test["labels"].numpy(),
            paths=np.asarray(test["paths"], dtype=object), threshold=np.asarray(threshold),
        )
    return results


def make_loader(config, split, transform, batch_size, workers, max_videos):
    data = config["data"]
    dataset = PhysionFullFutureOCPDataset(
        root=data.get("root"), split=split, video_glob=data.get("video_glob", "**/*_img.mp4"),
        clip_frames=int(data.get("clip_frames", 16)),
        current_frame_step=int(data.get("current_frame_step", 2)),
        future_frame_step=int(data.get("future_frame_step", 4)), clip_gap=int(data.get("clip_gap", 32)),
        transform=transform, max_videos=max_videos,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=data.get("pin_mem", True), persistent_workers=workers > 0,
        drop_last=False,
    )
    return loader


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--eval-config", default=None, help="OCP evaluation YAML")
    known, _ = config_parser.parse_known_args()
    eval_config = {}
    if known.eval_config:
        with open(known.eval_config) as handle:
            eval_config = yaml.safe_load(handle) or {}
    eval_launch = eval_config.get("launch", {})
    eval_task_launch = task_launch(eval_config, "full_future_ocp")
    eval_task = task_experiment(eval_config, "full_future_ocp")
    eval_experiment = eval_task
    evaluation = eval_experiment.get("evaluation", {})

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--config", default=eval_task_launch.get("training_config", eval_launch.get("training_config", eval_config.get("training_config"))), help="Training recipe YAML")
    parser.add_argument("--checkpoint", default=eval_task_launch.get("checkpoint", eval_launch.get("checkpoint", eval_config.get("checkpoint"))), help="Trained world-model checkpoint")
    parser.add_argument("--output-dir", default=eval_task_launch.get("output_dir", eval_task.get("output_dir", eval_config.get("output_dir"))))
    parser.add_argument("--device", default=evaluation.get("device", "cuda:0"))
    parser.add_argument("--batch-size", type=int, default=int(evaluation.get("batch_size", 2)))
    parser.add_argument("--num-workers", type=int, default=int(evaluation.get("num_workers", 4)))
    parser.add_argument("--readout-epochs", type=int, default=int(evaluation.get("readout_epochs", 300)))
    parser.add_argument("--readout-lr", type=float, default=float(evaluation.get("readout_lr", 3.0e-4)))
    parser.add_argument("--readout-weight-decay", type=float,
                        default=float(evaluation.get("readout_weight_decay", 1.0e-3)))
    parser.add_argument("--progress-every-batches", type=int,
                        default=int(evaluation.get("progress_every_batches", 10)))
    parser.add_argument("--readout-log-every-epochs", type=int,
                        default=int(evaluation.get("readout_log_every_epochs", 50)))
    parser.add_argument("--readout-device", default=evaluation.get("readout_device", "same"),
                        help="Readout training device: 'same', 'cpu', or a torch device such as cuda:0")
    parser.add_argument("--max-readout-videos", type=int, default=evaluation.get("max_readout_videos"))
    parser.add_argument("--max-test-videos", type=int, default=evaluation.get("max_test_videos"))
    parser.add_argument("--fixed-threshold", type=float, default=evaluation.get("fixed_threshold"),
                        help="Use this threshold; otherwise select it using readout_data_v1 only")
    parser.add_argument("--humans-dir", default=evaluation.get(
        "humans_dir", "/data/ABDUCTIVE-WORLD/physion_v2/extracted/humans"),
        help="Directory containing human_accuracy-*.csv files")
    parser.add_argument("--human-hard-threshold", type=float,
                        default=float(evaluation.get("human_hard_threshold", 0.5)),
                        help="Human-Hard includes stimuli whose response-weighted human accuracy is at or below this value")
    parser.add_argument("--seed", type=int, default=int(evaluation.get("seed", 239)))
    parser.add_argument(
        "--probe-protocol", choices=("unified", "legacy"),
        default=evaluation.get("probe_protocol", "unified"),
    )
    parser.add_argument(
        "--probe-validation-fraction", type=float,
        default=float(evaluation.get("probe_validation_fraction", 0.25)),
    )
    parser.add_argument(
        "--probe-variants", nargs="+",
        choices=("full", "current", "rollout_mean", "last_chunk", "delta"),
        default=evaluation.get(
            "probe_variants",
            ("full", "current", "rollout_mean", "last_chunk", "delta"),
        ),
    )
    args = parser.parse_args()
    if not args.config or not args.checkpoint:
        parser.error("provide --eval-config, or both --config and --checkpoint")
    if args.progress_every_batches <= 0 or args.readout_log_every_epochs <= 0:
        parser.error("progress logging intervals must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-8s][%(asctime)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with open(args.config) as handle:
        config = experiment_config(yaml.safe_load(handle))
    run_dir = Path(config.get("folder", Path(args.config).parent))
    if args.checkpoint and not Path(args.checkpoint).is_absolute() and "/" not in str(args.checkpoint):
        args.checkpoint = str(run_dir / args.checkpoint)
    if args.output_dir and not Path(args.output_dir).is_absolute():
        args.output_dir = str(run_dir / args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    readout_device = device if args.readout_device == "same" else torch.device(args.readout_device)
    if readout_device.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA is unavailable; falling back to CPU for readout training")
        readout_device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype_name = str(config.get("meta", {}).get("dtype", "bfloat16")).lower()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16 if dtype_name == "float16" else torch.float32
    use_amp = device.type == "cuda" and dtype_name in ("bfloat16", "float16")

    adapter_name = evaluation.get("world_model_adapter")
    LOGGER.info(
        "loading world model: adapter=%s checkpoint=%s device=%s",
        adapter_name or config.get("evaluation", {}).get("world_model_adapter"),
        args.checkpoint,
        device,
    )
    model, checkpoint_metadata = load_world_model(
        config, args.checkpoint, device, adapter_name=adapter_name
    )

    crop_size = int(config["data"].get("crop_size", 256))
    transform = make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0),
                                random_resize_scale=(1.0, 1.0), reprob=0.0, auto_augment=False,
                                motion_shift=False, crop_size=crop_size)
    LOGGER.info("building dataset: split=readout_data_v1")
    loader_started = time.monotonic()
    readout_loader = make_loader(config, "readout_data_v1", transform, args.batch_size,
                                 args.num_workers, args.max_readout_videos)
    LOGGER.info("dataset ready: split=readout_data_v1 samples=%d elapsed=%.1fs",
                len(readout_loader.dataset), time.monotonic() - loader_started)
    readout = extract_features(readout_loader, model, device, dtype, use_amp, "readout_data_v1",
                               args.progress_every_batches)
    LOGGER.info("building dataset: split=testdata_v1")
    loader_started = time.monotonic()
    test_loader = make_loader(config, "testdata_v1", transform, args.batch_size,
                              args.num_workers, args.max_test_videos)
    LOGGER.info("dataset ready: split=testdata_v1 samples=%d elapsed=%.1fs",
                len(test_loader.dataset), time.monotonic() - loader_started)
    test = extract_features(test_loader, model, device, dtype, use_amp, "testdata_v1",
                            args.progress_every_batches)
    output_dir = Path(args.output_dir or Path(args.checkpoint).parent / "evaluations/full_future_ocp")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.probe_protocol == "unified":
        probes = run_unified_probe_suite(readout, test, args, readout_device, output_dir)
        result = {
            "protocol": "unified_full_future_ocp_probe",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_epoch": int(checkpoint_metadata.get("checkpoint_epoch", -1)),
            "future_rgb_consumed": False,
            "label_definition": "any target_contacting_zone in [P+clip_gap, video_end)",
            "feature_definition": "canonical groups: current, rollout_mean, last_chunk, delta; full concatenates all four",
            "split_definition": "path-hash deterministic, class-stratified fit/threshold-validation; test never fits probe or threshold",
            "rollout_adapter": checkpoint_metadata.get(
                "rollout_adapter", "recipe-local world_model_adapter"
            ),
            "seed": int(args.seed),
            "future_frame_step": int(config["data"].get("future_frame_step", 4)),
            "clip_gap": int(config["data"].get("clip_gap", 0)),
            "num_readout": int(readout["labels"].numel()),
            "num_test": int(test["labels"].numel()),
            "test_label_positive_rate": float(test["labels"].mean()),
            "probes": probes,
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    classifier, mean, std = train_readout(readout["features"], readout["labels"], args.readout_epochs,
                                          args.readout_lr, args.readout_weight_decay,
                                          args.readout_log_every_epochs, readout_device)
    with torch.no_grad():
        readout_x = readout["features"].to(readout_device, non_blocking=True)
        test_x = test["features"].to(readout_device, non_blocking=True)
        readout_probs = torch.sigmoid(classifier((readout_x - mean) / std).squeeze(1)).cpu()
        test_probs = torch.sigmoid(classifier((test_x - mean) / std).squeeze(1)).cpu()
    threshold = args.fixed_threshold if args.fixed_threshold is not None else best_threshold(readout_probs, readout["labels"])
    metrics = binary_metrics(test_probs, test["labels"], threshold)
    human_metrics = {}
    test_human_accuracy = np.full(test["labels"].numel(), np.nan, dtype=np.float32)
    test_human_label = np.full(test["labels"].numel(), -1, dtype=np.int8)
    if args.humans_dir:
        human_accuracy = load_human_accuracy(args.humans_dir)
        human_metrics, test_human_accuracy, test_human_label = human_subset_metrics(
            test_probs, test["labels"], test["paths"], threshold, human_accuracy, args.human_hard_threshold,
        )
    LOGGER.info("test metrics computed: threshold=%.6f", threshold)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(checkpoint_metadata.get("checkpoint_epoch", -1)),
        "label_definition": "any target_contacting_zone in [P+clip_gap, video_end)",
        "feature_definition": "concat(current_mean, serial_future_mean, serial_last_chunk_mean, delta)",
        "rollout_adapter": checkpoint_metadata.get(
            "rollout_adapter", "recipe-local world_model_adapter"
        ),
        "future_frame_step": int(config["data"].get("future_frame_step", 4)),
        "clip_gap": int(config["data"].get("clip_gap", 0)),
        "num_readout": int(readout["labels"].numel()), "num_test": int(test["labels"].numel()),
        "test_label_positive_rate": float(test["labels"].mean()), "threshold": float(threshold),
        "threshold_source": "fixed" if args.fixed_threshold is not None else "readout_data_v1",
        "test_accuracy": float(metrics["accuracy"]),
        "test_balanced_accuracy": float(metrics["balanced_accuracy"]),
        "test_f1": float(metrics["f1"]), "test_auroc": auroc(test_probs.numpy(), test["labels"].numpy()),
        "test_precision": float(metrics["precision"]), "test_positive_recall": float(metrics["positive_recall"]),
        "test_negative_recall": float(metrics["negative_recall"]),
        "test_predicted_positive_rate": float(metrics["predicted_positive_rate"]),
        **human_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(output_dir / "test_predictions.npz", probabilities=test_probs.numpy(),
                        labels=test["labels"].numpy(), paths=np.asarray(test["paths"], dtype=object),
                        human_accuracy=test_human_accuracy,
                        human_label=test_human_label,
                        human_all_mask=np.isfinite(test_human_accuracy),
                        human_hard_mask=np.isfinite(test_human_accuracy) &
                        (test_human_accuracy <= args.human_hard_threshold))
    classifier_state = {key: value.detach().cpu() for key, value in classifier.state_dict().items()}
    torch.save({"classifier": classifier_state, "feature_mean": mean.detach().cpu(),
                "feature_std": std.detach().cpu(),
                "threshold": threshold}, output_dir / "readout.pt")
    LOGGER.info("evaluation outputs saved: output_dir=%s", output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
