"""Pure helpers for deterministic, auditable OCP probes."""

from __future__ import annotations

import hashlib

import torch


FULL_FUTURE_FEATURE_GROUPS = ("current", "rollout_mean", "last_chunk", "delta")
SINGLE_FUTURE_FEATURE_GROUPS = ("current", "predicted_future", "delta")


def stratified_probe_split(paths, labels, validation_fraction=0.25, seed=239):
    """Return deterministic fit/threshold-validation indices."""
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be in (0,1)")
    labels = torch.as_tensor(labels).long().cpu()
    if len(paths) != int(labels.numel()):
        raise ValueError("paths and labels must have the same length")
    if len(set(map(str, paths))) != len(paths):
        raise ValueError("probe split requires unique video paths")
    fit, validation = [], []
    for label in sorted(labels.unique().tolist()):
        members = [index for index, value in enumerate(labels.tolist()) if value == label]
        if len(members) < 2:
            raise ValueError(f"label {label} has fewer than two samples")
        ordered = sorted(
            members,
            key=lambda index: hashlib.sha256(
                f"{int(seed)}\0{paths[index]}".encode("utf-8")
            ).digest(),
        )
        validation_count = round(len(ordered) * float(validation_fraction))
        validation_count = min(max(1, validation_count), len(ordered) - 1)
        validation.extend(ordered[:validation_count])
        fit.extend(ordered[validation_count:])
    return torch.tensor(sorted(fit)), torch.tensor(sorted(validation))


def split_full_future_features(features):
    """Split the canonical four-way full-Future OCP feature concatenation."""
    if features.ndim != 2:
        raise ValueError("features must be [N,D]")
    if features.size(1) % len(FULL_FUTURE_FEATURE_GROUPS):
        raise ValueError("feature dimension is not divisible by four canonical groups")
    width = features.size(1) // len(FULL_FUTURE_FEATURE_GROUPS)
    chunks = features.split(width, dim=1)
    return dict(zip(FULL_FUTURE_FEATURE_GROUPS, chunks))


def select_probe_features(features, variant):
    """Select an auditable feature view without changing extracted rollouts."""
    groups = split_full_future_features(features)
    if variant == "full":
        return features
    if variant not in groups:
        choices = ("full",) + FULL_FUTURE_FEATURE_GROUPS
        raise ValueError(f"unknown probe variant {variant!r}; choose from {choices}")
    return groups[variant]


def split_single_future_features(features):
    """Split current, one predicted Future clip, and their delta."""
    if features.ndim != 2:
        raise ValueError("features must be [N,D]")
    if features.size(1) % len(SINGLE_FUTURE_FEATURE_GROUPS):
        raise ValueError("feature dimension is not divisible by three canonical groups")
    width = features.size(1) // len(SINGLE_FUTURE_FEATURE_GROUPS)
    chunks = features.split(width, dim=1)
    return dict(zip(SINGLE_FUTURE_FEATURE_GROUPS, chunks))


def select_single_future_probe_features(features, variant):
    """Select a single-Future probe view without redundant rollout groups."""
    groups = split_single_future_features(features)
    if variant == "full":
        return features
    if variant not in groups:
        choices = ("full",) + SINGLE_FUTURE_FEATURE_GROUPS
        raise ValueError(f"unknown probe variant {variant!r}; choose from {choices}")
    return groups[variant]
