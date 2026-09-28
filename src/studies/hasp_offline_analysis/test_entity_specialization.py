import torch

from src.studies.hasp_offline_analysis.entity_specialization import (
    ade_fde,
    identity_switch_rate,
    noun_frequency_buckets,
    slot_object_matching,
)


def test_matching_and_trajectory_metrics():
    target = torch.tensor([[[0., 0.], [1., 0.]], [[3., 0.], [4., 0.]]])
    predicted = target[[1, 0]].clone()
    pairs = slot_object_matching(predicted, target, torch.ones(2, 2, dtype=torch.bool))
    assert sorted(pairs) == [(0, 1), (1, 0)]
    metrics = ade_fde(predicted[[0]], target[[1]], torch.ones(1, 2, dtype=torch.bool))
    assert metrics["ade"] == 0.0 and metrics["fde"] == 0.0


def test_identity_switch_rate_ignores_occlusion_gap():
    result = identity_switch_rate([[0, 1], [-1, -1], [0, 1]])
    assert result["switches"] == 0
    switched = identity_switch_rate([[0], [1]])
    assert switched["switches"] == 1


def test_frequency_buckets_are_train_only_and_deterministic():
    rows = [{"noun_class": 1}] * 8 + [{"noun_class": 2}] * 2 + [{"noun_class": 3}]
    buckets = noun_frequency_buckets(rows)
    assert buckets[1] == "head"
    assert set(buckets.values()) == {"head", "medium", "tail"}
