import torch

from .run import _group


def test_interaction_group_prioritizes_contact():
    valid = torch.ones(4, dtype=torch.bool)
    assert _group(torch.tensor([1.0, 0.8, 0.6, 0.4]), valid, 2) == "contact"


def test_interaction_group_uses_distance_trend_for_no_contact():
    valid = torch.ones(4, dtype=torch.bool)
    assert _group(torch.tensor([1.0, 0.8, 0.6, 0.4]), valid, 16) == "approaching"
    assert _group(torch.tensor([0.4, 0.6, 0.8, 1.0]), valid, 16) == "separating"
