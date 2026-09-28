import torch

from src.studies.hasp_offline_analysis.physion_dynamic_selectivity import (
    dynamic_targets,
    temporal_intervention,
)


def test_dynamic_targets_speed_acceleration_and_ttc():
    state = torch.zeros(1, 3, 9)
    state[0, :, 3] = torch.tensor([1.0, 3.0, 5.0])
    row = {
        "future_object_state": state,
        "future_object_valid": torch.ones(1, 3, dtype=torch.bool),
        "future_time_to_contact": torch.tensor([[0.5]]),
        "future_time_to_contact_valid": torch.tensor([[True]]),
    }
    target = dynamic_targets(row)
    assert torch.isclose(target["speed"], torch.tensor(3.0))
    assert torch.isclose(target["acceleration"], torch.tensor(2.0))
    assert torch.isclose(target["ttc"], torch.tensor(0.5))
    assert target["contact_event"].item() == 1.0


def test_temporal_interventions_preserve_shape_and_expected_effects():
    context = torch.arange(2 * 8 * 1 * 1).reshape(2, 8, 1, 1).float()
    future = context + 100
    _, reversed_future = temporal_intervention(context, future, "future_reverse")
    assert torch.equal(reversed_future[:, 0], future[:, -1])
    _, static_future = temporal_intervention(context, future, "future_static")
    assert torch.allclose(static_future[:, 0], static_future[:, -1])
    _, repeated_future = temporal_intervention(context, future, "future_repeat")
    assert torch.equal(repeated_future[:, 0], repeated_future[:, -1])
    _, shuffled_a = temporal_intervention(context, future, "future_shuffle")
    _, shuffled_b = temporal_intervention(context, future, "future_shuffle")
    assert torch.equal(shuffled_a, shuffled_b)
