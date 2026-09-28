import torch
from src.studies.hasp_offline_analysis.clevrer_dynamic_selectivity import intervention, targets


def test_targets_compute_speed_acceleration_and_ttc():
    state = torch.zeros(6, 16, 8)
    state[0, :, 5] = torch.arange(16).float()
    valid = torch.zeros(1, 6, 16, dtype=torch.bool); valid[:, 0] = True
    row = {"state": state[None], "state_valid": valid, "first_contact_class": torch.full((1, 6, 6), 16, dtype=torch.long)}
    row["first_contact_class"][0, 0, 1] = 4
    out = targets(row, 0)
    assert out["speed"].item() > 0
    assert out["acceleration"].item() == 1.0
    assert torch.isclose(out["ttc"], torch.tensor(4 / 15))
    assert out["contact_event"].item() == 1.0


def test_interventions_are_deterministic_and_preserve_shape():
    c = torch.arange(8).reshape(1, 8, 1, 1).float(); f = c + 100
    assert torch.equal(intervention(c, f, "future_reverse")[1][:, 0], f[:, -1])
    assert torch.allclose(intervention(c, f, "future_static")[1][:, 0], intervention(c, f, "future_static")[1][:, -1])
    assert torch.equal(intervention(c, f, "future_shuffle")[1], intervention(c, f, "future_shuffle")[1])
