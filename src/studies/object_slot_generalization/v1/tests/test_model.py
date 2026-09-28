import torch

from src.studies.object_slot_generalization.v1.model import ObjectSlotModel, greedy_slot_matching, slot_losses, temporal_slot_consistency


def test_shapes_and_gradients():
    model = ObjectSlotModel(input_dim=32, hidden_dim=64, slots=4, iterations=2)
    x = torch.randn(2, 3, 25, 32, requires_grad=True)
    out = model(x)
    assert out.slots.shape == (2, 3, 4, 64)
    assert out.assignment.shape == (2, 3, 25, 4)
    assert out.patch_reconstruction.shape == x.shape
    assert out.slot_reconstruction.shape == (2, 3, 4, 25, 32)
    assert out.centroids.shape == (2, 3, 4, 2)
    losses = slot_losses(out, target_features=x.detach())
    losses["total"].backward()
    assert x.grad is not None


def test_matching():
    a = torch.eye(4).unsqueeze(0)
    b = a[:, [2, 0, 3, 1]]
    m = greedy_slot_matching(a, b)
    assert m.shape == (1, 4)
    assert sorted(m[0].tolist()) == [0, 1, 2, 3]


def test_temporal_consistency():
    x = torch.randn(2, 3, 4, 16, requires_grad=True)
    z = temporal_slot_consistency(x)
    z.backward()
    assert torch.isfinite(z)
    assert x.grad is not None
