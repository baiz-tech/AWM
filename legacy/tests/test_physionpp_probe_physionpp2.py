import torch

from src.experiments.vjepa2_clevrer_dynamics_decoder_seed239_physionpp2.scripts.physionpp.model import PhysionDecoder, loss


def test_physion_probe_shapes_and_loss():
    model = PhysionDecoder(hidden_dim=32, num_heads=4, ffn_dim=64)
    context = torch.randn(1, 8, 256, 1280)
    outputs = model(context, context)
    assert outputs["z_dyn"].shape == (1, 8, 8, 32)
    assert outputs["trajectory_2d"].shape == (1, 8, 16, 7)
    assert outputs["pair_distance_2d"].shape == (1, 28, 16)
    batch = {
        "object_present": torch.tensor([[True, True, False, False, False, False, False, False]]),
        "state_2d": torch.rand(1, 8, 16, 7),
        "state_valid": torch.ones(1, 8, 16, dtype=torch.bool),
        "pair_distance_2d": torch.rand(1, 8, 8, 16),
        "pair_valid": torch.ones(1, 8, 8, 16, dtype=torch.bool),
        "contact": torch.zeros(1, 8, 8, 16),
        "contact_valid": torch.ones(1, 8, 8, 16, dtype=torch.bool),
        "first_contact_class": torch.full((1, 8, 8), 16, dtype=torch.long),
        "ocp_label": torch.ones(1),
        "ocp_valid": torch.ones(1, dtype=torch.bool),
    }
    value, _ = loss(outputs, batch)
    assert torch.isfinite(value)
    value.backward()
