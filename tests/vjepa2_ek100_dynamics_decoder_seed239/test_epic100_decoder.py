import torch

from src.experiments.vjepa2_ek100_dynamics_decoder_seed239.scripts.epic100.event_data import event_windows, uniform_indices
from src.experiments.vjepa2_ek100_dynamics_decoder_seed239.scripts.epic100.model import Epic100Decoder, decoder_loss


def test_event_centered_sampling_matches_v4_protocol():
    current, future = event_windows(10.0, 22.0)
    assert current == (12.0, 16.0)
    assert future == (16.0, 22.0)
    current_indices = uniform_indices(*current, 30.0, 1000)
    future_indices = uniform_indices(*future, 30.0, 1000, require_unique=False)
    assert len(current_indices) == len(set(current_indices)) == 16
    assert current_indices.max() < future_indices.min()


def test_decoder_output_and_joint_loss_shapes():
    model = Epic100Decoder(
        num_categories=305, num_verbs=97, num_nouns=289, num_actions=3530,
        hidden_dim=32, num_heads=4, ffn_dim=64, num_slots=8,
    )
    context = torch.randn(1, 8, 256, 1280)
    outputs = model(context, context)
    assert outputs["z_dyn"].shape == (1, 8, 8, 32)
    assert outputs["trajectory_2d"].shape == (1, 8, 16, 8)
    assert outputs["pair_distance_2d"].shape == (1, 28, 16)
    batch = {
        "object_present": torch.tensor([[True] + [False] * 7]),
        "category": torch.tensor([[1] + [-1] * 7]),
        "state": torch.zeros(1, 8, 16, 8),
        "state_valid": torch.zeros(1, 8, 16, dtype=torch.bool),
        "pair_distance": torch.zeros(1, 8, 8, 16),
        "pair_valid": torch.zeros(1, 8, 8, 16, dtype=torch.bool),
        "verb_label": torch.tensor([0]), "noun_label": torch.tensor([0]),
        "action_label": torch.tensor([0]),
    }
    batch["state_valid"][:, 0] = True
    total, losses, assignment = decoder_loss(outputs, batch)
    assert torch.isfinite(total)
    assert assignment.shape == (1, 8)
    assert {"category", "center", "verb", "noun", "action"}.issubset(losses)
