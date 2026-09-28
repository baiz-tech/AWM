import torch

from src.experiments.awm_clevrer_current_only_seed239.model import CLEVRERDecoder


def test_current_only_forward_contract():
    model = CLEVRERDecoder(hidden_dim=32, num_heads=4, ffn_dim=64)
    outputs = model(torch.randn(2, 8, 256, 1280))
    assert outputs["z_dyn"].shape == (2, 8, 8, 32)
    assert outputs["trajectory_2d"].shape == (2, 6, 16, 8)
    assert outputs["pair_tokens"].shape == (2, 15, 32)


def test_current_only_decoder_rejects_future_argument():
    model = CLEVRERDecoder(hidden_dim=32, num_heads=4, ffn_dim=64)
    try:
        model(torch.randn(1, 8, 256, 1280), torch.randn(1, 8, 256, 1280))
    except TypeError:
        return
    raise AssertionError("current-only decoder accepted a future latent")
