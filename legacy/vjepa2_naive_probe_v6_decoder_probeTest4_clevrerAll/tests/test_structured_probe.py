import subprocess
from pathlib import Path

import torch

from recipe.shared.evaluate.clevrer_v2.qa_model import AloeLikeQA
from recipe.vjepa2_naive_probe_v6_decoder_probeTest4_clevrerAll.scripts.clevrer.model import (
    CLEVRERDecoder,
    structured_probe_loss,
    stable_pair_distance,
)
from recipe.vjepa2_naive_probe_v6_decoder_probeTest4_clevrerAll.scripts.clevrer.probe_adapter import (
    StructuredProbeRunner,
)
from recipe.vjepa2_naive_probe_v5_decoder_clevrerAll.scripts.clevrer.structured_probe_evaluation import (
    _binary_metrics,
)
from recipe.vjepa2_naive_probe_v6_decoder_probeTest4_clevrerAll.scripts.clevrer.sampling_protocol import sampled_indices


REPO_ROOT = Path(__file__).resolve().parents[3]

def test_sampling_protocol_is_exact_and_aligned():
    current, future = sampled_indices(0)
    assert current.tolist() == list(range(0, 96, 4))
    assert future.tolist() == list(range(96, 128, 4))


def test_public_launchers_support_three_output_modes_with_workspace_default():
    script = REPO_ROOT / "recipe/vjepa2_naive_probe_v5_decoder_clevrerAll/scripts/clevrer/run_all.sh"
    expected_roots = {
        None: "/data/shyang/outputs/vjepa2-baiz/vjepa2_naive_probe_v5_decoder_clevrerAll",
        "workspace": str(REPO_ROOT / "outputs/runs/vjepa2_naive_probe_v5_decoder_clevrerAll"),
        "data": "/data/shyang/outputs/vjepa2-baiz/vjepa2_naive_probe_v5_decoder_clevrerAll",
        "both": "/data/shyang/outputs/vjepa2-baiz/vjepa2_naive_probe_v5_decoder_clevrerAll",
    }
    for mode, expected_root in expected_roots.items():
        command = ["bash", str(script), "--dry-run"]
        if mode is not None:
            command[2:2] = ["--output-mode", mode]
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        expected_mode = "both" if mode is None else mode
        assert f"output_mode={expected_mode}" in result.stdout
        assert f"primary_experiment_root={expected_root}" in result.stdout


def test_structured_probe_shapes_and_validity_fallback(tmp_path):
    model = CLEVRERDecoder(
        input_dim=32,
        hidden_dim=16,
        num_heads=4,
        ffn_dim=32,
        num_slots=6,
        slot_depth=1,
        transition_depth=1,
        interaction_depth=1,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        model.probes.presence_head.weight.zero_()
        model.probes.presence_head.bias.fill_(-10.0)

    runner = StructuredProbeRunner(model, tmp_path / "probe.pt")
    outputs = runner(
        torch.randn(1, 8, 256, 32),
        torch.randn(1, 8, 256, 32),
    )

    assert outputs["object_tokens"].shape == (1, 6, 16)
    assert outputs["pair_tokens"].shape == (1, 15, 16)
    assert outputs["trajectory_2d"].shape == (1, 6, 32, 8)
    assert outputs["pair_distance_2d"].shape == (1, 15, 32)
    assert outputs["contact_gt_event"].shape == (1, 15, 32)
    assert outputs["first_contact_logits"].shape == (1, 15, 33)
    assert outputs["presence_logits"].shape == (1, 6, 32)
    assert outputs["presence_summary_logits"].shape == (1, 6)
    assert outputs["color_logits"].shape == (1, 6, 8)
    assert outputs["material_logits"].shape == (1, 6, 2)
    assert outputs["shape_logits"].shape == (1, 6, 3)
    tokens = torch.randn(1, 16, 256, 32)
    source_ids = torch.tensor([[0] * 12 + [1] * 4])
    assert model(tokens, source_ids)["z_dyn"].shape == (1, 16, 6, 16)
    assert outputs["object_valid"].sum().item() == 2
    assert outputs["pair_valid"].sum().item() == 1
    assert not hasattr(model.probes, "object_readout")
    assert not hasattr(model.probes, "time_readout")


def test_structured_supervised_outputs_join_main_qa_sequence():
    model = AloeLikeQA(
        visual_dim=32,
        vocabulary_size=20,
        answer_classes=0,
        pad_id=0,
        learned_tokens_per_step=2,
        use_probe_tokens=True,
        probe_token_mode="structured_supervised_sequence_v2",
        probe_hidden_dim=16,
        answer_head_activation="gelu",
        input_dim=4,
        num_layers=1,
        num_heads=4,
        ffn_dim=32,
        cls_mlp_size=16,
        dropout=0.0,
    ).eval()
    probe_outputs = {
        "object_tokens": torch.randn(1, 6, 16),
        "pair_tokens": torch.randn(1, 15, 16),
        "object_valid": torch.tensor([[True, True, False, False, False, False]]),
        "pair_valid": torch.tensor([[True] + [False] * 14]),
        "presence_logits": torch.randn(1, 6),
        "color_logits": torch.randn(1, 6, 8),
        "material_logits": torch.randn(1, 6, 2),
        "shape_logits": torch.randn(1, 6, 3),
        "trajectory_2d": torch.randn(1, 6, 16, 8),
        "pair_distance_2d": torch.rand(1, 15, 16),
        "contact_gt_event": torch.randn(1, 15, 16),
        "first_contact_logits": torch.randn(1, 15, 17),
    }
    tokens, pad_mask = model._structured_sequence_tokens(probe_outputs)

    assert tokens.shape == (1, 357, 16)
    assert pad_mask.shape == (1, 357)
    assert (~pad_mask).sum().item() == 2 + 2 * 16 + 1 + 16

    batch = {
        "cls_video_emb": torch.empty(0, 16, 256, 32),
        "cls_q_tokens": torch.empty(0, 32, dtype=torch.long),
        "cls_label": torch.empty(0, dtype=torch.long),
        "mc_video_emb": torch.randn(1, 16, 256, 32),
        "mc_q_tokens": torch.randint(1, 20, (1, 32)),
        "mc_label": torch.ones(1, dtype=torch.long),
        "mc_flag": torch.zeros(1, dtype=torch.long),
        "mc_probe_outputs": probe_outputs,
    }
    with torch.no_grad():
        outputs = model(batch)
    assert outputs["mc_answer_logits"].shape == (1,)
    assert torch.isfinite(outputs["mc_answer_logits"]).all()


def test_zero_pair_distance_has_finite_gradient():
    relative_center = torch.zeros(4, 2, requires_grad=True)
    distance = stable_pair_distance(relative_center)
    torch.nn.functional.smooth_l1_loss(distance, torch.ones_like(distance)).backward()

    assert distance.eq(0).all()
    assert torch.isfinite(relative_center.grad).all()


def test_binary_metrics_group_tied_scores():
    metrics = _binary_metrics(
        torch.tensor([2.0, 2.0, 0.0, -1.0]),
        torch.tensor([1, 0, 1, 0]),
    )

    assert abs(metrics["auroc"] - 0.625) < 1e-7
    assert abs(metrics["average_precision"] - 7.0 / 12.0) < 1e-7
