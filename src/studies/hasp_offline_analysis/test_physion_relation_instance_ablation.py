import torch

from src.studies.hasp_offline_analysis.physion_relation_instance_ablation import (
    mask_pair,
    paired_bootstrap,
    select_pairs,
)


def _row():
    valid = torch.zeros(8, 8, 16, dtype=torch.bool)
    contact = torch.zeros(8, 8, 16)
    for i, j in ((0, 1), (0, 2), (0, 3)):
        valid[i, j] = valid[j, i] = True
    contact[0, 1, 3] = contact[1, 0, 3] = 1
    return {"future_contact_valid": valid, "future_contact": contact}


def test_select_pairs_returns_triplet_and_area_metadata():
    support = torch.zeros(8, 256, dtype=torch.bool)
    support[0, :10] = True; support[1, 10:20] = True
    support[2, :5] = True; support[3, 20:30] = True
    selected = select_pairs(_row(), support, seed=239)
    assert selected is not None
    assert set(selected) == {"contact", "noncontact", "random", "areas"}
    assert selected["contact"] == (0, 1)


def test_mask_pair_changes_only_selected_patch_union():
    context = torch.ones(1, 8, 256, 2)
    support = torch.zeros(8, 256, dtype=torch.bool)
    support[0, :3] = True; support[1, 3:5] = True
    masked = mask_pair(context, support, (0, 1))
    assert torch.allclose(masked[:, :, :5], torch.ones(1, 8, 5, 2))
    context[:, :, :5] = 7
    masked = mask_pair(context, support, (0, 1))
    assert torch.all(masked[:, :, :5] != 7)
    assert torch.all(masked[:, :, 5:] == 1)


def test_paired_bootstrap_reports_finite_ci():
    result = paired_bootstrap([1.0, 2.0, 3.0], draws=100)
    assert result["n"] == 3
    assert result["ci95"][0] <= result["mean"] <= result["ci95"][1]
