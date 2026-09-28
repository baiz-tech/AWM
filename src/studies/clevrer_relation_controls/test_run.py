import torch

from .run import VARIANTS, _auroc


def test_control_variants_cover_requested_combinations():
    assert VARIANTS["Relation-only"] == ("relation",)
    assert VARIANTS["Entity+Dynamic"] == ("entity", "dynamic")
    assert VARIANTS["Full"] == ("entity", "dynamic", "relation")


def test_auroc_returns_none_for_single_class():
    assert _auroc(torch.ones(3), torch.randn(3)) is None
