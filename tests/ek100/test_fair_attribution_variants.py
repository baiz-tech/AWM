import torch
from src.studies.fair_attribution.model_variants import AttributionHead, AttributionExtractor, _pool512


def test_fixed_pool_and_head_parameter_match():
    x = torch.randn(2, 3, 4, 1280)
    assert _pool512(x.mean((1, 2))).shape == (2, 512)
    heads = [AttributionHead(5, 7, [(0, 0), (1, 2)]) for _ in range(5)]
    assert len({sum(p.numel() for p in h.parameters()) for h in heads}) == 1


def test_variant_requirements_and_e0_shape():
    ex = AttributionExtractor("E0")
    assert ex(torch.randn(2, 3, 4, 1280)).shape == (2, 512)
