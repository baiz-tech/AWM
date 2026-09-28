import torch

from src.studies.hasp_offline_analysis.clevrer_relation_instance_ablation import mask_pair


def test_mask_pair_preserves_structured_and_flattenable_shape():
    context = torch.ones(1, 8, 256, 1280)
    support = torch.zeros(6, 256, dtype=torch.bool)
    support[0, :4] = True
    support[1, 4:8] = True
    masked = mask_pair(context, support, (0, 1))
    assert masked.shape == context.shape
    assert masked.reshape(masked.size(0), -1, masked.size(-1)).shape == (1, 2048, 1280)
