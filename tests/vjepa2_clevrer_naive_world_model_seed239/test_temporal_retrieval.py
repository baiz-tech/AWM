import types
import unittest

import numpy as np
import torch

from src.data.physionpp.evaluate.eval_same_video_temporal_retrieval import (
    _candidate_offsets,
    _evaluate,
    _gather,
    _rank,
    _temporal_pool,
    extract,
)


class TemporalRetrievalTest(unittest.TestCase):
    def test_candidate_grid_and_token_pooling(self):
        self.assertEqual(_candidate_offsets(-8, 12, 4, 0), [-8, -4, 0, 4, 8, 12])
        with self.assertRaisesRegex(ValueError, "candidate offset grid"):
            _candidate_offsets(-8, 12, 4, 1)
        tokens = torch.arange(2 * 8 * 3.0).reshape(2, 8, 3)
        self.assertEqual(_temporal_pool(tokens, 4).shape, (2, 4, 3))
        self.assertEqual(_rank(torch.tensor([0.1, 0.8, 0.2]), 2), (2, 1))

    def test_single_rank_gather_is_path_stable_and_metrics_are_complete(self):
        local = {
            "model_temporal_rank": [2, 1],
            "current_temporal_rank": [3, 2],
            "model_global_rank": [2, 1],
            "current_global_rank": [3, 2],
            "model_temporal_top_offset": [4, 0],
            "current_temporal_top_offset": [8, 4],
            "model_global_top_offset": [4, 0],
            "current_global_top_offset": [8, 4],
            "paths": ["b", "a"],
        }
        merged = _gather(local, rank=0, world_size=1)
        np.testing.assert_array_equal(merged["paths"], np.asarray(["a", "b"], dtype=object))
        args = types.SimpleNamespace(
            offset_min=-8,
            offset_max=12,
            offset_step=4,
            correct_offset=0,
            bootstrap_samples=8,
        )
        result = _evaluate(merged, "temporal", args)
        self.assertEqual(result["model"]["num_queries"], 2)
        self.assertIn("random_chance", result)
        self.assertIn("model_minus_current_bootstrap", result)
        self.assertIn("model_minus_random_bootstrap", result)

    def test_extract_uses_unified_primitives_and_isolates_future_targets(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encode_current_calls = 0
                self.predict_next_calls = 0
                self.encode_target_calls = 0

            def encode_current(self, current):
                self.encode_current_calls += 1
                basis = torch.arange(1, 4, device=current.device).float()
                return basis.view(1, 1, 3).expand(current.size(0), 4, 3)

            def predict_next(self, context):
                self.predict_next_calls += 1
                return context + 0.1

            def encode_target(self, current, future):
                self.encode_target_calls += 1
                value = future.float().mean(dim=(1, 2, 3, 4))
                basis = torch.arange(1, 4, device=future.device).float()
                return value[:, None, None] * basis.view(1, 1, 3).expand(-1, 4, -1)

        offsets = [-8, -4, 0, 4, 8, 12]
        batch = {
            "frames": torch.arange(1 * 3 * 14 * 2 * 2.0).reshape(1, 3, 14, 2, 2),
            "current_positions": torch.tensor([[0, 1]]),
            "candidate_positions": torch.tensor(
                [[[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]]
            ),
            "path": ["video"],
        }
        args = types.SimpleNamespace(
            correct_offset=0,
            candidate_batch_size=2,
            num_time_steps=2,
        )
        model = FakeModel()
        result = extract(
            [batch], model, torch.device("cpu"), False, torch.float32, args, offsets
        )
        self.assertEqual(result["paths"], ["video"])
        self.assertEqual(model.encode_current_calls, 1)
        self.assertEqual(model.predict_next_calls, 1)
        self.assertEqual(model.encode_target_calls, 3)


if __name__ == "__main__":
    unittest.main()
