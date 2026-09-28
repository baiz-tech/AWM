import pickle
import tempfile
import unittest
from pathlib import Path

import torch

from src.data.physionpp.evaluate.diagnostics import select_probe_features, stratified_probe_split
from src.data.physionpp.evaluate.eval_ocp import (
    extract_features,
    read_ocp_label,
    rollout_from_context,
)
from src.experiments.vjepa2_clevrer_naive_world_model_seed239.model import NativeVJEPA2Predictor


class _FakeEncoder(torch.nn.Module):
    embed_dim = 8

    def forward(self, videos, masks=None):
        video = videos[0]
        batch, _, frames, height, width = video.shape
        temporal = frames // 2
        spatial = (height // 16) ** 2
        values = video.reshape(
            batch, video.size(1), temporal, 2, height, width
        ).mean((1, 3, 4, 5))
        basis = torch.arange(1, self.embed_dim + 1, device=video.device).float()
        tokens = (values[:, :, None, None] * basis).expand(
            batch, temporal, spatial, self.embed_dim
        ).reshape(batch, temporal * spatial, self.embed_dim)
        if masks is None:
            return [tokens]
        index = masks[0][0]
        selected = torch.gather(
            tokens, 1, index.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )
        return [[selected]]


class _FakePredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, contexts, masks_x, masks_y):
        return [[contexts[0][0] + self.bias]]


class FullFutureOCPTest(unittest.TestCase):
    def test_unified_probe_split_and_views_are_path_stable(self):
        paths = [f"video-{index}" for index in range(12)]
        labels = torch.tensor([0] * 6 + [1] * 6)
        fit, validation = stratified_probe_split(paths, labels, 0.25, seed=239)
        self.assertFalse(set(fit.tolist()) & set(validation.tolist()))
        self.assertEqual(set(fit.tolist()) | set(validation.tolist()), set(range(12)))
        features = torch.arange(2 * 16.0).reshape(2, 16)
        self.assertEqual(select_probe_features(features, "full").shape, (2, 16))
        torch.testing.assert_close(
            select_probe_features(features, "delta"), features[:, 12:16]
        )
        with self.assertRaisesRegex(ValueError, "validation_fraction"):
            stratified_probe_split(paths, labels, 1.0, seed=239)

    def test_closed_loop_rollout_uses_current_only_and_masks_padding(self):
        model = NativeVJEPA2Predictor(
            _FakeEncoder(), _FakePredictor(), clip_frames=4,
            tubelet_size=2, crop_size=32, patch_size=16,
        ).eval()
        current = torch.randn(2, 3, 4, 32, 32)
        chunk_mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
        prediction = model.rollout_to_length(current, chunk_mask)

        self.assertEqual(prediction.shape, (2, 3, 8, 8))
        self.assertTrue(torch.count_nonzero(prediction[1, 1:]).eq(0))
        torch.testing.assert_close(prediction[0, 1], prediction[0, 0] + 0.1)

        shared_prediction = rollout_from_context(
            model, model.encode_current(current), chunk_mask
        )
        torch.testing.assert_close(shared_prediction, prediction)

    def test_ocp_label_is_any_contact_from_p_plus_gap_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "sample_img.mp4"
            video.touch()
            frames = {
                f"{index:04d}": {
                    "labels": {"target_contacting_zone": index == 15}
                }
                for index in range(30)
            }
            with (Path(directory) / "sample.pkl").open("wb") as handle:
                pickle.dump({
                    "static": {"start_frame_for_prediction": 10},
                    "frames": frames,
                }, handle)

            self.assertEqual(read_ocp_label(video, clip_gap=0), 1)
            self.assertEqual(read_ocp_label(video, clip_gap=6), 0)

    def test_feature_definition_matches_multi_future_ocp(self):
        model = NativeVJEPA2Predictor(
            _FakeEncoder(), _FakePredictor(), clip_frames=4,
            tubelet_size=2, crop_size=32, patch_size=16,
        ).eval()
        batch = {
            "current": torch.randn(2, 3, 4, 32, 32),
            "chunk_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
            "chunk_frame_mask": torch.tensor([
                [[1, 1, 1, 1], [1, 0, 0, 0]],
                [[1, 1, 0, 0], [0, 0, 0, 0]],
            ], dtype=torch.bool),
            "num_future_chunks": torch.tensor([2, 1]),
            "contact_label": torch.tensor([1, 0]),
            "path": ["positive_img.mp4", "negative_img.mp4"],
        }
        result = extract_features(
            [batch], model, torch.device("cpu"), torch.float32, False,
            "test", log_every_batches=1,
        )

        # current + all serial Future + last serial chunk + delta.
        self.assertEqual(result["features"].shape, (2, 4 * _FakeEncoder.embed_dim))
        self.assertEqual(result["labels"].tolist(), [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
