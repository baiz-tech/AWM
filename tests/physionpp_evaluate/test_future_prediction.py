import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.physionpp.evaluate.eval_future_prediction import (
    METRIC_NAMES,
    extract_local_metrics,
    merge_metric_payloads,
    metric_rows,
    read_future_prediction_spec,
    summarize,
)


class FuturePredictionTests(unittest.TestCase):
    def test_standard_model_compares_prediction_and_teacher_target(self):
        class Model:
            def eval(self):
                return self

            def encode_current(self, current):
                return torch.ones(current.size(0), 2, 2)

            def predict_next(self, context):
                return context + 1.0

            def encode_target(self, current, future):
                self.target_shapes = (tuple(current.shape), tuple(future.shape))
                return torch.ones(current.size(0), 2, 2) + 2.0

        model = Model()
        result = extract_local_metrics(
            [{
                "current": torch.zeros(2, 3, 4, 8, 8),
                "future": torch.zeros(2, 3, 4, 8, 8),
                "path": ["a.mp4", "b.mp4"],
            }],
            model,
            torch.device("cpu"),
            torch.float32,
            False,
            1,
        )
        self.assertEqual(model.target_shapes, ((2, 3, 4, 8, 8), (2, 3, 4, 8, 8)))
        np.testing.assert_allclose(result["prediction"]["mse"], [1.0, 1.0])
        np.testing.assert_allclose(result["current_copy"]["mse"], [4.0, 4.0])

    def test_prediction_indices_follow_checkpoint_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "sample_img.mp4"
            video.touch()
            with (Path(directory) / "sample.pkl").open("wb") as handle:
                pickle.dump(
                    {"static": {"start_frame_for_prediction": 40}}, handle
                )
            spec = read_future_prediction_spec(
                video,
                clip_frames=4,
                current_frame_step=2,
                future_frame_step=4,
                clip_gap=0,
                video_length=80,
            )
        self.assertEqual(spec["current_indices"].tolist(), [32, 34, 36, 38])
        self.assertEqual(spec["future_indices"].tolist(), [40, 44, 48, 52])

    def test_metrics_are_per_video_and_match_known_values(self):
        target = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        source = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        rows = metric_rows(source, target)
        self.assertEqual(set(rows), set(METRIC_NAMES))
        torch.testing.assert_close(rows["mse"], torch.tensor([0.0, 2.0]))
        torch.testing.assert_close(rows["rmse"], torch.tensor([0.0, 2.0**0.5]))
        torch.testing.assert_close(rows["relative_l2"], torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(rows["cosine_similarity"], torch.tensor([1.0, 0.0]))

    def test_distributed_merge_sorts_and_rejects_duplicates(self):
        def payload(path, value):
            return {
                "paths": [path],
                "prediction": {
                    metric: np.asarray([value], dtype=np.float32)
                    for metric in METRIC_NAMES
                },
                "current_copy": {
                    metric: np.asarray([value + 1], dtype=np.float32)
                    for metric in METRIC_NAMES
                },
            }

        first, second = payload("b.mp4", 2), payload("a.mp4", 1)
        merged = merge_metric_payloads([first, second])
        self.assertEqual(merged["paths"].tolist(), ["a.mp4", "b.mp4"])
        self.assertEqual(merged["prediction"]["mse"].tolist(), [1.0, 2.0])
        with self.assertRaisesRegex(RuntimeError, "duplicate paths"):
            merge_metric_payloads([first, first])

    def test_summary_rejects_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            summarize(np.asarray([1.0, np.nan]))


if __name__ == "__main__":
    unittest.main()
