import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.data.physionpp.evaluate.diagnostics import (
    select_single_future_probe_features,
)
from src.data.physionpp.evaluate.eval_single_future_ocp import (
    _validate_checkpoint,
    extract_local_features,
    feature_result,
    merge_feature_payloads,
    read_single_future_spec,
    resolve_single_future_protocol,
)


class SingleFutureOCPTests(unittest.TestCase):
    def test_checkpoint_is_validated_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pt"
            with mock.patch("torch.distributed.get_rank", return_value=0), mock.patch(
                "torch.distributed.broadcast_object_list"
            ):
                with self.assertRaisesRegex(FileNotFoundError, str(missing)):
                    _validate_checkpoint(missing)

    def test_standard_model_predicts_exactly_one_future_clip(self):
        class Model:
            def __init__(self):
                self.encode_calls = 0
                self.predict_calls = 0

            def eval(self):
                return self

            def encode_current(self, current):
                self.encode_calls += 1
                return torch.ones(current.size(0), 4, 3)

            def predict_next(self, context):
                self.predict_calls += 1
                return context + 1.0

        model = Model()
        result = extract_local_features(
            [{
                "current": torch.zeros(2, 3, 4, 8, 8),
                "contact_label": torch.tensor([0, 1]),
                "path": ["a.mp4", "b.mp4"],
            }],
            model,
            torch.device("cpu"),
            torch.float32,
            False,
            "testdata_v1",
            1,
        )
        self.assertEqual(model.encode_calls, 1)
        self.assertEqual(model.predict_calls, 1)
        self.assertEqual(result["features"].shape, (2, 9))

    def _write_metadata(self, directory, contacts, missing=()):
        video = Path(directory) / "sample_img.mp4"
        video.touch()
        frames = {
            f"{index:04d}": {
                "labels": {"target_contacting_zone": index in contacts}
            }
            for index in range(40)
            if index not in missing
        }
        with (Path(directory) / "sample.pkl").open("wb") as handle:
            pickle.dump({
                "static": {"start_frame_for_prediction": 8},
                "frames": frames,
            }, handle)
        return video

    def _spec(self, video, label_scope="single_clip"):
        return read_single_future_spec(
            video,
            clip_frames=4,
            current_frame_step=2,
            future_frame_step=2,
            clip_gap=0,
            video_length=40,
            label_scope=label_scope,
        )

    def test_label_checks_intermediate_raw_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            video = self._write_metadata(directory, contacts={11})
            spec = self._spec(video)
        self.assertEqual(spec["future_indices"].tolist(), [8, 10, 12, 14])
        self.assertEqual((spec["label_start"], spec["label_end"]), (8, 14))
        self.assertEqual(spec["contact_label"], 1)

    def test_label_excludes_collision_after_last_sampled_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            video = self._write_metadata(directory, contacts={15})
            spec = self._spec(video)
        self.assertEqual(spec["contact_label"], 0)

    def test_all_future_label_includes_collision_after_predicted_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            video = self._write_metadata(directory, contacts={15})
            spec = self._spec(video, label_scope="all_future")
        self.assertEqual(spec["future_indices"].tolist(), [8, 10, 12, 14])
        self.assertEqual((spec["label_start"], spec["label_end"]), (8, 39))
        self.assertEqual(spec["label_scope"], "all_future")
        self.assertEqual(spec["contact_label"], 1)

    def test_missing_intermediate_frame_invalidates_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            video = self._write_metadata(directory, contacts=set(), missing={11})
            spec = self._spec(video)
        self.assertIsNone(spec)

    def test_all_future_label_requires_metadata_through_video_end(self):
        with tempfile.TemporaryDirectory() as directory:
            video = self._write_metadata(directory, contacts=set(), missing={20})
            clip_spec = self._spec(video)
            all_future_spec = self._spec(video, label_scope="all_future")
        self.assertIsNotNone(clip_spec)
        self.assertIsNone(all_future_spec)

    def test_feature_views_have_no_redundant_rollout_groups(self):
        current = torch.tensor([[[1.0, 3.0], [3.0, 5.0]]])
        predicted = current + 2.0
        result = feature_result(current, predicted, torch.tensor([1]), ["a.mp4"])
        self.assertEqual(result["features"].shape, (1, 6))
        torch.testing.assert_close(
            select_single_future_probe_features(result["features"], "current"),
            torch.tensor([[2.0, 4.0]]),
        )
        torch.testing.assert_close(
            select_single_future_probe_features(result["features"], "delta"),
            torch.tensor([[2.0, 2.0]]),
        )

    def test_distributed_merge_sorts_and_rejects_duplicates(self):
        payload_a = {
            "features": torch.tensor([[2.0]]),
            "labels": torch.tensor([0.0]),
            "paths": ["b.mp4"],
        }
        payload_b = {
            "features": torch.tensor([[1.0]]),
            "labels": torch.tensor([1.0]),
            "paths": ["a.mp4"],
        }
        merged = merge_feature_payloads([payload_a, payload_b])
        self.assertEqual(merged["paths"], ["a.mp4", "b.mp4"])
        with self.assertRaisesRegex(RuntimeError, "duplicate paths"):
            merge_feature_payloads([payload_a, payload_a])

    def test_protocol_prefers_model_attributes(self):
        config = {"data": {"future_frames": 16, "frame_step": 2}}

        class Model:
            clip_frames = 8
            current_frame_step = 3
            future_frame_step = 5

        self.assertEqual(
            resolve_single_future_protocol(config, Model()),
            {
                "_ocp_clip_frames": 8,
                "_ocp_current_frame_step": 3,
                "_ocp_future_frame_step": 5,
            },
        )

    def test_protocol_supports_recipe_config_fallbacks(self):
        config = {"data": {"future_frames": 8, "frame_step": 2}}
        protocol = resolve_single_future_protocol(config, object())
        self.assertEqual(
            protocol,
            {
                "_ocp_clip_frames": 8,
                "_ocp_current_frame_step": 2,
                "_ocp_future_frame_step": 2,
            },
        )
        self.assertEqual(config["data"]["_ocp_clip_frames"], 8)

    def test_protocol_rejects_non_positive_values(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            resolve_single_future_protocol(
                {"data": {"clip_frames": 0, "frame_step": 2}},
                object(),
            )


if __name__ == "__main__":
    unittest.main()
