import types
import unittest
from unittest.mock import patch

import torch

from src.data.physionpp.evaluate.world_model_adapter import load_world_model


class _ContractModel(torch.nn.Module):
    clip_frames = 4
    temporal_steps = 2
    spatial_tokens = 4
    tubelet_size = 2
    crop_size = 32
    patch_size = 16
    tokens_per_chunk = 8

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)

    def encode_current(self, current):
        return current

    def predict_next(self, context):
        return context

    def encode_target(self, current, future):
        return future


class WorldModelAdapterLoaderTest(unittest.TestCase):
    def test_loads_metadata_and_freezes_model(self):
        model = _ContractModel()
        module = types.SimpleNamespace(
            load_world_model=lambda config, checkpoint, device: (
                model,
                {"checkpoint_epoch": 3},
            )
        )
        config = {"evaluation": {"world_model_adapter": "recipe.fake_adapter"}}
        with patch("importlib.import_module", return_value=module):
            loaded, metadata = load_world_model(
                config, "checkpoint.pt", torch.device("cpu")
            )
        self.assertIs(loaded, model)
        self.assertEqual(metadata["checkpoint_epoch"], 3)
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_requires_explicit_adapter(self):
        with self.assertRaisesRegex(ValueError, "evaluation.world_model_adapter"):
            load_world_model({}, "checkpoint.pt", torch.device("cpu"))

    def test_rejects_invalid_return_contract(self):
        module = types.SimpleNamespace(load_world_model=lambda *args: torch.nn.Identity())
        config = {"evaluation": {"world_model_adapter": "recipe.fake_adapter"}}
        with patch("importlib.import_module", return_value=module):
            with self.assertRaisesRegex(TypeError, r"return \(model, metadata\)"):
                load_world_model(config, "checkpoint.pt", torch.device("cpu"))

    def test_rejects_model_without_unified_methods(self):
        module = types.SimpleNamespace(
            load_world_model=lambda *args: (torch.nn.Identity(), {})
        )
        config = {"evaluation": {"world_model_adapter": "recipe.fake_adapter"}}
        with patch("importlib.import_module", return_value=module):
            with self.assertRaisesRegex(AttributeError, "encode_current"):
                load_world_model(config, "checkpoint.pt", torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
