import tempfile
import unittest
from pathlib import Path

import torch

from src.experiments.vjepa2_clevrer_naive_world_model_seed239.train import save_checkpoint


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = torch.nn.Linear(2, 2)


class BestCheckpointTest(unittest.TestCase):
    def test_checkpoint_records_best_validation_state(self):
        model = _Model()
        optimizer = torch.optim.AdamW(model.predictor.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                scaler=None,
                epoch=3,
                step=17,
                config={"folder": directory},
                source_epoch=4,
                best_validation_loss=0.125,
                best_epoch=3,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(payload["epoch"], 3)
        self.assertEqual(payload["best_epoch"], 3)
        self.assertAlmostEqual(payload["best_validation_loss"], 0.125)
        self.assertIn("predictor", payload)


if __name__ == "__main__":
    unittest.main()
