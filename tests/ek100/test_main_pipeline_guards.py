import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.awm_ek100_multiscale_adapter_seed239.output_guard import ensure_no_artifacts
from src.experiments.awm_ek100_multiscale_adapter_seed239.train_readout import run


class TinyReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 15)

    def forward(self, x):
        logits = self.head(x)
        return logits[:, :5], logits[:, 5:10], logits[:, 10:], logits.square().mean()


def _uneven_worker(rank, init_file):
    dist.init_process_group("gloo", init_method=Path(init_file).as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(239)
        model = DDP(TinyReadout())
        size = 5 if rank == 0 else 3
        x = torch.ones(size, 2)
        labels = torch.zeros(size, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, labels, labels), batch_size=2)
        weights = torch.ones(5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        action_map = {(0, 0): 0}
        train_metrics = run(model, loader, torch.device("cpu"), action_map, 5, weights, True, optimizer)
        validation_metrics = run(model, loader, torch.device("cpu"), action_map, 5, weights)
        for metrics in (train_metrics, validation_metrics):
            assert math.isfinite(metrics["loss"])
            assert metrics["verb_top5"] == 1.0
            assert metrics["noun_top5"] == 1.0
    finally:
        dist.destroy_process_group()


class MainPipelineGuardTests(unittest.TestCase):
    def test_launcher_metadata_does_not_block_new_ek100_stage(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "stage"
            (output / "configs").mkdir(parents=True)
            (output / "logs").mkdir()
            (output / "checkpoints").mkdir()
            for name in ("command.txt", "manifest.json", "environment.json"):
                (output / name).touch()
            ensure_no_artifacts(output)

            for artifact in ("features-rank00000.pt", "best.pt", "history.json", "cache_manifest.json"):
                (output / artifact).touch()
                with self.assertRaisesRegex(FileExistsError, "already exist"):
                    ensure_no_artifacts(output)
                (output / artifact).unlink()

    def test_uneven_distributed_readout_train_and_validation(self):
        with TemporaryDirectory() as directory:
            mp.spawn(_uneven_worker, args=(str(Path(directory) / "rendezvous"),), nprocs=2)


if __name__ == "__main__":
    unittest.main()
