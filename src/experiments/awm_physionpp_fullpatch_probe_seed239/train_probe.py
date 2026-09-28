#!/usr/bin/env python3
"""DDP training for the Physion++ full-patch structured/OCP probe."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from .structured_probe import PhysionStructuredProbe, loss
from src.core.run_context import apply_cli_defaults, task_context

class Cache(Dataset):
    def __init__(self, root):
        self.files = sorted(Path(root).glob("sample_*.pt"))
        if not self.files: raise FileNotFoundError(f"no cache samples under {root}")
    def __len__(self): return len(self.files)
    def __getitem__(self, index): return torch.load(self.files[index], map_location="cpu", weights_only=False)

def setup():
    if "RANK" not in os.environ: return 0, 1, 0
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local); dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local}")); return rank, world, local

def run(model, loader, device, optimizer=None):
    model.train(optimizer is not None); sums, count = {}, 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items() if torch.is_tensor(value)}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            total, values = loss(model(batch["context"], batch["future"]), batch)
        if optimizer:
            optimizer.zero_grad(set_to_none=True); total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        count += len(batch["context"])
        for key, value in values.items(): sums[key] = sums.get(key, 0.) + float(value.detach()) * len(batch["context"])
    keys = sorted(sums); packed = torch.tensor([sums[key] for key in keys] + [count], device=device, dtype=torch.float64)
    if dist.is_initialized(): dist.all_reduce(packed)
    return {key: float(packed[i] / packed[-1].clamp_min(1)) for i, key in enumerate(keys)}

def main():
    ctx = task_context()
    parser = argparse.ArgumentParser(); parser.add_argument("--train-cache", required=True); parser.add_argument("--validation-cache", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=2); parser.add_argument("--learning-rate", type=float, default=2e-4); parser.add_argument("--seed", type=int, default=239)
    apply_cli_defaults(parser, ctx)
    args = parser.parse_args(); rank, world, local = setup(); torch.manual_seed(args.seed + rank); device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu"); output = Path(args.output_dir)
    if rank == 0: output.mkdir(parents=True, exist_ok=False)
    if dist.is_initialized(): dist.barrier()
    train_set, val_set = Cache(args.train_cache), Cache(args.validation_cache)
    train_sampler, val_sampler = DistributedSampler(train_set, world, rank, shuffle=True, seed=args.seed), DistributedSampler(val_set, world, rank, shuffle=False)
    train = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, num_workers=2, pin_memory=device.type == "cuda")
    validation = DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler, num_workers=2, pin_memory=device.type == "cuda")
    model = PhysionStructuredProbe().to(device); model = DDP(model, device_ids=[local]) if world > 1 else model
    optimizer, scheduler, best, history = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.04), None, float("inf"), []
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch); train_metrics, validation_metrics = run(model, train, device, optimizer), run(model, validation, device); scheduler.step(); record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}; history.append(record)
        if rank == 0:
            module = model.module if isinstance(model, DDP) else model
            config = {"input_dim": 1280, "hidden_dim": 256, "num_heads": 8, "object_layers": 3, "ffn_dim": 1024, "dropout": .1}
            payload = {"protocol": "physionpp_fullpatch_structured_probe_ocp_v1", "epoch": epoch, "model": module.state_dict(), "model_config": config, "validation_metrics": validation_metrics, "world_size": world}
            torch.save(payload, output / "latest.pt")
            if validation_metrics["total"] < best: best = validation_metrics["total"]; torch.save(payload, output / "best.pt")
            (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    if rank == 0: (output / "manifest.json").write_text(json.dumps({"protocol": "physionpp_fullpatch_structured_probe_ocp_v1", "best_validation_loss": best, "seed": args.seed, "world_size": world}, indent=2) + "\n")
    if dist.is_initialized(): dist.destroy_process_group()
if __name__ == "__main__": main()
