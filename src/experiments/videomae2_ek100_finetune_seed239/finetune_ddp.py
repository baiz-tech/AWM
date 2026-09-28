"""8-GPU DDP VideoMAEv2 EK100 fine-tuning with resumable checkpoints."""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .dataset import EK100Dataset, read_rows, split_indices
from .model import VideoMAEClassifier, build_encoder
from src.core.run_context import apply_cli_defaults, task_context


class GracefulStop(Exception):
    pass


def main():
    ctx = task_context()
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--workers", type=int)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    apply_cli_defaults(p, ctx)
    args = p.parse_args()

    rank = int(os.getenv("RANK", 0))
    world = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    cfg = yaml.safe_load(Path(args.config).read_text())
    base_seed = int(cfg.get("seed", 239))
    random.seed(base_seed + rank)
    np.random.seed(base_seed + rank)
    torch.manual_seed(base_seed + rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    data_cfg = cfg["data"]
    rows = read_rows(data_cfg["train_annotations"])
    train_idx, probe_idx = split_indices(data_cfg["train_annotations"], seed=base_seed)
    crop = int(data_cfg.get("crop_size", 224))
    train_ds = EK100Dataset(data_cfg["train_annotations"], data_cfg["video_root"], train_idx, crop)
    probe_ds = EK100Dataset(data_cfg["train_annotations"], data_cfg["video_root"], probe_idx, crop)
    final_ds = EK100Dataset(data_cfg["validation_annotations"], data_cfg["video_root"], None, crop)

    verbs = sorted({int(r["verb_class"]) for r in rows})
    nouns = sorted({int(r["noun_class"]) for r in rows})
    verb_to_idx = {v: i for i, v in enumerate(verbs)}
    noun_to_idx = {n: i for i, n in enumerate(nouns)}
    pairs = sorted({(verb_to_idx[int(r["verb_class"])], noun_to_idx[int(r["noun_class"])]) for r in rows})
    pair_to_idx = {pair: i for i, pair in enumerate(pairs)}
    batch_size = args.batch_size or int(cfg["training"].get("finetune_batch_size", 1))
    workers = args.workers if args.workers is not None else int(data_cfg.get("num_workers", 4))

    def make_loader(ds, shuffle):
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=shuffle, seed=base_seed)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=workers,
                          pin_memory=True, drop_last=shuffle, persistent_workers=workers > 0)

    encoder = build_encoder(cfg["model"], device)
    model = DDP(VideoMAEClassifier(encoder, len(verbs), len(nouns), len(pairs)).to(device),
                device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr or float(cfg["training"].get("learning_rate", 1e-4)),
                                  weight_decay=float(cfg["training"].get("weight_decay", 0.05)))
    epochs = args.epochs or int(cfg["training"].get("epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    out = Path(args.output)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_config.json").write_text(json.dumps({"config": str(Path(args.config).resolve()), "world_size": world,
                                                           "batch_size": batch_size, "epochs": epochs}, indent=2) + "\n")
    dist.barrier()

    history = []
    best_score = -1.0
    best_epoch = 0
    start_epoch = 0
    stop_requested = False

    resume_path = args.resume or (str(out / "latest.pt") if (out / "latest.pt").exists() else None)
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_score = float(ckpt.get("best_score", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        if rank == 0 and (out / "history.json").exists():
            history = json.loads((out / "history.json").read_text())
            logging.info("resumed from %s at epoch %d", resume_path, start_epoch)

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        if rank == 0:
            logging.warning("received signal %s; will save latest checkpoint and stop after current batch/epoch", signum)

    for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)

    def save_checkpoint(path, epoch, summary=None, score=None):
        if rank != 0:
            return
        state = {"epoch": epoch, "model": model.module.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "best_score": best_score,
                 "best_epoch": best_epoch, "protocol": "videomaev2_ek100_official_style_finetune_ddp",
                 "verb_vocabulary": verbs, "noun_vocabulary": nouns, "action_map": pair_to_idx,
                 "summary": summary}
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)

    def write_history():
        if rank == 0:
            tmp = out / "history.json.tmp"
            tmp.write_text(json.dumps(history, indent=2) + "\n")
            os.replace(tmp, out / "history.json")

    def run_epoch(ds, train, epoch):
        loader = make_loader(ds, train)
        loader.sampler.set_epoch(epoch)
        model.train(train)
        # loss_sum, count, verb@1, verb@5, noun@1, noun@5, action@1, action@5, action_valid
        sums = torch.zeros(9, device=device, dtype=torch.float64)
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch_no, batch in enumerate(loader, 1):
                yv = torch.tensor([verb_to_idx[int(x)] for x in batch["verb"]], device=device)
                yn = torch.tensor([noun_to_idx[int(x)] for x in batch["noun"]], device=device)
                pair_labels = []
                valid_action = []
                for v, n in zip(yv.tolist(), yn.tolist()):
                    key = (v, n)
                    pair_labels.append(pair_to_idx.get(key, -1))
                    valid_action.append(key in pair_to_idx)
                ya = torch.tensor([max(0, x) for x in pair_labels], device=device)
                video = batch["video"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                    logits = model(video)
                    loss = (torch.nn.functional.cross_entropy(logits[0], yv) +
                            torch.nn.functional.cross_entropy(logits[1], yn) +
                            0.2 * torch.nn.functional.cross_entropy(logits[2], ya))
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                def topk(logit, target):
                    k = min(5, logit.shape[1])
                    hit = logit.topk(k, 1).indices.eq(target[:, None])
                    return hit[:, 0].sum(), hit.any(1).sum()
                v1, v5 = topk(logits[0], yv); n1, n5 = topk(logits[1], yn)
                mask = torch.tensor(valid_action, device=device, dtype=torch.bool)
                if mask.any():
                    a1, a5 = topk(logits[2][mask], ya[mask]); av = mask.sum()
                else:
                    a1 = a5 = av = torch.tensor(0, device=device)
                bsz = yv.numel()
                sums += torch.tensor([float(loss.detach()) * bsz, bsz, v1, v5, n1, n5, a1, a5, av], device=device)
                if rank == 0 and (batch_no == 1 or batch_no % 50 == 0 or batch_no == len(loader)):
                    logging.info(json.dumps({"stage": "train" if train else "validation", "epoch": epoch + 1,
                                             "batch": batch_no, "batches": len(loader), "batch_loss": float(loss.detach())}))
        dist.all_reduce(sums)
        denom = sums[1].clamp_min(1)
        action_denom = sums[8].clamp_min(1)
        return {"loss": float(sums[0] / denom), "verb_top1": float(sums[2] / denom), "verb_top5": float(sums[3] / denom),
                "noun_top1": float(sums[4] / denom), "noun_top5": float(sums[5] / denom),
                "action_top1": float(sums[6] / action_denom), "action_top5": float(sums[7] / action_denom),
                "action_valid_samples": int(sums[8].item())}

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    completed = False
    for epoch in range(start_epoch, epochs):
        train_metrics = run_epoch(train_ds, True, epoch)
        val_metrics = run_epoch(probe_ds, False, epoch)
        scheduler.step()
        score = (val_metrics["verb_top1"] + val_metrics["noun_top1"]) / 2.0
        if rank == 0:
            row = {"stage": "epoch_summary", "epoch": epoch + 1, "train": train_metrics,
                   "validation": val_metrics, "lr": scheduler.get_last_lr()[0]}
            history.append(row)
            if score > best_score:
                best_score, best_epoch = score, epoch + 1
            write_history()
            save_checkpoint(out / "latest.pt", epoch + 1, row, score)
            logging.info(json.dumps(row))
            if best_epoch == epoch + 1:
                save_checkpoint(out / "best.pt", epoch + 1, row, score)
                logging.info(json.dumps({"stage": "checkpoint", "epoch": epoch + 1, "best_score": best_score,
                                         "path": str(out / "best.pt")}))
        # A signal is handled at an epoch boundary so every rank reaches the
        # same collectives and the just-finished epoch is safely persisted.
        stop_tensor = torch.tensor([int(stop_requested) if rank == 0 else 0], device=device)
        dist.broadcast(stop_tensor, src=0)
        dist.barrier()
        if bool(stop_tensor.item()):
            if rank == 0:
                logging.warning("training interrupted; latest checkpoint retained at %s", out / "latest.pt")
            break
    else:
        completed = True

    if completed:
        # Evaluate the in-memory best checkpoint on the official validation split.
        if rank == 0:
            best_payload = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        else:
            best_payload = None
        payload = [best_payload]
        dist.broadcast_object_list(payload, src=0)
        model.module.load_state_dict(payload[0]["model"])
        final_metrics = run_epoch(final_ds, False, epochs)
        if rank == 0:
            result = {"protocol": "videomaev2_ek100_official_style_finetune_ddp", "best_epoch": best_epoch,
                      "best_probe_score": best_score, "final_validation": final_metrics, "train_samples": len(train_ds),
                      "probe_validation_samples": len(probe_ds), "official_validation_samples": len(final_ds),
                      "world_size": world, "encoder_finetuned": True}
            (out / "metrics.json.tmp").write_text(json.dumps(result, indent=2) + "\n")
            os.replace(out / "metrics.json.tmp", out / "metrics.json")
            logging.info(json.dumps({"stage": "complete", "metrics_path": str(out / "metrics.json"), "result": result}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
