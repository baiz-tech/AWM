"""8-GPU joint object-slot training on existing Physion++ and EK100 caches.

Only the new object-slot module is optimized.  Existing V12/V23 checkpoints
are read as data caches, never overwritten.  EK100 supervision is noun-only
here so that the result measures object transfer rather than a relation head.
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from .model import ObjectSlotModel, slot_losses


class Physion(Dataset):
    def __init__(self, root, limit=0):
        self.files = sorted(Path(root).glob('sample_*.pt'))
        if limit: self.files = self.files[:limit]
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        z = torch.load(self.files[i], map_location='cpu', weights_only=False)
        return z['context'].float(), z['current_object_valid'].float(), torch.tensor(-1)


class EK100(Dataset):
    def __init__(self, path, limit=0):
        path = Path(path)
        if path.is_dir():
            rank = int(os.environ.get('RANK', '0'))
            shards = sorted(path.glob('features-rank*.pt'))
            path = shards[rank] if len(shards) > 1 and rank < len(shards) else shards[0]
        z = torch.load(path, map_location='cpu', weights_only=False)
        self.x = z['grid'].float()
        self.n = z['noun'].long()
        if limit: self.x, self.n = self.x[:limit], self.n[:limit]
    def __len__(self): return len(self.n)
    def __getitem__(self, i): return self.x[i], torch.zeros(8), self.n[i]


def setup():
    if 'RANK' not in os.environ: return 0, 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    r, w, l = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(l); dist.init_process_group('nccl')
    return r, w, torch.device('cuda', l)


def train_epoch(model, loaders, device, optimizer, noun_head):
    model.train(); noun_head.train(); total = 0.0; count = 0; noun_correct = 0; noun_count = 0
    # Alternating batches avoids EK100 dominating the smaller Physion cache.
    for name, loader in loaders:
        for x, presence, noun in loader:
            x, presence, noun = x.to(device, non_blocking=True), presence.to(device, non_blocking=True), noun.to(device, non_blocking=True)
            out = model(x)
            target_presence = presence[:, None, :].expand(-1, x.size(1), -1)
            # Presence is available for Physion; for EK it is deliberately omitted.
            base = slot_losses(out, target_features=x, target_presence=target_presence if name == 'physion' else None)
            # Fast differentiable proxy during training. Exact permutation
            # matching remains an offline diagnostic; Python Hungarian-style
            # matching inside every batch is prohibitively slow at this scale.
            if x.size(1) > 1:
                consistency = 1.0 - F.cosine_similarity(out.slots[:, 1:], out.slots[:, :-1], dim=-1).mean()
            else:
                consistency = x.sum() * 0.0
            loss = base['total'] + 0.2 * consistency
            if name == 'ek100':
                feat = out.slots.amax(2).mean(1)
                logits = noun_head(feat)
                loss = loss + F.cross_entropy(logits, noun)
                noun_correct += int((logits.argmax(1) == noun).sum()); noun_count += len(noun)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(noun_head.parameters()), 1.0); optimizer.step()
            total += float(loss.detach()) * len(x); count += len(x)
    return total / max(count, 1), noun_correct / max(noun_count, 1)


@torch.no_grad()
def evaluate(model, loader, device, noun_head):
    model.eval(); noun_head.eval(); correct = count = 0
    for x, _, noun in loader:
        x, noun = x.to(device), noun.to(device); out = model(x)
        logits = noun_head(out.slots.amax(2).mean(1)); correct += int((logits.argmax(1) == noun).sum()); count += len(noun)
    stat = torch.tensor([correct, count], device=device, dtype=torch.float64)
    if dist.is_initialized(): dist.all_reduce(stat)
    return float(stat[0] / stat[1].clamp_min(1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--physion-train', required=True); p.add_argument('--ek100-train', required=True); p.add_argument('--ek100-validation', required=True)
    p.add_argument('--output', required=True); p.add_argument('--epochs', type=int, default=5); p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--limit-per-rank', type=int, default=0); p.add_argument('--seed', type=int, default=239)
    a = p.parse_args(); rank, world, device = setup(); seed = a.seed + rank; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pt, ev = Physion(a.physion_train, a.limit_per_rank), EK100(a.ek100_train, a.limit_per_rank)
    ep = EK100(a.ek100_validation, a.limit_per_rank)
    ps = DistributedSampler(pt, world, rank, shuffle=True, seed=a.seed) if world > 1 else None
    # EK100 files are already rank-sharded. Do not shard them a second time.
    es = None
    vs = None
    pl = DataLoader(pt, a.batch_size, sampler=ps, shuffle=ps is None, num_workers=1, pin_memory=device.type == 'cuda')
    el = DataLoader(ev, a.batch_size, shuffle=True, num_workers=1, pin_memory=device.type == 'cuda')
    vl = DataLoader(ep, a.batch_size, shuffle=False, num_workers=1, pin_memory=device.type == 'cuda')
    # V23 cache stores the original EK100 noun ids (0..299); do not assume
    # the reduced 289-class vocabulary used by a separate readout protocol.
    local_max = torch.tensor([int(ev.n.max().item())], device=device)
    if dist.is_initialized(): dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
    noun_classes = int(local_max.item()) + 1
    model = ObjectSlotModel().to(device); noun_head = torch.nn.Linear(256, noun_classes).to(device)
    if world > 1:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
        noun_head = DDP(noun_head, device_ids=[device.index], find_unused_parameters=True)
    opt = torch.optim.AdamW(list(model.parameters()) + list(noun_head.parameters()), lr=1e-4, weight_decay=1e-4)
    out = Path(a.output)
    if rank == 0: out.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized(): dist.barrier()
    history = []
    for epoch in range(1, a.epochs + 1):
        if ps: ps.set_epoch(epoch)
        loss, train_noun = train_epoch(model, [('physion', pl), ('ek100', el)], device, opt, noun_head)
        val_noun = evaluate(model, vl, device, noun_head)
        rec = {'epoch': epoch, 'train_loss': loss, 'train_noun_top1': train_noun, 'ek100_validation_noun_top1': val_noun}
        if rank == 0:
            history.append(rec); (out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
            m = model.module if hasattr(model, 'module') else model; h = noun_head.module if hasattr(noun_head, 'module') else noun_head
            torch.save({'model': m.state_dict(), 'noun_head': h.state_dict(), 'epoch': epoch, 'protocol': 'object_slot_generalization_joint_v1'}, out / 'latest.pt')
            print(rec, flush=True)
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__': main()
