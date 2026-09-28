from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader

from src.data.ek100.labels import encode_labels
from src.data.ek100.dataset import read_rows
from .model_variants import AttributionExtractor, AttributionHead, VARIANTS, load_components
from src.core.run_context import apply_cli_defaults, task_context


class GridDataset(Dataset):
    def __init__(self, path: str):
        files = sorted(Path(path).glob("features-rank*.pt"))
        if not files: raise FileNotFoundError(f"no features-rank*.pt under {path}")
        data = [torch.load(p, map_location="cpu", weights_only=False) for p in files]
        self.x = torch.cat([z["grid"].float() for z in data]); self.v = torch.cat([z["verb"].long() for z in data]); self.n = torch.cat([z["noun"].long() for z in data])
    def __len__(self): return len(self.v)
    def __getitem__(self, i): return self.x[i], self.v[i], self.n[i]


def metrics(logits, verb, noun, action, action_target):
    out = {"verb_top1": float((logits[0].argmax(1) == verb).float().mean()), "verb_top5": float((logits[0].topk(min(5, logits[0].size(1)), 1).indices == verb[:, None]).any(1).float().mean()), "noun_top1": float((logits[1].argmax(1) == noun).float().mean()), "noun_top5": float((logits[1].topk(min(5, logits[1].size(1)), 1).indices == noun[:, None]).any(1).float().mean())}
    valid = action_target >= 0
    out["action_top1"] = float((action[valid].argmax(1) == action_target[valid]).float().mean()) if valid.any() else None
    out["samples"] = len(verb)
    return out


def run_variant(name, extractor, head, loader, device, action_map, train=False, optimizer=None):
    total = {k: 0.0 for k in ("loss", "verb_top1", "verb_top5", "noun_top1", "noun_top5", "action_top1")}; count = 0
    head.train(train)
    for x, v, n in loader:
        x, v, n = x.to(device), v.to(device), n.to(device)
        target = torch.tensor([action_map.get((int(a), int(b)), -1) for a, b in zip(v.cpu(), n.cpu())], device=device)
        feature = extractor(x)
        logits = head(feature)
        valid = target >= 0
        loss = F.cross_entropy(logits[0], v) + F.cross_entropy(logits[1], n)
        if valid.any(): loss = loss + 0.2 * F.cross_entropy(logits[2][valid], target[valid])
        if train:
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        m = metrics(logits, v, n, logits[2], target); b = len(v); count += b
        total["loss"] += float(loss.detach()) * b
        for k in total:
            if k != "loss" and m[k] is not None: total[k] += m[k] * b
    return {k: (v / count if k != "action_top1" or count else v) for k, v in total.items()}


def main():
    ctx=task_context();p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--train", required=True); p.add_argument("--validation", required=True); p.add_argument("--readout", required=True); p.add_argument("--adapter", required=True); p.add_argument("--output", required=True); p.add_argument("--variant", choices=VARIANTS, required=True); p.add_argument("--epochs", type=int, default=12); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--seed", type=int, default=239); p.add_argument("--device", default="cuda"); apply_cli_defaults(p,ctx);a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); device = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    cfg = yaml.safe_load(Path(a.config).read_text()); rows = read_rows(cfg["data"]["train_annotations"]); vv = sorted({int(r["verb_class"]) for r in rows}); nv = sorted({int(r["noun_class"]) for r in rows}); pairs = sorted({(vv.index(int(r["verb_class"])), nv.index(int(r["noun_class"]))) for r in rows}); action_map = {pair: i for i, pair in enumerate(pairs)}
    tr, va = GridDataset(a.train), GridDataset(a.validation); tr.v, tr.n = encode_labels(tr.v, vv), encode_labels(tr.n, nv); va.v, va.n = encode_labels(va.v, vv), encode_labels(va.n, nv)
    comp = load_components(a.readout, a.adapter, device); ex = AttributionExtractor(a.variant, comp.readout_world, comp.shared).to(device); head = AttributionHead(len(vv), len(nv), list(action_map)).to(device); opt = torch.optim.AdamW(head.parameters(), lr=float(cfg["training"].get("attribution_learning_rate", 3e-4)), weight_decay=float(cfg["training"].get("weight_decay", 5e-4)))
    train_loader, val_loader = DataLoader(tr, a.batch_size, shuffle=True), DataLoader(va, a.batch_size)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True); history = []; best = -1.0
    for epoch in range(1, a.epochs + 1):
        tm = run_variant(a.variant, ex, head, train_loader, device, action_map, True, opt); vm = run_variant(a.variant, ex, head, val_loader, device, action_map); history.append({"epoch": epoch, "train": tm, "validation": vm}); score = (vm["verb_top1"] + vm["noun_top1"]) / 2
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if score > best: best = score; torch.save({"model": head.state_dict(), "variant": a.variant, "history": history[-1], "action_map": action_map}, out / "best.pt")
    print(json.dumps(history[-1], indent=2))

if __name__ == "__main__": main()
