"""Held-out Physion++ evaluation for the v2 object-supervised OCSE."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from .model import OCSE


def match(pred, target, valid):
    ids = valid.nonzero(as_tuple=False).flatten()
    if not len(ids): return []
    cost = (pred[:, None] - target[ids][None]).abs().mean(-1)
    used, pairs = set(), []
    for i in cost.amin(-1).argsort().tolist():
        choices = [j for j in cost[i].argsort().tolist() if j not in used]
        if choices: pairs.append((i, int(ids[choices[0]]))); used.add(choices[0])
    return pairs


@torch.no_grad()
def collect(model, files, device):
    rows = []
    for p in files:
        z = torch.load(p, map_location='cpu', weights_only=False)
        out = model(z['context'].float().unsqueeze(0).to(device))
        pred = out.state[0, 0].float().cpu(); logits = out.presence_logits[0, 0].float().cpu()
        target, valid = z['current_object_state'].float(), z['current_object_valid'].bool()
        pairs = match(pred, target, valid)
        matched = torch.zeros(pred.size(0), dtype=torch.bool)
        for i, j in pairs: matched[i] = True; rows.append({'pred': pred[i], 'target': target[j], 'presence': logits[i], 'positive': 1})
        for i in (~matched).nonzero(as_tuple=False).flatten(): rows.append({'pred': pred[i], 'target': torch.zeros(9), 'presence': logits[i], 'positive': 0})
    return rows


def r2(pred, target, cols):
    a, b = pred[:, cols], target[:, cols]
    return float(1 - ((a - b) ** 2).sum() / ((b - b.mean(0)) ** 2).sum().clamp_min(1e-8))


def auroc(labels, scores):
    """Rank-based AUROC without an sklearn dependency."""
    labels, scores = labels.bool(), scores.float()
    pos, neg = int(labels.sum()), int((~labels).sum())
    if not pos or not neg: return float('nan')
    order = scores.argsort(); ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    # Ties are rare for logits; averaging ranks makes the definition exact.
    for value in scores.unique():
        tied = scores == value
        if tied.sum() > 1: ranks[tied] = ranks[tied].mean()
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    device=torch.device(a.device); payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False); model=OCSE().to(device);model.load_state_dict(payload['model']);model.eval();files=sorted(Path(a.cache).glob('sample_*.pt'));files=files[:a.limit] if a.limit else files;rows=collect(model,files,device)
    pred=torch.stack([x['pred'] for x in rows]);target=torch.stack([x['target'] for x in rows]);positive=torch.tensor([x['positive'] for x in rows]);score=torch.tensor([x['presence'] for x in rows]);
    auc=auroc(positive,score)
    matched=positive.bool();result={'protocol':'ocse_v2_physion_object_eval_v1','checkpoint':str(Path(a.checkpoint).resolve()),'cache':str(Path(a.cache).resolve()),'samples':len(files),'matched_slot_rows':int(matched.sum()),'presence_auroc':auc,'position_r2':r2(pred[matched],target[matched],[0,1,2]) if matched.any() else None,'velocity_r2':r2(pred[matched],target[matched],[3,4,5]) if matched.any() else None,'extent_r2':r2(pred[matched],target[matched],[6,7,8]) if matched.any() else None,'matched_state_mae':float((pred[matched]-target[matched]).abs().mean()) if matched.any() else None,'note':'Matching is based on predicted-vs-target 9-D state; no Dynamic/Relation features are used.'}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
