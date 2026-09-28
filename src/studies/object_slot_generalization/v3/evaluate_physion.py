"""Evaluate v3 grounded object slots on held-out Physion++ cache."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from .model import GroundedObjectSlots


SCALE = torch.tensor([.20, .08, .01, .12, .06, .02, .09, .08, .12])


def match(pred, target, valid, scale):
    ids = valid.nonzero(as_tuple=False).flatten()
    if not len(ids): return []
    cost = ((pred.detach()[:, None] - target[ids][None]).abs() / scale).mean(-1)
    used, pairs = set(), []
    for i in cost.amin(-1).argsort().tolist():
        choices = [j for j in cost[i].argsort().tolist() if j not in used]
        if choices:
            pairs.append((i, int(ids[choices[0]]))); used.add(choices[0])
    return pairs


def auroc(y, s):
    y, s = y.bool(), s.float(); p, n = int(y.sum()), int((~y).sum())
    if not p or not n: return float('nan')
    order = s.argsort(); ranks = torch.empty_like(s, dtype=torch.float64); ranks[order] = torch.arange(1, len(s)+1, dtype=torch.float64)
    for v in s.unique():
        m = s == v
        if int(m.sum()) > 1: ranks[m] = ranks[m].mean()
    return float((ranks[y].sum() - p * (p+1) / 2) / (p*n))


def r2(pred, target, cols):
    a, b = pred[:, cols], target[:, cols]
    return float(1 - ((a-b)**2).sum() / ((b-b.mean(0))**2).sum().clamp_min(1e-8))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--cache',required=True); p.add_argument('--output',required=True); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--limit',type=int,default=0); a=p.parse_args()
    device=torch.device(a.device); payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False); model=GroundedObjectSlots().to(device); model.load_state_dict(payload['model']); model.eval(); files=sorted(Path(a.cache).glob('sample_*.pt')); files=files[:a.limit] if a.limit else files
    pred_rows=[]; target_rows=[]; labels=[]; scores=[]; match_errors=[]
    for path in files:
        z=torch.load(path,map_location='cpu',weights_only=False); out=model(z['context'].float().unsqueeze(0).to(device)); pred=out.state[0,-1].float().cpu(); logits=out.presence_logits[0,-1].float().cpu(); target=z['current_object_state'].float(); valid=z['current_object_valid'].bool(); pairs=match(pred,target,valid,SCALE)
        used=set()
        for i,j in pairs:
            used.add(i); pred_rows.append(pred[i]); target_rows.append(target[j]); labels.append(1); scores.append(float(logits[i])); match_errors.append(float((pred[i]-target[j]).abs().mean()))
        for i in range(len(pred)):
            if i not in used: labels.append(0); scores.append(float(logits[i]))
    pp=torch.stack(pred_rows); tt=torch.stack(target_rows); y=torch.tensor(labels); s=torch.tensor(scores)
    result={'protocol':'ogs_v3_physion_object_eval_v1','checkpoint':str(Path(a.checkpoint).resolve()),'cache':str(Path(a.cache).resolve()),'samples':len(files),'matched_objects':len(pp),'presence_auroc':auroc(y,s),'position_r2':r2(pp,tt,[0,1,2]),'velocity_r2':r2(pp,tt,[3,4,5]),'extent_r2':r2(pp,tt,[6,7,8]),'position_mae':float((pp[:,:3]-tt[:,:3]).abs().mean()),'velocity_mae':float((pp[:,3:6]-tt[:,3:6]).abs().mean()),'extent_mae':float((pp[:,6:9]-tt[:,6:9]).abs().mean()),'matched_state_mae':float(np.mean(match_errors))}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__': main()
