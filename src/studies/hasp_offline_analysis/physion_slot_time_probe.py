#!/usr/bin/env python3
"""Slot-level Entity and per-time Dynamic probes for Physion++.

Entity probes use Hungarian assignments from the frozen structured probe.
Dynamic probes operate on the unpooled [slot, future-time, feature] state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from src.core.run_context import apply_cli_defaults, task_context


def load(root, limit=None):
    paths = sorted(Path(root).glob("sample_*.pt")); paths = paths[:limit] if limit else paths
    return [torch.load(p, map_location="cpu", weights_only=False) for p in paths]


def assignment(row):
    valid_obj = row["future_object_valid"].bool().any(-1)
    target_ids = torch.where(valid_obj)[0].tolist()
    if not target_ids: return torch.full((8,), -1, dtype=torch.long)
    pred = row["trajectory_prediction"].float()
    target = row["future_object_state"].float()[target_ids]
    mask = row["future_object_valid"].bool()[target_ids]
    weights = pred.new_tensor([2., 2., 2., .5, .5, .5, 1., 1., 1.])
    costs = []
    for slot in range(8):
        row_cost = []
        for j in range(len(target_ids)):
            m = mask[j]
            diff = (pred[slot] - target[j]).abs()
            row_cost.append(float((diff[m] * weights).sum() / (m.sum() * weights.sum()).clamp_min(1)))
        costs.append(row_cost)
    slots, target_columns = linear_sum_assignment(np.asarray(costs))
    out = torch.full((8,), -1, dtype=torch.long)
    for slot, target_column in zip(slots, target_columns): out[int(slot)] = target_ids[int(target_column)]
    return out


def fit_linear(x, y, tx, ty, epochs=100, mlp=False):
    x, y, tx, ty = [torch.as_tensor(z).float() for z in (x, y, tx, ty)]
    xm, xs = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-5)
    x, tx = (x - xm) / xs, (tx - xm) / xs
    out_dim = y.shape[1] if y.ndim > 1 else 1
    if y.ndim == 1: y, ty = y[:, None], ty[:, None]
    head = (torch.nn.Sequential(torch.nn.Linear(x.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, out_dim)) if mlp else torch.nn.Linear(x.shape[1], out_dim))
    opt = torch.optim.AdamW(head.parameters(), lr=.01 if mlp else .02, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True); loss = torch.nn.functional.smooth_l1_loss(head(x), y); loss.backward(); opt.step()
    with torch.no_grad():
        pred = head(tx); mae = (pred - ty).abs().mean(0); ss_res = ((pred - ty) ** 2).sum(0); ss_tot = ((ty - ty.mean(0)) ** 2).sum(0).clamp_min(1e-8)
    return {"mae": float(mae.mean()), "mae_dims": mae.tolist(), "r2": float((1 - ss_res / ss_tot).mean())}


def fit_binary(x, y, tx, ty):
    x, y, tx, ty = [torch.as_tensor(z).float() for z in (x, y, tx, ty)]
    xm, xs = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-5); x, tx = (x-xm)/xs, (tx-xm)/xs
    head = torch.nn.Linear(x.shape[1], 1); opt = torch.optim.AdamW(head.parameters(), lr=.02, weight_decay=1e-4); pos=y.sum().clamp_min(1); w=((len(y)-pos)/pos).clamp(max=50)
    for _ in range(100):
        opt.zero_grad(set_to_none=True); loss=torch.nn.functional.binary_cross_entropy_with_logits(head(x).squeeze(-1),y,pos_weight=w); loss.backward(); opt.step()
    with torch.no_grad():
        score=head(tx).squeeze(-1); pred=score.sigmoid()>0.5; target=ty>0.5; order=score.argsort().argsort().float()+1; p,n=int(target.sum()),int((~target).sum()); auc=None if not p or not n else float((order[target].sum()-p*(p+1)/2)/(p*n))
    return {"accuracy": float((pred==target).float().mean()), "auroc": auc}


def collect_entity(data):
    assignments = [assignment(row) for row in data]
    xs, presence, pos, extent, velocity = [], [], [], [], []
    for row, a in zip(data, assignments):
        state=row["future_object_state"].float(); valid=row["future_object_valid"].bool()
        for slot in range(8):
            xs.append(row["object_tokens"][slot].float()); target=int(a[slot]); ok=target>=0; presence.append(float(ok))
            if ok:
                m=valid[target]; pos.append((slot, state[target,m,:3].mean(0))); extent.append((slot,state[target,m,6:9].mean(0))); velocity.append((slot,state[target,m,3:6].mean(0)))
    # Repeat only matched rows for continuous targets.
    def matched(field):
        z=[]
        for row, a in zip(data, assignments):
            state=row["future_object_state"].float(); valid=row["future_object_valid"].bool()
            for slot in range(8):
                target=int(a[slot])
                if target>=0: z.append((row["object_tokens"][slot].float(), state[target,valid[target],field].mean(0)))
        return torch.stack([x for x, _ in z]), torch.stack([y for _, y in z])
    xpres=torch.stack(xs); ypres=torch.tensor(presence); xp,yp=matched(slice(0,3)); xe,ye=matched(slice(6,9)); xv,yv=matched(slice(3,6))
    return {"presence":(xpres,ypres),"position":(xp,yp),"extent":(xe,ye),"velocity":(xv,yv)}


def collect_dynamic(data, max_points=30000):
    x=[]; pos=[]; vel=[]; speed=[]
    for row in data:
        a=assignment(row); state=row["future_object_state"].float(); valid=row["future_object_valid"].bool(); tf=row["time_features"].float()
        for slot in range(8):
            target=int(a[slot])
            if target<0: continue
            for t in range(16):
                if valid[target,t]:
                    x.append(tf[slot,t]); pos.append(state[target,t,:3]); vel.append(state[target,t,3:6]); speed.append(state[target,t,3:6].norm())
    if len(x)>max_points:
        g=torch.Generator().manual_seed(239); idx=torch.randperm(len(x),generator=g)[:max_points]; x=[x[i] for i in idx]; pos=[pos[i] for i in idx]; vel=[vel[i] for i in idx]; speed=[speed[i] for i in idx]
    return torch.stack(x), torch.stack(pos), torch.stack(vel), torch.stack(speed)


def main():
    ctx=task_context();p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--test',required=True);p.add_argument('--output',required=True);p.add_argument('--max-train',type=int);p.add_argument('--max-test',type=int);apply_cli_defaults(p,ctx);a=p.parse_args(); tr,te=load(a.train,a.max_train),load(a.test,a.max_test)
    result={"protocol":"physion_slot_time_factor_probe_v1","train_samples":len(tr),"test_samples":len(te),"entity":{},"dynamic":{}}
    etr,ete=collect_entity(tr),collect_entity(te)
    for name in ('presence','position','extent','velocity'):
        x,y=etr[name];tx,ty=ete[name];result['entity'][name]=fit_binary(x,y,tx,ty) if name=='presence' else fit_linear(x,y,tx,ty)
    dx,dp,dv,ds=collect_dynamic(tr);tx,tp,tv,ts=collect_dynamic(te)
    result['dynamic']['position']=fit_linear(dx,dp,tx,tp,mlp=True)
    result['dynamic']['velocity']=fit_linear(dx,dv,tx,tv,mlp=True)
    result['dynamic']['speed']=fit_linear(dx,ds,tx,ts,mlp=True)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__': main()
