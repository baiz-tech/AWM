#!/usr/bin/env python3
"""Pair-level relation factor probes for Physion++."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from src.core.run_context import apply_cli_defaults, task_context


PAIRS = list(itertools.combinations(range(8), 2))


def load(root, limit=None):
    paths = sorted(Path(root).glob("sample_*.pt")); paths = paths[:limit] if limit else paths
    return [torch.load(p, map_location="cpu", weights_only=False) for p in paths]


def assignment(row):
    valid_obj = row["future_object_valid"].bool().any(-1); ids = torch.where(valid_obj)[0].tolist()
    if not ids: return torch.full((8,), -1, dtype=torch.long)
    pred, target = row["trajectory_prediction"].float(), row["future_object_state"].float()[ids]
    mask = row["future_object_valid"].bool()[ids]; weights = pred.new_tensor([2.,2.,2.,.5,.5,.5,1.,1.,1.]); cost=[]
    for s in range(8):
        cost.append([float((((pred[s]-target[j]).abs()*weights)[mask[j]]).sum()/((mask[j].sum()*weights.sum()).clamp_min(1))) for j in range(len(ids))])
    si, ti = linear_sum_assignment(np.asarray(cost)); out=torch.full((8,),-1,dtype=torch.long)
    for s,j in zip(si,ti): out[int(s)]=ids[int(j)]
    return out


def fit_regression(x,y,tx,ty,mlp=True,epochs=100):
    x,y,tx,ty=[torch.as_tensor(z).float() for z in (x,y,tx,ty)]; xm=x.mean(0,keepdim=True); xs=x.std(0,keepdim=True).clamp_min(1e-5); x,tx=(x-xm)/xs,(tx-xm)/xs
    out=y.shape[1] if y.ndim>1 else 1
    if y.ndim==1:y,ty=y[:,None],ty[:,None]
    head=torch.nn.Sequential(torch.nn.Linear(x.shape[1],256),torch.nn.GELU(),torch.nn.Linear(256,out)) if mlp else torch.nn.Linear(x.shape[1],out); opt=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=1e-4)
    for _ in range(epochs):opt.zero_grad(set_to_none=True);loss=torch.nn.functional.smooth_l1_loss(head(x),y);loss.backward();opt.step()
    with torch.no_grad(): pred=head(tx); mae=(pred-ty).abs().mean(); r2=1-((pred-ty)**2).sum()/(((ty-ty.mean(0))**2).sum().clamp_min(1e-8))
    return {"mae":float(mae),"r2":float(r2)}


def fit_binary(x,y,tx,ty,epochs=100):
    x,y,tx,ty=[torch.as_tensor(z).float() for z in (x,y,tx,ty)]; xm=x.mean(0,keepdim=True);xs=x.std(0,keepdim=True).clamp_min(1e-5);x,tx=(x-xm)/xs,(tx-xm)/xs;head=torch.nn.Sequential(torch.nn.Linear(x.shape[1],256),torch.nn.GELU(),torch.nn.Linear(256,1));opt=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=1e-4);pos=y.sum().clamp_min(1);w=((len(y)-pos)/pos).clamp(max=50)
    for _ in range(epochs):opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(head(x).squeeze(-1),y,pos_weight=w);loss.backward();opt.step()
    with torch.no_grad():score=head(tx).squeeze(-1);pred=score.sigmoid()>0.5;target=ty>0.5;order=score.argsort().argsort().float()+1;p,n=int(target.sum()),int((~target).sum());auc=None if not p or not n else float((order[target].sum()-p*(p+1)/2)/(p*n))
    return {"accuracy":float((pred==target).float().mean()),"auroc":auc}


def collect(data, max_points=120000):
    static_x, static_dist, static_contact, static_ttc, time_x, time_dist, time_contact = [], [], [], [], [], [], []
    for row in data:
        a=assignment(row); distance=row["future_pair_distance"].float(); dvalid=row["future_pair_valid"].bool(); contact=row["future_contact"].float(); cvalid=row["future_contact_valid"].bool(); ttc=row["future_time_to_contact"].float(); ttc_valid=row["future_time_to_contact_valid"].bool()
        for pi,(s1,s2) in enumerate(PAIRS):
            t1,t2=int(a[s1]),int(a[s2])
            if t1<0 or t2<0:continue
            dv=dvalid[t1,t2];cv=cvalid[t1,t2];
            if dv.any():
                static_x.append(row["pair_tokens"][pi].float()); static_dist.append(distance[t1,t2][dv].mean()); static_contact.append(float((contact[t1,t2][cv]>.5).any()) if cv.any() else 0.)
                static_ttc.append(float(ttc[t1,t2]) if ttc_valid[t1,t2] else 1.0)
                for t in range(16):
                    if dv[t]:
                        tubelet = min(t // 2, contact.shape[-1] - 1)
                        time_x.append(row["pair_time_features"][pi,t].float());time_dist.append(distance[t1,t2,t]);time_contact.append(float(contact[t1,t2,tubelet]>.5) if cv[tubelet] else 0.)
    if len(time_x)>max_points:
        g=torch.Generator().manual_seed(239);idx=torch.randperm(len(time_x),generator=g)[:max_points];time_x=[time_x[i] for i in idx];time_dist=[time_dist[i] for i in idx];time_contact=[time_contact[i] for i in idx]
    return torch.stack(static_x),torch.tensor(static_dist),torch.tensor(static_contact),torch.tensor(static_ttc),torch.stack(time_x),torch.tensor(time_dist),torch.tensor(time_contact)


def main():
    ctx=task_context();p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--test',required=True);p.add_argument('--output',required=True);p.add_argument('--max-train',type=int);p.add_argument('--max-test',type=int);apply_cli_defaults(p,ctx);a=p.parse_args();tr,te=load(a.train,a.max_train),load(a.test,a.max_test);sx,sd,sc,st,tx,td,tc=collect(tr);tsx,tsd,tsc,tst,ttx,ttd,ttc=collect(te)
    result={"protocol":"physion_pair_relation_probe_v1","train_samples":len(tr),"test_samples":len(te),"static_pair":{},"time_pair":{}}
    result['static_pair']['distance']=fit_regression(sx,sd,tsx,tsd);result['static_pair']['contact']=fit_binary(sx,sc,tsx,tsc);result['static_pair']['time_to_contact']=fit_regression(sx,st,tsx,tst);result['time_pair']['distance']=fit_regression(tx,td,ttx,ttd);result['time_pair']['contact']=fit_binary(tx,tc,ttx,ttc)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
