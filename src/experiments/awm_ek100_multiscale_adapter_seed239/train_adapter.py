#!/usr/bin/env python3
"""Train the V23-style shared adapter using EK100 only."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np, torch, torch.distributed as dist, torch.nn.functional as F, yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from src.data.ek100.labels import encode_labels
from src.data.ek100.dataset import read_rows
from .train_readout import GridDataset
from .model import EK100OnlyWorldModel
from src.core.run_context import apply_cli_defaults, task_context

def setup():
    if 'RANK' not in os.environ: return 0, 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    r,w,l=(int(os.environ[x]) for x in ('RANK','WORLD_SIZE','LOCAL_RANK')); torch.cuda.set_device(l); dist.init_process_group('nccl'); return r,w,torch.device('cuda',l)

def main():
    ctx=task_context();p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--train',required=True); p.add_argument('--validation',required=True); p.add_argument('--readout',required=True); p.add_argument('--output',required=True); p.add_argument('--epochs',type=int,default=12); p.add_argument('--seed',type=int,default=239); apply_cli_defaults(p,ctx);a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); rank,world,dev=setup(); torch.manual_seed(a.seed+rank); np.random.seed(a.seed+rank)
    ro=torch.load(a.readout,map_location='cpu',weights_only=False); pairs=[None]*len(ro['action_map'])
    for pair,i in ro['action_map'].items(): pairs[i]=pair
    tr,va=GridDataset(a.train,rank,world),GridDataset(a.validation,rank,world); rows=read_rows(cfg['data']['train_annotations']); vv=ro['verb_vocabulary']; nv=ro['noun_vocabulary']; tr.v,tr.n=encode_labels(tr.v,vv),encode_labels(tr.n,nv); va.v,va.n=encode_labels(va.v,vv),encode_labels(va.n,nv)
    model=EK100OnlyWorldModel(ro['verb_classes'],ro['noun_classes'],pairs).to(dev); model.readout.load_state_dict(ro['model']); model=DDP(model,device_ids=[dev.index]) if world>1 else model
    train=DataLoader(tr, batch_size=cfg['training']['adapter_batch_size'], shuffle=True, num_workers=0); val=DataLoader(va,batch_size=cfg['training']['adapter_batch_size'],shuffle=False,num_workers=0); opt=torch.optim.AdamW((model.module if hasattr(model,'module') else model).shared.parameters(),lr=cfg['training']['adapter_learning_rate'],weight_decay=cfg['training']['weight_decay']); out=Path(a.output); hist=[]
    if rank==0: out.mkdir(parents=True,exist_ok=False)
    if dist.is_initialized(): dist.barrier()
    best=float('inf')
    for epoch in range(1,a.epochs+1):
        model.train(); sums=[0.,0.,0.,0.]
        for x,v,n in train:
            x,v,n=x.to(dev),v.to(dev),n.to(dev); vl,nl,al,cons=model(x); action=torch.tensor([ro['action_map'].get((int(i),int(j)),-1) for i,j in zip(v.cpu(),n.cpu())],device=dev); ok=action>=0; loss=F.cross_entropy(vl,v)+F.cross_entropy(nl,n)+.2*(F.cross_entropy(al[ok],action[ok]) if ok.any() else x.sum()*0)+.02*cons; opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_((model.module if hasattr(model,'module') else model).shared.parameters(),1.); opt.step(); sums[0]+=float(loss.detach()); sums[1]+=len(v); sums[2]+=float((vl.argmax(1)==v).sum()); sums[3]+=float((nl.argmax(1)==n).sum())
        model.eval(); validation=torch.zeros(4,device=dev,dtype=torch.float64)
        with torch.no_grad():
            for x,v,n in val:
                x,v,n=x.to(dev),v.to(dev),n.to(dev); vl,nl,al,cons=model(x); action=torch.tensor([ro['action_map'].get((int(i),int(j)),-1) for i,j in zip(v.cpu(),n.cpu())],device=dev); ok=action>=0; loss=F.cross_entropy(vl,v)+F.cross_entropy(nl,n)+.2*(F.cross_entropy(al[ok],action[ok]) if ok.any() else x.sum()*0)+.02*cons; validation += torch.tensor([float(loss)*len(v),len(v),float((vl.argmax(1)==v).sum()),float((nl.argmax(1)==n).sum())],device=dev,dtype=torch.float64)
        if dist.is_initialized(): dist.all_reduce(validation)
        if rank==0:
            record={'epoch':epoch,'train_loss':sums[0]/max(sums[1],1),'train_verb_top1':sums[2]/max(sums[1],1),'train_noun_top1':sums[3]/max(sums[1],1),'validation_loss':float(validation[0]/validation[1].clamp_min(1)),'validation_verb_top1':float(validation[2]/validation[1].clamp_min(1)),'validation_noun_top1':float(validation[3]/validation[1].clamp_min(1))}; hist.append(record); z=model.module if hasattr(model,'module') else model; state={'protocol':'v23_style_ek100_only','epoch':epoch,'shared':z.shared.state_dict(),'readout':str(Path(a.readout).resolve()),'validation':record}; torch.save(state,out/'latest.pt');
            if record['validation_loss'] < best: best=record['validation_loss']; torch.save(state,out/'best.pt')
            (out/'history.json').write_text(json.dumps(hist,indent=2))
    if dist.is_initialized(): dist.destroy_process_group()
if __name__=='__main__': main()
