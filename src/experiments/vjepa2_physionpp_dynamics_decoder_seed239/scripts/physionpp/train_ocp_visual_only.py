#!/usr/bin/env python3
"""Train OCP from raw context + predicted-future visual latents only."""
from __future__ import annotations
import argparse, hashlib, json, os, random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from .ocp_readout import VisualOnlyOCPReadout
from .train_ocp_readout import metrics
from src.core.run_context import apply_cli_defaults, task_context

def cname(path): return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:16]+'.pt'
class DS(Dataset):
    def __init__(self, root, targets, split):
        t=torch.load(targets,map_location='cpu',weights_only=False); self.labels=t['ocp_label'].float(); self.valid=t['ocp_valid'].bool(); self.files=[Path(root)/split/cname(x) for x in t['path']]
        if any(not x.is_file() for x in self.files): raise FileNotFoundError('latent cache is incomplete')
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        r=torch.load(self.files[i],map_location='cpu',weights_only=True); return r['context_tokens'],r['future_tokens'],self.labels[i],self.valid[i]

def main():
    ctx=task_context(); p=argparse.ArgumentParser(); p.add_argument('--cache-root',type=Path,required=True); p.add_argument('--train-targets',type=Path,required=True); p.add_argument('--validation-targets',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--num-workers',type=int,default=2); p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--seed',type=int,default=239); apply_cli_defaults(p,ctx); a=p.parse_args()
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1)); local=int(os.environ.get('LOCAL_RANK',0));
    if world>1: torch.cuda.set_device(local); dist.init_process_group('nccl')
    random.seed(a.seed+rank); np.random.seed(a.seed+rank); torch.manual_seed(a.seed+rank); dev=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    tr,va=DS(a.cache_root,a.train_targets,'data_v1'),DS(a.cache_root,a.validation_targets,'readout_data_v1'); ts=DistributedSampler(tr,num_replicas=world,rank=rank,shuffle=True,seed=a.seed) if world>1 else None; vs=DistributedSampler(va,num_replicas=world,rank=rank,shuffle=False) if world>1 else None; kw={'batch_size':a.batch_size,'num_workers':a.num_workers,'pin_memory':dev.type=='cuda'}; tl=DataLoader(tr,sampler=ts,shuffle=ts is None,**kw); vl=DataLoader(va,sampler=vs,shuffle=False,**kw)
    net=VisualOnlyOCPReadout().to(dev); net=DDP(net,device_ids=[local]) if world>1 else net; opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=.04); a.output_dir.mkdir(parents=True,exist_ok=True); history=[]; best=float('inf')
    for epoch in range(1,a.epochs+1):
        if ts: ts.set_epoch(epoch)
        net.train(); tsum=0.; tcount=0
        for c,f,y,v in tl:
            c,f,y,v=c.to(dev).float(),f.to(dev).float(),y.to(dev),v.bool().to(dev); use=v
            if not use.any(): continue
            opt.zero_grad(set_to_none=True); z=net(torch.cat([c,f],1)); loss=torch.nn.functional.binary_cross_entropy_with_logits(z[use],y[use]); loss.backward(); opt.step(); tsum+=float(loss)*int(use.sum()); tcount+=int(use.sum())
        net.eval(); vsum=0.; vcount=0; scores=[]; labels=[]
        with torch.no_grad():
            for c,f,y,v in vl:
                c,f,y,v=c.to(dev).float(),f.to(dev).float(),y.to(dev),v.bool().to(dev); use=v; z=net(torch.cat([c,f],1));
                if use.any(): vsum+=float(torch.nn.functional.binary_cross_entropy_with_logits(z[use],y[use]))*int(use.sum()); vcount+=int(use.sum()); scores.append(z[use].cpu()); labels.append(y[use].cpu())
        if world>1:
            st=torch.tensor([vsum,vcount],dtype=torch.float64,device=dev); dist.all_reduce(st); vsum,vcount=st.tolist(); gathered=[None]*world; local_scores=torch.cat(scores).tolist() if scores else []; local_labels=torch.cat(labels).tolist() if labels else []; dist.all_gather_object(gathered,(local_scores,local_labels)); scores=[torch.tensor(g[0],dtype=torch.float32) for g in gathered if g[0]]; labels=[torch.tensor(g[1],dtype=torch.float32) for g in gathered if g[1]]
        score=torch.cat(scores) if scores else torch.empty(0); label=torch.cat(labels) if labels else torch.empty(0); mm=metrics(score,label) if score.numel() else {k:float('nan') for k in ('accuracy','balanced_accuracy','auroc','positive_recall','negative_recall')}; row={'epoch':epoch,'train_loss':tsum/max(tcount,1),'validation_loss':vsum/max(vcount,1),'train_samples':tcount,'validation_samples':vcount,**mm}; history.append(row)
        if rank==0:
            obj=net.module if isinstance(net,DDP) else net; payload={'protocol':'physionpp3_visual_only_ocp_readout_v1','epoch':epoch,'model':obj.state_dict(),'model_config':{'visual_dim':1280,'hidden_dim':256},'metrics':row}; torch.save(payload,a.output_dir/'latest.pt');
            if row['validation_loss']<best: best=row['validation_loss']; torch.save(payload,a.output_dir/'best.pt')
            (a.output_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n'); print(json.dumps(row),flush=True)
    if world>1: dist.destroy_process_group()
if __name__=='__main__': main()
