#!/usr/bin/env python3
"""Train current-only future structured probe; future latent is never passed to the model."""
from __future__ import annotations
import argparse, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from src.training.distributed import ExactDistributedSampler
from .model_current_only import CurrentOnlyPhysionDecoder
from .model import loss

KEYS=('object_present','state_2d','state_valid','object_type','color_rgb','is_target','pair_distance_2d','pair_valid','contact','contact_valid','first_contact_class')
def cname(path): return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:16]+'.pt'
class DS(Dataset):
    def __init__(self,cache,targets,split,max_videos=None):
        self.cache=Path(cache)/split; t=torch.load(targets,map_location='cpu',weights_only=False); n=len(t['path']); n=min(n,int(max_videos)) if max_videos else n; self.paths=t['path'][:n]; self.t={k:v[:n] for k,v in t.items() if k in KEYS}; self.files=[self.cache/cname(p) for p in self.paths]
        missing=[str(x) for x in self.files if not x.is_file()]
        if missing: raise FileNotFoundError(f'missing cache: {missing[0]}')
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        r=torch.load(self.files[i],map_location='cpu',weights_only=True); return {'context':r['context_tokens'],**{k:v[i] for k,v in self.t.items()}}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache-root',type=Path,required=True); p.add_argument('--train-targets',type=Path,required=True); p.add_argument('--validation-targets',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch-size',type=int,default=2); p.add_argument('--num-workers',type=int,default=2); p.add_argument('--seed',type=int,default=239); p.add_argument('--learning-rate',type=float,default=2e-4); p.add_argument('--max-train-videos',type=int); p.add_argument('--max-validation-videos',type=int); a=p.parse_args()
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1)); local=int(os.environ.get('LOCAL_RANK',0));
    if world>1: torch.cuda.set_device(local); dist.init_process_group('nccl')
    random.seed(a.seed+rank); np.random.seed(a.seed+rank); torch.manual_seed(a.seed+rank); dev=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu'); vocab=torch.load(a.train_targets,map_location='cpu',weights_only=False).get('object_type_vocab',['unknown']); tr,va=DS(a.cache_root,a.train_targets,'data_v1',a.max_train_videos),DS(a.cache_root,a.validation_targets,'readout_data_v1',a.max_validation_videos); ts=DistributedSampler(tr,num_replicas=world,rank=rank,shuffle=True,seed=a.seed) if world>1 else None; vs=ExactDistributedSampler(va,rank=rank,world_size=world) if world>1 else None; kw={'batch_size':a.batch_size,'num_workers':a.num_workers,'pin_memory':dev.type=='cuda'}; tl=DataLoader(tr,sampler=ts,shuffle=ts is None,**kw); vl=DataLoader(va,sampler=vs,shuffle=False,**kw); model=CurrentOnlyPhysionDecoder(num_object_types=len(vocab)).to(dev); model=DDP(model,device_ids=[local]) if world>1 else model; opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.04); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs); a.output_dir.mkdir(parents=True,exist_ok=True); history=[]; best=float('inf'); names=('total','presence','target','object_type','color','center','geometry','velocity','distance','contact','first_contact')
    for epoch in range(1,a.epochs+1):
        if ts: ts.set_epoch(epoch)
        model.train(); sums={k:0. for k in names}; count=0
        for batch in tl:
            batch={k:v.to(dev) for k,v in batch.items()}; opt.zero_grad(set_to_none=True); value,m=loss(model(batch['context']),batch); value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); bs=batch['context'].size(0); count+=bs
            for k in names: sums[k]+=float(m[k].detach())*bs
        sch.step(); model.eval(); vsums={k:0. for k in names}; vcount=0
        with torch.no_grad():
            for batch in vl:
                batch={k:v.to(dev) for k,v in batch.items()}; _,m=loss(model(batch['context']),batch); bs=batch['context'].size(0); vcount+=bs
                for k in names: vsums[k]+=float(m[k])*bs
        if world>1:
            st=torch.tensor([sums[k] for k in names]+[count]+[vsums[k] for k in names]+[vcount],dtype=torch.float64,device=dev); dist.all_reduce(st); sums={k:float(st[i]) for i,k in enumerate(names)}; count=int(st[len(names)]); off=len(names)+1; vsums={k:float(st[off+i]) for i,k in enumerate(names)}; vcount=int(st[off+len(names)])
        row={'epoch':epoch,'train':{k:sums[k]/max(count,1) for k in names},'validation':{k:vsums[k]/max(vcount,1) for k in names},'train_samples':count,'validation_samples':vcount}; row['train_loss']=row['train']['total']; row['validation_loss']=row['validation']['total']; history.append(row)
        if rank==0:
            obj=model.module if isinstance(model,DDP) else model; out={'protocol':'physionpp3_current_only_structured_probe_v1','epoch':epoch,'model':obj.state_dict(),'model_config':{'input_dim':1280,'hidden_dim':256,'num_heads':8,'ffn_dim':1024,'num_slots':8,'num_object_types':len(vocab)},'object_type_vocab':vocab,'metrics':row}; torch.save(out,a.output_dir/'latest.pt');
            if row['validation_loss']<best: best=row['validation_loss']; torch.save(out,a.output_dir/'best.pt')
            (a.output_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n'); print(json.dumps(row),flush=True)
    if world>1: dist.destroy_process_group()
if __name__=='__main__': main()
