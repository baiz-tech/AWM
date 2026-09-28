#!/usr/bin/env python3
"""Train an independent OCP readout on frozen Physion++ structured probes."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from .model import PhysionDecoder
from .ocp_readout import OCPReadout
from src.core.run_context import apply_cli_defaults, task_context

def metrics(scores, labels, threshold=0.5):
    labels=labels.bool(); pred=scores.sigmoid().ge(threshold); tp=((pred)&labels).sum().item(); tn=((~pred)&(~labels)).sum().item(); fp=(pred&(~labels)).sum().item(); fn=((~pred)&labels).sum().item(); pos=tp+fn; neg=tn+fp; n=pos+neg
    auroc=float('nan')
    if pos and neg:
        order=torch.argsort(scores); s=scores[order]; y=labels[order]; rank_sum=0.; start=0
        while start<n:
            end=start+1
            while end<n and s[end]==s[start]: end+=1
            rank_sum += ((start+1+end)/2)*int(y[start:end].sum()); start=end
        auroc=(rank_sum-pos*(pos+1)/2)/(pos*neg)
    return {'accuracy':(tp+tn)/n if n else float('nan'),'balanced_accuracy':((tp/pos if pos else float('nan'))+(tn/neg if neg else float('nan')))/2 if pos and neg else float('nan'),'auroc':auroc,'positive_recall':tp/pos if pos else float('nan'),'negative_recall':tn/neg if neg else float('nan'),'tp':tp,'tn':tn,'fp':fp,'fn':fn,'samples':n}

def cname(path): return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:16]+'.pt'
class DS(Dataset):
    def __init__(self, cache_root, targets, split):
        t=torch.load(targets,map_location='cpu',weights_only=False); self.labels=t['ocp_label'].float(); self.valid=t['ocp_valid'].bool(); self.files=[Path(cache_root)/split/cname(p) for p in t['path']]
        missing=[str(x) for x in self.files if not x.is_file()]
        if missing: raise FileNotFoundError(f'missing latent cache files, first={missing[0]}')
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        r=torch.load(self.files[i],map_location='cpu',weights_only=True); return r['context_tokens'],r['future_tokens'],self.labels[i],self.valid[i]

def main():
    ctx=task_context(); p=argparse.ArgumentParser(); p.add_argument('--cache-root',type=Path,required=True); p.add_argument('--train-targets',type=Path,required=True); p.add_argument('--validation-targets',type=Path,required=True); p.add_argument('--probe-checkpoint',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--num-workers',type=int,default=2); p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--seed',type=int,default=239); p.add_argument('--probe-dim',type=int,default=256); p.add_argument('--visual-dim',type=int,default=1280); p.add_argument('--hidden-dim',type=int,default=256); apply_cli_defaults(p,ctx); a=p.parse_args()
    rank=int(__import__('os').environ.get('RANK',0)); world=int(__import__('os').environ.get('WORLD_SIZE',1)); local=int(__import__('os').environ.get('LOCAL_RANK',0));
    if world>1: torch.cuda.set_device(local); dist.init_process_group('nccl')
    random.seed(a.seed+rank); np.random.seed(a.seed+rank); torch.manual_seed(a.seed+rank); dev=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    payload=torch.load(a.probe_checkpoint,map_location='cpu',weights_only=False); probe=PhysionDecoder(**payload.get('model_config',{})).to(dev); probe.load_state_dict(payload['model'],strict=True); probe.eval()
    for q in probe.parameters(): q.requires_grad_(False)
    tr,va=DS(a.cache_root,a.train_targets,'data_v1'),DS(a.cache_root,a.validation_targets,'readout_data_v1'); ts=DistributedSampler(tr,num_replicas=world,rank=rank,shuffle=True,seed=a.seed) if world>1 else None; vs=DistributedSampler(va,num_replicas=world,rank=rank,shuffle=False) if world>1 else None
    kw={'batch_size':a.batch_size,'num_workers':a.num_workers,'pin_memory':dev.type=='cuda'}; tl=DataLoader(tr,sampler=ts,shuffle=ts is None,**kw); vl=DataLoader(va,sampler=vs,shuffle=False,**kw)
    net=OCPReadout(a.visual_dim,a.probe_dim,a.hidden_dim).to(dev); net=DDP(net,device_ids=[local]) if world>1 else net; opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=.04); a.output_dir.mkdir(parents=True,exist_ok=True); history=[]; best=float('inf')
    for epoch in range(1,a.epochs+1):
        if ts: ts.set_epoch(epoch)
        net.train(); total=0.; count=0
        for context,future,label,valid in tl:
            valid=valid.bool().to(dev); context,future,label=context.to(dev),future.to(dev),label.to(dev); opt.zero_grad(set_to_none=True)
            with torch.no_grad(): po=probe(context.float(),future.float())
            logits=net(torch.cat([context,future],1),po); use=valid
            if not use.any(): continue
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[use],label[use]); loss.backward(); opt.step(); total+=float(loss)*int(use.sum()); count+=int(use.sum())
        if world>1: dist.barrier()
        net.eval(); vsum=0.; vcount=0; val_scores=[]; val_labels=[]
        with torch.no_grad():
            for context,future,label,valid in vl:
                valid=valid.bool().to(dev); context=context.to(dev).float(); future=future.to(dev).float(); label=label.to(dev); po=probe(context,future); logits=net(torch.cat([context,future],1),po); use=valid
                if use.any(): vsum+=float(torch.nn.functional.binary_cross_entropy_with_logits(logits[use],label[use]))*int(use.sum()); vcount+=int(use.sum()); val_scores.append(logits[use].detach().cpu()); val_labels.append(label[use].detach().cpu())
        if world>1:
            stats=torch.tensor([vsum,vcount],dtype=torch.float64,device=dev); dist.all_reduce(stats); vsum,vcount=stats.tolist(); gathered=[None]*world; local_scores=torch.cat(val_scores).tolist() if val_scores else []; local_labels=torch.cat(val_labels).tolist() if val_labels else []; dist.all_gather_object(gathered,(local_scores,local_labels)); val_scores=[torch.tensor(g[0],dtype=torch.float32) for g in gathered if g[0]]; val_labels=[torch.tensor(g[1],dtype=torch.float32) for g in gathered if g[1]]
        score=torch.cat(val_scores) if val_scores else torch.empty(0); label=torch.cat(val_labels) if val_labels else torch.empty(0); cls=metrics(score,label) if score.numel() else {'accuracy':float('nan'),'balanced_accuracy':float('nan'),'auroc':float('nan'),'positive_recall':float('nan'),'negative_recall':float('nan'),'tp':0,'tn':0,'fp':0,'fn':0,'samples':0}
        row={'epoch':epoch,'train_loss':total/max(count,1),'validation_loss':vsum/max(vcount,1),'train_samples':count,'validation_samples':vcount,**{k:cls[k] for k in ('accuracy','balanced_accuracy','auroc','positive_recall','negative_recall')}}; history.append(row)
        if rank==0:
            obj=net.module if isinstance(net,DDP) else net; out={'protocol':'physionpp3_independent_ocp_readout_v1','epoch':epoch,'model':obj.state_dict(),'model_config':{'visual_dim':a.visual_dim,'probe_dim':a.probe_dim,'hidden_dim':a.hidden_dim},'probe_checkpoint':str(a.probe_checkpoint.resolve()),'metrics':row}; torch.save(out,a.output_dir/'latest.pt');
            if row['validation_loss']<best: best=row['validation_loss']; torch.save(out,a.output_dir/'best.pt')
            (a.output_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n'); print(json.dumps(row),flush=True)
    if world>1: dist.destroy_process_group()
if __name__=='__main__': main()
