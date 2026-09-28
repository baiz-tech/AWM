from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from .model import OCSE, ocse_loss, matched_object_loss

class Physion(Dataset):
    def __init__(self, root): self.files = sorted(Path(root).glob('sample_*.pt'))
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        z=torch.load(self.files[i],map_location='cpu',weights_only=False)
        return z['context'].float(),z['current_object_valid'].float(),z['current_object_state'].float(),torch.tensor(-1)

class EK100(Dataset):
    def __init__(self, root):
        root=Path(root); rank=int(os.environ.get('RANK','0')); shards=sorted(root.glob('features-rank*.pt'))
        if not shards: raise FileNotFoundError(root)
        p=shards[rank] if len(shards)>1 and rank<len(shards) else shards[0]
        z=torch.load(p,map_location='cpu',weights_only=False); self.x=z['grid'].float(); self.n=z['noun'].long()
    def __len__(self): return len(self.n)
    def __getitem__(self,i): return self.x[i],torch.zeros(8),torch.zeros(8,9),self.n[i]

def setup():
    if 'RANK' not in os.environ:return 0,1,torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    r,w,l=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']),int(os.environ['LOCAL_RANK']);torch.cuda.set_device(l);dist.init_process_group('nccl');return r,w,torch.device('cuda',l)

def run(model,noun_head,loaders,device,opt,train=True):
    model.train(train); noun_head.train(train); total=correct=count=0
    for name,dl in loaders:
        for x,pres,state,noun in dl:
            x,pres,state,noun=x.to(device,non_blocking=True),pres.to(device,non_blocking=True),state.to(device,non_blocking=True),noun.to(device,non_blocking=True)
            out=model(x); target=pres[:,None,:].expand(-1,x.size(1),-1)
            # Physion cache exposes object-level validity [B,T,S], whereas
            # OCSE objectness is patch-level [B,T,N]; do not mix these spaces.
            # Object-level presence supervision will be added after Hungarian
            # state matching. For now train the gate with reconstruction and
            # ownership regularizers only.
            ls=ocse_loss(out,target_features=x,target_objectness=None)
            loss=ls['total']
            if name=='physion':
                obj=matched_object_loss(out,state,pres); loss=loss+obj['object_total']
            if name=='ek100':
                logits=noun_head(out.slots.amax(2).mean(1)); loss=loss+F.cross_entropy(logits,noun);correct+=int((logits.argmax(1)==noun).sum());count+=len(noun)
            if train: opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(noun_head.parameters()),1.);opt.step()
            total+=float(loss.detach())*len(x)
    stats=torch.tensor([total,max(count,1),correct],device=device,dtype=torch.float64)
    if dist.is_initialized():dist.all_reduce(stats)
    return float(stats[0]/max(len(loaders[0][1].dataset),1)),float(stats[2]/stats[1].clamp_min(1))

@torch.no_grad()
def eval_noun(model,head,dl,device):
    model.eval();head.eval();a=b=0
    for x,_,_,n in dl:
        x,n=x.to(device),n.to(device);o=model(x);a+=int((head(o.slots.amax(2).mean(1)).argmax(1)==n).sum());b+=len(n)
    s=torch.tensor([a,b],device=device,dtype=torch.float64)
    if dist.is_initialized():dist.all_reduce(s)
    return float(s[0]/s[1].clamp_min(1))

def main():
    p=argparse.ArgumentParser();p.add_argument('--physion-train',required=True);p.add_argument('--ek100-train',required=True);p.add_argument('--ek100-validation',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=239);a=p.parse_args()
    rank,world,device=setup();random.seed(a.seed+rank);np.random.seed(a.seed+rank);torch.manual_seed(a.seed+rank)
    pt,et,ev=Physion(a.physion_train),EK100(a.ek100_train),EK100(a.ek100_validation);ps=DistributedSampler(pt,world,rank,shuffle=True,seed=a.seed) if world>1 else None
    pl=DataLoader(pt,a.batch_size,sampler=ps,shuffle=ps is None,num_workers=1,pin_memory=device.type=='cuda');el=DataLoader(et,a.batch_size,shuffle=True,num_workers=1,pin_memory=device.type=='cuda');vl=DataLoader(ev,a.batch_size,shuffle=False,num_workers=1,pin_memory=device.type=='cuda')
    local_max=torch.tensor([int(et.n.max())],device=device);dist.all_reduce(local_max,op=dist.ReduceOp.MAX) if dist.is_initialized() else None
    m=OCSE().to(device);h=torch.nn.Linear(256,int(local_max.item())+1).to(device)
    if world>1:m=DDP(m,device_ids=[device.index],find_unused_parameters=True);h=DDP(h,device_ids=[device.index],find_unused_parameters=True)
    opt=torch.optim.AdamW(list(m.parameters())+list(h.parameters()),lr=1e-4,weight_decay=1e-4);out=Path(a.output)
    if rank==0:out.mkdir(parents=True,exist_ok=True)
    if dist.is_initialized():dist.barrier()
    hist=[]
    for e in range(1,a.epochs+1):
        if ps:ps.set_epoch(e)
        loss,noun=run(m,h,[('physion',pl),('ek100',el)],device,opt,True);val=eval_noun(m,h,vl,device)
        if rank==0:
            rec={'epoch':e,'train_loss':loss,'train_noun_top1':noun,'ek100_validation_noun_top1':val};hist.append(rec);print(rec,flush=True);(out/'history.json').write_text(json.dumps(hist,indent=2)+'\n');mm=m.module if hasattr(m,'module') else m;hh=h.module if hasattr(h,'module') else h;torch.save({'model':mm.state_dict(),'noun_head':hh.state_dict(),'epoch':e,'protocol':'ocse_v2_joint_physion_ek100'},out/'latest.pt')
    if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
