#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os
from contextlib import nullcontext
from pathlib import Path
import numpy as np,torch,torch.distributed as dist,torch.nn.functional as F,yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from src.data.ek100.labels import encode_labels
from src.data.ek100.dataset import read_rows
from .readout import EK100MultiScaleReadout
from src.core.run_context import apply_cli_defaults, task_context
from .output_guard import ensure_no_artifacts
class GridDataset(Dataset):
 def __init__(self,path,rank=0,world=1):
  paths=sorted(Path(path).glob('features-rank*.pt'))
  # The cache is already sharded by DDP rank.  Loading every 8x8 shard in
  # every worker would replicate about 620 GB of host RAM, so each rank owns
  # exactly one shard during distributed training.
  if world>1:
   assert len(paths)==world,(len(paths),world);paths=[paths[rank]]
  d=[torch.load(p,map_location='cpu',weights_only=False) for p in paths];self.x=torch.cat([z['grid'].float() for z in d]);self.v=torch.cat([z['verb'].long() for z in d]);self.n=torch.cat([z['noun'].long() for z in d])
 def __len__(self):return len(self.v)
 def __getitem__(self,i):return self.x[i],self.v[i],self.n[i]
def setup():
 if 'RANK' not in os.environ:return 0,1,torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 r,w,l=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']),int(os.environ['LOCAL_RANK']);torch.cuda.set_device(l);dist.init_process_group('nccl');return r,w,torch.device('cuda',l)
def run(m,dl,dev,amap,nc,weights,train=False,opt=None):
 m.train(train);s=torch.zeros(9,device=dev,dtype=torch.float64);sup=torch.zeros(nc,device=dev,dtype=torch.float64);cor=torch.zeros_like(sup)
 forward=m.module if not train and isinstance(m,DDP) else m
 with m.join() if train and isinstance(m,DDP) else nullcontext():
  for x,v,n in dl:
   x,v,n=x.to(dev),v.to(dev),n.to(dev);a=torch.tensor([amap.get((int(i),int(j)),-1) for i,j in zip(v.cpu(),n.cpu())],device=dev);vl,nl,al,consistency=forward(x);ok=a>=0;weighted=F.cross_entropy(nl,n,weight=weights);noun_loss=.5*F.cross_entropy(nl,n)+.5*weighted;loss=F.cross_entropy(vl,v)+noun_loss+.2*(F.cross_entropy(al[ok],a[ok]) if ok.any() else x.sum()*0.)+.02*consistency
   if train:opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
   tv,tn=vl.topk(5,1).indices,nl.topk(5,1).indices;b=len(v);s+=torch.tensor([loss.detach().item()*b,b,int(ok.sum()),(vl.argmax(1)==v).sum(),(tv==v[:,None]).any(1).sum(),(nl.argmax(1)==n).sum(),(tn==n[:,None]).any(1).sum(),(al[ok].argmax(1)==a[ok]).sum() if ok.any() else 0,consistency.detach().item()*b],device=dev,dtype=torch.float64);sup+=torch.bincount(n,minlength=nc);cor+=torch.bincount(n[nl.argmax(1)==n],minlength=nc)
 if dist.is_initialized():dist.all_reduce(s);dist.all_reduce(sup);dist.all_reduce(cor)
 active=sup>0;c=max(float(s[1]),1);ac=max(float(s[2]),1);return {'loss':float(s[0]/c),'slot_consistency':float(s[8]/c),'verb_top1':float(s[3]/c),'verb_top5':float(s[4]/c),'noun_top1':float(s[5]/c),'noun_top5':float(s[6]/c),'action_top1':float(s[7]/ac),'noun_balanced_accuracy':float((cor[active]/sup[active]).mean())}
def main():
 ctx=task_context();p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=239);p.add_argument('--epochs',type=int,default=30);apply_cli_defaults(p,ctx);a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());r,w,dev=setup();torch.manual_seed(a.seed+r);np.random.seed(a.seed+r);tr,va=GridDataset(a.train,r,w),GridDataset(a.validation,r,w);rows=read_rows(cfg['data']['train_annotations']);vv=sorted({int(x['verb_class']) for x in rows});nv=sorted({int(x['noun_class']) for x in rows});tr.v,tr.n=encode_labels(tr.v,vv),encode_labels(tr.n,nv);va.v,va.n=encode_labels(va.v,vv),encode_labels(va.n,nv);local_pairs=list({(int(x),int(y)) for x,y in zip(tr.v.tolist(),tr.n.tolist())})
 if dist.is_initialized():
  gathered=[None]*w;dist.all_gather_object(gathered,local_pairs);pairs=sorted({pair for part in gathered for pair in part})
 else:pairs=sorted(local_pairs)
 amap={x:i for i,x in enumerate(pairs)};counts=torch.bincount(tr.n,minlength=len(nv)).float().to(dev)
 if dist.is_initialized():dist.all_reduce(counts)
 weights=((1-.9999)/(1-.9999**counts)).clamp(max=5);weights/=weights.mean();m=EK100MultiScaleReadout(len(vv),len(nv),pairs).to(dev);m=DDP(m,device_ids=[dev.index]) if w>1 else m;tl=DataLoader(tr,64,shuffle=True);vl=DataLoader(va,64);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=5e-4);out=Path(a.output);best=(-1.,-1.);hist=[]
 ensure_no_artifacts(out)
 if r==0:out.mkdir(parents=True,exist_ok=True)
 if dist.is_initialized():dist.barrier()
 for e in range(1,a.epochs+1):
  tm,vm=run(m,tl,dev,amap,len(nv),weights,True,opt),run(m,vl,dev,amap,len(nv),weights);hist.append({'epoch':e,'train':tm,'validation':vm});score=(.5*(vm['verb_top1']+vm['noun_top1']),vm['noun_balanced_accuracy'])
  if r==0:
   (out/'history.json').write_text(json.dumps(hist,indent=2))
   if score>best:best=score;z=m.module if hasattr(m,'module') else m;torch.save({'model':z.state_dict(),'verb_vocabulary':vv,'noun_vocabulary':nv,'verb_classes':len(vv),'noun_classes':len(nv),'action_map':amap,'validation':vm},out/'best.pt')
 if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
