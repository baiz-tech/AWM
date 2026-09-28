#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,random,time
from pathlib import Path
import numpy as np,torch,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
from recipe.shared.distributed import ExactDistributedSampler
from .model_target_only import TargetOnlyDecoder,target_loss
KEYS=('state_2d','state_valid','object_present','ocp_label','ocp_valid')
def cname(p):return hashlib.sha1(str(Path(p).resolve()).encode()).hexdigest()[:16]+'.pt'
class DS(Dataset):
 def __init__(self,cache,targets,split):
  t=torch.load(targets,map_location='cpu',weights_only=False);self.files=[];self.t={k:t[k] for k in KEYS}
  for p in t['path']:
   f=Path(cache)/split/cname(p)
   if not f.is_file():raise FileNotFoundError(f)
   self.files.append(f)
 def __len__(self):return len(self.files)
 def __getitem__(self,i):
  r=torch.load(self.files[i],map_location='cpu',weights_only=True);return {'context':r['context_tokens'],'future':r['future_tokens'],**{k:v[i] for k,v in self.t.items()}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--train-targets',type=Path,required=True);p.add_argument('--validation-targets',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--num-workers',type=int,default=2);p.add_argument('--seed',type=int,default=239);a=p.parse_args();rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local);dist.init_process_group('nccl') if world>1 else None;random.seed(a.seed+rank);np.random.seed(a.seed+rank);torch.manual_seed(a.seed+rank);dev=torch.device(f'cuda:{local}');tr=DS(a.cache_root,a.train_targets,'data_v1');va=DS(a.cache_root,a.validation_targets,'readout_data_v1');ts=DistributedSampler(tr,num_replicas=world,rank=rank,seed=a.seed);vs=ExactDistributedSampler(va,rank=rank,world_size=world);opts={'batch_size':a.batch_size,'num_workers':a.num_workers,'pin_memory':True};tl=DataLoader(tr,sampler=ts,**opts);vl=DataLoader(va,sampler=vs,**opts);m=TargetOnlyDecoder().to(dev);m=DDP(m,device_ids=[local]) if world>1 else m;opt=torch.optim.AdamW(m.parameters(),2e-4,weight_decay=.04);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs);a.output_dir.mkdir(parents=True,exist_ok=True);best=float('inf');hist=[]
 for ep in range(1,a.epochs+1):
  start=time.time();ts.set_epoch(ep);m.train();s=torch.zeros(5,dtype=torch.float64,device=dev)
  for b in tl:
   b={k:v.to(dev) for k,v in b.items()};opt.zero_grad(set_to_none=True);v,x=target_loss(m(b['context'],b['future']),b);v.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();bs=b['context'].size(0);s+=torch.tensor([float(x['total']),float(x['presence']),float(x['geometry']),float(x['ocp']),bs],device=dev,dtype=torch.float64)*torch.tensor([bs,bs,bs,bs,1],device=dev)
  if world>1:dist.all_reduce(s)
  sch.step();m.eval();q=torch.zeros(5,dtype=torch.float64,device=dev)
  with torch.no_grad():
   for b in vl:
    b={k:v.to(dev) for k,v in b.items()};v,x=target_loss(m(b['context'],b['future']),b);bs=b['context'].size(0);q+=torch.tensor([float(x['total']),float(x['presence']),float(x['geometry']),float(x['ocp']),bs],device=dev,dtype=torch.float64)*torch.tensor([bs,bs,bs,bs,1],device=dev)
  if world>1:dist.all_reduce(q)
  row={'epoch':ep,'seconds':time.time()-start,'train':dict(zip(('total','presence','geometry','ocp'),(s[:4]/s[4]).tolist())),'validation':dict(zip(('total','presence','geometry','ocp'),(q[:4]/q[4]).tolist()))};hist.append(row)
  if rank==0:
   print(json.dumps(row),flush=True);mod=m.module if isinstance(m,DDP) else m;pay={'protocol':'physionpp_target_object_only_decoder_v1','epoch':ep,'model':mod.state_dict(),'model_config':{},'validation':row,'seed':a.seed};torch.save(pay,a.output_dir/'latest.pt');
   if row['validation']['total']<best:best=row['validation']['total'];torch.save(pay,a.output_dir/'best.pt')
   (a.output_dir/'history.json').write_text(json.dumps(hist,indent=2)+'\n')
 if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
