#!/usr/bin/env python3
"""Train the Physion++ decoder and structured probe with DDP."""
from __future__ import annotations
import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np,torch,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
from src.training.distributed import ExactDistributedSampler
try:
    from .model import PhysionDecoder, loss
except ImportError:
    from .model import PhysionDecoder, loss

KEYS=('object_present','state_2d','state_valid','pair_distance_2d','pair_valid','contact','contact_valid','first_contact_class','ocp_label','ocp_valid')
def cache_name(path):return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:16]+'.pt'
class DS(Dataset):
 def __init__(self,cache,targets,split,max_videos=None):
  self.cache=Path(cache)/split; t=torch.load(targets,map_location='cpu',weights_only=False); n=len(t['path']); keep=torch.arange(n)<(int(max_videos) if max_videos else n); self.paths=[t['path'][i] for i in range(n) if keep[i]];self.t={k:v[keep] for k,v in t.items() if k in KEYS}; self.files=[]
  for i,path in enumerate(self.paths):
   f=self.cache/cache_name(path)
   if not f.is_file():raise FileNotFoundError(f'missing cache: {f}')
   self.files.append(f)
 def __len__(self):return len(self.files)
 def __getitem__(self,i):
  r=torch.load(self.files[i],map_location='cpu',weights_only=True);return {'context':r['context_tokens'],'future':r['future_tokens'],**{k:v[i] for k,v in self.t.items()}}
def setup(seed):
 if 'RANK' in os.environ:
  rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local);dist.init_process_group('nccl')
 else:rank=0;world=1;local=0
 random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank);return rank,world,local
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--train-targets',type=Path,required=True);p.add_argument('--validation-targets',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--num-workers',type=int,default=2);p.add_argument('--seed',type=int,default=239);p.add_argument('--learning-rate',type=float,default=2e-4);p.add_argument('--weight-decay',type=float,default=.04);p.add_argument('--max-train-videos',type=int);p.add_argument('--max-validation-videos',type=int);a=p.parse_args();rank,world,local=setup(a.seed);device=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu');train=DS(a.cache_root,a.train_targets,'data_v1',a.max_train_videos);val=DS(a.cache_root,a.validation_targets,'readout_data_v1',a.max_validation_videos);ts=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=a.seed);vs=ExactDistributedSampler(val,rank=rank,world_size=world);opts={'batch_size':a.batch_size,'num_workers':a.num_workers,'pin_memory':device.type=='cuda','persistent_workers':a.num_workers>0};tl=DataLoader(train,sampler=ts,**opts);vl=DataLoader(val,sampler=vs,**opts);model=PhysionDecoder().to(device);model=DDP(model,device_ids=[local]) if world>1 else model;opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs);a.output_dir.mkdir(parents=True,exist_ok=True);best=float('inf');history=[]
 metric_names=('total','presence','center','geometry','velocity','distance','contact','first_contact','ocp')
 for epoch in range(1,a.epochs+1):
  epoch_start=time.time(); train_sums={k:0.0 for k in metric_names}; train_count=0; train_batches=0; train_no_valid=0; train_valid_objects=0; train_valid_frames=0
  ts.set_epoch(epoch);model.train();total=0.;count=0
  for b in tl:
   b={k:v.to(device) for k,v in b.items()};opt.zero_grad(set_to_none=True);o=model(b['context'],b['future']);value,metrics=loss(o,b)
   if not torch.isfinite(value): raise FloatingPointError(f'non-finite loss at epoch={epoch}, batch={train_batches}')
   value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();bs=b['context'].size(0);train_count+=bs;train_batches+=1
   for k in metric_names: train_sums[k]+=float(metrics[k].detach())*bs
   valid_per_sample=(b['state_valid'] & b['object_present'][:,:,None]).any((1,2));train_no_valid+=int((~valid_per_sample).sum());train_valid_objects+=int((b['object_present'] & b['state_valid'].any(2)).sum());train_valid_frames+=int((b['state_valid'] & b['object_present'][:,:,None]).sum())
  train_stats=torch.tensor([train_sums[k] for k in metric_names]+[train_count,train_batches,train_no_valid,train_valid_objects,train_valid_frames],dtype=torch.float64,device=device)
  if dist.is_initialized():dist.all_reduce(train_stats)
  sch.step();model.eval(); val_sums={k:0.0 for k in metric_names};val_count=0;val_batches=0;val_no_valid=0;val_valid_objects=0;val_valid_frames=0
  with torch.no_grad():
   for b in vl:
    b={k:v.to(device) for k,v in b.items()};value,metrics=loss(model(b['context'],b['future']),b);bs=b['context'].size(0);val_count+=bs;val_batches+=1
    for k in metric_names: val_sums[k]+=float(metrics[k])*bs
    valid_per_sample=(b['state_valid'] & b['object_present'][:,:,None]).any((1,2));val_no_valid+=int((~valid_per_sample).sum());val_valid_objects+=int((b['object_present'] & b['state_valid'].any(2)).sum());val_valid_frames+=int((b['state_valid'] & b['object_present'][:,:,None]).sum())
  validation_stats=torch.tensor([val_sums[k] for k in metric_names]+[val_count,val_batches,val_no_valid,val_valid_objects,val_valid_frames],dtype=torch.float64,device=device)
  if dist.is_initialized():dist.all_reduce(validation_stats)
  tn=train_stats[len(metric_names)];vn=validation_stats[len(metric_names)];row={'epoch':epoch,'learning_rate':sch.get_last_lr()[0],'seconds':time.time()-epoch_start,'train':{k:float(train_stats[i]/tn.clamp_min(1)) for i,k in enumerate(metric_names)},'validation':{k:float(validation_stats[i]/vn.clamp_min(1)) for i,k in enumerate(metric_names)},'train_stats':{'samples':int(tn),'batches':int(train_stats[len(metric_names)+1]),'samples_without_valid_box':int(train_stats[len(metric_names)+2]),'objects_with_valid_box':int(train_stats[len(metric_names)+3]),'valid_object_frames':int(train_stats[len(metric_names)+4])},'validation_stats':{'samples':int(vn),'batches':int(validation_stats[len(metric_names)+1]),'samples_without_valid_box':int(validation_stats[len(metric_names)+2]),'objects_with_valid_box':int(validation_stats[len(metric_names)+3]),'valid_object_frames':int(validation_stats[len(metric_names)+4])}};row['train_loss']=row['train']['total'];row['validation_loss']=row['validation']['total'];history.append(row)
  if rank==0: print(json.dumps(row,sort_keys=True),flush=True)
  if rank==0:
   module=model.module if isinstance(model,DDP) else model;payload={'protocol':'physionpp_fullpatch_dynamics_decoder_v1','epoch':epoch,'model':module.state_dict(),'model_config':{'input_dim':1280,'hidden_dim':256,'num_heads':8,'ffn_dim':1024,'num_slots':8},'validation':row,'seed':a.seed};torch.save(payload,a.output_dir/'latest.pt');
   if row['validation_loss']<best:best=row['validation_loss'];torch.save(payload,a.output_dir/'best.pt')
   (a.output_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n')
 if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
