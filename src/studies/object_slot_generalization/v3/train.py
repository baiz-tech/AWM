from __future__ import annotations
import argparse,json,os
from pathlib import Path
import torch,torch.distributed as dist,torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from .model import GroundedObjectSlots,object_state_loss,self_supervised_loss
PHYSION_SCALE=torch.tensor([.20,.08,.01,.12,.06,.02,.09,.08,.12])
class Physion(Dataset):
 def __init__(self,p):self.f=sorted(Path(p).glob('sample_*.pt'))
 def __len__(self):return len(self.f)
 def __getitem__(self,i):
  z=torch.load(self.f[i],map_location='cpu',weights_only=False);return z['context'].float(),z['current_object_state'].float(),z['current_object_valid'].bool(),torch.tensor(-1)
class EK(Dataset):
 def __init__(self,p):
  f=sorted(Path(p).glob('features-rank*.pt'));r=int(os.environ.get('RANK',0));z=torch.load(f[r%len(f)],map_location='cpu',weights_only=False);self.x,self.n=z['grid'].float(),z['noun'].long()
 def __len__(self):return len(self.n)
 def __getitem__(self,i):return self.x[i],torch.zeros(8,9),torch.zeros(8,dtype=torch.bool),self.n[i]
def setup():
 if 'RANK' not in os.environ:return 0,1,torch.device('cuda')
 r,w,l=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']),int(os.environ['LOCAL_RANK']);torch.cuda.set_device(l);dist.init_process_group('nccl');return r,w,torch.device('cuda',l)
def main():
 p=argparse.ArgumentParser();p.add_argument('--physion',required=True);p.add_argument('--ektrain',required=True);p.add_argument('--ekval',required=True);p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch',type=int,default=8);a=p.parse_args();r,w,d=setup();pt,et,ev=Physion(a.physion),EK(a.ektrain),EK(a.ekval);ps=DistributedSampler(pt,w,r,shuffle=True) if w>1 else None;pl=DataLoader(pt,a.batch,sampler=ps,shuffle=ps is None,num_workers=1);el=DataLoader(et,a.batch,shuffle=True,num_workers=1);vl=DataLoader(ev,a.batch,num_workers=1);classes=torch.tensor([max(int(et.n.max()),int(ev.n.max()))+1],device=d);dist.all_reduce(classes,op=dist.ReduceOp.MAX) if w>1 else None;n=int(classes.item());m=GroundedObjectSlots().to(d);h=torch.nn.Linear(256,n).to(d)
 if w>1:m=DDP(m,device_ids=[d.index],find_unused_parameters=True);h=DDP(h,device_ids=[d.index],find_unused_parameters=True)
 opt=torch.optim.AdamW(list(m.parameters())+list(h.parameters()),lr=1e-4,weight_decay=1e-4);out=Path(a.out);out.mkdir(parents=True,exist_ok=True) if r==0 else None;dist.barrier() if w>1 else None;hist=[];scale=PHYSION_SCALE.to(d)
 for e in range(1,a.epochs+1):
  if ps:ps.set_epoch(e)
  m.train();h.train();loss=cor=cnt=0
  for name,dl in [('p',pl),('e',el)]:
   for x,state,valid,noun in dl:
    x,state,valid,noun=x.to(d),state.to(d),valid.to(d),noun.to(d);o=m(x);l=self_supervised_loss(o,x)
    if name=='p':l=l+object_state_loss(o,state,valid,scale)['total']
    else:q=h(o.slots.amax(2).mean(1));l=l+F.cross_entropy(q,noun);cor+=int((q.argmax(1)==noun).sum());cnt+=len(noun)
    opt.zero_grad(set_to_none=True);l.backward();torch.nn.utils.clip_grad_norm_(list(m.parameters())+list(h.parameters()),1);opt.step();loss+=float(l.detach())
  m.eval();h.eval();vc=vn=0
  with torch.no_grad():
   for x,_,_,noun in vl:x,noun=x.to(d),noun.to(d);q=h(m(x).slots.amax(2).mean(1));vc+=int((q.argmax(1)==noun).sum());vn+=len(noun)
  stat=torch.tensor([loss,cor,cnt,vc,vn],device=d,dtype=torch.float64);dist.all_reduce(stat) if w>1 else None
  if r==0:
   rec={'epoch':e,'loss':float(stat[0]),'train_noun_top1':float(stat[1]/stat[2].clamp_min(1)),'validation_noun_top1':float(stat[3]/stat[4].clamp_min(1))};hist.append(rec);print(rec,flush=True);(out/'history.json').write_text(json.dumps(hist,indent=2)+'\n');mm=m.module if hasattr(m,'module') else m;hh=h.module if hasattr(h,'module') else h;torch.save({'model':mm.state_dict(),'noun_head':hh.state_dict(),'epoch':e,'protocol':'ogs_v3_hungarian'},out/'latest.pt')
 if w>1:dist.destroy_process_group()
if __name__=='__main__':main()
