#!/usr/bin/env python3
"""Train the official-style three-query attentive EK100 readout on Orca tokens."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from external.vjepa2.src.models.attentive_pooler import AttentivePooler
from src.core.run_context import apply_cli_defaults, task_context

class Cached(Dataset):
    def __init__(self, root, vm, nm, pm): self.files=sorted(Path(root).glob('*.pt')); self.vm=vm; self.nm=nm; self.pm=pm
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        r=torch.load(self.files[i],map_location='cpu',weights_only=True); v=self.vm.get(r['verb'],-1); n=self.nm.get(r['noun'],-1)
        return r['video_tokens'].float(),v,n,self.pm.get((v,n),-1)

class Readout(nn.Module):
    def __init__(self,dim,nv,nnoun,na,heads,depth):
        super().__init__(); self.pooler=AttentivePooler(num_queries=3,embed_dim=dim,num_heads=heads,depth=depth,use_activation_checkpointing=False); self.verb=nn.Linear(dim,nv); self.noun=nn.Linear(dim,nnoun); self.action=nn.Linear(dim,na)
    def forward(self,x): q=self.pooler(x); return self.verb(q[:,0]),self.noun(q[:,1]),self.action(q[:,2])

def scan(root):
    files=sorted(Path(root).glob('*.pt'))
    if not files: raise FileNotFoundError(root)
    rows=[]
    for f in files:
        r=torch.load(f,map_location='cpu',weights_only=True)
        rows.append({'verb':r['verb'],'noun':r['noun'],'video_tokens':r['video_tokens'][:1]})
    return rows
def main():
    ctx=task_context();p=argparse.ArgumentParser(); p.add_argument('--cache-root',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--workers',type=int,default=4); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--num-heads',type=int,default=8); p.add_argument('--depth',type=int,default=2); p.add_argument('--seed',type=int,default=239); apply_cli_defaults(p,ctx);a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta=scan(a.cache_root/'train'); verbs=sorted({r['verb'] for r in meta}); nouns=sorted({r['noun'] for r in meta}); vm={v:i for i,v in enumerate(verbs)}; nm={n:i for i,n in enumerate(nouns)}; pairs=sorted({(vm[r['verb']],nm[r['noun']]) for r in meta}); pm={x:i for i,x in enumerate(pairs)}; dim=int(meta[0]['video_tokens'].shape[-1]); del meta
    datasets={s:Cached(a.cache_root/s,vm,nm,pm) for s in ('train','probe_validation','final_validation')}; loaders={s:DataLoader(d,batch_size=a.batch_size,shuffle=s=='train',num_workers=a.workers,pin_memory=True) for s,d in datasets.items()}
    net=Readout(dim,len(verbs),len(nouns),len(pairs),a.num_heads,a.depth).to(device); opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=a.weight_decay); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs); out=a.output_dir; out.mkdir(parents=True,exist_ok=True); history=[]; best=-1.
    def run(split,train):
        net.train(train); sums=torch.zeros(9,device=device,dtype=torch.float64)
        with torch.enable_grad() if train else torch.no_grad():
          for x,yv,yn,ya in loaders[split]:
            x,yv,yn,ya=x.to(device),yv.to(device),yn.to(device),ya.to(device); z=net(x); loss=nn.functional.cross_entropy(z[0],yv)+nn.functional.cross_entropy(z[1],yn)+.2*nn.functional.cross_entropy(z[2],ya.clamp_min(0))
            if train: opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
            def hits(q,y): h=q.topk(min(5,q.shape[1]),1).indices.eq(y[:,None]); return h[:,0].sum(),h.any(1).sum()
            v1,v5=hits(z[0],yv); n1,n5=hits(z[1],yn); m=ya.ge(0); a1,a5=hits(z[2][m],ya[m]) if m.any() else (ya.new_tensor(0),ya.new_tensor(0)); sums+=torch.tensor([float(loss)*len(yv),len(yv),v1,v5,n1,n5,a1,a5,m.sum()],device=device)
        d=sums[1].clamp_min(1); ad=sums[8].clamp_min(1); return {'loss':float(sums[0]/d),'verb_top1':float(sums[2]/d),'verb_top5':float(sums[3]/d),'noun_top1':float(sums[4]/d),'noun_top5':float(sums[5]/d),'action_top1':float(sums[6]/ad),'action_top5':float(sums[7]/ad),'action_valid_samples':int(sums[8])}
    for epoch in range(1,a.epochs+1):
        tr=run('train',True); va=run('probe_validation',False); sch.step(); row={'epoch':epoch,'train':tr,'validation':va,'lr':sch.get_last_lr()[0]}; history.append(row); (out/'history.json.tmp').write_text(json.dumps(history,indent=2)+'\n'); (out/'history.json.tmp').replace(out/'history.json'); print(json.dumps(row),flush=True); score=(va['verb_top1']+va['noun_top1'])/2
        if score>best: best=score; torch.save({'model':net.state_dict(),'verbs':verbs,'nouns':nouns,'action_map':pm,'best_epoch':epoch,'best_probe_score':best},out/'best.pt')
    ck=torch.load(out/'best.pt',map_location=device,weights_only=False); net.load_state_dict(ck['model']); final=run('final_validation',False); result={'protocol':'orca_ek100_frozen_tokens_attentive_readout','encoder':'Orca-4B','encoder_frozen':True,'readout':'AttentivePooler(3 queries)+verb/noun/action heads','best_epoch':ck['best_epoch'],'best_probe_score':best,'final_validation':final,'train_samples':len(datasets['train']),'probe_validation_samples':len(datasets['probe_validation']),'official_validation_samples':len(datasets['final_validation'])}; (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()
