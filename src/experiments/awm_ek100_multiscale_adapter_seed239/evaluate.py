#!/usr/bin/env python3
"""Official EK100 test evaluation for V23 shared residual state."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from src.data.ek100.labels import encode_labels
from .train_readout import GridDataset
from .model import EK100OnlyWorldModel
from src.core.run_context import apply_cli_defaults, task_context
def topk(x,y,k):return (x.topk(min(k,x.size(1)),1).indices==y[:,None]).any(1)
def main():
 ctx=task_context();p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--readout',required=True);p.add_argument('--features',required=True);p.add_argument('--output',required=True);apply_cli_defaults(p,ctx);a=p.parse_args();out=Path(a.output)
 if out.exists():raise FileExistsError(out)
 ek=torch.load(a.readout,map_location='cpu',weights_only=False);pairs=[None]*len(ek['action_map'])
 for pair,i in ek['action_map'].items():pairs[i]=pair
 d=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');m=EK100OnlyWorldModel(ek['verb_classes'],ek['noun_classes'],pairs).to(d);m.readout.load_state_dict(ek['model']);m.shared.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['shared']);m.eval();ds=GridDataset(a.features);vs=[];ns=[];acts=[]
 with torch.no_grad():
  for s in range(0,len(ds),64):
   v,n,ac,_=m(ds.x[s:s+64].to(d));vs.append(v.cpu());ns.append(n.cpu());acts.append(ac.cpu())
 v,n,ac=torch.cat(vs),torch.cat(ns),torch.cat(acts);verb=encode_labels(ds.v,ek['verb_vocabulary']);noun=encode_labels(ds.n,ek['noun_vocabulary']);action=torch.tensor([ek['action_map'].get((int(x),int(y)),-1) for x,y in zip(verb,noun)]);ok=action>=0;result={'protocol':'v23_dualdomain_shared_highres_world_ek100','checkpoint':str(Path(a.checkpoint).resolve()),'samples':len(ds),'verb_top1':float((v.argmax(1)==verb).float().mean()),'verb_top5':float(topk(v,verb,5).float().mean()),'noun_top1':float((n.argmax(1)==noun).float().mean()),'noun_top5':float(topk(n,noun,5).float().mean()),'action_top1':float((ac.argmax(1)[ok]==action[ok]).float().mean()),'action_top5':float(topk(ac[ok],action[ok],5).float().mean()),'unknown_noun_count':int((noun<0).sum())};out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
