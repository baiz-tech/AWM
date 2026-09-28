from __future__ import annotations
import argparse,json
from pathlib import Path
import torch, yaml
from .model import DirectHeads
from src.core.run_context import apply_cli_defaults, task_context

def load(path):
 d=[torch.load(p,map_location='cpu',weights_only=False) for p in sorted(Path(path).glob('features-*.pt'))]; return torch.cat([x['features'].float() for x in d]),torch.cat([x['verb'] for x in d]),torch.cat([x['noun'] for x in d])
def topk(logits,y,k): return (logits.topk(k,1).indices==y[:,None]).any(1).float().mean().item()
def main():
 ctx=task_context();p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--train',required=True); p.add_argument('--validation',required=True); p.add_argument('--final',required=True); p.add_argument('--output',required=True); p.add_argument('--epochs',type=int); apply_cli_defaults(p,ctx);a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); tx,tv,tn=load(a.train); vx,vv,vn=load(a.validation); ex,ev,en=load(a.final); verbs=sorted(set(tv.tolist())); nouns=sorted(set(tn.tolist())); vm={x:i for i,x in enumerate(verbs)}; nm={x:i for i,x in enumerate(nouns)}; tv=torch.tensor([vm[int(x)] for x in tv]); vv=torch.tensor([vm.get(int(x),-1) for x in vv]); ev=torch.tensor([vm.get(int(x),-1) for x in ev]); tn=torch.tensor([nm[int(x)] for x in tn]); vn=torch.tensor([nm.get(int(x),-1) for x in vn]); en=torch.tensor([nm.get(int(x),-1) for x in en]); pairs=sorted({(int(v),int(n)) for v,n in zip(tv.tolist(),tn.tolist())}); pm={p:i for i,p in enumerate(pairs)}; ta=torch.tensor([pm[(int(v),int(n))] for v,n in zip(tv.tolist(),tn.tolist())]); va=torch.tensor([pm.get((int(v),int(n)),-1) for v,n in zip(vv.tolist(),vn.tolist())]); ea=torch.tensor([pm.get((int(v),int(n)),-1) for v,n in zip(ev.tolist(),en.tolist())]); model=DirectHeads(tx.size(1),len(verbs),len(nouns),len(pairs)); opt=torch.optim.AdamW(model.parameters(),lr=cfg['training'].get('learning_rate',.001),weight_decay=cfg['training'].get('weight_decay',.001)); best=-1; state=None; hist=[]
 epochs = a.epochs if a.epochs is not None else int(cfg['training'].get('epochs',30))
 for epoch in range(1,epochs+1):
  model.train(); z=model(tx); ok=ta>=0; loss=torch.nn.functional.cross_entropy(z[0],tv)+torch.nn.functional.cross_entropy(z[1],tn)+.2*torch.nn.functional.cross_entropy(z[2][ok],ta[ok]); opt.zero_grad(); loss.backward(); opt.step(); model.eval();
  with torch.no_grad(): q=model(vx); score=(topk(q[0],vv,1)+topk(q[1],vn,1))/2
  hist.append({'epoch':epoch,'train_loss':float(loss),'validation_verb_top1':topk(q[0],vv,1),'validation_noun_top1':topk(q[1],vn,1)}); 
  if score>best: best=score; state={k:v.clone() for k,v in model.state_dict().items()}
 model.load_state_dict(state); model.eval()
 with torch.no_grad(): q=model(ex)
 result={'protocol':'videomaev2_ek100_direct_heads','verb_top1':topk(q[0],ev,1),'verb_top5':topk(q[0],ev,5),'noun_top1':topk(q[1],en,1),'noun_top5':topk(q[1],en,5),'action_top1':topk(q[2][ea>=0],ea[ea>=0],1),'action_top5':topk(q[2][ea>=0],ea[ea>=0],5),'train_samples':len(tv),'validation_samples':len(vv),'final_validation_samples':len(ev)}; out=Path(a.output); out.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'verb_vocabulary':verbs,'noun_vocabulary':nouns,'action_map':pm,'protocol':result['protocol']},out/'best.pt'); (out/'history.json').write_text(json.dumps(hist,indent=2)+'\n'); (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
