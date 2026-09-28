"""VideoMAEv2-style end-to-end EK100 fine-tuning (train/probe/final)."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,torch,yaml
from torch.utils.data import DataLoader
from .dataset import EK100Dataset,split_indices,read_rows
from .model import build_encoder,DirectHeads

def topk(z,y,k): return float((z.topk(min(k,z.size(1)),1).indices==y[:,None]).any(1).float().mean())
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',required=True); p.add_argument('--epochs',type=int); p.add_argument('--batch-size',type=int); p.add_argument('--lr',type=float); a=p.parse_args(); c=yaml.safe_load(Path(a.config).read_text()); seed=int(c.get('seed',239)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); d=c['data']; train_rows=read_rows(d['train_annotations']); tr_ids,va_ids=split_indices(d['train_annotations'],seed=seed); tr=EK100Dataset(d['train_annotations'],d['video_root'],tr_ids,d.get('crop_size',224)); va=EK100Dataset(d['train_annotations'],d['video_root'],va_ids,d.get('crop_size',224)); final=EK100Dataset(d['validation_annotations'],d['video_root'],None,d.get('crop_size',224)); verbs=sorted({int(x['verb_class']) for x in train_rows}); nouns=sorted({int(x['noun_class']) for x in train_rows}); vm={x:i for i,x in enumerate(verbs)}; nm={x:i for i,x in enumerate(nouns)}; pairs=sorted({(vm[int(x['verb_class'])],nm[int(x['noun_class'])]) for x in train_rows}); pm={x:i for i,x in enumerate(pairs)}; enc=build_encoder(c['model'],dev); enc.head= torch.nn.Identity(); model=DirectHeads(enc.embed_dim,len(verbs),len(nouns),len(pairs)).to(dev); model.encoder=enc; opt=torch.optim.AdamW(model.parameters(),lr=a.lr or c['training'].get('learning_rate',1e-4),weight_decay=c['training'].get('weight_decay',.05)); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs or c['training'].get('epochs',30)); bs=a.batch_size or c['training'].get('finetune_batch_size',1)
 def run(ds,training=False):
  model.train(training); sums=[0.,0.,0.,0.];
  for b in DataLoader(ds,bs,shuffle=training,num_workers=d.get('num_workers',4),pin_memory=True):
   yv=torch.tensor([vm[int(x)] for x in b['verb']],device=dev); yn=torch.tensor([nm[int(x)] for x in b['noun']],device=dev); ya=torch.tensor([pm[(int(v),int(n))] for v,n in zip(yv.tolist(),yn.tolist())],device=dev); z=model.encoder.patch_embed(b['video'].to(dev)); z=z+model.encoder.pos_embed.to(z); z=model.encoder.pos_drop(z)
   for block in model.encoder.blocks: z=block(z)
   feat=model.encoder.fc_norm(z.mean(1)); q=model.verb(feat),model.noun(feat),model.action(feat); loss=torch.nn.functional.cross_entropy(q[0],yv)+torch.nn.functional.cross_entropy(q[1],yn)+.2*torch.nn.functional.cross_entropy(q[2],ya)
   if training: opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
   sums[0]+=float(loss)*len(yv); sums[1]+=len(yv); sums[2]+=float((q[0].argmax(1)==yv).sum()); sums[3]+=float((q[1].argmax(1)==yn).sum())
  return {'loss':sums[0]/sums[1],'verb_top1':sums[2]/sums[1],'noun_top1':sums[3]/sums[1]}
 epochs=a.epochs or c['training'].get('epochs',30); best=-1; state=None; hist=[]
 for e in range(1,epochs+1):
  tm=run(tr,True); vv=run(va); sch.step(); score=(vv['verb_top1']+vv['noun_top1'])/2; hist.append({'epoch':e,'train':tm,'validation':vv,'lr':sch.get_last_lr()[0]}); print(json.dumps(hist[-1]),flush=True)
  if score>best: best=score; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(state); fm=run(final); result={'protocol':'videomaev2_ek100_official_style_finetune','final_validation':fm,'train_samples':len(tr),'probe_validation_samples':len(va),'official_validation_samples':len(final),'encoder_finetuned':True}; out=Path(a.output); out.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'verb_vocabulary':verbs,'noun_vocabulary':nouns,'action_map':pm,'protocol':result['protocol']},out/'best.pt'); (out/'history.json').write_text(json.dumps(hist,indent=2)+'\n'); (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
