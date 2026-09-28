#!/usr/bin/env python3
"""Evaluate Physion++ slot correspondence and probe semantic readouts."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from .model import PhysionDecoder, match_objects
from .model_current_only import CurrentOnlyPhysionDecoder
from src.core.run_context import apply_cli_defaults, task_context

def cname(path): return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:16]+'.pt'

class DS(Dataset):
    def __init__(self, cache_root, targets, split):
        t=torch.load(targets,map_location='cpu',weights_only=False); self.t=t; self.files=[Path(cache_root)/split/cname(p) for p in t['path']]
        missing=[str(x) for x in self.files if not x.is_file()]
        if missing: raise FileNotFoundError(f'missing latent cache: {missing[0]}')
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        r=torch.load(self.files[i],map_location='cpu',weights_only=True)
        keys=('object_present','state_2d','state_valid','object_type','color_rgb','is_target','pair_distance_2d','pair_valid','contact','contact_valid','first_contact_class')
        return r['context_tokens'],r['future_tokens'],{k:self.t[k][i] for k in keys}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache-root',type=Path,required=True); p.add_argument('--targets',type=Path,required=True); p.add_argument('--split',default='readout_data_v1'); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--num-workers',type=int,default=2); p.add_argument('--max-samples',type=int); p.add_argument('--model-kind',choices=('full','current_only'),default='full'); apply_cli_defaults(p,task_context()); a=p.parse_args()
    t=torch.load(a.targets,map_location='cpu',weights_only=False); vocab=t.get('object_type_vocab',['unknown']); ds=DS(a.cache_root,a.targets,a.split); 
    if a.max_samples: ds.files=ds.files[:a.max_samples]
    loader=DataLoader(ds,batch_size=a.batch_size,num_workers=a.num_workers); payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False); model_cls=CurrentOnlyPhysionDecoder if a.model_kind=='current_only' else PhysionDecoder; model=model_cls(**payload.get('model_config',{})); model.load_state_dict(payload['model'],strict=True); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(dev).eval()
    total_slots=matched=presence_tp=presence_fp=presence_fn=0; target_ok=target_n=type_ok=type_n=0; color_sum=color_n=0.; center_sum=center_n=0.; geometry_sum=geometry_n=0.; velocity_sum=velocity_n=0.; slot_consistency_hits=identity_frames=switch_events=transition_count=0; confusion=torch.zeros(len(vocab),len(vocab),dtype=torch.long)
    with torch.no_grad():
      for context,future,b in loader:
        context,future=context.to(dev).float(),future.to(dev).float(); b={k:v.to(dev) for k,v in b.items()}; out=model(context) if a.model_kind=='current_only' else model(context,future); assign=match_objects(out,b); matched_mask=assign.ge(0); present=b['object_present'].bool(); total_slots+=matched_mask.numel(); matched+=int(matched_mask.sum()); pred_present=out['presence_logits'].sigmoid().ge(.5); presence_tp+=int((pred_present&matched_mask).sum()); presence_fp+=int((pred_present&~matched_mask).sum()); presence_fn+=int((~pred_present&matched_mask).sum())
        rows=matched_mask.nonzero(as_tuple=False)
        if len(rows):
          bi,pi=rows.T; ti=assign[bi,pi]; target=b['is_target'][bi,ti].bool(); target_pred=out['target_logits'][bi,pi].sigmoid().ge(.5); target_ok+=int((target_pred==target).sum()); target_n+=target.numel(); typ=b['object_type'][bi,ti].long(); typ_pred=out['object_type_logits'][bi,pi].argmax(-1); type_ok+=int((typ_pred==typ).sum()); type_n+=typ.numel();
          for x,y in zip(typ_pred.cpu(),typ.cpu()):
            if 0<=int(y)<len(vocab) and 0<=int(x)<len(vocab): confusion[int(y),int(x)]+=1
          color_sum+=float((out['color_pred'][bi,pi]-b['color_rgb'][bi,ti]).abs().sum()); color_n+=int(out['color_pred'][bi,pi].numel()); valid=b['state_valid'][bi,ti].bool(); pred=out['trajectory_2d'][bi,pi]; true=b['state_2d'][bi,ti]; center_sum+=float((pred[...,:2]-true[...,:2]).abs()[valid.unsqueeze(-1).expand_as(pred[...,:2])].sum()); center_n+=int(valid.sum())*2; geometry_sum+=float((pred[...,2:5]-true[...,2:5]).abs()[valid.unsqueeze(-1).expand_as(pred[...,2:5])].sum()); geometry_n+=int(valid.sum())*3; velocity_sum+=float((pred[...,5:7]-true[...,5:7]).abs()[valid.unsqueeze(-1).expand_as(pred[...,5:7])].sum()); velocity_n+=int(valid.sum())*2
          # For each matched slot, measure whether the nearest GT identity stays constant over time.
          for s,gt in zip(pi.tolist(),ti.tolist()):
            pv=b['state_valid'][bi[0],:,:].bool() if False else None
          for sample in range(context.size(0)):
            mm=matched_mask[sample]
            for slot in torch.nonzero(mm,as_tuple=False).flatten().tolist():
              gt=int(assign[sample,slot]); valid_gt=b['state_valid'][sample].bool(); pred_xy=out['trajectory_2d'][sample,slot,:,:2]; true_xy=b['state_2d'][sample,:,:,:2]; active=valid_gt[:, :].any(1) if valid_gt.ndim==2 else valid_gt
              if active.any():
                d=torch.linalg.vector_norm(pred_xy[:,None]-true_xy.permute(1,0,2),dim=-1); nearest=d[:,active].argmin(1); ids=torch.nonzero(active,as_tuple=False).flatten()[nearest]; slot_consistency_hits+=int((ids==gt).sum()); identity_frames+=ids.numel();
                if ids.numel()>1: switch_events+=int((ids[1:]!=ids[:-1]).sum()); transition_count+=ids.numel()-1
    precision=presence_tp/max(presence_tp+presence_fp,1); recall=presence_tp/max(presence_tp+presence_fn,1); result={'protocol':'physionpp3_interpretability_v2','checkpoint':str(a.checkpoint.resolve()),'targets':str(a.targets.resolve()),'split':a.split,'samples':len(ds),'object_slots_matched':matched,'total_slots':total_slots,'presence_f1':2*precision*recall/max(precision+recall,1e-12),'slot_occupancy_rate':matched/max(total_slots,1),'slot_identity_consistency':slot_consistency_hits/max(identity_frames,1),'slot_id_switch_count':switch_events/max(matched,1),'slot_id_switch_rate':switch_events/max(transition_count,1),'target_accuracy':target_ok/max(target_n,1),'object_type_accuracy':type_ok/max(type_n,1),'object_type_vocab':vocab,'object_type_confusion_matrix':confusion.tolist(),'color_rgb_mae':color_sum/max(color_n,1),'center_mae':center_sum/max(center_n,1),'geometry_mae':geometry_sum/max(geometry_n,1),'velocity_mae':velocity_sum/max(velocity_n,1)}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
